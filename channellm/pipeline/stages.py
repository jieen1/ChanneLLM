"""三阶段拓扑定义(设计文档 §4)。

Stage0 Thinker(LLM_AR)→ Stage1 Talker(LLM_AR)→ Stage2 Code2Wav(非自回归)。
单进程单卡共驻;跨阶段用进程内队列/共享内存,不做分布式(设计文档 §4 跨阶段传输)。
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class StageId(str, enum.Enum):
    THINKER = "thinker"
    TALKER = "talker"
    CODE2WAV = "code2wav"


PIPELINE_ORDER = (StageId.THINKER, StageId.TALKER, StageId.CODE2WAV)

# vllm-omni 基线参数(deploy/minicpmo_4_5.yaml,实测后校准)
CODEC_CHUNK_FRAMES = 25  # 首包延迟主旋钮
CODEC_LEFT_CONTEXT_FRAMES = 3
# 官方 duplex 首次 TTS 调用会 force-flush：有左上下文及至少 5 个新 codec
# token 即可送入同一个 Token2Wav stream，避免把首包硬等到完整 25 帧。
CODEC_INITIAL_MIN_AUDIO_FRAMES = 5
# MiniCPM-o 官方流式 Token2wav 在首个 codec phrase 前插入同数量的静音码，
# 使 25 个新帧立刻满足 25+3 的首块窗口。不能用文本 tokenizer 的 4218 推断；
# 这是官方 `modeling_minicpmo.py` 明示的 S3 codec silence code。
CODEC_STREAM_SILENCE_TOKEN = 4218


@dataclasses.dataclass(frozen=True)
class PipelineChunk:
    """发往某个 stage 的增量输入或其最终输出。

    ``stage`` 始终是消费者 stage。Code2Wav 的 chunk 同时也是编排层交给
    播放层的 PCM 产物；这样所有跨层数据都有同一组 epoch 标识，旧轮次不会
    因为队列中残留而穿透到播放端。
    """

    stage: StageId
    source: StageId | None = None
    payload: Any = None
    turn_epoch: int = 0
    speech_id: str = ""
    final: bool = False


@dataclasses.dataclass
class StageRequestState:
    """一个请求在三个引擎的统一身份(九件事 #1)。

    参考 vllm_omni/engine/orchestrator.py 的 OrchestratorRequestState:
    每个 stage 各有 request_id/replica_id,但共享同一个状态对象,
    否则请求会丢失、错乱。
    """

    request_id: str
    turn_epoch: int
    speech_id: str = ""
    stage_request_ids: dict[StageId, str] = dataclasses.field(default_factory=dict)
    finished_stages: set[StageId] = dataclasses.field(default_factory=set)
    cancelled: bool = False
    # Talker 产出的 codec token 必须攒到 Code2Wav 的处理单元。这个缓冲只
    # 属于该请求，绝不能跨 turn 复用。
    codec_buffer: list[Any] = dataclasses.field(default_factory=list)
    codec_prefix_seeded: bool = False
    initial_codec_flush_attempted: bool = False
    terminal_emitted: bool = False
