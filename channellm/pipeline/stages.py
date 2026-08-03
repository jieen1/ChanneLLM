"""三阶段拓扑定义(设计文档 §4)。

Stage0 Thinker(LLM_AR)→ Stage1 Talker(LLM_AR)→ Stage2 Code2Wav(非自回归)。
单进程单卡共驻;跨阶段用进程内队列/共享内存,不做分布式(设计文档 §4 跨阶段传输)。
"""

from __future__ import annotations

import dataclasses
import enum


class StageId(str, enum.Enum):
    THINKER = "thinker"
    TALKER = "talker"
    CODE2WAV = "code2wav"


PIPELINE_ORDER = (StageId.THINKER, StageId.TALKER, StageId.CODE2WAV)

# vllm-omni 基线参数(deploy/minicpmo_4_5.yaml,实测后校准)
CODEC_CHUNK_FRAMES = 25  # 首包延迟主旋钮
CODEC_LEFT_CONTEXT_FRAMES = 3


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
