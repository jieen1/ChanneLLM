"""延迟 trace schema —— P0 测量地基(设计文档 §7 延迟预算、§P0)。

每条记录是一个 (anchor, 时间戳) 观测,携带 trace_id / turn_epoch / speech_id,
序列化为 JSONL,由 channellm.metrics.latency 聚合成 waterfall。

时钟纪律:
- ts_ns 一律取 time.monotonic_ns() —— 单机单时钟域,延迟计算只认它。
- wall_ns 仅用于跨机对照与人工排查,禁止参与延迟数学。
- 禁止把各段 nominal 值相加当成总延迟(设计文档 R2)。
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


class Anchor:
    """全链路锚点,覆盖设计文档 §7 的每一段。"""

    # 采集与上行(客户端/媒体层,P5 才接入,先留位)
    CAPTURE_CHUNK_READY = "capture_chunk_ready"
    UPLINK_SENT = "uplink_sent"
    DOWNLINK_RECEIVED = "downlink_received"

    # 输入处理
    RESAMPLE_DONE = "resample_done"
    CHUNK_ALIGNED = "chunk_aligned"  # 1.0s chunk 对齐完成,即将喂 encoder
    STREAMING_PREFILL_START = "streaming_prefill_start"
    STREAMING_PREFILL_DONE = "streaming_prefill_done"

    # EOU 与决策
    EOU_DETECTED = "eou_detected"  # 用户说完(产品层判定/模型观测)
    SPEAK_DECISION = "speak_decision"  # 模型决定开口(is_listen False)
    BARGE_IN_DETECTED = "barge_in_detected"  # 用户在播放中开口
    PLAYOUT_MUTED = "playout_muted"  # 本地播放实际静音

    # Thinker
    THINKER_PREFILL_START = "thinker_prefill_start"
    THINKER_PREFILL_DONE = "thinker_prefill_done"
    FIRST_TOKEN_DECODED = "first_token_decoded"
    FIRST_PHRASE_READY = "first_phrase_ready"

    # Talker / Code2Wav
    STREAMING_GENERATE_START = "streaming_generate_start"
    STREAMING_GENERATE_DONE = "streaming_generate_done"
    TALKER_CHUNK_READY = "talker_chunk_ready"  # 25 帧 codec 攒齐
    CODE2WAV_FIRST_PCM = "code2wav_first_pcm"  # 首个 PCM sample 产出
    PCM_QUALITY_REJECTED = "pcm_quality_rejected"  # 硬门禁在发布前拒绝 PCM

    # 播放
    PUBLISHED = "published"  # 交给传输层
    DEVICE_PLAYOUT_START = "device_playout_start"

    # 会话生命周期
    SESSION_PREPARE_DONE = "session_prepare_done"
    LOAD_DONE = "load_done"  # 权重加载+初始化完成(冷启动基准)
    FIRST_FORWARD_DONE = "first_forward_done"  # 首次真实 forward(显存测量场景 2)


class Segment:
    """waterfall 分段定义:(名称, 起始锚点, 结束锚点)。

    EOU_TO_FIRST_AUDIO 是产品级指标(设计文档 §2);串行基线里用本地 PCM 产出近似,
    接入 LiveKit 后拆成 local/remote 两条口径。
    """

    EOU_TO_FIRST_PCM_LOCAL = (
        "eou_to_first_pcm_local",
        Anchor.EOU_DETECTED,
        Anchor.CODE2WAV_FIRST_PCM,
    )
    EOU_TO_SPEAK_DECISION = (
        "eou_to_speak_decision",
        Anchor.EOU_DETECTED,
        Anchor.SPEAK_DECISION,
    )
    SPEAK_DECISION_TO_FIRST_PCM = (
        "speak_decision_to_first_pcm",
        Anchor.SPEAK_DECISION,
        Anchor.CODE2WAV_FIRST_PCM,
    )
    PREFILL_TAIL = (
        "prefill_tail_after_eou",
        Anchor.EOU_DETECTED,
        Anchor.THINKER_PREFILL_DONE,
    )
    FIRST_TOKEN = (
        "first_token_decode",
        Anchor.THINKER_PREFILL_DONE,
        Anchor.FIRST_TOKEN_DECODED,
    )
    TALKER_FIRST_CHUNK = (
        "talker_first_chunk",
        Anchor.FIRST_TOKEN_DECODED,
        Anchor.TALKER_CHUNK_READY,
    )
    CODE2WAV_FIRST = (
        "code2wav_first",
        Anchor.TALKER_CHUNK_READY,
        Anchor.CODE2WAV_FIRST_PCM,
    )
    BARGE_IN_TO_SILENCE = (
        "barge_in_to_silence",
        Anchor.BARGE_IN_DETECTED,
        Anchor.PLAYOUT_MUTED,
    )
    CHUNK_PREFILL = (
        "chunk_streaming_prefill",
        Anchor.STREAMING_PREFILL_START,
        Anchor.STREAMING_PREFILL_DONE,
    )

    ALL = [
        EOU_TO_FIRST_PCM_LOCAL,
        EOU_TO_SPEAK_DECISION,
        SPEAK_DECISION_TO_FIRST_PCM,
        PREFILL_TAIL,
        FIRST_TOKEN,
        TALKER_FIRST_CHUNK,
        CODE2WAV_FIRST,
        BARGE_IN_TO_SILENCE,
        CHUNK_PREFILL,
    ]


@dataclasses.dataclass
class TraceRecord:
    anchor: str
    ts_ns: int
    trace_id: str
    turn_epoch: int = 0
    speech_id: str = ""
    wall_ns: int = 0
    session_id: str = ""
    tags: dict[str, str] = dataclasses.field(default_factory=dict)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> TraceRecord:
        data = json.loads(line)
        known = {f.name for f in dataclasses.fields(TraceRecord)}
        clean = {k: v for k, v in data.items() if k in known}
        return TraceRecord(**clean)


def match_key(record: TraceRecord) -> tuple[str, int]:
    """分段配对键:同一 trace_id + turn_epoch 内找 start→end。"""
    return (record.trace_id, record.turn_epoch)
