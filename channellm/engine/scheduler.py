"""L1 连续批处理调度器骨架(P1)。

单卡对话场景并发低(1 个实时会话 + 可能的 backchannel/预热),调度器
保持最小:准入控制 + 有界并发,防止 Code2Wav 大批量前向挤掉 Talker 的
单 token 节拍(设计文档 §8 计算 QoS)。
"""

from __future__ import annotations

import dataclasses
import enum


class RequestPhase(str, enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"


@dataclasses.dataclass
class RequestState:
    request_id: str
    stage: str  # thinker / talker / code2wav
    phase: RequestPhase = RequestPhase.PREFILL
    resumable: bool = False  # 下游请求生成完不关闭,挂等更多上游输入(九件事 #3)
    priority: int = 0  # Talker 解码 > Code2Wav 批量


class Scheduler:
    """P1 实现:admit/step/preempt。先落接口,禁止提前复杂化。"""

    def __init__(self, max_concurrent: int = 4) -> None:
        self.max_concurrent = max_concurrent
        self._active: dict[str, RequestState] = {}

    def admit(self, request: RequestState) -> bool:
        if len(self._active) >= self.max_concurrent:
            return False
        self._active[request.request_id] = request
        return True

    def release(self, request_id: str) -> RequestState | None:
        return self._active.pop(request_id, None)

    def active(self) -> list[RequestState]:
        return sorted(self._active.values(), key=lambda r: -r.priority)
