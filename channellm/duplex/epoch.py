"""epoch 端到端取消的标识与丢弃策略(设计文档 §5)。

所有 LLM token、codec chunk、audio chunk 都携带 (turn_epoch, speech_id)。
旧 epoch 的产物无条件丢弃 —— 覆盖四类状态域:已生成未播、传输队列中、
客户端 jitter buffer、以及 Input/Reply/Notification/Task 四个域。

新 response 到来时先 barge-in + cancel 旧的,不能 await 旧任务完成
(参考 vllm-omni vllm_omni/experimental/fullduplex/core/runtime.py 的 cancel-not-await 注释)。
"""

from __future__ import annotations

import dataclasses
import itertools


@dataclasses.dataclass(frozen=True)
class EpochTag:
    turn_epoch: int
    speech_id: str = ""

    def __str__(self) -> str:
        return f"e{self.turn_epoch}:{self.speech_id}"


class EpochGuard:
    """单一说话权下的 epoch 推进与过期判定。"""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._current: EpochTag | None = None

    @property
    def current(self) -> EpochTag | None:
        return self._current

    def advance(self, speech_id: str = "") -> EpochTag:
        """开启新一轮(barge-in 时调用):返回新 tag 并使旧 tag 全部过期。"""
        self._current = EpochTag(turn_epoch=next(self._counter), speech_id=speech_id)
        return self._current

    def is_stale(self, tag: EpochTag) -> bool:
        """旧 epoch 无条件丢弃;未 advance 过时一切皆旧。"""
        if self._current is None:
            return True
        return tag.turn_epoch != self._current.turn_epoch

    def accept(self, tag: EpochTag) -> bool:
        return not self.is_stale(tag)
