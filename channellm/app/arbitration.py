"""播报仲裁 —— 用户语音永远优先(设计文档 §6)。

任务结果进通知队列,等 idle 窗口播报;幂等、可合并(dedup_key)。
backchannel 与真实状态一致:无任务时不得说"我去办"。
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Notification:
    dedup_key: str
    text: str
    created_ns: int = 0
    played: bool = False


class PlaybackArbiter:
    """同步核心的仲裁状态机;媒体循环驱动 tick。"""

    def __init__(self) -> None:
        self.user_speaking = False
        self.agent_speaking = False
        self._queue: list[Notification] = []
        self._seen: set[str] = set()

    def on_user_speech_start(self) -> None:
        self.user_speaking = True

    def on_user_speech_end(self) -> None:
        self.user_speaking = False

    def on_agent_speech_start(self) -> None:
        self.agent_speaking = True

    def on_agent_speech_end(self) -> None:
        self.agent_speaking = False

    def enqueue(self, notification: Notification) -> bool:
        """幂等入队;重复 dedup_key 直接丢弃。"""
        if notification.dedup_key in self._seen:
            return False
        self._seen.add(notification.dedup_key)
        self._queue.append(notification)
        return True

    def next_playable(self) -> Notification | None:
        """只有空闲窗口(无人说话)才播报。"""
        if self.user_speaking or self.agent_speaking or not self._queue:
            return None
        notification = self._queue.pop(0)
        notification.played = True
        return notification

    def pending(self) -> int:
        return len(self._queue)
