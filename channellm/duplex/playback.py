"""本地媒体适配器的有界待播缓冲。

它不负责驱动声卡；上层的设备/LiveKit writer 通过 ``drain`` 获取当前 epoch
的 PCM。关键语义是 ``mute`` 同步清空所有待播项，给 barge-in 提供可验证的
媒体边界，而不是仅停止后续 GPU 输出。
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

from channellm.duplex.epoch import EpochTag


class BufferedPlaybackSink:
    """线程安全、有界的当前回合 PCM 待播队列。"""

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = Lock()
        self._items: deque[tuple[Any, EpochTag]] = deque()
        self.dropped_oldest = 0
        self.muted_items = 0

    def mute(self) -> None:
        """barge-in 临界区调用：同步丢弃扬声器/网络 writer 尚未取走的 PCM。"""
        with self._lock:
            self.muted_items += len(self._items)
            self._items.clear()

    def publish(self, pcm: Any, tag: EpochTag) -> None:
        with self._lock:
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped_oldest += 1
            self._items.append((pcm, tag))

    def drain(self, max_items: int | None = None) -> list[tuple[Any, EpochTag]]:
        """由设备 writer 取走当前待播项；``None`` 表示全部。"""
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be non-negative")
        with self._lock:
            count = len(self._items) if max_items is None else min(max_items, len(self._items))
            return [self._items.popleft() for _ in range(count)]

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)
