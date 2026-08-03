"""本地媒体适配器的有界待播缓冲。

它不负责驱动声卡；上层的设备/LiveKit writer 通过 ``drain`` 获取当前 epoch
的 PCM。关键语义是 ``mute`` 同步清空所有待播项，给 barge-in 提供可验证的
媒体边界，而不是仅停止后续 GPU 输出。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from threading import Lock
from typing import Any, Protocol

from channellm.duplex.epoch import EpochTag


class PlayoutLifecycle(Protocol):
    """由 L3 runtime 实现的媒体播放生命周期回调。"""

    def on_device_playout_start(self, tag: EpochTag) -> bool: ...

    def on_device_playout_finished(self, tag: EpochTag) -> bool: ...


class BufferedPlaybackSink:
    """线程安全、有界的当前回合 PCM 待播队列。"""

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = Lock()
        self._items: deque[tuple[Any, EpochTag]] = deque()
        self._finished_tags: set[EpochTag] = set()
        self.dropped_oldest = 0
        self.muted_items = 0

    def mute(self) -> None:
        """barge-in 临界区调用：同步丢弃扬声器/网络 writer 尚未取走的 PCM。"""
        with self._lock:
            self.muted_items += len(self._items)
            self._items.clear()
            self._finished_tags.clear()

    def publish(self, pcm: Any, tag: EpochTag) -> None:
        with self._lock:
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped_oldest += 1
            self._items.append((pcm, tag))

    def finish(self, tag: EpochTag) -> None:
        """生产端不再为 ``tag`` 发布 PCM；writer 排空后才能结束播放。"""
        with self._lock:
            self._finished_tags.add(tag)

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

    def is_finished_and_drained(self, tag: EpochTag) -> bool:
        """只在终止已知且该回合没有待写帧时返回真。"""
        with self._lock:
            return tag in self._finished_tags and not any(
                item_tag == tag for _pcm, item_tag in self._items
            )


class PcmPlayoutPump:
    """把 ``BufferedPlaybackSink`` 接到实际媒体 writer 的单线程桥。

    设备首帧锚点紧贴 writer handoff，而不是 GPU 产出或队列入队时刻；它不声称
    已测得物理 DAC 的首 sample。LiveKit/声卡适配器只需提供这个回调。
    """

    def __init__(
        self,
        sink: BufferedPlaybackSink,
        lifecycle: PlayoutLifecycle,
        write: Callable[[Any, EpochTag], None],
    ) -> None:
        self.sink = sink
        self.lifecycle = lifecycle
        self.write = write

    def pump(self, max_items: int | None = None) -> int:
        """写出至多 ``max_items`` 帧，返回实际交给 writer 的帧数。"""
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be non-negative")
        written = 0
        while max_items is None or written < max_items:
            items = self.sink.drain(1)
            if not items:
                break
            pcm, tag = items[0]
            if not self.lifecycle.on_device_playout_start(tag):
                continue
            self.write(pcm, tag)
            written += 1
            if self.sink.is_finished_and_drained(tag):
                self.lifecycle.on_device_playout_finished(tag)
        return written
