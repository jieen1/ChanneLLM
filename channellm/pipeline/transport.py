"""跨阶段传输 —— 单进程单卡,有界队列 + 背压(设计文档 §4)。

不做分布式那套。三个事实/决策:
- 有界队列:上游快于下游时不能无限堆(R10 无限反压 → 延迟漂移)。
- 首块与后续块不同超时:首块 3000ms,后续 300ms(参考实现基线)。
- overrun 策略可配:drop_oldest(保实时性,默认)/ block(保完整性)。
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class OverrunPolicy(str, enum.Enum):
    DROP_OLDEST = "drop_oldest"
    BLOCK = "block"
    DROP_NEWEST = "drop_newest"


@dataclasses.dataclass
class ChannelStats:
    enqueued: int = 0
    dequeued: int = 0
    dropped_oldest: int = 0
    dropped_newest: int = 0
    timeouts: int = 0


class ChunkChannel(Generic[T]):
    """有界 chunk 通道。asyncio 队列 + 可观测计数。"""

    def __init__(
        self,
        name: str,
        capacity: int = 16,
        policy: OverrunPolicy = OverrunPolicy.DROP_OLDEST,
        first_timeout_s: float = 3.0,
        next_timeout_s: float = 0.3,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.name = name
        self.capacity = capacity
        self.policy = policy
        self.first_timeout_s = first_timeout_s
        self.next_timeout_s = next_timeout_s
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity)
        self._saw_first = False
        self.stats = ChannelStats()

    async def put(self, item: T) -> None:
        if self._queue.full():
            if self.policy is OverrunPolicy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self.stats.dropped_oldest += 1
                except asyncio.QueueEmpty:
                    pass
            elif self.policy is OverrunPolicy.DROP_NEWEST:
                self.stats.dropped_newest += 1
                return
            # BLOCK: fall through to awaiting put
        await self._queue.put(item)
        self.stats.enqueued += 1

    def put_nowait(self, item: T) -> None:
        if self._queue.full():
            if self.policy is OverrunPolicy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self.stats.dropped_oldest += 1
                except asyncio.QueueEmpty:
                    pass
            elif self.policy is OverrunPolicy.DROP_NEWEST:
                self.stats.dropped_newest += 1
                return
        self._queue.put_nowait(item)
        self.stats.enqueued += 1

    async def get(self, is_first: bool | None = None) -> T | None:
        """取一块;超时返回 None 并计数。is_first=None 时自动跟踪首块。"""
        if is_first is None:
            is_first = not self._saw_first
        timeout = self.first_timeout_s if is_first else self.next_timeout_s
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError:
            self.stats.timeouts += 1
            return None
        self._saw_first = True
        self.stats.dequeued += 1
        return item

    def qsize(self) -> int:
        return self._queue.qsize()

    def get_nowait(self) -> T | None:
        """同步消费一个已就绪 chunk；空队列不计为超时。"""
        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._saw_first = True
        self.stats.dequeued += 1
        return item

    def discard(self, predicate: Callable[[T], bool]) -> int:
        """删除匹配的已排队项，不把旧 epoch 留给下一回合。"""
        kept: list[T] = []
        removed = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if predicate(item):
                removed += 1
            else:
                kept.append(item)
        for item in kept:
            self._queue.put_nowait(item)
        return removed
