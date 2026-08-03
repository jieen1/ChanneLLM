"""把单 GPU 模型工作移出实时输入线程的有界队列运行时。

输入/媒体线程只做 epoch 推进、mute 和入队；Thinker/Talker/Code2Wav 仍在一个
顺序 worker 中执行。新回合不等待旧 GPU 调用完成：旧调用返回后会被
``RealtimeRuntime`` 的 tag 校验丢弃，而新回合的模型状态复位按队列顺序发生，
不会与旧调用并发修改同一份 KV/vocoder cache。
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections import deque
from typing import Any

from channellm.duplex.epoch import EpochTag


@dataclasses.dataclass(frozen=True)
class QueueStats:
    enqueued: int = 0
    processed: int = 0
    dropped_stale: int = 0
    dropped_overrun: int = 0
    failures: int = 0


@dataclasses.dataclass(frozen=True)
class _BeginTurn:
    tag: EpochTag


@dataclasses.dataclass(frozen=True)
class _AudioChunk:
    tag: EpochTag
    pcm: Any


class QueuedDuplexRuntime:
    """单 worker 的全双工输入前端；每个实例绑定一个 ``DuplexPipelineDriver``。"""

    def __init__(self, driver: Any, *, capacity: int = 16, start: bool = True) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.driver = driver
        self.capacity = capacity
        self.stats = QueueStats()
        self._stats_lock = threading.Lock()
        self.failures: list[BaseException] = []
        self._jobs: deque[_BeginTurn | _AudioChunk] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._inflight = 0
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    @property
    def active_tag(self) -> EpochTag | None:
        return self.driver.runtime.active_tag

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            if self._closed:
                raise RuntimeError("queued runtime is closed")
            self._thread = threading.Thread(
                target=self._run,
                name="channellm-duplex-gpu",
                daemon=True,
            )
            self._thread.start()

    def begin_turn(self, speech_id: str = "") -> EpochTag:
        """立即取消旧回合；昂贵的模型状态复位由 worker 串行完成。"""
        tag = self.driver.runtime.begin_turn(speech_id)
        with self._condition:
            dropped = len(self._jobs)
            self._jobs.clear()
            self._add_stats(dropped_stale=dropped, enqueued=1)
            self._jobs.append(_BeginTurn(tag))
            self._condition.notify()
        return tag

    def on_eou(self, tag: EpochTag) -> bool:
        """EOU 是控制面锚点，不必等待 GPU worker。"""
        return self.driver.on_eou(tag)

    def submit_audio(self, tag: EpochTag, pcm: Any) -> bool:
        """有界入队；队满时丢最旧的未处理输入以保留实时性。"""
        if self.driver.runtime.active_tag != tag:
            return False
        with self._condition:
            if self._closed or self.driver.runtime.active_tag != tag:
                return False
            audio_count = sum(isinstance(job, _AudioChunk) for job in self._jobs)
            if audio_count >= self.capacity:
                for job in self._jobs:
                    if isinstance(job, _AudioChunk):
                        self._jobs.remove(job)
                        self._add_stats(dropped_overrun=1)
                        break
            self._jobs.append(_AudioChunk(tag, pcm))
            self._add_stats(enqueued=1)
            self._condition.notify()
        return True

    def pending(self) -> int:
        with self._condition:
            return sum(isinstance(job, _AudioChunk) for job in self._jobs)

    def wait_idle(self, timeout_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._jobs or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout_s: float = 1.0) -> bool:
        with self._condition:
            self._closed = True
            self._jobs.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            return not thread.is_alive()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._jobs and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                job = self._jobs.popleft()
                self._inflight += 1
            try:
                self._process(job)
            except BaseException as exc:  # worker 必须活着继续接收下一回合
                self.failures.append(exc)
                self._add_stats(failures=1)
            finally:
                with self._condition:
                    self._inflight -= 1
                    self._condition.notify_all()

    def _process(self, job: _BeginTurn | _AudioChunk) -> None:
        if self.driver.runtime.active_tag != job.tag:
            self._add_stats(dropped_stale=1)
            return
        if isinstance(job, _BeginTurn):
            self.driver.reset_for_turn(job.tag)
        else:
            self.driver.process_audio_chunk(job.tag, job.pcm)
        self._add_stats(processed=1)

    def _add_stats(self, **changes: int) -> None:
        with self._stats_lock:
            self.stats = dataclasses.replace(
                self.stats,
                **{name: getattr(self.stats, name) + value for name, value in changes.items()},
            )
