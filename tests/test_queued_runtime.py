from __future__ import annotations

import threading
import time

import torch

from channellm.duplex.driver import DuplexPipelineDriver
from channellm.duplex.queued_runtime import QueuedDuplexRuntime
from channellm.duplex.runtime import RealtimeRuntime
from channellm.pipeline.orchestrator import Orchestrator


class _Decision:
    is_listen = True
    end_of_turn = False


class _BlockingSession:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[bytes] = []

    def on_chunk(self, pcm: bytes) -> _Decision:
        self.calls.append(pcm)
        if pcm == b"old":
            self.started.set()
            assert self.release.wait(1.0)
        return _Decision()

    def latest_unit_conditioning(self):
        return torch.empty(0, dtype=torch.long), torch.empty((0, 1))


class _Talker:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _Code2Wav:
    def __init__(self) -> None:
        self.reset_calls = 0

    def stream_reset(self) -> None:
        self.reset_calls += 1


class _Sink:
    def __init__(self) -> None:
        self.mute_calls = 0

    def mute(self) -> None:
        self.mute_calls += 1

    def publish(self, _pcm, _tag) -> None:
        raise AssertionError("listen-only fake must not publish PCM")


def _make_runtime(*, capacity: int = 16, start: bool = True):
    sink = _Sink()
    session = _BlockingSession()
    driver = DuplexPipelineDriver(
        RealtimeRuntime(Orchestrator(), sink), session, _Talker(), _Code2Wav()
    )
    return QueuedDuplexRuntime(driver, capacity=capacity, start=start), session, sink


def test_barge_in_returns_without_waiting_for_inflight_model_work():
    queued, session, sink = _make_runtime()
    try:
        old_tag = queued.begin_turn("old")
        assert queued.submit_audio(old_tag, b"old")
        assert session.started.wait(0.5)

        start = time.monotonic()
        new_tag = queued.begin_turn("new")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        assert queued.submit_audio(new_tag, b"new")

        session.release.set()
        assert queued.wait_idle(1.0)
    finally:
        session.release.set()
        assert queued.close()

    assert new_tag.turn_epoch == old_tag.turn_epoch + 1
    assert sink.mute_calls == 1
    assert session.calls == [b"old", b"new"]
    assert queued.failures == []


def test_new_turn_discards_unstarted_old_audio_before_model_execution():
    queued, session, _sink = _make_runtime(start=False)
    try:
        old_tag = queued.begin_turn("old")
        assert queued.submit_audio(old_tag, b"old")
        new_tag = queued.begin_turn("new")
        assert queued.submit_audio(new_tag, b"new")
        queued.start()
        assert queued.wait_idle(1.0)
    finally:
        session.release.set()
        assert queued.close()

    assert queued.active_tag == new_tag
    assert session.calls == [b"new"]
    assert queued.stats.dropped_stale >= 2


def test_input_queue_drops_oldest_unprocessed_audio_on_overrun():
    queued, session, _sink = _make_runtime(capacity=2, start=False)
    try:
        tag = queued.begin_turn("current")
        assert queued.submit_audio(tag, b"first")
        assert queued.submit_audio(tag, b"second")
        assert queued.submit_audio(tag, b"third")
        queued.start()
        assert queued.wait_idle(1.0)
    finally:
        session.release.set()
        assert queued.close()

    assert session.calls == [b"second", b"third"]
    assert queued.stats.dropped_overrun == 1
