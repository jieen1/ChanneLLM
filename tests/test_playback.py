from __future__ import annotations

import asyncio

import pytest

from channellm.app.event_store import EventKind, EventStore
from channellm.duplex.epoch import EpochTag
from channellm.duplex.playback import AsyncPcmPlayoutPump, BufferedPlaybackSink, PcmPlayoutPump
from channellm.duplex.runtime import RealtimeRuntime
from channellm.pipeline.orchestrator import Orchestrator
from channellm.pipeline.stages import StageId


def test_buffered_sink_mute_discards_already_published_audio() -> None:
    sink = BufferedPlaybackSink(capacity=2)
    old = EpochTag(1, "speech-old")
    sink.publish(b"old-1", old)
    sink.publish(b"old-2", old)

    sink.mute()

    assert sink.qsize() == 0
    assert sink.muted_items == 2
    assert sink.drain() == []


def test_buffered_sink_drops_oldest_under_backpressure() -> None:
    sink = BufferedPlaybackSink(capacity=2)
    tag = EpochTag(1, "speech")
    sink.publish(b"one", tag)
    sink.publish(b"two", tag)
    sink.publish(b"three", tag)

    assert sink.dropped_oldest == 1
    assert sink.drain() == [(b"two", tag), (b"three", tag)]


def test_playout_pump_marks_device_boundaries_at_writer_handoff() -> None:
    sink = BufferedPlaybackSink()
    runtime = RealtimeRuntime(Orchestrator(), sink)
    tag = runtime.begin_turn("speech")
    runtime.submit_stage_output(tag, stage=StageId.CODE2WAV, payload=b"one")
    runtime.submit_stage_output(tag, stage=StageId.CODE2WAV, payload=b"two")
    assert runtime.finish_turn(tag)
    written: list[tuple[bytes, EpochTag]] = []

    pump = PcmPlayoutPump(sink, runtime, lambda pcm, item_tag: written.append((pcm, item_tag)))
    assert pump.pump() == 2

    assert written == [(b"one", tag), (b"two", tag)]
    assert runtime.state_machine.reply.playout_tag is None


def test_barge_in_clears_buffer_before_playout_pump_can_handoff_old_pcm() -> None:
    sink = BufferedPlaybackSink()
    runtime = RealtimeRuntime(Orchestrator(), sink)
    old_tag = runtime.begin_turn("old")
    runtime.submit_stage_output(old_tag, stage=StageId.CODE2WAV, payload=b"old")
    assert runtime.finish_turn(old_tag)
    runtime.begin_turn("new")
    written: list[tuple[bytes, EpochTag]] = []

    assert PcmPlayoutPump(sink, runtime, lambda pcm, tag: written.append((pcm, tag))).pump() == 0

    assert written == []


def test_writer_failure_does_not_record_played_fact(tmp_path) -> None:
    sink = BufferedPlaybackSink()
    with EventStore(tmp_path / "events.sqlite") as store:
        runtime = RealtimeRuntime(Orchestrator(), sink, event_store=store)
        tag = runtime.begin_turn("writer-failure")
        runtime.submit_stage_output(tag, stage=StageId.CODE2WAV, payload=b"pcm")

        def fail_write(_pcm: bytes, _tag: EpochTag) -> None:
            raise OSError("device unavailable")

        with pytest.raises(OSError, match="device unavailable"):
            PcmPlayoutPump(sink, runtime, fail_write).pump()
        events = list(store.iterate())

    assert [event.kind for event in events] == [EventKind.AGENT_SPEECH_PLANNED.value]
    assert sink.qsize() == 0
    assert runtime.state_machine.reply.playout_tag is None


def test_async_pump_records_playout_only_after_successful_handoff() -> None:
    sink = BufferedPlaybackSink()
    runtime = RealtimeRuntime(Orchestrator(), sink)
    tag = runtime.begin_turn("async-writer")
    runtime.submit_stage_output(tag, stage=StageId.CODE2WAV, payload=b"pcm")
    assert runtime.finish_turn(tag)
    written: list[bytes] = []

    async def write(pcm: bytes, _tag: EpochTag) -> None:
        written.append(pcm)

    assert asyncio.run(AsyncPcmPlayoutPump(sink, runtime, write).pump()) == 1
    assert written == [b"pcm"]
    assert runtime.state_machine.reply.playout_tag is None


def test_async_writer_failure_clears_playout_state() -> None:
    sink = BufferedPlaybackSink()
    runtime = RealtimeRuntime(Orchestrator(), sink)
    tag = runtime.begin_turn("async-writer-failure")
    runtime.submit_stage_output(tag, stage=StageId.CODE2WAV, payload=b"pcm")
    assert runtime.finish_turn(tag)

    async def fail_write(_pcm: bytes, _tag: EpochTag) -> None:
        raise OSError("transport unavailable")

    with pytest.raises(OSError, match="transport unavailable"):
        asyncio.run(AsyncPcmPlayoutPump(sink, runtime, fail_write).pump())

    assert sink.qsize() == 0
    assert runtime.state_machine.reply.playout_tag is None
