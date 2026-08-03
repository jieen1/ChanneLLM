from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from channellm.app.event_store import EventKind, EventStore
from channellm.duplex.epoch import EpochTag
from channellm.duplex.livekit import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    LiveKitAudioInput,
    LiveKitAudioOutput,
    LiveKitBufferedPlaybackSink,
    connect_room,
    pcm_to_int16_bytes,
    received_pcm,
)
from channellm.duplex.playback import AsyncPcmPlayoutPump
from channellm.duplex.runtime import RealtimeRuntime
from channellm.pipeline.orchestrator import Orchestrator
from channellm.pipeline.stages import StageId


@dataclass
class _AudioFrame:
    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int


class _Source:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.frames: list[_AudioFrame] = []
        self.clear_calls = 0

    async def capture_frame(self, frame: _AudioFrame) -> None:
        self.frames.append(frame)

    def clear_queue(self) -> None:
        self.clear_calls += 1


class _TrackFactory:
    @staticmethod
    def create_audio_track(name: str, source: _Source):
        return (name, source)


class _Participant:
    def __init__(self) -> None:
        self.published = []

    async def publish_track(self, track) -> None:
        self.published.append(track)


class _Room:
    def __init__(self) -> None:
        self.local_participant = _Participant()
        self.connected: tuple[str, str] | None = None

    async def connect(self, url: str, token: str) -> None:
        self.connected = (url, token)


class _AudioStream:
    @staticmethod
    def from_track(**kwargs):
        return _Stream(kwargs)


class _Stream:
    def __init__(self, kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self._events = [_Event(_AudioFrame(b"\x01\x00\x02\x00", 16_000, 1, 2))]

    def __aiter__(self):
        self._iter = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class _Event:
    frame: _AudioFrame


class _Rtc:
    AudioFrame = _AudioFrame
    AudioSource = _Source
    LocalAudioTrack = _TrackFactory
    AudioStream = _AudioStream
    Room = _Room


def test_output_preserves_pcm_geometry_and_publishes_track() -> None:
    room = _Room()
    output = asyncio.run(LiveKitAudioOutput.create_and_publish(room, rtc_module=_Rtc))
    assert room.local_participant.published == [("channellm-agent-audio", output.source)]
    assert output.source.kwargs == {
        "sample_rate": OUTPUT_SAMPLE_RATE,
        "num_channels": 1,
        "queue_size_ms": 50,
    }

    asyncio.run(output.write(np.array([0.0, 0.5, -0.5], dtype=np.float32), EpochTag(1)))
    frame = output.source.frames[0]
    assert (frame.sample_rate, frame.num_channels, frame.samples_per_channel) == (24_000, 1, 3)
    np.testing.assert_array_equal(np.frombuffer(frame.data, dtype="<i2"), [0, 16384, -16384])


def test_output_rejects_nonfinite_and_overfull_pcm_without_clipping() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        pcm_to_int16_bytes(np.array([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="exceeds full scale"):
        pcm_to_int16_bytes(np.array([1.01], dtype=np.float32))


def test_mute_clears_local_and_livekit_queues() -> None:
    output, _track = LiveKitAudioOutput.create(rtc_module=_Rtc)
    sink = LiveKitBufferedPlaybackSink(output)
    sink.publish(b"pcm", EpochTag(1))
    sink.mute()
    assert sink.qsize() == 0
    assert output.source.clear_calls == 1


def test_input_requests_16khz_mono_and_copies_frame_samples() -> None:
    async def collect():
        adapter = LiveKitAudioInput(_Rtc, capacity=3)
        return [frame async for frame in adapter.frames(track=object())]

    frames = asyncio.run(collect())
    assert len(frames) == 1
    assert frames[0].sample_rate == INPUT_SAMPLE_RATE
    assert frames[0].channels == 1
    np.testing.assert_array_equal(frames[0].samples, [1, 2])


def test_received_pcm_rejects_malformed_geometry() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        received_pcm(_AudioFrame(b"\x00\x00", 16_000, 1, 2))


def test_connect_room_is_outbound_and_requires_credentials() -> None:
    room = asyncio.run(connect_room("wss://example.test", "token", rtc_module=_Rtc))
    assert room.connected == ("wss://example.test", "token")
    with pytest.raises(ValueError, match="required"):
        asyncio.run(connect_room("", "token", rtc_module=_Rtc))


def test_runtime_async_livekit_path_records_played_only_after_capture(tmp_path) -> None:
    output, _track = LiveKitAudioOutput.create(rtc_module=_Rtc)
    sink = LiveKitBufferedPlaybackSink(output)
    with EventStore(tmp_path / "events.sqlite") as store:
        runtime = RealtimeRuntime(Orchestrator(), sink, event_store=store)
        tag = runtime.begin_turn("livekit-output")
        runtime.submit_stage_output(
            tag,
            StageId.CODE2WAV,
            np.array([0.0, 0.25, -0.25], dtype=np.float32),
        )
        assert runtime.finish_turn(tag)
        assert asyncio.run(AsyncPcmPlayoutPump(sink, runtime, output.write).pump()) == 1
        events = list(store.iterate())

    assert [event.kind for event in events] == [
        EventKind.AGENT_SPEECH_PLANNED.value,
        EventKind.AGENT_SPEECH_ACTUALLY_PLAYED.value,
    ]
    assert len(output.source.frames) == 1
    assert output.source.frames[0].sample_rate == OUTPUT_SAMPLE_RATE
