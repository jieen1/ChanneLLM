"""流式应用层原语测试(无 GPU):PCM 编解码、出站队列、barge-in 清空语义。"""

from __future__ import annotations

import numpy as np
import pytest

from channellm.app.stream_server import (
    StreamingPlaybackSink,
    float32_to_pcm16,
    pcm16_to_float32,
)
from channellm.duplex.epoch import EpochTag


def test_pcm16_roundtrip_preserves_waveform() -> None:
    wave = np.array([0.0, 0.5, -0.5, 0.999, -1.0, 1.0], dtype=np.float32)
    decoded = pcm16_to_float32(float32_to_pcm16(wave))
    np.testing.assert_allclose(decoded, wave, atol=1e-3)


def test_pcm16_clips_out_of_range_and_rejects_odd_bytes() -> None:
    decoded = pcm16_to_float32(float32_to_pcm16(np.array([1.5, -1.5], dtype=np.float32)))
    assert decoded[0] <= 1.0 and decoded[1] >= -1.0
    with pytest.raises(ValueError):
        pcm16_to_float32(b"\x00\x00\x00")


def test_sink_publish_orders_audio_with_epoch() -> None:
    sink = StreamingPlaybackSink()
    sink.publish(np.zeros(480, dtype=np.float32), EpochTag(turn_epoch=7, speech_id="s"))
    sink.publish(np.zeros(240, dtype=np.float32), EpochTag(turn_epoch=7, speech_id="s"))
    items = sink.drain()
    assert [item.kind for item in items] == ["audio", "audio"]
    assert items[0].epoch == 7
    assert len(items[0].payload) == 480 * 2
    assert len(items[1].payload) == 240 * 2
    assert sink.drain() == []


def test_sink_mute_drops_pending_audio_and_posts_clear() -> None:
    sink = StreamingPlaybackSink()
    sink.publish(np.zeros(480, dtype=np.float32), EpochTag(turn_epoch=3))
    sink.mute()
    items = sink.drain()
    assert [item.kind for item in items] == ["clear"]
    sink.publish(np.zeros(480, dtype=np.float32), EpochTag(turn_epoch=4))
    items = sink.drain()
    assert [item.kind for item in items] == ["audio"]
    assert items[0].epoch == 4


def test_sink_capacity_drops_oldest() -> None:
    sink = StreamingPlaybackSink(capacity=2)
    for i in range(4):
        sink.publish(np.zeros(16, dtype=np.float32), EpochTag(turn_epoch=i))
    items = sink.drain()
    assert [item.epoch for item in items] == [2, 3]


def test_sink_waker_fires_from_worker_thread() -> None:
    import threading

    sink = StreamingPlaybackSink()
    fired = threading.Event()
    sink.set_waker(fired.set)
    sink.publish(np.zeros(16, dtype=np.float32), EpochTag(turn_epoch=1))
    assert fired.is_set()


def test_epoch_framing_little_endian() -> None:
    epoch = 0x1234
    frame = epoch.to_bytes(2, "little") + float32_to_pcm16(np.zeros(8, dtype=np.float32))
    assert frame[0] == 0x34 and frame[1] == 0x12
    assert len(frame) == 2 + 8 * 2


def test_sink_post_control_orders_with_audio() -> None:
    sink = StreamingPlaybackSink()
    sink.post_control({"type": "voiceprint", "enrolled": True})
    sink.publish(np.zeros(16, dtype=np.float32), EpochTag(turn_epoch=1))
    items = sink.drain()
    assert [item.kind for item in items] == ["control", "audio"]
    assert items[0].meta == {"type": "voiceprint", "enrolled": True}


def test_sink_pending_count_tracks_queue() -> None:
    sink = StreamingPlaybackSink()
    assert sink.pending_count == 0
    sink.publish(np.zeros(16, dtype=np.float32), EpochTag(turn_epoch=1))
    sink.post_control({"type": "gate", "state": "open"})
    assert sink.pending_count == 2
    sink.drain()
    assert sink.pending_count == 0
