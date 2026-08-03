from __future__ import annotations

from channellm.duplex.epoch import EpochTag
from channellm.duplex.playback import BufferedPlaybackSink


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
