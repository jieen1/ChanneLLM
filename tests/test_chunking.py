import numpy as np

from channellm.audio.chunking import StreamChunker, pcm16_to_float32, resample_linear


def test_exact_chunks():
    chunker = StreamChunker(sample_rate=16000, chunk_seconds=1.0)
    wave = np.zeros(32000, dtype=np.float32)  # exactly 2 seconds
    chunks = list(chunker.feed(wave))
    assert len(chunks) == 2
    assert all(len(c) == 16000 for c in chunks)
    assert chunker.flush_tail() is None


def test_partial_feeds_accumulate():
    chunker = StreamChunker()
    out = []
    for _ in range(5):
        out.extend(chunker.feed(np.zeros(8000, dtype=np.float32)))  # 0.5s each
    assert len(out) == 2  # 5 * 0.5s = 2.5s -> 2 full chunks
    tail = chunker.flush_tail()
    assert tail is not None and len(tail) == 16000  # padded tail


def test_pcm16_conversion():
    pcm = np.array([0, 32767, -32768], dtype=np.int16)
    wave = pcm16_to_float32(pcm)
    assert wave.dtype == np.float32
    assert abs(wave[1] - 1.0) < 1e-3


def test_resample_linear_identity_and_ratio():
    wave = np.linspace(0, 1, 8000, dtype=np.float32)
    assert resample_linear(wave, 8000, 8000) is wave
    up = resample_linear(wave, 8000, 16000)
    assert len(up) == 16000
