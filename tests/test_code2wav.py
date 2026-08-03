from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
import torch

from channellm.engine.code2wav import Code2Wav


class _FakeToken2Wav:
    stream_cache = object()

    def __init__(self, output: object) -> None:
        self.output = output

    def stream(self, *_args: object, **_kwargs: object) -> object:
        return self.output


def _streaming_code2wav(output: object) -> Code2Wav:
    code2wav = Code2Wav.__new__(Code2Wav)
    code2wav.t2w = _FakeToken2Wav(output)
    code2wav.ref_wav_path = "unused.wav"
    return code2wav


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def test_stream_chunk_accepts_finite_24khz_wav_bytes() -> None:
    expected = np.array([0.1, -0.2], dtype=np.float32)
    code2wav = _streaming_code2wav(_wav_bytes(expected, 24_000))

    actual = code2wav.stream_chunk([1, 2, 3])

    np.testing.assert_allclose(actual, expected)


def test_stream_chunk_rejects_wrong_wav_sample_rate_before_playback() -> None:
    code2wav = _streaming_code2wav(_wav_bytes(np.zeros(8, dtype=np.float32), 16_000))

    with pytest.raises(RuntimeError, match="应输出 24kHz,得到 16000"):
        code2wav.stream_chunk([1])


def test_stream_chunk_rejects_nonfinite_tensor_before_playback() -> None:
    code2wav = _streaming_code2wav(torch.tensor([0.0, float("nan")]))

    with pytest.raises(RuntimeError, match="非有限 PCM"):
        code2wav.stream_chunk([1])
