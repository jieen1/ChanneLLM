from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from channellm.engine.code2wav import Code2Wav, PcmQualityError


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

    with pytest.raises(PcmQualityError, match="应输出 24kHz,得到 16000"):
        code2wav.stream_chunk([1])


def test_stream_chunk_rejects_nonfinite_tensor_before_playback() -> None:
    code2wav = _streaming_code2wav(torch.tensor([0.0, float("nan")]))

    with pytest.raises(PcmQualityError, match="contains non-finite"):
        code2wav.stream_chunk([1])


@pytest.mark.parametrize(
    ("wave", "failure"),
    [
        (torch.tensor([0.0, 1.0]), "peak 1.00000"),
        (torch.tensor([0.0, 0.999]), "clipped ratio"),
        (torch.tensor([0.0, 0.9]), "sample step 0.90000"),
        (torch.full((8,), 0.2), "dc offset 0.20000"),
    ],
)
def test_stream_chunk_rejects_audible_pcm_integrity_failures_before_playback(
    wave: torch.Tensor, failure: str
) -> None:
    code2wav = _streaming_code2wav(wave)

    with pytest.raises(PcmQualityError, match=failure):
        code2wav.stream_chunk([1])


def test_stream_chunk_normalizes_non_clipped_near_full_scale_pcm() -> None:
    code2wav = _streaming_code2wav(
        torch.tensor([0.5, 0.99, 0.5, -0.3, -0.8, -0.6, -0.3])
    )

    actual = code2wav.stream_chunk([1])

    assert np.abs(actual).max() == pytest.approx(0.97)


class _MutableFlowDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cnn_cache_buffer", torch.tensor([1.0]))
        self.register_buffer("att_cache_buffer", torch.tensor([2.0]))
        estimator = torch.nn.Module()
        estimator.register_buffer("cnn_cache_buffer", torch.tensor([3.0]))
        estimator.register_buffer("att_cache_buffer", torch.tensor([4.0]))
        self.estimator = estimator


class _StatefulToken2Wav:
    def __init__(self) -> None:
        self.flow = SimpleNamespace(decoder=_MutableFlowDecoder())
        self.stream_cache = None
        self.hift_cache_dict = {}

    def set_stream_cache(self, _ref_wav_path: str):
        decoder = self.flow.decoder
        return {"estimator_cnn_cache": decoder.cnn_cache_buffer}, {"speech": torch.zeros(0)}


def test_stream_reset_restores_mutated_flow_module_cache_buffers() -> None:
    code2wav = Code2Wav.__new__(Code2Wav)
    code2wav.t2w = _StatefulToken2Wav()
    code2wav.ref_wav_path = "unused.wav"
    code2wav._stream_base = None
    code2wav._stream_module_cache_base = ()

    code2wav.stream_reset()
    decoder = code2wav.t2w.flow.decoder
    expected = {name: buffer.clone() for name, buffer in decoder.named_buffers()}
    for _name, buffer in decoder.named_buffers():
        buffer.add_(10)

    code2wav.stream_reset()

    for name, buffer in decoder.named_buffers():
        torch.testing.assert_close(buffer, expected[name])
    assert code2wav.t2w.stream_cache["estimator_cnn_cache"] is not decoder.cnn_cache_buffer
