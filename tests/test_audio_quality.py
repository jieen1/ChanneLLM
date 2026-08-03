from __future__ import annotations

import numpy as np
import pytest

from channellm.audio.quality import inspect_signal


def test_signal_quality_accepts_finite_unclipped_audio() -> None:
    waveform = 0.1 * np.sin(2 * np.pi * 440 * np.arange(24_000) / 24_000)
    quality = inspect_signal(waveform.astype(np.float32), 24_000)

    assert quality.duration_s == 1.0
    assert quality.rms == pytest.approx(0.1 / np.sqrt(2))
    assert quality.peak == pytest.approx(0.1)
    assert quality.clipped_ratio == 0.0
    assert quality.dc_offset == pytest.approx(0.0, abs=1e-6)
    assert quality.max_step < 0.02
    assert quality.failures() == ()


def test_signal_quality_rejects_silence_clipping_and_nonfinite() -> None:
    silence = inspect_signal(np.zeros(24_000, dtype=np.float32), 24_000)
    clipped = inspect_signal(np.ones(24_000, dtype=np.float32), 24_000)
    nonfinite = inspect_signal(np.array([0.0, np.nan], dtype=np.float32), 24_000)

    assert "rms 0.00000 < 0.01000" in silence.failures()
    assert "peak 1.00000 > 0.99900" in clipped.failures()
    assert "clipped ratio 1.000000 > 0.000000" in clipped.failures()
    assert "contains non-finite samples" in nonfinite.failures()


def test_signal_quality_rejects_dc_bias_and_obvious_sample_pop() -> None:
    biased = inspect_signal(np.full(24_000, 0.2, dtype=np.float32), 24_000)
    popped = inspect_signal(np.array([0.0, 0.9, 0.0], dtype=np.float32), 24_000)

    assert "dc offset 0.20000 exceeds 0.10000" in biased.failures()
    assert "sample step 0.90000 > 0.80000" in popped.failures()
