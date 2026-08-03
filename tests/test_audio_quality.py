from __future__ import annotations

import numpy as np
import pytest

from channellm.audio.quality import inspect_signal


def test_signal_quality_accepts_finite_unclipped_audio() -> None:
    quality = inspect_signal(np.full(24_000, 0.1, dtype=np.float32), 24_000)

    assert quality.duration_s == 1.0
    assert quality.rms == pytest.approx(0.1)
    assert quality.peak == pytest.approx(0.1)
    assert quality.clipped_ratio == 0.0
    assert quality.failures() == ()


def test_signal_quality_rejects_silence_clipping_and_nonfinite() -> None:
    silence = inspect_signal(np.zeros(24_000, dtype=np.float32), 24_000)
    clipped = inspect_signal(np.ones(24_000, dtype=np.float32), 24_000)
    nonfinite = inspect_signal(np.array([0.0, np.nan], dtype=np.float32), 24_000)

    assert "rms 0.00000 < 0.01000" in silence.failures()
    assert "peak 1.00000 > 0.99900" in clipped.failures()
    assert "clipped ratio 1.000000 > 0.000000" in clipped.failures()
    assert "contains non-finite samples" in nonfinite.failures()
