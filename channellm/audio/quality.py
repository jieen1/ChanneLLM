"""音频信号完整性门禁。

这不是主观语音质量/可懂度评分器：它只拒绝已知会让试听结论失真的输出，
例如空音频、NaN、削波和几乎静音。真实对话质量仍需保留回放样本与人工评审。
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SignalQuality:
    duration_s: float
    rms: float
    peak: float
    clipped_ratio: float
    finite: bool

    def failures(
        self,
        *,
        min_duration_s: float = 0.5,
        min_rms: float = 0.01,
        max_peak: float = 0.999,
        max_clipped_ratio: float = 0.0,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.finite:
            failures.append("contains non-finite samples")
        if self.duration_s < min_duration_s:
            failures.append(f"duration {self.duration_s:.3f}s < {min_duration_s:.3f}s")
        if self.rms < min_rms:
            failures.append(f"rms {self.rms:.5f} < {min_rms:.5f}")
        if self.peak > max_peak:
            failures.append(f"peak {self.peak:.5f} > {max_peak:.5f}")
        if self.clipped_ratio > max_clipped_ratio:
            failures.append(
                f"clipped ratio {self.clipped_ratio:.6f} > {max_clipped_ratio:.6f}"
            )
        return tuple(failures)


def inspect_signal(wave: np.ndarray, sample_rate: int) -> SignalQuality:
    """计算可复现的基础信号指标，不修改输入。"""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(samples).all())
    if not len(samples) or not finite:
        return SignalQuality(
            duration_s=len(samples) / sample_rate,
            rms=0.0,
            peak=0.0,
            clipped_ratio=0.0,
            finite=finite,
        )
    abs_samples = np.abs(samples)
    return SignalQuality(
        duration_s=len(samples) / sample_rate,
        rms=float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        peak=float(abs_samples.max()),
        clipped_ratio=float(np.mean(abs_samples >= 0.999)),
        finite=True,
    )
