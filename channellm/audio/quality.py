"""音频信号完整性门禁。

这不是主观语音质量/可懂度评分器：它只拒绝已知会让试听结论失真的输出，
例如空音频、NaN、削波、几乎静音、明显直流偏置和采样突变。真实对话质量
仍需保留回放样本与人工评审；这些指标不能替代可懂度或自然度评分。
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
    dc_offset: float
    max_step: float
    finite: bool

    def failures(
        self,
        *,
        min_duration_s: float = 0.5,
        min_rms: float = 0.01,
        max_peak: float = 0.999,
        max_clipped_ratio: float = 0.0,
        max_dc_offset: float = 0.1,
        max_step: float = 0.8,
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
        if abs(self.dc_offset) > max_dc_offset:
            failures.append(
                f"dc offset {self.dc_offset:.5f} exceeds {max_dc_offset:.5f}"
            )
        if self.max_step > max_step:
            failures.append(f"sample step {self.max_step:.5f} > {max_step:.5f}")
        return tuple(failures)

    def review_warnings(
        self,
        *,
        recommended_max_peak: float = 0.98,
    ) -> tuple[str, ...]:
        """返回不会证明音质合格、但需要保留样本人工复核的风险。"""
        warnings: list[str] = []
        if self.peak >= recommended_max_peak:
            warnings.append(
                f"peak {self.peak:.5f} >= review threshold {recommended_max_peak:.5f}"
            )
        return tuple(warnings)


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
            dc_offset=0.0,
            max_step=0.0,
            finite=finite,
        )
    abs_samples = np.abs(samples)
    return SignalQuality(
        duration_s=len(samples) / sample_rate,
        rms=float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))),
        peak=float(abs_samples.max()),
        clipped_ratio=float(np.mean(abs_samples >= 0.999)),
        dc_offset=float(np.mean(samples, dtype=np.float64)),
        max_step=float(np.abs(np.diff(samples)).max()) if len(samples) > 1 else 0.0,
        finite=True,
    )
