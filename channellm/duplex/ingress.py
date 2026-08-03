"""实时媒体 PCM 进入单 GPU duplex runtime 的质量边界。

LiveKit/设备适配器把解码后的 PCM frame 交给此模块；这里仅做确定的通道归一、
16kHz 整块组装和有界提交。实时路径故意拒绝隐式线性重采样，避免把 P0 回放的
低保真便捷函数误带入生产语音输入。
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from channellm.audio.chunking import TARGET_SAMPLE_RATE, StreamChunker, pcm16_to_float32
from channellm.duplex.epoch import EpochTag


class AudioSubmitter(Protocol):
    """同步控制面 + 有界音频提交的最小接口。"""

    def begin_turn(self, speech_id: str = "") -> EpochTag: ...

    def submit_audio(self, tag: EpochTag, pcm: Any) -> bool: ...

    def on_eou(self, tag: EpochTag) -> bool: ...


class PcmIngress:
    """将外部 PCM frame 变成 MiniCPM-o 所需的 16kHz 单声道完整 unit。"""

    def __init__(self, submitter: AudioSubmitter, *, chunk_seconds: float = 1.0) -> None:
        self.submitter = submitter
        self.chunk_seconds = chunk_seconds
        self._tag: EpochTag | None = None
        self._chunker = StreamChunker(TARGET_SAMPLE_RATE, chunk_seconds)

    @property
    def active_tag(self) -> EpochTag | None:
        return self._tag

    def begin_speech(self, speech_id: str = "") -> EpochTag:
        """开始一个新输入回合，并无条件丢弃上个未凑满的输入尾帧。"""
        self._chunker = StreamChunker(TARGET_SAMPLE_RATE, self.chunk_seconds)
        self._tag = self.submitter.begin_turn(speech_id)
        return self._tag

    def push_frame(
        self,
        pcm: np.ndarray,
        *,
        sample_rate: int,
        channels: int = 1,
    ) -> int:
        """提交一个已解码 frame，返回成功进入 GPU 输入队列的完整 unit 数。"""
        tag = self._require_active()
        wave = _normalize_frame(pcm, sample_rate=sample_rate, channels=channels)
        submitted = 0
        for chunk in self._chunker.feed(wave):
            if not self.submitter.submit_audio(tag, chunk):
                break
            submitted += 1
        return submitted

    def end_speech(self) -> bool:
        """补齐唯一尾块并记录 EOU；重复结束或无活跃输入会失败。"""
        tag = self._require_active()
        tail = self._chunker.flush_tail(pad_silence=True)
        if tail is not None and not self.submitter.submit_audio(tag, tail):
            return False
        self._tag = None
        return self.submitter.on_eou(tag)

    def _require_active(self) -> EpochTag:
        if self._tag is None:
            raise RuntimeError("begin_speech must be called before submitting PCM")
        return self._tag


def _normalize_frame(pcm: np.ndarray, *, sample_rate: int, channels: int) -> np.ndarray:
    """严格输入验证；高保真重采样应由媒体 SDK/专用适配器显式完成。"""
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"real-time PCM must be explicitly resampled to {TARGET_SAMPLE_RATE}Hz, "
            f"got {sample_rate}Hz"
        )
    if channels <= 0:
        raise ValueError("channels must be positive")
    values = np.asarray(pcm)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    if values.ndim == 2:
        if values.shape[1] != channels:
            raise ValueError("PCM frame channel count does not match its shape")
        values = values.reshape(-1, channels)
    elif values.ndim == 1:
        if values.size % channels:
            raise ValueError("interleaved PCM length is not divisible by channels")
        values = values.reshape(-1, channels)
    else:
        raise ValueError("PCM frame must be one- or two-dimensional")
    if values.dtype == np.int16:
        wave = pcm16_to_float32(values)
    elif np.issubdtype(values.dtype, np.floating):
        wave = values.astype(np.float32, copy=False)
    else:
        raise TypeError("PCM frame must use int16 or floating-point samples")
    if not np.isfinite(wave).all():
        raise ValueError("PCM frame contains non-finite samples")
    return wave.mean(axis=1, dtype=np.float32)
