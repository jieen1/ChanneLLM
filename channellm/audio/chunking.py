"""16kHz 音频分块 —— 对齐 MiniCPM-o 的 audio_chunk_length=1.0(设计文档 §1)。

模型配置事实:audio_chunk_length=1.0s、audio_pool_step=5(Whisper encoder
每 pool_step 帧下采样一次)。P0 回放集一律重采样到 16kHz 单声道再分块,
禁止用 8kHz 窄带源(PSTN 口径会恶化中文擦音/塞擦音,设计文档"明确不做")。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

TARGET_SAMPLE_RATE = 16000


def pcm16_to_float32(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def float32_to_pcm16(wave: np.ndarray) -> np.ndarray:
    clipped = np.clip(wave, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample_linear(wave: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """numpy-only 线性插值重采样(P0 回放用;生产路径再评估更高质量方案)。"""
    if src_rate == dst_rate:
        return wave
    if wave.ndim != 1:
        raise ValueError("expect mono waveform")
    n_dst = int(round(len(wave) * dst_rate / src_rate))
    if n_dst == 0:
        return np.zeros(0, dtype=wave.dtype)
    src_idx = np.linspace(0.0, len(wave) - 1, n_dst)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(wave) - 1)
    frac = (src_idx - lo).astype(wave.dtype)
    return wave[lo] * (1 - frac) + wave[hi] * frac


def load_wav_mono16k(path: str) -> np.ndarray:
    """读取任意采样率 wav → float32 mono 16kHz。"""
    import soundfile as sf

    wave, rate = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return resample_linear(wave, rate, TARGET_SAMPLE_RATE)


class StreamChunker:
    """把任意长度的音频流切成固定时长 chunk,尾部不足一块先攒着。

    与官方 streaming_prefill 的调用节奏一致:每次喂一个 audio_chunk_length。
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE, chunk_seconds: float = 1.0) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = int(round(sample_rate * chunk_seconds))
        if self.chunk_samples <= 0:
            raise ValueError("chunk_seconds too small")
        self._buffer = np.zeros(0, dtype=np.float32)

    def feed(self, wave: np.ndarray) -> Iterator[np.ndarray]:
        """喂入一段波形,yield 出所有攒满的 chunk。"""
        if wave.dtype != np.float32:
            wave = wave.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, wave])
        while len(self._buffer) >= self.chunk_samples:
            chunk, self._buffer = (
                self._buffer[: self.chunk_samples],
                self._buffer[self.chunk_samples :],
            )
            yield chunk

    def flush_tail(self, pad_silence: bool = True) -> np.ndarray | None:
        """收尾:不足一块的残余。pad_silence=True 时补零凑满(官方按整块喂)。"""
        if len(self._buffer) == 0:
            return None
        tail = self._buffer
        self._buffer = np.zeros(0, dtype=np.float32)
        if pad_silence and len(tail) < self.chunk_samples:
            tail = np.concatenate(
                [tail, np.zeros(self.chunk_samples - len(tail), dtype=np.float32)]
            )
        return tail
