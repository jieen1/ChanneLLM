"""可选的 LiveKit PCM 适配器。

本模块不把 LiveKit 变成核心 runtime 依赖：只有实际创建远端输入/输出时才导入
``livekit.rtc``。输入明确请求 SDK 重采样为 16kHz 单声道；输出保持 Code2Wav 的
24kHz 单声道。它不实现 EOU/VAD，也不替代 iOS 的 AEC——二者必须由客户端和
会话控制面提供，不能在 worker 端伪造。
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from channellm.duplex.epoch import EpochTag
from channellm.duplex.playback import BufferedPlaybackSink

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class ReceivedPcmFrame:
    """LiveKit 解码后的 int16 interleaved PCM，供 ``PcmIngress`` 消费。"""

    samples: np.ndarray
    sample_rate: int
    channels: int


def load_rtc() -> Any:
    """延迟加载 LiveKit SDK，并给出可操作的部署错误。"""
    try:
        return importlib.import_module("livekit.rtc")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LiveKit Python SDK is required for remote media; install the supported "
            "livekit package before starting the P5 worker"
        ) from exc


def received_pcm(frame: Any) -> ReceivedPcmFrame:
    """从 SDK ``AudioFrame`` 复制 int16 数据，避免 frame 生命周期泄漏。"""
    sample_rate = int(frame.sample_rate)
    channels = int(frame.num_channels)
    samples_per_channel = int(frame.samples_per_channel)
    if sample_rate <= 0 or channels <= 0 or samples_per_channel < 0:
        raise ValueError("LiveKit AudioFrame has invalid audio geometry")
    samples = np.frombuffer(frame.data, dtype="<i2")
    expected = samples_per_channel * channels
    if samples.size != expected:
        raise ValueError(
            f"LiveKit AudioFrame contains {samples.size} samples, expected {expected}"
        )
    return ReceivedPcmFrame(samples=samples.copy(), sample_rate=sample_rate, channels=channels)


def pcm_to_int16_bytes(pcm: Any) -> bytes:
    """无损接收 int16，或严格量化已通过质量门禁的 float PCM。

    超幅或非有限 float 绝不在媒体适配器中静默 clip；那会掩盖上游质量故障。
    """
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        raw = bytes(pcm)
        if len(raw) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM byte payload must contain whole int16 samples")
        return raw
    samples = np.asarray(pcm)
    if samples.ndim > 1:
        raise ValueError("LiveKit output PCM must be mono one-dimensional samples")
    samples = samples.reshape(-1)
    if samples.dtype == np.int16:
        return samples.astype("<i2", copy=False).tobytes()
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError("LiveKit output PCM must be int16, float, or int16 bytes")
    floats = samples.astype(np.float32, copy=False)
    if not np.isfinite(floats).all():
        raise ValueError("LiveKit output PCM contains non-finite samples")
    if floats.size and np.abs(floats).max() > 1.0:
        raise ValueError("LiveKit output PCM exceeds full scale")
    return np.rint(floats * np.iinfo(np.int16).max).astype("<i2").tobytes()


class LiveKitAudioInput:
    """将一个远端音轨转为显式 16kHz PCM frame 异步流。"""

    def __init__(self, rtc_module: Any | None = None, *, capacity: int = 8) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.rtc = rtc_module or load_rtc()
        self.capacity = capacity

    async def frames(self, track: Any) -> AsyncIterator[ReceivedPcmFrame]:
        stream = self.rtc.AudioStream.from_track(
            track=track,
            sample_rate=INPUT_SAMPLE_RATE,
            num_channels=1,
            capacity=self.capacity,
        )
        try:
            async for event in stream:
                frame = received_pcm(event.frame)
                if frame.sample_rate != INPUT_SAMPLE_RATE or frame.channels != 1:
                    raise RuntimeError(
                        "LiveKit AudioStream did not honor requested 16kHz mono format"
                    )
                yield frame
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result


class LiveKitAudioOutput:
    """24kHz PCM -> LiveKit ``AudioSource.capture_frame`` 的异步 writer。"""

    def __init__(self, source: Any, rtc_module: Any) -> None:
        self.source = source
        self.rtc = rtc_module

    @classmethod
    def create(
        cls,
        *,
        track_name: str = "channellm-agent-audio",
        queue_size_ms: int = 50,
        rtc_module: Any | None = None,
    ) -> tuple[LiveKitAudioOutput, Any]:
        if queue_size_ms <= 0 or queue_size_ms % 10:
            raise ValueError("queue_size_ms must be a positive multiple of 10")
        rtc = rtc_module or load_rtc()
        source = rtc.AudioSource(
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            queue_size_ms=queue_size_ms,
        )
        track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
        return cls(source, rtc), track

    @classmethod
    async def create_and_publish(
        cls,
        room: Any,
        *,
        track_name: str = "channellm-agent-audio",
        queue_size_ms: int = 50,
        rtc_module: Any | None = None,
    ) -> LiveKitAudioOutput:
        output, track = cls.create(
            track_name=track_name,
            queue_size_ms=queue_size_ms,
            rtc_module=rtc_module,
        )
        await room.local_participant.publish_track(track)
        return output

    async def write(self, pcm: Any, _tag: EpochTag) -> None:
        data = pcm_to_int16_bytes(pcm)
        samples_per_channel = len(data) // np.dtype(np.int16).itemsize
        frame = self.rtc.AudioFrame(
            data=data,
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=samples_per_channel,
        )
        await self.source.capture_frame(frame)

    def clear_queue(self) -> None:
        """barge-in 时同步丢弃 SDK 尚未播放的下行帧。"""
        self.source.clear_queue()


class LiveKitBufferedPlaybackSink(BufferedPlaybackSink):
    """把本地 epoch mute 同步传播到 LiveKit 的音频 source 队列。"""

    def __init__(self, output: LiveKitAudioOutput, *, capacity: int = 64) -> None:
        super().__init__(capacity=capacity)
        self.output = output

    def mute(self) -> None:
        super().mute()
        self.output.clear_queue()


async def connect_room(url: str, token: str, *, rtc_module: Any | None = None) -> Any:
    """由 worker 主动向 LiveKit URL 建立出站连接。"""
    if not url or not token:
        raise ValueError("LiveKit URL and participant token are required")
    rtc = rtc_module or load_rtc()
    room = rtc.Room()
    await room.connect(url, token)
    return room
