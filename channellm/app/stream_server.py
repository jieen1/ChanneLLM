"""流式语音应用层 —— WebSocket 字节流与 duplex runtime 边界的桥(MVP)。

协议(单端口,WebSocket 升级前同端口提供 web 客户端静态文件):

上行:
- 二进制帧 = PCM16 16kHz 单声道(小端 int16),直接进 ``PcmIngress``;
- 文本帧 = JSON 控制:``{"type": "eou"}`` 标注端点(可选,duplex 模型
  本身逐 chunk 自决 listen/speak)。

下行:
- 二进制帧 = 2 字节小端 epoch + PCM16 24kHz 单声道;epoch 单调递增,
  客户端丢弃 epoch 小于已见最大值的帧(barge-in 后的陈旧音频);
- 文本帧 = JSON 控制:``{"type": "clear"}``(barge-in,清空播放缓冲)、
  ``{"type": "turn", "epoch": n}``(新回合开始)、``{"type": "reply", ...}``。

单会话 MVP:模型一次装载常驻,同一时刻只服务一个连接(第二个连接收到
1013);会话状态(Thinker/Talker KV)按连接新建,断连即释放。Code2Wav 的
流式 cache 是有状态单例,多会话并发前必须先拆 per-session vocoder。
"""

from __future__ import annotations

import dataclasses
import struct
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from channellm.duplex.epoch import EpochTag

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
_EPOCH_STRUCT = struct.Struct("<H")


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    """上行 PCM16 帧 → float32 波形;空帧返回空数组。"""
    if len(payload) % 2 != 0:
        raise ValueError("PCM16 帧必须是偶数字节")
    return np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(wave: Any) -> bytes:
    """float32 波形 → 下行 PCM16 字节(饱和截断,与媒体层一致)。"""
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


@dataclasses.dataclass
class OutboundItem:
    kind: str  # "audio" | "clear" | "turn" | "reply"
    epoch: int = 0
    payload: bytes = b""
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


class StreamingPlaybackSink:
    """PlaybackSink → 出站队列。``publish``/``mute`` 由 GPU worker 线程调用,
    通过 ``call_soon_threadsafe`` 唤醒 asyncio 发送协程。"""

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._items: deque[OutboundItem] = deque()
        self._waker: Callable[[], None] | None = None

    def set_waker(self, waker: Callable[[], None]) -> None:
        self._waker = waker

    def _wake(self) -> None:
        waker = self._waker
        if waker is not None:
            waker()

    def publish(self, pcm: Any, tag: EpochTag) -> None:
        data = float32_to_pcm16(pcm)
        if not data:
            return
        with self._lock:
            if len(self._items) >= self.capacity:
                self._items.popleft()
            self._items.append(OutboundItem("audio", tag.turn_epoch & 0xFFFF, data))
        self._wake()

    def mute(self) -> None:
        """barge-in:丢弃未发送音频并通知客户端清空播放缓冲。"""
        with self._lock:
            self._items.clear()
            self._items.append(OutboundItem("clear"))
        self._wake()

    def post_turn(self, tag: EpochTag) -> None:
        with self._lock:
            self._items.append(OutboundItem("turn", tag.turn_epoch & 0xFFFF))
        self._wake()

    def post_reply(self, tag: EpochTag, text: str) -> None:
        with self._lock:
            self._items.append(OutboundItem("reply", tag.turn_epoch & 0xFFFF, meta={"text": text}))
        self._wake()

    def drain(self) -> list[OutboundItem]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items


class VoiceSession:
    """单个 WebSocket 连接的 duplex 会话(模型共享,状态独立)。"""

    def __init__(self, models: Any) -> None:
        from channellm.duplex.driver import DuplexPipelineDriver
        from channellm.duplex.ingress import PcmIngress
        from channellm.duplex.queued_runtime import QueuedDuplexRuntime
        from channellm.duplex.runtime import RealtimeRuntime
        from channellm.engine.duplex_session import DuplexSession
        from channellm.engine.graph_decode import GraphDecodeSession
        from channellm.engine.talker import TalkerStream
        from channellm.engine.talker_graph_decode import TalkerGraphDecodeSession
        from channellm.pipeline.orchestrator import Orchestrator
        from channellm.pipeline.stages import CODEC_INITIAL_MIN_AUDIO_FRAMES

        self.models = models
        models.audio_front.reset()

        kv = models.make_thinker_kv()
        graph = GraphDecodeSession(models.thinker, kv)
        try:
            graph.capture()  # 必须在空 KV/prepare 之前
        except Exception as exc:
            # 捕获失败(如显存碎片/前序错误)不应拖垮会话:降级 eager decode。
            print(f"[session] think graph capture 失败,降级 eager: {exc!r}", flush=True)
            graph = None
        self.session = DuplexSession(models.thinker, kv, models.audio_front, graph=graph)
        self.session.prepare()

        self.talker_stream = TalkerStream(
            models.talker, early_first_frames=CODEC_INITIAL_MIN_AUDIO_FRAMES
        )
        self.talker_stream.graph = TalkerGraphDecodeSession(
            models.talker, self.talker_stream._kv
        )

        self.sink = StreamingPlaybackSink()
        self.runtime = RealtimeRuntime(
            Orchestrator(codec_initial_min_audio_frames=CODEC_INITIAL_MIN_AUDIO_FRAMES),
            self.sink,
        )
        self.driver = DuplexPipelineDriver(
            self.runtime,
            self.session,
            self.talker_stream,
            models.code2wav,
            response_text=lambda: models.audio_front.tokenizer.decode(
                self.session.res_ids, skip_special_tokens=True
            ),
        )
        self.queued = QueuedDuplexRuntime(self.driver)
        # chunk 尺寸必须对齐 processor 的流式块(16480/16000=1.03s),
        # 否则 mel 分帧错位会改变 duplex 决策行为。
        proc = models.audio_front.model.processor
        chunk_seconds = proc.get_streaming_chunk_size() / INPUT_SAMPLE_RATE
        self.ingress = PcmIngress(self.queued, chunk_seconds=chunk_seconds)
        self.tag = self.ingress.begin_speech("ws-session")
        self.sink.post_turn(self.tag)

        # 会话级遥测:上下行样本数 + 每 chunk 决策可见性。
        self.uplink_samples = 0
        self.downlink_samples = 0
        orig_publish = self.sink.publish

        def counted_publish(pcm, tag):
            self.downlink_samples += int(np.asarray(pcm).size)
            orig_publish(pcm, tag)

        self.sink.publish = counted_publish
        orig_process = self.driver.process_audio_chunk

        def logged_process(tag, pcm):
            decision = orig_process(tag, pcm)
            if decision is not None:
                kind = "LISTEN" if decision.is_listen else "SPEAK"
                print(
                    f"[chunk] {kind} tokens={decision.n_speak_tokens} "
                    f"embed={decision.cost_embed_ms:.0f}ms "
                    f"decision={decision.cost_decision_ms:.0f}ms",
                    flush=True,
                )
            return decision

        self.driver.process_audio_chunk = logged_process

    def feed_pcm16(self, payload: bytes) -> int:
        """提交一个上行 PCM16 帧,返回凑满并进入 GPU 队列的完整 unit 数。"""
        if len(payload) < 2:
            return 0
        i16 = np.frombuffer(payload, dtype=np.int16)
        self.uplink_samples += i16.size
        return int(self.ingress.push_frame(i16, sample_rate=INPUT_SAMPLE_RATE))

    def mark_eou(self) -> None:
        self.queued.on_eou(self.tag)

    def close(self, timeout_s: float = 2.0) -> None:
        """确定性释放会话资源:worker、KV 页、graph 会话。

        不等待 Python GC:每会话 ~268MB Talker KV + graph pool,靠 GC 回收
        会在多会话下累积并污染后续 capture(实测第 8 会话 capture 失效)。
        """
        self.queued.close(timeout_s=timeout_s)
        print(
            f"[session] summary: uplink={self.uplink_samples / INPUT_SAMPLE_RATE:.1f}s "
            f"downlink={self.downlink_samples / OUTPUT_SAMPLE_RATE:.1f}s",
            flush=True,
        )
        try:
            self.models.pool.free_seq(self.session.kv.seq)
        except Exception:  # 释放失败不应掩盖断连清理
            pass
        # 显式断开 graph 会话引用,CUDAGraph 随引用计数立即析构
        self.session.graph = None
        self.talker_stream.graph = None
        self._dropped = (self.queued, self.driver, self.runtime, self.ingress)
        import gc

        gc.collect()
