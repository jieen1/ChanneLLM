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
from pathlib import Path
from typing import Any

import numpy as np

from channellm.duplex.epoch import EpochTag

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
_EPOCH_STRUCT = struct.Struct("<H")

# 服务端端点检测(能量 VAD):话音后静默达到阈值即提前冲刷半块,消除
# 1.03s chunk 的对齐等待(平均 ~515ms)。阈值对齐业界端点检测常见取值
# (OpenAI Realtime server_vad 默认 silence 500ms)。
VAD_VOICE_RMS = 0.008
VAD_SILENCE_MS = 450


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
        self._vad_voiced = False
        self._vad_silent_samples = 0
        self._gate_speaking = False
        # Silero VAD 噪声门:加载失败自动降级能量阈值,不阻塞会话。
        try:
            from channellm.audio.vad import VoiceGate

            vad_path = Path(__file__).resolve().parents[2] / "assets/silero_vad.onnx"
            self._voice_gate = VoiceGate(
                vad_path, min_silence_ms=int(VAD_SILENCE_MS)
            )
        except Exception as exc:  # noqa: BLE001 - 降级路径必须吞掉一切加载错误
            print(f"[session] Silero VAD 不可用,降级能量阈值: {exc!r}", flush=True)
            self._voice_gate = None
        orig_publish = self.sink.publish

        def counted_publish(pcm, tag):
            self.downlink_samples += int(np.asarray(pcm).size)
            orig_publish(pcm, tag)

        self.sink.publish = counted_publish
        orig_process = self.driver.process_audio_chunk

        import time as _time

        self._t0 = _time.monotonic()

        def logged_process(tag, pcm):
            decision = orig_process(tag, pcm)
            if decision is not None:
                kind = "LISTEN" if decision.is_listen else "SPEAK"
                print(
                    f"[chunk@{_time.monotonic() - self._t0:6.2f}s] {kind} "
                    f"tokens={decision.n_speak_tokens} "
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
        # 多回合:上一回复 finish_turn 后 active_tag 清空,持续输入必须进入
        # 新回合(顺带复位 talker/code2wav 流式状态),否则 submit_audio 全被拒。
        if self.queued.active_tag is None:
            self.tag = self.ingress.refresh_turn("ws-session")
            self.sink.post_turn(self.tag)
            print("[session] next turn (previous reply ended)", flush=True)
        wave = i16.astype(np.float32, copy=False) / 32768.0
        if self._voice_gate is not None:
            gated = self._voice_gate.feed(wave)
            if self._voice_gate.speaking != self._gate_speaking:
                self._gate_speaking = self._voice_gate.speaking
                print(f"[vad] speaking={self._gate_speaking}", flush=True)
            if gated.size == 0:
                return 0  # 仍在 pad 前瞻缓冲内
            if self._voice_gate.speech_ended:
                self._voice_gate.speech_ended = False
                if self.ingress.flush_partial():
                    print("[vad] Silero 话音结束,提前冲刷半块", flush=True)
            return int(self.ingress.push_frame(gated, sample_rate=INPUT_SAMPLE_RATE))
        self._vad_feed(i16)  # Silero 不可用时的能量阈值降级
        return int(self.ingress.push_frame(wave, sample_rate=INPUT_SAMPLE_RATE))

    def _vad_feed(self, i16: np.ndarray) -> None:
        """能量 VAD:检测到"说话后静默"即提前冲刷,消除 chunk 对齐等待。"""
        rms = float(np.sqrt(np.mean(i16.astype(np.float32) ** 2))) / 32768.0
        if rms > VAD_VOICE_RMS:
            self._vad_voiced = True
            self._vad_silent_samples = 0
            return
        if not self._vad_voiced:
            return
        self._vad_silent_samples += i16.size
        if self._vad_silent_samples >= int(INPUT_SAMPLE_RATE * VAD_SILENCE_MS / 1000):
            self._vad_voiced = False
            self._vad_silent_samples = 0
            if self.ingress.flush_partial():
                import time as _time

                print(
                    f"[vad@{_time.monotonic() - self._t0:6.2f}s] 话音结束,提前冲刷半块"
                    f"(silence>={VAD_SILENCE_MS}ms)",
                    flush=True,
                )

    def mark_eou(self) -> None:
        self.queued.on_eou(self.tag)

    def close(self, timeout_s: float = 2.0) -> None:
        """确定性释放会话资源:worker、KV 页、graph 会话。

        不等待 Python GC:每会话 ~268MB Talker KV + graph pool,靠 GC 回收
        会在多会话下累积并污染后续 capture(实测第 8 会话 capture 失效)。
        """
        self.queued.close(timeout_s=timeout_s)
        stats = self.queued.stats
        print(
            f"[session] summary: uplink={self.uplink_samples / INPUT_SAMPLE_RATE:.1f}s "
            f"downlink={self.downlink_samples / OUTPUT_SAMPLE_RATE:.1f}s "
            f"queue(enq={stats.enqueued} proc={stats.processed} "
            f"drop_over={stats.dropped_overrun} drop_stale={stats.dropped_stale} "
            f"fail={stats.failures})",
            flush=True,
        )
        for failure in self.queued.failures[:2]:
            import traceback

            print(
                "[session] worker failure: "
                + "".join(
                    traceback.format_exception(type(failure), failure, failure.__traceback__)
                )[-1500:],
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
