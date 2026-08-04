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
import time
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

# 声纹门:只在模型回复播放中门控候选语音,防环境音 barge-in 打断回复。
# 阈值来自实测分离度(campplus 同人不同段 ~0.6,人声 vs 噪声 ~0.05)。
VOICEPRINT_THRESHOLD = 0.35
VOICEPRINT_CONFIRM_S = 0.3     # 候选语音段挂起确认窗
VOICEPRINT_MIN_ENROLL_S = 1.5  # 注册最少有效语音
VOICEPRINT_MAX_ENROLL_S = 12.0  # 注册采集上限(防无界缓冲)
VOICEPRINT_PATH = Path(__file__).resolve().parents[2] / "voiceprint.npy"
# 回复"仍在播放"的判定:合成远快于实时(Talker ~5x),下行音频以突发送达,
# 客户端实际播放窗口 = 已发送音频时长 + 网络/起播余量,而非"最后发布时间"。
_REPLY_PLAYBACK_MARGIN_S = 0.3


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

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)

    def post_control(self, meta: dict[str, Any]) -> None:
        """JSON 控制消息统一走入站队列(与音频同序,避免并发 send)。"""
        with self._lock:
            self._items.append(OutboundItem("control", meta=meta))
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

    def __init__(self, models: Any, voiceprint: Any = None) -> None:
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
        self._last_publish_t: float | None = None
        self._published_since_clear = 0  # 上次 barge-in 后已发布的下行样本数
        # 声纹门:仅在模型回复播放中过滤非目标说话人,空闲收听零额外延迟。
        self._voiceprint = voiceprint
        self._speaker_gate = None
        if voiceprint is not None:
            from channellm.audio.speaker import SpeakerGate

            self._speaker_gate = SpeakerGate(
                voiceprint.embedder,
                voiceprint.embedding,
                threshold=VOICEPRINT_THRESHOLD,
                confirm_s=VOICEPRINT_CONFIRM_S,
            )
        self._enrolling = False
        self._enroll_runs: list[np.ndarray] = []
        self._enroll_run_parts: list[np.ndarray] = []
        self._enroll_voiced_n = 0
        self._idle_chunks = 0
        # Silero VAD 噪声门:加载失败自动降级能量阈值,不阻塞会话。
        try:
            from channellm.audio.vad import VoiceGate

            vad_path = Path(__file__).resolve().parents[2] / "assets/silero_vad.onnx"
            self._voice_gate = VoiceGate(
                vad_path, min_silence_ms=int(VAD_SILENCE_MS)
            )
            self._force_listen(True)  # 启动保护:首次真实语音前禁止自发说话
        except Exception as exc:  # noqa: BLE001 - 降级路径必须吞掉一切加载错误
            print(f"[session] Silero VAD 不可用,降级能量阈值: {exc!r}", flush=True)
            self._voice_gate = None
        orig_publish = self.sink.publish

        def counted_publish(pcm, tag):
            n = int(np.asarray(pcm).size)
            self.downlink_samples += n
            self._published_since_clear += n
            self._last_publish_t = time.monotonic()
            orig_publish(pcm, tag)

        self.sink.publish = counted_publish
        orig_mute = self.sink.mute

        def clearing_mute():
            self._published_since_clear = 0  # barge-in:客户端缓冲已清空
            orig_mute()

        self.sink.mute = clearing_mute
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
            if self._voice_gate is not None and not self._gate_speaking:
                self._force_listen(True)  # 回复结束:禁言直到下一次真实语音
            self._idle_chunks = 0
            print("[session] next turn (previous reply ended)", flush=True)
        wave = i16.astype(np.float32, copy=False) / 32768.0
        if self._voice_gate is not None:
            gated = self._voice_gate.feed(wave)
            if self._voice_gate.speaking != self._gate_speaking:
                self._gate_speaking = self._voice_gate.speaking
                print(f"[vad] speaking={self._gate_speaking}", flush=True)
                if self._gate_speaking:
                    self._force_listen(False)  # 真实语音到达:允许模型抢话
                    self._idle_chunks = 0
            if gated.size == 0:
                return 0  # 仍在 pad 前瞻缓冲内
            if self._voice_gate.speech_ended:
                self._voice_gate.speech_ended = False
                if self.ingress.flush_partial():
                    print("[vad] Silero 话音结束,提前冲刷半块", flush=True)
            gated = self._apply_voiceprint(gated)
            if gated.size == 0:
                return 0  # 声纹门扣留/拦截:不下发
            self._track_idle()
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

    def _force_listen(self, on: bool) -> None:
        """强制 duplex 决策为 listen(官方 force_listen_count 机制)。

        静默中模型会自发采样出 SPEAK tokens=1 伪回复并自我循环(每秒一次
        next turn,用户听到的是死寂+频繁打断)。无真实语音因由时不允许抢话:
        回复结束/会话开始 → 禁言,真实语音到达(VoicedGate 上升沿)→ 解除。
        """
        s = self.session
        want = s._generate_count + (1 << 30) if on else 0
        if (s.params.force_listen_count > s._generate_count) != on:
            print(f"[duplex] force-listen={'on' if on else 'off'}", flush=True)
        s.params.force_listen_count = want

    def _track_idle(self) -> None:
        """兜底:听完用户语音后模型迟迟不回复,8 个静默块后重新禁言。"""
        reply_recent = (
            self._last_publish_t is not None
            and time.monotonic() - self._last_publish_t < 2.0
        )
        if self._gate_speaking or reply_recent:
            self._idle_chunks = 0
            return
        self._idle_chunks += 1
        if self._idle_chunks == 8:
            self._force_listen(True)

    def _reply_active(self) -> bool:
        """回复音频仍在出站或客户端播放中(声纹门只在此期间工作)。

        合成远快于实时,下行以突发送达:播放窗口 ≈ 已发送音频时长,
        从最后一次发布时刻起算,再加网络/起播余量。
        """
        if self.sink.pending_count > 0:
            return True
        last = self._last_publish_t
        if last is None:
            return False
        playback_left = self._published_since_clear / OUTPUT_SAMPLE_RATE
        return (time.monotonic() - last) < playback_left + _REPLY_PLAYBACK_MARGIN_S

    def _apply_voiceprint(self, gated: np.ndarray) -> np.ndarray:
        if self._enrolling:
            self._enroll_collect(gated)
            return gated[:0]  # 注册期间音频不下发,模型保持安静
        gate = self._speaker_gate
        if gate is None or gate.print_emb is None:
            return gated
        out = gate.feed(gated, reply_active=self._reply_active())
        event, gate.event = gate.event, None
        if event is not None:
            labels = {
                "open": "声纹确认,放行",
                "muted": "声纹不符,已拦截",
                "dropped": "短语音未及确认,丢弃",
            }
            print(f"[voiceprint] {labels.get(event, event)}", flush=True)
            self.sink.post_control({"type": "gate", "state": event})
        return out

    def enroll_start(self) -> None:
        if self._voiceprint is None:
            self.sink.post_control({"type": "enroll_failed", "reason": "embedder_unavailable"})
            return
        self._enrolling = True
        self._enroll_runs = []
        self._enroll_run_parts = []
        self._enroll_voiced_n = 0
        self.sink.post_control({"type": "enroll_started"})
        print("[voiceprint] 注册开始", flush=True)

    def _enroll_collect(self, gated: np.ndarray) -> None:
        from channellm.audio.speaker import bridge_voiced_mask

        voiced = bridge_voiced_mask(gated)
        if not voiced.any():
            self._enroll_close_run()
            return
        change = np.flatnonzero(np.diff(voiced.astype(np.int8))) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [gated.size]])
        cap = int(VOICEPRINT_MAX_ENROLL_S * INPUT_SAMPLE_RATE)
        for s, e in zip(starts, ends):
            if voiced[s]:
                if self._enroll_voiced_n + (e - s) <= cap:
                    self._enroll_run_parts.append(gated[s:e])
                    self._enroll_voiced_n += e - s
            else:
                self._enroll_close_run()

    def _enroll_close_run(self) -> None:
        if self._enroll_run_parts:
            self._enroll_runs.append(np.concatenate(self._enroll_run_parts))
            self._enroll_run_parts = []

    def enroll_end(self) -> None:
        if not self._enrolling:
            return
        self._enrolling = False
        self._enroll_close_run()
        voiced_s = self._enroll_voiced_n / INPUT_SAMPLE_RATE
        if voiced_s < VOICEPRINT_MIN_ENROLL_S:
            self.sink.post_control(
                {"type": "enroll_failed", "reason": "too_short", "voiced_s": round(voiced_s, 2)}
            )
            print(f"[voiceprint] 注册失败:有效语音仅 {voiced_s:.1f}s", flush=True)
            return
        # run 可能被停顿切碎到 <0.3s(嵌入下限):拼接后按 ~2s 切块求均值,
        # 只要总有效语音达标就必然产出可靠嵌入。
        full = np.concatenate(self._enroll_runs)
        chunk_n = 2 * INPUT_SAMPLE_RATE
        chunks = [full[i:i + chunk_n] for i in range(0, len(full), chunk_n)]
        emb = self._voiceprint.embedder.embed_average(chunks)
        if emb is None:
            self.sink.post_control({"type": "enroll_failed", "reason": "no_valid_segment"})
            print("[voiceprint] 注册失败:无有效语音段", flush=True)
            return
        self._voiceprint.save(emb)
        if self._speaker_gate is not None:
            self._speaker_gate.print_emb = emb
        self.sink.post_control({"type": "enrolled", "voiced_s": round(voiced_s, 2)})
        print(f"[voiceprint] 注册完成({voiced_s:.1f}s 语音)→ {VOICEPRINT_PATH.name}", flush=True)

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
