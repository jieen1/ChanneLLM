"""单进程全双工会话运行时的控制面。

这里不拥有模型或媒体 SDK：引擎把各 stage 的增量产物提交进来，播放端实现
``PlaybackSink``。运行时负责 L3/L4 不变量——epoch 取消不等待、旧音频绝不
播放、事实事件与延迟锚点在同一回合上记录。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from channellm.app.event_store import EventKind, EventStore
from channellm.duplex.epoch import EpochGuard, EpochTag
from channellm.duplex.session import SessionStateMachine
from channellm.pipeline.orchestrator import Orchestrator, new_request_id
from channellm.pipeline.stages import StageId
from channellm.tracing.schema import Anchor


class PlaybackSink(Protocol):
    """媒体适配器的最小契约；LiveKit 与本地扬声器都可实现它。"""

    def mute(self) -> None:
        """立即丢弃本地待播缓冲。不得等待旧 response 收尾。"""

    def publish(self, pcm: Any, tag: EpochTag) -> None:
        """发布一个已经合成的 PCM 块。"""


@dataclass
class _ActiveTurn:
    request_id: str
    tag: EpochTag
    trace_id: str
    planned: bool = False
    speak_decision: bool = False
    streaming_prefill_started: bool = False
    streaming_prefill_done: bool = False
    first_token_decoded: bool = False
    talker_chunk_ready: bool = False
    response_text: str = ""
    cleaned: bool = False


class RealtimeRuntime:
    """把 L2 stage 输出接入 epoch、播放、事件与 trace 的会话主循环。

    此对象刻意保持同步且无后台任务：同进程的音频循环可以在每个 chunk 边界
    调用它，而实际 GPU stage 和网络发送由外层驱动。这样 barge-in 的关键路径
    只有 ``cancel + mute + epoch advance``，不会被任何 await 阻塞。
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        sink: PlaybackSink,
        trace_recorder: Any | None = None,
        event_store: EventStore | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.sink = sink
        self.trace_recorder = trace_recorder
        self.event_store = event_store
        self.epoch_guard = EpochGuard()
        self.state_machine = SessionStateMachine()
        self._active: _ActiveTurn | None = None
        self._cancelled_request_ids: list[str] = []
        self._playout_traces: dict[EpochTag, str] = {}
        self._playout_requests: dict[EpochTag, str] = {}
        self._playout_texts: dict[EpochTag, str] = {}
        self._playout_played_event_seqs: dict[EpochTag, int] = {}
        self._started_playout_tags: set[EpochTag] = set()
        self._pending_playout: EpochTag | None = None
        # 输入控制、GPU worker 与媒体 writer 会并发调用本对象。临界区只覆盖
        # epoch/事件状态，不包裹模型 forward，因而 barge-in 不会等待 GPU 工作。
        self._lock = RLock()

    @property
    def active_tag(self) -> EpochTag | None:
        with self._lock:
            return self._active.tag if self._active is not None else None

    def begin_turn(self, speech_id: str = "") -> EpochTag:
        """开始输入回合；若正在回复，先同步打断旧回合且绝不等待它。"""
        with self._lock:
            previous = self._active
            if previous is not None:
                self._barge_in_active(previous)
            elif self._pending_playout is not None:
                self._barge_in_pending_playout(self._pending_playout)

            tag = self.epoch_guard.advance(speech_id)
            request_id = new_request_id()
            self.orchestrator.submit_initial(request_id, tag.turn_epoch, tag.speech_id)
            self.state_machine.on_input_start(tag)
            self._active = _ActiveTurn(
                request_id=request_id,
                tag=tag,
                trace_id=self._new_trace_id(),
            )
            return tag

    def reap_cancelled(self) -> int:
        """回收已经 cancel 的请求，不等待模型任务完成。

        由外层媒体循环在安全 tick 调用；把这一步与 barge-in 的同步临界区
        分开，既满足 cancel-not-await，又避免 cancelled request 无限滞留。
        """
        with self._lock:
            request_ids = self._cancelled_request_ids
            self._cancelled_request_ids = []
            for request_id in request_ids:
                self.orchestrator.cleanup(request_id)
            return len(request_ids)

    def on_eou(self, tag: EpochTag | None = None) -> bool:
        """记录模型的 EOU 观察；过期观察不能改变当前会话状态。"""
        with self._lock:
            active = self._active
            if active is None or not self._is_current(tag or active.tag):
                return False
            self.state_machine.on_eou()
            self._anchor(Anchor.EOU_DETECTED, active.tag, active.trace_id)
            return True

    def on_speak_decision(self, tag: EpochTag) -> bool:
        """记录 MiniCPM-o 首次决定开口的时刻。

        这是 Thinker 已作出说话判断、尚未进入 Talker 生成的边界。不能借用
        ``TALKER_CHUNK_READY`` 的时刻，否则会把 Talker 首块生成耗时伪装成
        Thinker 决策耗时，污染 EOU waterfall。
        """
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            if not active.speak_decision:
                active.speak_decision = True
                self._anchor(Anchor.SPEAK_DECISION, tag, active.trace_id)
            return True

    def on_streaming_prefill_start(self, tag: EpochTag, *, ts_ns: int | None = None) -> bool:
        """记录 Thinker 开始处理一个输入 unit 的时刻。"""
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            if not active.streaming_prefill_started:
                active.streaming_prefill_started = True
                self._anchor(Anchor.STREAMING_PREFILL_START, tag, active.trace_id, ts_ns=ts_ns)
            return True

    def on_streaming_prefill_done(self, tag: EpochTag, *, ts_ns: int | None = None) -> bool:
        """记录当前回复首个 Thinker token 之前的输入 prefill 已完成。"""
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            if not active.streaming_prefill_done:
                active.streaming_prefill_done = True
                self._anchor(Anchor.THINKER_PREFILL_DONE, tag, active.trace_id, ts_ns=ts_ns)
            return True

    def on_first_token_decoded(self, tag: EpochTag, *, ts_ns: int | None = None) -> bool:
        """记录当前回复的首个 Thinker token 已由模型采样。"""
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            if not active.first_token_decoded:
                active.first_token_decoded = True
                self._anchor(Anchor.FIRST_TOKEN_DECODED, tag, active.trace_id, ts_ns=ts_ns)
            return True

    def set_response_text(self, tag: EpochTag, text: str) -> bool:
        """保存本回合最终文本，供真正播放后写入 L4 事实。

        这不是 transcript 的权威写入：只有媒体 writer 报告首帧 handoff 后，文本
        才会伴随 ``AgentSpeechActuallyPlayed`` 进入事件日志。
        """
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            active.response_text = text.strip()
            if active.planned:
                self._playout_texts[tag] = active.response_text
                played_seq = self._playout_played_event_seqs.get(tag)
                if played_seq is not None and self.event_store is not None:
                    self._playout_played_event_seqs[tag] = self.event_store.supersede(
                        played_seq,
                        EventKind.AGENT_SPEECH_ACTUALLY_PLAYED,
                        payload={"text": active.response_text},
                        turn_id=self._playout_requests.get(tag),
                        speech_id=tag.speech_id,
                    )
            return True

    def on_device_playout_start(self, tag: EpochTag) -> bool:
        """由媒体 writer 在设备/下行真正取走首块 PCM 时调用。

        即使回复 stage 已结束，最新 epoch 的已发布 PCM 仍可被设备播放；因此
        trace id 独立保存，不依赖仍存在的 ``_active`` 请求。被 barge-in 作废
        的 tag 会在 ``begin_turn`` 被移除，不能留下虚假的设备播放锚点。
        """
        with self._lock:
            trace_id = self._playout_traces.get(tag)
            if trace_id is None or not self._is_current(tag):
                return False
            if tag not in self._started_playout_tags:
                self._started_playout_tags.add(tag)
                self._anchor(Anchor.DEVICE_PLAYOUT_START, tag, trace_id)
                if self.event_store is not None:
                    self._playout_played_event_seqs[tag] = self.event_store.append(
                        EventKind.AGENT_SPEECH_ACTUALLY_PLAYED,
                        payload={"text": self._playout_texts.get(tag, "")},
                        turn_id=self._playout_requests.get(tag),
                        speech_id=tag.speech_id,
                    )
            return True

    def on_device_playout_finished(self, tag: EpochTag) -> bool:
        """媒体 writer 已播放完当前回复的全部待播 PCM。"""
        with self._lock:
            if self._pending_playout != tag:
                return False
            self._pending_playout = None
            self._playout_traces.pop(tag, None)
            self._playout_requests.pop(tag, None)
            self._playout_texts.pop(tag, None)
            self._playout_played_event_seqs.pop(tag, None)
            self._started_playout_tags.discard(tag)
            self.state_machine.on_playout_finished(tag)
            return True

    def submit_stage_output(
        self,
        tag: EpochTag,
        stage: StageId,
        payload: Any = None,
        *,
        final: bool = False,
    ) -> list[Any]:
        """推进一个 stage 的产物，并只播放当前 epoch 的真实 PCM。

        返回值是编排器产生的下游输入，供外层立即提交到下一个模型 stage。
        过期回合在调用编排器前即丢弃，避免旧 token/codec 再占用 GPU。
        """
        with self._lock:
            active = self._active
            if active is None or not self._is_current(tag) or active.tag != tag:
                return []

            emitted = self._submit_to_orchestrator(active.request_id, stage, payload, final)
            for chunk in emitted:
                if not self._is_current(tag):
                    break
                self._record_pipeline_progress(active, chunk)
                if self._is_pcm_output(chunk):
                    self._publish_pcm(active, chunk)
            return emitted

    def finish_turn(self, tag: EpochTag) -> bool:
        """结束当前回合，即使没有音频产出也清理 resumable 请求。"""
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            if not active.cleaned:
                self.orchestrator.cleanup(active.request_id)
                active.cleaned = True
            if active.planned:
                finish = getattr(self.sink, "finish", None)
                if callable(finish):
                    finish(tag)
            self.state_machine.on_reply_done()
            self._active = None
            return True

    def abort_turn(self, tag: EpochTag, reason: str) -> bool:
        """因不可安全播放的输出终止当前回复，不等待 GPU worker 收尾。

        已经由设备取走的健康 PCM 仍是事实，不能倒写；尚在本地缓冲的内容会被
        mute。后续同 epoch 的模型输出因 ``_active`` 已清空而无法再发布。
        """
        with self._lock:
            active = self._active
            if active is None or active.tag != tag or not self._is_current(tag):
                return False
            self.orchestrator.cancel(active.request_id)
            self._cancelled_request_ids.append(active.request_id)
            self.sink.mute()
            self._anchor(
                Anchor.PCM_QUALITY_REJECTED,
                tag,
                active.trace_id,
                reason=reason,
            )
            self._anchor(Anchor.PLAYOUT_MUTED, tag, active.trace_id)
            if self.event_store is not None:
                self.event_store.append(
                    EventKind.AGENT_SPEECH_REJECTED,
                    payload={"reason": reason},
                    turn_id=active.request_id,
                    speech_id=tag.speech_id,
                )
            self._clear_playout_state(tag)
            self.state_machine.on_barge_in()
            self.state_machine.on_reply_done()
            self._active = None
            return True

    def _submit_to_orchestrator(
        self, request_id: str, stage: StageId, payload: Any, final: bool
    ) -> list[Any]:
        # 测试替身及早期引擎适配器仅有旧的三参接口；生产 Orchestrator 支持
        # final，用其保证尾包冲刷。兼容分支不改变正常运行时语义。
        if final:
            try:
                return self.orchestrator.submit_update(
                    request_id, stage, payload, final=True
                )
            except TypeError as exc:
                if "final" not in str(exc):
                    raise
        return self.orchestrator.submit_update(request_id, stage, payload)

    def _publish_pcm(self, active: _ActiveTurn, chunk: Any) -> None:
        payload = getattr(chunk, "payload", None)
        if payload is None:
            return
        tag = _chunk_tag(chunk, active.tag)
        if tag != active.tag or not self._is_current(tag):
            return
        if not active.planned:
            active.planned = True
            self._playout_traces[active.tag] = active.trace_id
            self._playout_requests[active.tag] = active.request_id
            self._playout_texts[active.tag] = active.response_text
            self._pending_playout = active.tag
            self.state_machine.on_speak_start(active.tag)
            self._event(EventKind.AGENT_SPEECH_PLANNED, active)
            self._anchor(Anchor.CODE2WAV_FIRST_PCM, active.tag, active.trace_id)
        self.sink.publish(payload, active.tag)
        self._anchor(Anchor.PUBLISHED, active.tag, active.trace_id)

    def _record_pipeline_progress(self, active: _ActiveTurn, chunk: Any) -> None:
        if (
            getattr(chunk, "source", None) is StageId.TALKER
            and not active.talker_chunk_ready
            and getattr(chunk, "payload", None) is not None
        ):
            active.talker_chunk_ready = True
            self._anchor(Anchor.TALKER_CHUNK_READY, active.tag, active.trace_id)

    @staticmethod
    def _is_pcm_output(chunk: Any) -> bool:
        if getattr(chunk, "stage", None) is not StageId.CODE2WAV:
            return False
        # L2 的 Talker→Code2Wav 消息是 codec token，不能当 PCM 提前播放。
        source = getattr(chunk, "source", None)
        return source is None or source is StageId.CODE2WAV

    def _is_current(self, tag: EpochTag) -> bool:
        return self.epoch_guard.accept(tag)

    def _anchor(
        self,
        anchor: str,
        tag: EpochTag,
        trace_id: str,
        *,
        ts_ns: int | None = None,
        **extra: Any,
    ) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.anchor(
                anchor,
                trace_id=trace_id,
                turn_epoch=tag.turn_epoch,
                speech_id=tag.speech_id,
                ts_ns=ts_ns,
                **extra,
            )

    def _new_trace_id(self) -> str:
        if self.trace_recorder is not None:
            return self.trace_recorder.new_trace()
        return uuid.uuid4().hex[:16]

    def _event(self, kind: EventKind, active: _ActiveTurn) -> None:
        if self.event_store is not None:
            self.event_store.append(
                kind,
                turn_id=active.request_id,
                speech_id=active.tag.speech_id,
            )

    def _barge_in_active(self, active: _ActiveTurn) -> None:
        self._anchor(Anchor.BARGE_IN_DETECTED, active.tag, active.trace_id)
        self.state_machine.on_barge_in()
        self.orchestrator.cancel(active.request_id)
        self._cancelled_request_ids.append(active.request_id)
        self.sink.mute()
        self._anchor(Anchor.PLAYOUT_MUTED, active.tag, active.trace_id)
        self._clear_playout_state(active.tag)

    def _barge_in_pending_playout(self, tag: EpochTag) -> None:
        trace_id = self._playout_traces.get(tag)
        if trace_id is not None:
            self._anchor(Anchor.BARGE_IN_DETECTED, tag, trace_id)
        self.state_machine.on_barge_in()
        self.sink.mute()
        if trace_id is not None:
            self._anchor(Anchor.PLAYOUT_MUTED, tag, trace_id)
        self._clear_playout_state(tag)

    def _clear_playout_state(self, tag: EpochTag) -> None:
        self._playout_traces.pop(tag, None)
        self._playout_requests.pop(tag, None)
        self._playout_texts.pop(tag, None)
        self._playout_played_event_seqs.pop(tag, None)
        self._started_playout_tags.discard(tag)
        if self._pending_playout == tag:
            self._pending_playout = None


def _chunk_tag(chunk: Any, fallback: EpochTag) -> EpochTag:
    """兼容 L2 原生 chunk 与只提供 ``tag`` 的媒体适配器。"""
    tag = getattr(chunk, "tag", None)
    if isinstance(tag, EpochTag):
        return tag
    turn_epoch = getattr(chunk, "turn_epoch", fallback.turn_epoch)
    speech_id = getattr(chunk, "speech_id", fallback.speech_id)
    return EpochTag(turn_epoch=turn_epoch, speech_id=speech_id)
