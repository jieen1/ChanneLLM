"""L3 会话状态：Input / Reply / Notification / Task 四个互不抢占的状态域。"""

from __future__ import annotations

import dataclasses
import enum

from channellm.duplex.epoch import EpochTag


class TurnPhase(str, enum.Enum):
    LISTENING = "listening"
    THINKING = "thinking"  # EOU 后、开口前
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"  # barge-in 后旧 epoch 收尾


@dataclasses.dataclass
class InputDomain:
    active_tag: EpochTag | None = None
    eou_observed: bool = False


@dataclasses.dataclass
class ReplyDomain:
    generation_tag: EpochTag | None = None
    playout_tag: EpochTag | None = None
    playback_cursor_ms: int = 0


@dataclasses.dataclass
class NotificationDomain:
    pending_keys: set[str] = dataclasses.field(default_factory=set)
    active_key: str | None = None


@dataclasses.dataclass
class TaskDomain:
    pending_task_ids: set[str] = dataclasses.field(default_factory=set)


class SessionStateMachine:
    """只处理状态归属；模型、媒体和任务 worker 保持各自独立。

    关键不变量：新输入可取消 Reply 域，但不能清空已经落盘的 Task 或等待 idle
    窗口的 Notification。这样打断语音不会丢任务，也不会错误地把任务结果当作
    当前回复的一部分。
    """

    def __init__(self) -> None:
        self.phase = TurnPhase.LISTENING
        self.input = InputDomain()
        self.reply = ReplyDomain()
        self.notification = NotificationDomain()
        self.task = TaskDomain()

    @property
    def playback_cursor_ms(self) -> int:
        return self.reply.playback_cursor_ms

    @property
    def pending_replies(self) -> list[str]:
        """兼容旧读取口径；回复状态由单一 active generation 表示。"""
        return [str(self.reply.generation_tag)] if self.reply.generation_tag else []

    def on_input_start(self, tag: EpochTag) -> None:
        self.input.active_tag = tag
        self.input.eou_observed = False
        self.phase = TurnPhase.LISTENING

    def on_eou(self) -> None:
        self.input.eou_observed = True
        self.phase = TurnPhase.THINKING

    def on_speak_start(self, tag: EpochTag | None = None) -> None:
        self.reply.generation_tag = tag or self.input.active_tag
        self.reply.playout_tag = tag or self.input.active_tag
        self.phase = TurnPhase.SPEAKING

    def on_barge_in(self) -> None:
        """只清 Reply 域；Input 会由随后 ``on_input_start`` 建立新 epoch。"""
        self.reply.generation_tag = None
        self.reply.playout_tag = None
        self.reply.playback_cursor_ms = 0
        self.phase = TurnPhase.INTERRUPTED

    def on_reply_done(self) -> None:
        """模型生成结束；待播媒体仍由 ``reply.playout_tag`` 追踪。"""
        self.reply.generation_tag = None
        self.phase = TurnPhase.LISTENING

    def on_playout_finished(self, tag: EpochTag) -> None:
        if self.reply.playout_tag == tag:
            self.reply.playout_tag = None
            self.reply.playback_cursor_ms = 0

    def on_notification_enqueued(self, dedup_key: str) -> None:
        self.notification.pending_keys.add(dedup_key)

    def on_notification_started(self, dedup_key: str) -> None:
        self.notification.pending_keys.discard(dedup_key)
        self.notification.active_key = dedup_key

    def on_notification_finished(self, dedup_key: str) -> None:
        if self.notification.active_key == dedup_key:
            self.notification.active_key = None

    def on_task_enqueued(self, task_id: str) -> None:
        self.task.pending_task_ids.add(task_id)

    def on_task_finished(self, task_id: str) -> None:
        self.task.pending_task_ids.discard(task_id)
