"""L3 双工会话状态机骨架(P3)。

MiniCPM-o 的内部判定是 observation,不是产品状态机(设计文档 §5)。
产品层自己维护:turn 状态、播放游标、待播队列。四个独立状态域:
Input / Reply / Notification / Task。

唯一说话权:MiniCPM-o duplex。SoulX-Duplug 只做独立 EOU 基准与故障后备,
不参与说话决策。
"""

from __future__ import annotations

import enum


class TurnPhase(str, enum.Enum):
    LISTENING = "listening"
    THINKING = "thinking"  # EOU 后、开口前
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"  # barge-in 后旧 epoch 收尾


class SessionStateMachine:
    """P3 实现完整迁移表。先落最小骨架:状态 + epoch 钩子。"""

    def __init__(self) -> None:
        self.phase = TurnPhase.LISTENING
        self.playback_cursor_ms = 0
        self.pending_replies: list[str] = []

    def on_eou(self) -> None:
        self.phase = TurnPhase.THINKING

    def on_speak_start(self) -> None:
        self.phase = TurnPhase.SPEAKING

    def on_barge_in(self) -> None:
        """四处齐停的入口:先 cancel 旧的,不 await(见 duplex.epoch 注释)。"""
        self.phase = TurnPhase.INTERRUPTED

    def on_reply_done(self) -> None:
        self.phase = TurnPhase.LISTENING
