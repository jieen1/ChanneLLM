"""从 L4 事实事件重建投影视图与 token-budgeted 上下文。

事件日志是唯一事实源；本模块的 Markdown 和 ``ContextSnapshot`` 都是可丢弃、
可重建的投影。尤其不能把 ``AgentSpeechPlanned`` 当作已经说出口的上下文，
否则被 barge-in 丢弃的回复会在后续回合中变成伪事实。
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable

from channellm.app.event_store import Event, EventKind

TokenCounter = Callable[[str], int]


@dataclasses.dataclass(frozen=True)
class ContextEntry:
    """一个已纳入模型上下文的、可追溯到单个事件的文本事实。"""

    seq: int
    text: str
    tokens: int


@dataclasses.dataclass(frozen=True)
class ContextSnapshot:
    """固定 token 上限的会话上下文投影。"""

    session_epoch: int
    budget_tokens: int
    used_tokens: int
    entries: tuple[ContextEntry, ...]

    @property
    def text(self) -> str:
        return "\n".join(entry.text for entry in self.entries)


def estimate_tokens(text: str) -> int:
    """无 tokenizer 时的保守估算；生产调用应注入目标模型 tokenizer。"""
    pieces = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
    return len(pieces)


def effective_events(
    events: Iterable[Event], *, session_epoch: int
) -> list[Event]:
    """过滤其他 session、修订前版本与事件元标记，保留当前有效事实。"""
    scoped = [event for event in events if event.session_epoch == session_epoch]
    scoped.sort(key=lambda event: event.seq)
    superseded = {event.supersedes for event in scoped if event.supersedes is not None}
    return [
        event
        for event in scoped
        if event.seq not in superseded and event.kind != EventKind.SUPERSEDED.value
    ]


def render_event(event: Event) -> str | None:
    """把一个有效事件映射成用户可见/模型可用的事实文本。"""
    text = _payload_text(event.payload, "text")
    if event.kind == EventKind.USER_SPEECH_FINAL.value and text:
        return f"用户: {text}"
    if event.kind == EventKind.USER_BACKCHANNEL_OBSERVED.value and text:
        return f"用户附和: {text}"
    if event.kind == EventKind.AGENT_SPEECH_ACTUALLY_PLAYED.value and text:
        return f"助手: {text}"
    if event.kind == EventKind.TASK_ENQUEUED.value:
        description = _payload_text(event.payload, "description")
        if description:
            return f"任务已登记: {description}"
    if event.kind == EventKind.TASK_RESULT_READY.value:
        result = _payload_text(event.payload, "result") or text
        if result:
            return f"任务结果: {result}"
    return None


def render_markdown(events: Iterable[Event], *, session_epoch: int) -> str:
    """生成可随时重建的 Markdown 投影，不改变事件库。"""
    lines = ["# 会话投影", ""]
    for event in effective_events(events, session_epoch=session_epoch):
        rendered = render_event(event)
        if rendered is not None:
            lines.append(f"- {rendered}")
    return "\n".join(lines) + "\n"


def build_context_snapshot(
    events: Iterable[Event],
    *,
    session_epoch: int,
    budget_tokens: int,
    token_counter: TokenCounter = estimate_tokens,
) -> ContextSnapshot:
    """在预算内选择最新的完整事实，而不是按固定回合数硬截断。"""
    if budget_tokens < 0:
        raise ValueError("budget_tokens must be non-negative")

    selected: list[ContextEntry] = []
    used_tokens = 0
    rendered = (
        (event, render_event(event))
        for event in effective_events(events, session_epoch=session_epoch)
    )
    for event, text in reversed(list(rendered)):
        if text is None:
            continue
        tokens = token_counter(text)
        if tokens < 0:
            raise ValueError("token_counter must not return a negative value")
        if used_tokens + tokens > budget_tokens:
            continue
        selected.append(ContextEntry(seq=event.seq, text=text, tokens=tokens))
        used_tokens += tokens
    selected.reverse()
    return ContextSnapshot(
        session_epoch=session_epoch,
        budget_tokens=budget_tokens,
        used_tokens=used_tokens,
        entries=tuple(selected),
    )


def _payload_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) else ""
