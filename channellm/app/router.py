"""多标签 Router —— CHAT/NOTE/TASK 不互斥(设计文档 §6)。

「查一下 X,然后提醒我」两者都要。契约:
- 输出是多标签 proposal,不是三选一;全部记录。
- task candidate 必须单独确认;低置信人名/数字(R11)在有副作用动作前必须确认。
- P4 才接真实判定(LLM 或规则);这里先落契约与可测试的骨架。
"""

from __future__ import annotations

import dataclasses
import enum


class Label(str, enum.Enum):
    CHAT = "chat"
    NOTE = "note"
    TASK = "task"


@dataclasses.dataclass
class EntityMention:
    text: str
    entity_type: str  # person / number / proper_noun / ...
    confidence: float = 1.0


@dataclasses.dataclass
class RouteProposal:
    labels: set[Label]
    confidence: dict[Label, float] = dataclasses.field(default_factory=dict)
    task_candidate: bool = False
    needs_confirmation: bool = False
    confirmation_reason: str = ""
    entities: list[EntityMention] = dataclasses.field(default_factory=list)


CONFIRM_THRESHOLD = 0.8


def propose(
    text: str,
    labels: set[Label],
    confidence: dict[Label, float] | None = None,
    entities: list[EntityMention] | None = None,
) -> RouteProposal:
    """从上游判定结果构造 proposal,并强制确认策略。"""
    if not labels:
        labels = {Label.CHAT}
    confidence = confidence or {label: 1.0 for label in labels}
    entities = entities or []

    low_confidence = [entity for entity in entities if entity.confidence < CONFIRM_THRESHOLD]
    task_candidate = Label.TASK in labels
    needs_confirmation = bool(task_candidate and low_confidence)
    reason = (
        "low-confidence entities: " + ", ".join(e.text for e in low_confidence)
        if needs_confirmation
        else ""
    )
    return RouteProposal(
        labels=set(labels),
        confidence=confidence,
        task_candidate=task_candidate,
        needs_confirmation=needs_confirmation,
        confirmation_reason=reason,
        entities=entities,
    )
