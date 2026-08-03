"""任务派发 —— 双时钟不变量(设计文档 §3)。

1. 会话主循环不 await 外部模型/HTTP/长事务:task enqueue 落盘即返回。
2. 网络发送由独立 dispatch worker 负责(崩溃不影响实时媒体)。
3. task result 只进通知队列,播不播由仲裁器决定。
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable

from channellm.app.event_store import EventKind, EventStore


@dataclasses.dataclass
class Task:
    task_id: str
    description: str
    confirmed: bool  # 低置信实体必须确认后才允许有副作用动作(R11)


class TaskDispatcher:
    """enqueue 同步落盘;dispatch 循环独立运行。"""

    def __init__(self, store: EventStore, send: Callable[[Task], None] | None = None) -> None:
        self.store = store
        self.send = send  # P4: 外部大模型 API / MCP

    def enqueue(self, description: str, turn_id: str, confirmed: bool) -> str:
        task_id = uuid.uuid4().hex[:12]
        self.store.append(
            EventKind.TASK_ENQUEUED,
            payload={"description": description, "confirmed": confirmed},
            turn_id=turn_id,
            task_id=task_id,
        )
        return task_id

    def dispatch_one(self, task: Task) -> None:
        if not task.confirmed:
            raise ValueError("unconfirmed task must not trigger side effects")
        if self.send is not None:
            self.send(task)
