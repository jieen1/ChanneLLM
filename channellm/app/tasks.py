"""任务派发 —— 双时钟不变量(设计文档 §3)。

1. 会话主循环不 await 外部模型/HTTP/长事务:task enqueue 落盘即返回。
2. 网络发送由独立 dispatch worker 负责(崩溃不影响实时媒体)。
3. task result 只进通知队列,播不播由仲裁器决定。
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable
from typing import Any

from channellm.app.arbitration import Notification
from channellm.app.event_store import EventKind, EventStore


@dataclasses.dataclass
class Task:
    task_id: str
    description: str
    confirmed: bool  # 低置信实体必须确认后才允许有副作用动作(R11)
    turn_id: str | None = None


class TaskDispatcher:
    """enqueue 同步落盘;dispatch 循环独立运行。"""

    def __init__(
        self,
        store: EventStore,
        send: Callable[[Task], Any] | None = None,
        notify: Callable[[Notification], bool] | None = None,
    ) -> None:
        self.store = store
        self.send = send  # P4: 外部大模型 API / MCP
        self.notify = notify

    def enqueue(self, description: str, turn_id: str, confirmed: bool) -> str:
        task_id = uuid.uuid4().hex[:12]
        self.store.append(
            EventKind.TASK_ENQUEUED,
            payload={"description": description, "confirmed": confirmed},
            turn_id=turn_id,
            task_id=task_id,
        )
        return task_id

    def dispatch_one(self, task: Task) -> Any:
        """在独立 worker 调用外部执行器，并把结果写回 L4/通知队列。"""
        if not task.confirmed:
            raise ValueError("unconfirmed task must not trigger side effects")
        if self.send is None:
            return None
        result = self.send(task)
        self.record_result(task, result)
        return result

    def record_result(self, task: Task, result: Any) -> None:
        """持久化结果，并可选进入由 arbiter 控制的通知队列。"""
        self.store.append(
            EventKind.TASK_RESULT_READY,
            payload={"result": result},
            turn_id=task.turn_id,
            task_id=task.task_id,
        )
        if self.notify is not None:
            text = result if isinstance(result, str) else str(result)
            self.notify(Notification(dedup_key=f"task-result:{task.task_id}", text=text))
