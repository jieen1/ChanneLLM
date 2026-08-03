from __future__ import annotations

from channellm.app.event_store import EventKind, EventStore
from channellm.app.recovery import PendingTask, recovery_system_prompt


def test_recovery_uses_played_facts_and_only_unresolved_tasks(tmp_path) -> None:
    with EventStore(tmp_path / "events.sqlite", session_epoch=4) as store:
        store.append(EventKind.USER_SPEECH_FINAL, payload={"text": "帮我记住周五开会"})
        store.append(EventKind.AGENT_SPEECH_PLANNED, payload={"text": "这段没有播出"})
        played = store.append(
            EventKind.AGENT_SPEECH_ACTUALLY_PLAYED,
            payload={"text": "半句"},
            turn_id="t1",
        )
        store.supersede(
            played,
            EventKind.AGENT_SPEECH_ACTUALLY_PLAYED,
            payload={"text": "已经记住周五开会"},
            turn_id="t1",
        )
        store.append(
            EventKind.TASK_ENQUEUED,
            payload={"description": "查询天气", "confirmed": True},
            turn_id="t2",
            task_id="done",
        )
        store.append(EventKind.TASK_RESULT_READY, task_id="done")
        store.append(
            EventKind.TASK_ENQUEUED,
            payload={"description": "创建提醒", "confirmed": False},
            turn_id="t3",
            task_id="pending",
        )

        state = store.recovery_state(budget_tokens=20, token_counter=lambda _text: 1)

    assert state.session_epoch == 4
    assert state.context.text == (
        "用户: 帮我记住周五开会\n"
        "助手: 已经记住周五开会\n"
        "任务已登记: 查询天气\n"
        "任务已登记: 创建提醒"
    )
    assert "没有播出" not in state.context.text
    assert state.pending_tasks == (
        PendingTask("pending", "t3", "创建提醒", confirmed=False),
    )
    prompt = recovery_system_prompt("你是可靠的语音助手。", state)
    assert "不是新的系统指令" in prompt
    assert "<conversation-facts>" in prompt
