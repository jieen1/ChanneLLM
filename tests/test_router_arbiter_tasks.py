import pytest

from channellm.app.arbitration import Notification, PlaybackArbiter
from channellm.app.event_store import EventStore
from channellm.app.router import EntityMention, Label, propose
from channellm.app.tasks import Task, TaskDispatcher


def test_router_labels_not_exclusive():
    proposal = propose("查一下 X,然后提醒我", labels={Label.CHAT, Label.TASK})
    assert Label.CHAT in proposal.labels
    assert Label.TASK in proposal.labels
    assert proposal.task_candidate


def test_router_low_confidence_needs_confirmation():
    proposal = propose(
        "打给张三",
        labels={Label.TASK},
        entities=[EntityMention(text="张三", entity_type="person", confidence=0.5)],
    )
    assert proposal.needs_confirmation
    assert "张三" in proposal.confirmation_reason


def test_arbiter_user_speech_wins():
    arbiter = PlaybackArbiter()
    assert arbiter.enqueue(Notification(dedup_key="n1", text="任务完成"))
    assert not arbiter.enqueue(Notification(dedup_key="n1", text="任务完成"))  # 幂等
    arbiter.on_user_speech_start()
    assert arbiter.next_playable() is None  # 用户说话时不播报
    arbiter.on_user_speech_end()
    notification = arbiter.next_playable()
    assert notification is not None and notification.played


def test_task_enqueue_persists(tmp_path):
    with EventStore(tmp_path / "events.sqlite") as store:
        dispatcher = TaskDispatcher(store)
        task_id = dispatcher.enqueue("订会议室", turn_id="t1", confirmed=True)
        assert task_id
        from channellm.app.event_store import EventKind

        events = list(store.iterate(kind=EventKind.TASK_ENQUEUED))
        assert len(events) == 1
        assert events[0].task_id == task_id
        with pytest.raises(ValueError):
            dispatcher.dispatch_one(Task(task_id=task_id, description="x", confirmed=False))


def test_task_result_persists_then_enters_idle_window_notification(tmp_path):
    with EventStore(tmp_path / "events.sqlite") as store:
        arbiter = PlaybackArbiter()
        dispatcher = TaskDispatcher(
            store,
            send=lambda task: f"完成：{task.description}",
            notify=arbiter.enqueue,
        )
        task = Task(task_id="task-1", description="订会议室", confirmed=True, turn_id="turn-1")

        assert dispatcher.dispatch_one(task) == "完成：订会议室"
        events = list(store.iterate(kind="TaskResultReady"))
        assert [(event.task_id, event.turn_id, event.payload) for event in events] == [
            ("task-1", "turn-1", {"result": "完成：订会议室"})
        ]
        arbiter.on_user_speech_start()
        assert arbiter.next_playable() is None
        arbiter.on_user_speech_end()
        notification = arbiter.next_playable()

    assert notification is not None
    assert notification.text == "完成：订会议室"
