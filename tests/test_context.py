from channellm.app.context import build_context_snapshot
from channellm.app.event_store import EventKind, EventStore


def test_markdown_projection_keeps_played_speech_and_excludes_unplayed_plan(tmp_path):
    with EventStore(tmp_path / "events.sqlite", session_epoch=3) as store:
        original = store.append(
            EventKind.USER_SPEECH_FINAL, payload={"text": "旧说法"}, turn_id="t1"
        )
        store.supersede(
            original,
            EventKind.USER_SPEECH_FINAL,
            payload={"text": "修订后的说法"},
            turn_id="t1",
        )
        store.append(EventKind.AGENT_SPEECH_PLANNED, payload={"text": "未播回复"})
        store.append(EventKind.AGENT_SPEECH_ACTUALLY_PLAYED, payload={"text": "已播回复"})

        markdown = store.markdown_projection()

    assert "修订后的说法" in markdown
    assert "已播回复" in markdown
    assert "旧说法" not in markdown
    assert "未播回复" not in markdown


def test_context_snapshot_is_token_budgeted_and_session_isolated(tmp_path):
    path = tmp_path / "events.sqlite"
    with EventStore(path, session_epoch=1) as store:
        store.append(EventKind.USER_SPEECH_FINAL, payload={"text": "甲"})
        store.append(EventKind.AGENT_SPEECH_ACTUALLY_PLAYED, payload={"text": "乙"})
    with EventStore(path, session_epoch=2) as store:
        store.append(EventKind.USER_SPEECH_FINAL, payload={"text": "其他会话"})
        snapshot = store.context_snapshot(
            budget_tokens=2,
            token_counter=lambda _text: 1,
            session_epoch=1,
        )

    assert snapshot.used_tokens == 2
    assert [entry.text for entry in snapshot.entries] == ["用户: 甲", "助手: 乙"]
    assert "其他会话" not in snapshot.text


def test_context_snapshot_skips_oversized_fact_without_exceeding_budget(tmp_path):
    with EventStore(tmp_path / "events.sqlite", session_epoch=1) as store:
        store.append(EventKind.USER_SPEECH_FINAL, payload={"text": "短"})
        store.append(EventKind.TASK_RESULT_READY, payload={"result": "很长的结果"})
        snapshot = build_context_snapshot(
            store.iterate(),
            session_epoch=1,
            budget_tokens=1,
            token_counter=lambda text: 2 if "很长" in text else 1,
        )

    assert snapshot.used_tokens == 1
    assert [entry.text for entry in snapshot.entries] == ["用户: 短"]
