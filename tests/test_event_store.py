import sqlite3

from channellm.app.event_store import EventKind, EventStore


def test_wal_mode(tmp_path):
    with EventStore(tmp_path / "events.sqlite") as store:
        conn = sqlite3.connect(str(store.path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
    assert mode.lower() == "wal"


def test_append_and_iterate(tmp_path):
    with EventStore(tmp_path / "events.sqlite", session_epoch=1) as store:
        seq1 = store.append(
            EventKind.USER_SPEECH_FINAL,
            payload={"text": "帮我查一下明天天气"},
            turn_id="t1",
            speech_id="s1",
        )
        seq2 = store.append(EventKind.AGENT_SPEECH_PLANNED, payload={"text": "好的"}, turn_id="t1")
        events = list(store.iterate())
        assert [e.seq for e in events] == [seq1, seq2]
        assert events[0].payload["text"] == "帮我查一下明天天气"
        assert events[0].session_epoch == 1


def test_supersede_revision(tmp_path):
    with EventStore(tmp_path / "events.sqlite") as store:
        old = store.append(EventKind.USER_SPEECH_FINAL, payload={"text": "打给张三"}, turn_id="t1")
        new = store.supersede(
            old, EventKind.USER_SPEECH_FINAL, payload={"text": "打给张珊"}, turn_id="t1"
        )
        latest = store.latest_of_turn("t1", EventKind.USER_SPEECH_FINAL)
        assert latest is not None
        assert latest.seq == new
        assert latest.payload["text"] == "打给张珊"
        superseded = list(store.iterate(kind=EventKind.SUPERSEDED))
        assert len(superseded) == 1
        assert superseded[0].payload["by"] == new
