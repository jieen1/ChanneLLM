"""事件存储 —— L4 事实源(设计文档 §6)。

事实源是事件日志,不是 transcript:markdown 不能当权威(附和可能被计入、
打断后有文本从未播放、ASR 后续会修订、append 无事务与顺序保证)。

authority:SQLite WAL,append-only,单写者。
修订语义:ASR 修订/重说 = 追加新事件并 supersedes 旧 seq,旧事件永不删除。
markdown/文本视图只是投影,可随时从事件重建。
"""

from __future__ import annotations

import enum
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class EventKind(str, enum.Enum):
    USER_SPEECH_FINAL = "UserSpeechFinal"
    USER_BACKCHANNEL_OBSERVED = "UserBackchannelObserved"
    AGENT_SPEECH_PLANNED = "AgentSpeechPlanned"
    AGENT_SPEECH_ACTUALLY_PLAYED = "AgentSpeechActuallyPlayed"  # 与 planned 分开
    AGENT_SPEECH_REJECTED = "AgentSpeechRejected"  # PCM 硬门禁拒绝，未交付播放
    TASK_ENQUEUED = "TaskEnqueued"
    TASK_RESULT_READY = "TaskResultReady"
    SUPERSEDED = "Superseded"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_epoch INTEGER NOT NULL,
    ts_ns        INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    turn_id      TEXT,
    speech_id    TEXT,
    task_id      TEXT,
    supersedes   INTEGER REFERENCES events(seq),
    payload      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_turn ON events(turn_id);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
"""


@dataclass
class Event:
    seq: int
    session_epoch: int
    ts_ns: int
    kind: str
    turn_id: str | None = None
    speech_id: str | None = None
    task_id: str | None = None
    supersedes: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class EventStore:
    """单写者事件日志。一个实例独占一个 db 文件。"""

    def __init__(self, path: str | Path, session_epoch: int = 0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_epoch = session_epoch
        # L3 GPU worker 与媒体 writer 都会追加事实；连接允许跨线程，但所有
        # connection 操作仍由此锁串行，保持 SQLite WAL 的单写者不变量。
        self._write_lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    def append(
        self,
        kind: EventKind | str,
        payload: dict[str, Any] | None = None,
        turn_id: str | None = None,
        speech_id: str | None = None,
        task_id: str | None = None,
        supersedes: int | None = None,
        ts_ns: int | None = None,
    ) -> int:
        kind_value = kind.value if isinstance(kind, EventKind) else str(kind)
        with self._write_lock:
            cursor = self._conn.execute(
                "INSERT INTO events (session_epoch, ts_ns, kind, turn_id, speech_id,"
                " task_id, supersedes, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_epoch,
                    ts_ns if ts_ns is not None else time.time_ns(),
                    kind_value,
                    turn_id,
                    speech_id,
                    task_id,
                    supersedes,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def supersede(
        self,
        old_seq: int,
        kind: EventKind | str,
        payload: dict[str, Any] | None = None,
        turn_id: str | None = None,
        speech_id: str | None = None,
    ) -> int:
        """修订:标记旧事件被替代,并追加新版本。返回新事件 seq。"""
        new_seq = self.append(
            kind, payload=payload, turn_id=turn_id, speech_id=speech_id, supersedes=old_seq
        )
        self.append(
            EventKind.SUPERSEDED, payload={"by": new_seq}, turn_id=turn_id, supersedes=old_seq
        )
        return new_seq

    def iterate(
        self,
        since_seq: int = 0,
        kind: EventKind | str | None = None,
        turn_id: str | None = None,
    ) -> Iterator[Event]:
        query = (
            "SELECT seq, session_epoch, ts_ns, kind, turn_id, speech_id, task_id,"
            " supersedes, payload FROM events WHERE seq > ?"
        )
        args: list[Any] = [since_seq]
        if kind is not None:
            query += " AND kind = ?"
            args.append(kind.value if isinstance(kind, EventKind) else str(kind))
        if turn_id is not None:
            query += " AND turn_id = ?"
            args.append(turn_id)
        query += " ORDER BY seq"
        with self._write_lock:
            rows = list(self._conn.execute(query, args))
        for row in rows:
            yield Event(
                seq=row[0],
                session_epoch=row[1],
                ts_ns=row[2],
                kind=row[3],
                turn_id=row[4],
                speech_id=row[5],
                task_id=row[6],
                supersedes=row[7],
                payload=json.loads(row[8]),
            )

    def latest_of_turn(self, turn_id: str, kind: EventKind | str) -> Event | None:
        """取该 turn 某类事件的最新有效版本(未被 supersede 链淘汰的以最后追加为准)。"""
        events = list(self.iterate(kind=kind, turn_id=turn_id))
        return events[-1] if events else None

    def markdown_projection(self, session_epoch: int | None = None) -> str:
        """从 append-only 事实事件重建 Markdown；投影本身不落库。"""
        from channellm.app.context import render_markdown

        return render_markdown(
            self.iterate(),
            session_epoch=self.session_epoch if session_epoch is None else session_epoch,
        )

    def context_snapshot(
        self,
        budget_tokens: int,
        token_counter: Any | None = None,
        session_epoch: int | None = None,
    ) -> Any:
        """按 token 预算取当前 session 的有效事实，而非按回合数硬截断。"""
        from channellm.app.context import build_context_snapshot

        kwargs = {} if token_counter is None else {"token_counter": token_counter}
        return build_context_snapshot(
            self.iterate(),
            session_epoch=self.session_epoch if session_epoch is None else session_epoch,
            budget_tokens=budget_tokens,
            **kwargs,
        )

    def recovery_state(
        self, budget_tokens: int, token_counter: Any | None = None
    ) -> Any:
        """重启时重建已确认上下文与未完成任务；不恢复未播音频。"""
        from channellm.app.recovery import recover_session

        kwargs = {} if token_counter is None else {"token_counter": token_counter}
        return recover_session(self, budget_tokens=budget_tokens, **kwargs)

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
