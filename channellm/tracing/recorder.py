"""JSONL trace 记录器。线程安全,单写者顺序落盘。

用法:
    rec = TraceRecorder("traces/run1.jsonl", session_id="s1", tags={"loc": "local"})
    trace_id = rec.new_trace()
    rec.anchor(Anchor.EOU_DETECTED, trace_id=trace_id, turn_epoch=3)
    ...
    rec.close()
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from channellm.tracing.schema import TraceRecord


class TraceRecorder:
    def __init__(
        self,
        out_path: str | Path,
        session_id: str = "",
        tags: dict[str, str] | None = None,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.base_tags = dict(tags or {})
        self._lock = threading.Lock()
        self._fh = open(self.out_path, "a", encoding="utf-8")
        self._closed = False

    def new_trace(self) -> str:
        return uuid.uuid4().hex[:16]

    def anchor(
        self,
        anchor: str,
        trace_id: str = "",
        turn_epoch: int = 0,
        speech_id: str = "",
        tags: dict[str, str] | None = None,
        **extra: Any,
    ) -> TraceRecord:
        merged = {**self.base_tags, **(tags or {})}
        record = TraceRecord(
            anchor=anchor,
            ts_ns=time.monotonic_ns(),
            wall_ns=time.time_ns(),
            trace_id=trace_id,
            turn_epoch=turn_epoch,
            speech_id=speech_id,
            session_id=self.session_id,
            tags=merged,
            extra=extra,
        )
        line = record.to_json()
        with self._lock:
            if self._closed:
                raise RuntimeError("recorder closed")
            self._fh.write(line + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._fh.close()
                self._closed = True

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_records(path: str | Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(TraceRecord.from_json(line))
    records.sort(key=lambda r: r.ts_ns)
    return records


def dump_records(records: Iterable[TraceRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json() + "\n")
