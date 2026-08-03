#!/usr/bin/env python
"""P0 —— 串行基线 waterfall 聚合(设计文档 §P0 验收产出)。

跨多个 trace JSONL 聚合分段延迟:p50/p95/p99,local/remote、cold/warm
分开报;禁止只报均值。cold = 每个文件的第一次运行(首次 forward 口径),
warm = 其余;也可用 tags 里自带的 temp 标签。

用法:
    python scripts/p0_waterfall.py traces/*.jsonl [--group-by loc,category]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from channellm.metrics.latency import format_waterfall, waterfall  # noqa: E402
from channellm.tracing.recorder import load_records  # noqa: E402
from channellm.tracing.schema import TraceRecord  # noqa: E402


def mark_cold_warm(records: list[TraceRecord]) -> list[TraceRecord]:
    """按 (session_id, trace_id) 首现顺序打 cold/warm 标签。"""
    seen: set[tuple[str, str]] = set()
    order: list[tuple[str, str]] = []
    for record in records:
        key = (record.session_id, record.trace_id)
        if key not in seen:
            seen.add(key)
            order.append(key)
    rank = {key: idx for idx, key in enumerate(order)}
    for record in records:
        key = (record.session_id, record.trace_id)
        record.tags.setdefault("temp", "cold" if rank[key] == 0 else "warm")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--group-by",
        default="temp,loc",
        help="逗号分隔的 tag 键(默认 temp,loc)",
    )
    parser.add_argument("--report", type=Path, default=None, help="另存 markdown 报告")
    args = parser.parse_args()

    records: list[TraceRecord] = []
    for path in args.traces:
        records.extend(load_records(path))
    if not records:
        print("没有读到任何 trace 记录", file=sys.stderr)
        return 1

    records = mark_cold_warm(records)
    group_by = [key.strip() for key in args.group_by.split(",") if key.strip()]
    report = waterfall(records, group_by=tuple(group_by))
    table = format_waterfall(report, group_labels=group_by)
    print(table)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            f"# P0 串行基线 waterfall\n\nsources: {', '.join(map(str, args.traces))}\n\n{table}\n",
            encoding="utf-8",
        )
        print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
