"""分段延迟统计:p50/p95/p99,禁止只报均值(设计文档 §2 报告规则)。

配对规则:同一 (trace_id, turn_epoch) 内,取第一个 start 锚点与
其后第一个 end 锚点组成一个样本;end 早于 start 的配对丢弃。
local/remote、cold/warm 通过 tags 分开统计。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from channellm.tracing.schema import Segment, TraceRecord, match_key

NS_PER_MS = 1_000_000


def percentiles(values: Sequence[float], ps: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    """nearest-rank 百分位。样本不足时返回 nan 而不是假数字。"""
    out: dict[str, float] = {"n": float(len(values))}
    if not values:
        for p in ps:
            out[f"p{int(p)}"] = float("nan")
        return out
    ordered = sorted(values)
    for p in ps:
        rank = max(1, min(len(ordered), math.ceil(p / 100 * len(ordered))))
        out[f"p{int(p)}"] = ordered[rank - 1]
    return out


def segment_latencies(
    records: Iterable[TraceRecord],
    start_anchor: str,
    end_anchor: str,
) -> list[float]:
    """返回该分段的全部样本(ms)。"""
    starts: dict[tuple, int] = {}
    done: dict[tuple, bool] = {}
    samples: list[float] = []
    for record in sorted(records, key=lambda r: r.ts_ns):
        key = match_key(record)
        if record.anchor == start_anchor and key not in starts:
            starts[key] = record.ts_ns
        elif record.anchor == end_anchor and key in starts and not done.get(key):
            delta_ms = (record.ts_ns - starts[key]) / NS_PER_MS
            if delta_ms >= 0:
                samples.append(delta_ms)
            done[key] = True
    return samples


def _group_key(record: TraceRecord, group_by: Sequence[str]) -> tuple:
    return tuple(record.tags.get(k, "-") for k in group_by)


def waterfall(
    records: Iterable[TraceRecord],
    segments: Sequence[tuple[str, str, str]] = Segment.ALL,
    group_by: Sequence[str] = (),
) -> dict[tuple, dict[str, dict[str, float]]]:
    """按 group_by tags 分组,输出 {group: {segment_name: {p50,p95,p99,n}}}。"""
    records = list(records)
    groups: dict[tuple, list[TraceRecord]] = defaultdict(list)
    if group_by:
        for record in records:
            groups[_group_key(record, group_by)].append(record)
    else:
        groups[()] = records

    report: dict[tuple, dict[str, dict[str, float]]] = {}
    for group, group_records in groups.items():
        report[group] = {}
        for name, start_anchor, end_anchor in segments:
            samples = segment_latencies(group_records, start_anchor, end_anchor)
            report[group][name] = percentiles(samples)
    return report


def format_waterfall(
    report: dict[tuple, dict[str, dict[str, float]]],
    group_labels: Sequence[str] = (),
) -> str:
    """渲染成对齐的文本表(markdown 友好)。"""
    lines: list[str] = []
    header = ["segment", "n", "p50(ms)", "p95(ms)", "p99(ms)"]
    if group_labels:
        header = list(group_labels) + header
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for group in sorted(report, key=lambda g: str(g)):
        for name, stats in report[group].items():
            if stats["n"] == 0:
                continue
            row = [name, str(int(stats["n"]))]
            row += [f"{stats[f'p{p}']:.1f}" for p in (50, 95, 99)]
            if group_labels:
                row = list(map(str, group)) + row
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
