#!/usr/bin/env python
"""P0 —— 串行回放的诚实 waterfall(按 chunk 语义还原,设计文档 §P0)。

锚点对版本的 eou_to_first_pcm_local 在串行回放里是口径假象:eou_detected 与
code2wav_first_pcm 在同一循环迭代内先后打下,差值只有零点几毫秒,掩盖了
真实流式系统里"静音等待 + 决策 chunk 算力"这两大块成本。

本脚本从 trace JSONL 按 chunk 语义重建每条回放的时间账:

- eou_chunk          用户说完所在的 chunk(manifest 口径,eou_detected 锚点)
- decision_chunk     模型决定开口的 chunk(首个 is_listen=False 的 generate)
- first_pcm_chunk    首个非静音 PCM 产出的 chunk
- stream_wait_s      决策 chunk 流完时刻 - eou_offset_s:真实系统里从用户说完
                     到决策 chunk 的最后一个采样点到达,必须等过的流时间
                     (manifest 秒级标注,chunk 内小数部分也算)
- prefill_decision   决策 chunk 的 streaming_prefill 实测耗时
- decision_compute_s 决策 chunk 的 cost_llm + cost_tts + cost_token2wav
- tail_wait_s        (first_pcm - decision) × chunk 时长:首包若是静音,
                     真实系统里还要再流过的时间
- serial_total_s     四者之和 = 真实全双工系统 EOU→首包的对照系

回放本身以机器速度跑,锚点 ts 差值仅用于拆 prefill/generate 内部耗时,
不直接当产品延迟用。p50/p95/p99 需要足够样本;n<10 时只报 min/median/max。

用法:
    python scripts/p0_serial_report.py traces/p0_serial_run3.jsonl \
        --out artifacts/p0/waterfall.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from channellm.tracing.recorder import load_records  # noqa: E402
from channellm.tracing.schema import Anchor, TraceRecord  # noqa: E402


@dataclass
class GenerateObs:
    chunk_idx: int
    is_listen: bool
    end_of_turn: bool
    n_tokens: int
    cost_llm: float
    cost_tts: float
    cost_token2wav: float
    ts_ns: int


@dataclass
class TraceStats:
    trace_id: str
    category: str = ""
    eou_chunk: int | None = None
    eou_source: str = ""
    eou_offset_s: float | None = None
    decision_chunk: int | None = None
    decision: GenerateObs | None = None
    first_pcm_chunk: int | None = None
    total_chunks: int = 0
    load_ms: float | None = None
    prefill_ms_by_chunk: dict[int, float] = field(default_factory=dict)
    generates: list[GenerateObs] = field(default_factory=list)
    replay_eou_to_pcm_ms: float | None = None  # 锚点 ts 差,回放假象,仅留档


def env_line() -> str:
    parts: list[str] = []
    try:
        import torch

        parts.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            parts.append(torch.cuda.get_device_name(0))
    except Exception:
        parts.append("torch n/a")
    try:
        import transformers

        parts.append(f"transformers {transformers.__version__}")
    except Exception:
        pass
    return " · ".join(parts)


def analyze_trace(trace_id: str, records: list[TraceRecord]) -> TraceStats:
    stats = TraceStats(trace_id=trace_id)
    current_chunk = -1
    prefill_start: dict[int, int] = {}
    eou_ts: int | None = None
    pcm_ts: int | None = None

    for record in records:
        if record.tags.get("category"):
            stats.category = record.tags["category"]
        anchor = record.anchor
        extra = record.extra

        if anchor == Anchor.LOAD_DONE and stats.load_ms is None:
            stats.load_ms = extra.get("load_ms")
        elif anchor == Anchor.CHUNK_ALIGNED:
            current_chunk = int(extra.get("chunk_idx", current_chunk + 1))
            stats.total_chunks = max(stats.total_chunks, current_chunk + 1)
        elif anchor == Anchor.STREAMING_PREFILL_START:
            prefill_start[int(extra.get("chunk_idx", current_chunk))] = record.ts_ns
        elif anchor == Anchor.STREAMING_PREFILL_DONE:
            chunk_idx = int(extra.get("chunk_idx", current_chunk))
            start = prefill_start.get(chunk_idx)
            if start is not None:
                stats.prefill_ms_by_chunk[chunk_idx] = (record.ts_ns - start) / 1e6
        elif anchor == Anchor.STREAMING_GENERATE_DONE:
            obs = GenerateObs(
                chunk_idx=int(extra.get("chunk_idx", -1)),
                is_listen=bool(extra.get("is_listen", True)),
                end_of_turn=bool(extra.get("end_of_turn", False)),
                n_tokens=int(extra.get("n_tokens", 0)),
                cost_llm=float(extra.get("cost_llm", 0.0)),
                cost_tts=float(extra.get("cost_tts", 0.0)),
                cost_token2wav=float(extra.get("cost_token2wav", 0.0)),
                ts_ns=record.ts_ns,
            )
            stats.generates.append(obs)
            if not obs.is_listen and stats.decision is None:
                stats.decision = obs
                stats.decision_chunk = obs.chunk_idx
        elif anchor == Anchor.EOU_DETECTED:
            stats.eou_chunk = current_chunk
            stats.eou_source = str(extra.get("eou_source", ""))
            stats.eou_offset_s = extra.get("eou_offset_s")
            eou_ts = record.ts_ns
        elif anchor == Anchor.CODE2WAV_FIRST_PCM:
            stats.first_pcm_chunk = int(extra.get("chunk_idx", current_chunk))
            pcm_ts = record.ts_ns

    if eou_ts is not None and pcm_ts is not None and pcm_ts >= eou_ts:
        stats.replay_eou_to_pcm_ms = (pcm_ts - eou_ts) / 1e6
    return stats


@dataclass
class SerialMetrics:
    stats: TraceStats
    stream_wait_s: float | None
    prefill_decision_s: float | None
    decision_compute_s: float | None
    tail_wait_s: float | None
    serial_total_s: float | None


def serial_metrics(stats: TraceStats, chunk_seconds: float) -> SerialMetrics:
    # 真实系统里,决策 chunk 的最后一个采样点在 (decision_chunk+1)×chunk_s 时刻才到达;
    # 从 manifest 标注的用户说完时刻到这一刻,是必须等过的流时间(含 chunk 内小数段)。
    stream_wait_s = None
    if stats.eou_offset_s is not None and stats.decision_chunk is not None:
        stream_wait_s = max(0.0, (stats.decision_chunk + 1) * chunk_seconds - stats.eou_offset_s)

    prefill_decision_s = None
    if stats.decision_chunk is not None:
        prefill_ms = stats.prefill_ms_by_chunk.get(stats.decision_chunk)
        if prefill_ms is not None:
            prefill_decision_s = prefill_ms / 1000

    decision_compute_s = None
    if stats.decision is not None:
        decision = stats.decision
        decision_compute_s = decision.cost_llm + decision.cost_tts + decision.cost_token2wav

    tail_wait_s = None
    if (
        stats.first_pcm_chunk is not None
        and stats.decision_chunk is not None
        and stats.first_pcm_chunk >= stats.decision_chunk
    ):
        tail_wait_s = (stats.first_pcm_chunk - stats.decision_chunk) * chunk_seconds

    serial_total_s = None
    parts = [stream_wait_s, prefill_decision_s, decision_compute_s, tail_wait_s]
    if all(part is not None for part in parts):
        serial_total_s = sum(parts)

    return SerialMetrics(
        stats, stream_wait_s, prefill_decision_s, decision_compute_s, tail_wait_s, serial_total_s
    )


def fmt(value: float | None, unit: str = "s", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def summarize(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) < 10:
        return (
            f"min {min(values):.3f} / median {statistics.median(values):.3f} / "
            f"max {max(values):.3f} (n={len(values)},样本不足,不报 p95/p99)"
        )
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return (
        f"p50 {quantiles[49]:.3f} / p95 {quantiles[94]:.3f} / p99 {quantiles[98]:.3f} "
        f"(n={len(values)})"
    )


def build_report(
    traces: list[TraceStats],
    chunk_seconds: float,
    source_paths: list[Path],
) -> str:
    metrics = [serial_metrics(stats, chunk_seconds) for stats in traces]
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines: list[str] = []
    lines.append("# P0 串行回放 waterfall(诚实口径)")
    lines.append("")
    lines.append(f"- 生成时间:{now}")
    lines.append(f"- trace 来源:{', '.join(str(p) for p in source_paths)}")
    lines.append(f"- 环境:{env_line()}")
    lines.append("- 模型:MiniCPM-o 4.5 官方 duplex 路径 + ChanneLLM transformers 5 兼容垫片")
    lines.append(f"- chunk 时长:{chunk_seconds:.1f}s(16kHz mono)")
    lines.append("")
    lines.append("## 口径说明")
    lines.append("")
    lines.append("回放以机器速度串行喂 chunk,`eou_detected` 与 `code2wav_first_pcm` 锚点 ts 差")
    lines.append("只反映脚本内部先后,不是产品延迟。真实全双工系统里,音频按 1× 实时流入,")
    lines.append("EOU→首包 = 静音对齐等待 + 决策 chunk 算力(+ 首包静音尾巴等待),")
    lines.append("即下表 serial_total。")
    lines.append("")
    lines.append("## 逐条明细")
    lines.append("")
    lines.append(
        "| category | eou_offset_s | decision_chunk | first_pcm_chunk | 流等待 | prefill "
        "| 决策算力 (llm+tts+t2w) | 尾部等待 | **serial_total** | 回放锚点差(留档) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for item in metrics:
        stats = item.stats
        decision = stats.decision
        costs = (
            f"{fmt(item.decision_compute_s)} "
            f"({decision.cost_llm:.2f}+{decision.cost_tts:.2f}+{decision.cost_token2wav:.2f})"
            if decision is not None
            else "n/a"
        )
        eou_at = (
            f"{stats.eou_offset_s:.1f} (chunk {stats.eou_chunk})"
            if stats.eou_offset_s is not None
            else "n/a"
        )
        lines.append(
            f"| {stats.category or stats.trace_id[:8]} | {eou_at} "
            f"| {stats.decision_chunk if stats.decision_chunk is not None else 'n/a'} "
            f"| {stats.first_pcm_chunk if stats.first_pcm_chunk is not None else 'n/a'} "
            f"| {fmt(item.stream_wait_s)} | {fmt(item.prefill_decision_s)} "
            f"| {costs} | {fmt(item.tail_wait_s)} "
            f"| **{fmt(item.serial_total_s)}** "
            f"| {fmt(stats.replay_eou_to_pcm_ms, 'ms', 1)} |"
        )
    lines.append("")

    totals = [m.serial_total_s for m in metrics if m.serial_total_s is not None]
    waits = [m.stream_wait_s for m in metrics if m.stream_wait_s is not None]
    computes = [m.decision_compute_s for m in metrics if m.decision_compute_s is not None]
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- **EOU→首包(serial_total)**:{summarize(totals)}")
    lines.append(f"- 流等待(EOU→决策 chunk 流完):{summarize(waits)}")
    lines.append(f"- 决策 chunk 算力(llm+tts+t2w):{summarize(computes)}")

    warm_prefills = [
        ms
        for stats in traces
        for chunk_idx, ms in stats.prefill_ms_by_chunk.items()
        if chunk_idx > 0
    ]
    cold_prefills = [
        ms
        for stats in traces
        for chunk_idx, ms in stats.prefill_ms_by_chunk.items()
        if chunk_idx == 0
    ]
    lines.append(f"- streaming_prefill warm(chunk≥1):{summarize(warm_prefills)} ms")
    lines.append(f"- streaming_prefill cold(chunk 0):{summarize(cold_prefills)} ms")

    t2w_first: list[float] = []
    t2w_rest: list[float] = []
    for stats in traces:
        seen_t2w = False
        for obs in stats.generates:
            if obs.cost_token2wav > 0:
                (t2w_rest if seen_t2w else t2w_first).append(obs.cost_token2wav)
                seen_t2w = True
    lines.append(f"- token2wav 首次调用(含冷启动):{summarize(t2w_first)}")
    lines.append(f"- token2wav 后续调用:{summarize(t2w_rest)}")

    loads = [stats.load_ms for stats in traces if stats.load_ms is not None]
    if loads:
        lines.append(f"- 模型加载 load_ms:{summarize(loads)} ms(进程内首次 vs 复载)")
    lines.append("")

    lines.append("## 解读")
    lines.append("")
    lines.append("- **流等待是 chunk 粒度的税**:决策发生在 EOU 所在 chunk 内(模型反应快),")
    lines.append("  但该 chunk 必须完整流完才能处理,贡献 0.4–0.8s。P2 编排若把决策 chunk 粒度")
    lines.append("  减半,这部分直接减半。")
    lines.append("- **决策算力 warm 约 0.4–0.6s**:token2wav 首次调用含 vocoder 冷启动(最高")
    lines.append("  0.78s),会话内预热(如 prepare 时跑一次 ref audio)可消掉。")
    lines.append("- **eou_offset_s 为机器估算**(末帧有声 +0.1s),正式报告前建议人工复核。")
    lines.append("- **回放锚点差(亚毫秒)不是延迟**,仅作锚点链路完整性留档。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/p0/waterfall.md"))
    args = parser.parse_args()

    records: list[TraceRecord] = []
    for path in args.traces:
        records.extend(load_records(path))
    if not records:
        print("没有读到任何 trace 记录", file=sys.stderr)
        return 1

    by_trace: dict[str, list[TraceRecord]] = {}
    for record in records:
        if record.trace_id:
            by_trace.setdefault(record.trace_id, []).append(record)

    traces = [analyze_trace(trace_id, group) for trace_id, group in by_trace.items()]
    report = build_report(traces, args.chunk_seconds, args.traces)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[已写入 {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
