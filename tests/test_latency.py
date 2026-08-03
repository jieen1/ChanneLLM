import math

from channellm.metrics.latency import format_waterfall, percentiles, segment_latencies, waterfall
from channellm.tracing.recorder import dump_records
from channellm.tracing.schema import Anchor, Segment, TraceRecord


def make_pair(trace_id: str, start_ns: int, end_ns: int) -> list[TraceRecord]:
    return [
        TraceRecord(anchor=Anchor.EOU_DETECTED, ts_ns=start_ns, trace_id=trace_id),
        TraceRecord(anchor=Anchor.CODE2WAV_FIRST_PCM, ts_ns=end_ns, trace_id=trace_id),
    ]


def test_percentiles_nearest_rank():
    values = list(range(1, 101))  # 1..100
    stats = percentiles(values)
    assert stats["n"] == 100
    assert stats["p50"] == 50
    assert stats["p95"] == 95
    assert stats["p99"] == 99


def test_percentiles_empty():
    stats = percentiles([])
    assert stats["n"] == 0
    assert math.isnan(stats["p50"])


def test_segment_pairs_by_trace():
    records = make_pair("a", 0, 1_000_000) + make_pair("b", 0, 2_000_000)
    # 乱序喂入也不影响
    records = records[::-1]
    samples = segment_latencies(records, Anchor.EOU_DETECTED, Anchor.CODE2WAV_FIRST_PCM)
    assert sorted(samples) == [1.0, 2.0]


def test_waterfall_and_format(tmp_path):
    records = make_pair("a", 0, 1_000_000)
    records[0].tags = {"temp": "cold"}
    records[1].tags = {"temp": "cold"}
    dump_records(records, tmp_path / "t.jsonl")
    report = waterfall(records, segments=[Segment.EOU_TO_FIRST_PCM_LOCAL], group_by=("temp",))
    assert ("cold",) in report
    stats = report[("cold",)]["eou_to_first_pcm_local"]
    assert stats["n"] == 1
    table = format_waterfall(report, group_labels=("temp",))
    assert "eou_to_first_pcm_local" in table
