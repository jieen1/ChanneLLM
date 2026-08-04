import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)  # 用当前测试解释器,避免耦合 .venv 路径

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_waterfall_cli_on_synthetic_traces(tmp_path):
    from channellm.tracing.recorder import dump_records
    from channellm.tracing.schema import Anchor, TraceRecord

    records = []
    for i in range(3):
        trace_id = f"t{i}"
        records.append(
            TraceRecord(
                anchor=Anchor.EOU_DETECTED,
                ts_ns=0,
                trace_id=trace_id,
                session_id="s",
                tags={"loc": "local"},
            )
        )
        records.append(
            TraceRecord(
                anchor=Anchor.CODE2WAV_FIRST_PCM,
                ts_ns=(i + 1) * 1_000_000,
                trace_id=trace_id,
                session_id="s",
                tags={"loc": "local"},
            )
        )
    trace_path = tmp_path / "syn.jsonl"
    dump_records(records, trace_path)

    result = subprocess.run(
        [
            str(PYTHON),
            str(REPO_ROOT / "scripts/p0_waterfall.py"),
            str(trace_path),
            "--report",
            str(tmp_path / "report.md"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "eou_to_first_pcm_local" in result.stdout
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# P0")


def test_preflight_base_mode_exit_zero():
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/preflight.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "python" in result.stdout


def test_p0_run_help():
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/p0_run_official_duplex.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--manifest" in result.stdout


def test_p1_duplex_loop_help_documents_repeatable_benchmarking():
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/p1_duplex_loop.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--repeat" in result.stdout
    assert "--queued-runtime" in result.stdout
    assert "--realtime-input" in result.stdout
    assert "--eou-offset-s" in result.stdout
    assert "--vllm-omni-codec-bridge" in result.stdout


def test_p1_duplex_loop_rejects_empty_benchmark_batch():
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/p1_duplex_loop.py"), "--repeat", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--repeat 必须至少为 1" in result.stderr


def test_talker_benchmark_help_documents_repeatable_measurement():
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/p1_talker_bench.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--repeat" in result.stdout
    assert "25" in result.stdout


def test_graph_decode_check_help_documents_quality_mode() -> None:
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/p1_graph_decode_check.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--dtype" in result.stdout
    assert "--tokens" in result.stdout
    assert "fp32" in result.stdout
    assert "--kv-backend" in result.stdout
    assert "static" in result.stdout


def test_graph_decode_check_rejects_empty_decode_length() -> None:
    result = subprocess.run(
        [str(PYTHON), str(REPO_ROOT / "scripts/p1_graph_decode_check.py"), "--tokens", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--tokens 必须至少为 1" in result.stderr


def test_select_bucket_picks_smallest_fitting_width() -> None:
    from channellm.engine.static_graph_decode import DEFAULT_BUCKETS, select_bucket

    assert select_bucket(1, DEFAULT_BUCKETS) == 256
    assert select_bucket(256, DEFAULT_BUCKETS) == 256
    assert select_bucket(257, DEFAULT_BUCKETS) == 512
    assert select_bucket(40960, DEFAULT_BUCKETS) == 40960


def test_select_bucket_rejects_overflow_and_nonpositive_length() -> None:
    import pytest

    from channellm.engine.static_graph_decode import DEFAULT_BUCKETS, select_bucket

    with pytest.raises(MemoryError):
        select_bucket(40961, DEFAULT_BUCKETS)
    with pytest.raises(ValueError):
        select_bucket(0, DEFAULT_BUCKETS)
