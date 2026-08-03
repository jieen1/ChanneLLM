#!/usr/bin/env python
"""真实权重 Talker 25 帧 duplex phrase 隔离基准。

先执行一个首 unit 建立相同的流状态，再测非末 unit；该 unit 按 MiniCPM-o 合约
必须生成 25 帧。输出仅是本机 GPU 的可审计微基准，不能替代共驻端到端 SLO。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

SNAPSHOT_GLOB = Path.home() / ".cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots"


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile / 100) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup must be non-negative")

    snapshots = sorted(SNAPSHOT_GLOB.glob("*/"))
    if not snapshots:
        raise FileNotFoundError("MiniCPM-o 4.5 snapshot was not found")
    from channellm.engine.talker import TalkerStream, load_talker_weights

    device = torch.device("cuda")
    talker = load_talker_weights(snapshots[0], device=device, dtype=torch.bfloat16)
    token_ids = torch.tensor([1, 2, 3], dtype=torch.long, device=device)
    hidden_states = torch.zeros(
        (len(token_ids), talker.config.llm_dim), dtype=torch.bfloat16, device=device
    )
    stream = TalkerStream(talker)

    def run_phrase() -> float:
        stream.reset()
        stream.push(token_ids, hidden_states)  # 首 unit 允许 EOS，不纳入计时。
        torch.cuda.synchronize()
        start_ns = time.monotonic_ns()
        frames = stream.push(token_ids, hidden_states)
        torch.cuda.synchronize()
        if len(frames) != 25:
            raise AssertionError(f"expected a complete 25-frame phrase, got {len(frames)}")
        return (time.monotonic_ns() - start_ns) / 1_000_000

    for _ in range(args.warmup):
        run_phrase()
    samples = [run_phrase() for _ in range(args.repeat)]
    print(f"samples={len(samples)} frames=25")
    for percentile in (50, 95, 99):
        print(f"p{percentile}_ms={_percentile(samples, percentile):.1f}")
    print(f"per_token_p50_ms={_percentile(samples, 50) / 25:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
