#!/usr/bin/env python
"""P1 Talker CUDA graph decode 质量门禁 —— 贪心 codec 流 eager/graph 对照。

固定 seed 的确定性条件化(模拟 Thinker unit 的 token_ids + hidden),多 unit
续写;eager 路径走 Talker 官方 forward_embeds(SDPA),graph 路径 prefill 同
eager、decode 帧走 TalkerGraphDecodeSession replay。要求贪心 codec 序列
逐帧一致 —— 不一致即门禁失败。

用法:
    python scripts/p1_talker_graph_check.py [--units 4] [--frames 25]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

SNAP_DIR = Path.home() / ".cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=4, help="模拟的 Thinker unit 数")
    parser.add_argument("--frames", type=int, default=25, help="每 unit 最多 decode 帧数")
    parser.add_argument("--unit-tokens", type=int, default=10, help="每 unit 条件 token 数")
    return parser


def make_conditioning(cfg, unit: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """确定性伪 unit 条件:token ids + thinker 隐层(固定 seed)。"""
    gen = torch.Generator(device="cpu").manual_seed(1000 + unit)
    ids = torch.randint(0, 1000, (10,), generator=gen)
    hidden = torch.randn(10, cfg.llm_dim, generator=gen) * 0.5
    return ids.to(device), hidden.to(device=device, dtype=torch.bfloat16)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.units < 1 or args.frames < 1:
        parser.error("--units/--frames 必须至少为 1")

    model_dir = sorted(SNAP_DIR.glob("*/"))[0]
    from channellm.engine.blocks import TorchStaticKV
    from channellm.engine.talker import TalkerConfig, load_talker_weights
    from channellm.engine.talker_graph_decode import TalkerGraphDecodeSession

    cfg = TalkerConfig.from_official(model_dir / "config.json")
    talker = load_talker_weights(model_dir, device="cuda", dtype=torch.bfloat16, config=cfg)

    def mkv():
        return TorchStaticKV(
            cfg.num_hidden_layers, cfg.max_position_embeddings,
            cfg.num_kv_heads, cfg.head_dim, device="cuda", dtype=torch.bfloat16,
        )

    kv_eager = mkv()
    kv_graph = mkv()
    graph = TalkerGraphDecodeSession(talker, kv_graph)

    eager_seq: list[int] = []
    graph_seq: list[int] = []
    torch.cuda.synchronize()
    t0 = time.time()
    graph_s = 0.0
    eager_s = 0.0
    for unit in range(args.units):
        ids, hidden = make_conditioning(cfg, unit, "cuda")
        cond = talker.build_condition(ids, hidden, duplex=True)

        # eager 路径:prefill + 贪心 decode
        h = talker.forward_embeds(cond, kv_eager)
        logits = talker.head_code(h[-1])
        for _ in range(args.frames):
            tok = int(logits[-1].argmax().item())
            if tok == cfg.codec_eos_token_id:
                break
            eager_seq.append(tok)
            torch.cuda.synchronize()
            te0 = time.time()
            emb = talker.emb_code(torch.tensor([tok], device="cuda"))
            h = talker.forward_embeds(emb, kv_eager)
            logits = talker.head_code(h[-1])
            torch.cuda.synchronize()
            eager_s += time.time() - te0

        # graph 路径:prefill 与 eager 完全相同,decode 走 replay
        h = talker.forward_embeds(cond, kv_graph)
        logits = talker.head_code(h[-1])
        for _ in range(args.frames):
            tok = int(logits[-1].argmax().item())
            if tok == cfg.codec_eos_token_id:
                break
            graph_seq.append(tok)
            torch.cuda.synchronize()
            tg0 = time.time()
            tok_g, logits = graph.step(tok)
            torch.cuda.synchronize()
            graph_s += time.time() - tg0
            assert tok_g == int(logits.argmax().item())
    total_s = time.time() - t0

    m = min(len(eager_seq), len(graph_seq))
    mismatch = next((i for i in range(m) if eager_seq[i] != graph_seq[i]), None)
    if mismatch is None and len(eager_seq) == len(graph_seq):
        print(f"[parity] PASS: eager/graph {m} 帧 codec 逐帧一致")
    else:
        print(
            f"[parity] FAIL: 第 {mismatch + 1 if mismatch is not None else m + 1} 帧分歧 "
            f"(eager={eager_seq[mismatch] if mismatch is not None else None} "
            f"graph={graph_seq[mismatch] if mismatch is not None else None}) "
            f"len eager={len(eager_seq)} graph={len(graph_seq)}"
        )
    n = len(graph_seq)
    print(
        f"[graph] 捕获 {graph.capture_count} 次 共 {graph.capture_ms:.0f}ms; "
        f"graph 段 {n} 帧 / {graph_s:.2f}s = {n / graph_s:.1f} 帧/s "
        f"({graph_s / n * 1000:.2f} ms/帧); "
        f"eager 对照 {n} 帧 / {eager_s:.2f}s = {n / eager_s:.1f} 帧/s "
        f"({eager_s / n * 1000:.2f} ms/帧); 加速比 {eager_s / graph_s:.2f}x; "
        f"总耗时 {total_s:.1f}s"
    )
    print(f"[graph] codec 前 20 帧: {graph_seq[:20]}")
    return 0 if (mismatch is None and len(eager_seq) == len(graph_seq)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
