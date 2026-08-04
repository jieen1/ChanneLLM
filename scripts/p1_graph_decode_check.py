#!/usr/bin/env python
"""P1 CUDA graph decode 验证:质量优先地检查 replay 是否保持 eager 语义。"""

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
    parser.add_argument(
        "--prompt",
        default="请用一句话介绍西湖。",
        help="用于 eager/graph 对照的固定提示词",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=60,
        help="prefill 之后继续贪心 decode 的 token 数",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="默认 fp32 质量模式；bf16 仅用于性能诊断，未通过长序列 parity",
    )
    return parser


def build(model_dir, dtype: torch.dtype):
    from channellm.engine.thinker import ThinkerConfig, load_thinker_weights

    thinker = load_thinker_weights(model_dir, device="cuda", dtype=dtype)
    cfg = ThinkerConfig.from_official(model_dir / "config.json")
    return thinker, cfg


def make_kv(cfg, dtype: torch.dtype):
    from channellm.engine.blocks import SparkinferPagedKV
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn

    pool = PagedKVPool(
        cfg.num_hidden_layers,
        512,
        64,
        cfg.num_kv_heads,
        cfg.head_dim,
        dtype=dtype,
        device="cuda",
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=cfg.num_q_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            page_size=64,
            dtype=dtype,
        ),
        "cuda",
    )
    return SparkinferPagedKV(pool, attn)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens 必须至少为 1")

    model_dir = sorted(SNAP_DIR.glob("*/"))[0]
    from transformers import AutoTokenizer

    from channellm.engine.graph_decode import GraphDecodeSession

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    thinker, cfg = build(model_dir, dtype=dtype)
    graph_kv = make_kv(cfg, dtype)
    eager_kv = make_kv(cfg, dtype)
    prompt_ids = torch.tensor(prompt, dtype=torch.long, device="cuda")

    torch.cuda.synchronize()
    t0 = time.time()
    g = GraphDecodeSession(thinker, graph_kv)
    g.capture()
    graph_logits = thinker.forward(prompt_ids, graph_kv)
    torch.cuda.synchronize()
    cap_ms = (time.time() - t0) * 1000

    eager_logits = thinker.forward(prompt_ids, eager_kv)
    graph_first = int(graph_logits[-1].argmax().item())
    eager_first = int(eager_logits[-1].argmax().item())
    graph_tokens = [graph_first]
    eager_tokens = [eager_first]
    cur_graph = graph_first
    cur_eager = eager_first

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.tokens):
        cur_graph = g.step(cur_graph)
        eager_logits = thinker.forward(
            torch.tensor([cur_eager], dtype=torch.long, device="cuda"), eager_kv
        )
        cur_eager = int(eager_logits[-1].argmax().item())
        graph_tokens.append(cur_graph)
        eager_tokens.append(cur_eager)
    torch.cuda.synchronize()
    graph_s = time.time() - t0
    mismatch = next(
        (
            idx
            for idx, (eager_token, graph_token) in enumerate(zip(eager_tokens, graph_tokens))
            if eager_token != graph_token
        ),
        None,
    )
    if mismatch is None:
        print(f"[parity] PASS({args.dtype}): eager/graph {len(graph_tokens)} tokens 一致")
    else:
        print(
            f"[parity] {'FAIL' if args.dtype == 'fp32' else 'REVIEW'}({args.dtype}): "
            f"第 {mismatch + 1} 个 token 分歧 "
            f"(eager={eager_tokens[mismatch]} graph={graph_tokens[mismatch]})"
        )
    print(
        f"[graph] capture(warmup 2 步) {cap_ms:.0f}ms; {args.tokens} tok / {graph_s:.2f}s "
        f"= {args.tokens / graph_s:.1f} tok/s ({graph_s / args.tokens * 1000:.1f} ms/tok)"
    )
    print(f"[graph] 输出: {tok.decode(graph_tokens, skip_special_tokens=True)[:120]!r}")
    return 0 if mismatch is None or args.dtype == "bf16" else 1


if __name__ == "__main__":
    raise SystemExit(main())
