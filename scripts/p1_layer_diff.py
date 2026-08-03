#!/usr/bin/env python
"""P1 诊断 —— 自研 Thinker 与参考 Qwen3 逐层 hidden states 对比。

定位 logits 分歧的第一层:embed 之后逐层比 max|Δ|,正常 bf16 噪声应在
1e-2 量级;某层突然跳大即 bug 所在。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
from p1_thinker_parity import find_snapshot, load_reference_qwen3  # noqa: E402


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_dir = find_snapshot()

    from transformers import AutoTokenizer

    from channellm.engine.thinker import (
        SparkinferPagedKV,
        load_thinker_weights,
    )
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ids = tokenizer("请用三句话介绍一下杭州。", return_tensors="pt").input_ids[0].to(device)

    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype)
    reference = load_reference_qwen3(model_dir, device, dtype)

    config = thinker.config
    pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=64,
        page_size=64,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=64,
            dtype=dtype,
        ),
        device,
    )
    kv = SparkinferPagedKV(pool, attn)
    our_logits, our_hiddens = thinker(ids, kv, output_hidden_states=True)

    with torch.no_grad():
        ref_out = reference(input_ids=ids.unsqueeze(0), output_hidden_states=True)
    ref_hiddens = ref_out.hidden_states  # (embed, L0..L35, norm_out)

    names = ["embed"] + [f"layer{i:02d}" for i in range(config.num_hidden_layers)]
    print(f"{'层':<10}{'max|Δ|':>12}{'mean|Δ|':>12}{'cos':>10}")
    for i, name in enumerate(names):
        ours = our_hiddens[i].float()
        ref = ref_hiddens[i][0].float()
        diff = (ours - ref).abs()
        cos = torch.nn.functional.cosine_similarity(
            ours.reshape(-1), ref.reshape(-1), dim=0
        ).item()
        print(f"{name:<10}{diff.max().item():>12.5f}{diff.mean().item():>12.5f}{cos:>10.6f}")

    ours_norm = our_hiddens[-1]  # 我们的最后一层输出(未 norm)
    ref_last = ref_hiddens[-2][0]  # 参考的最后一层输出(未 norm)
    diff = (ours_norm.float() - ref_last.float()).abs()
    print(f"{'raw_last':<10}{diff.max().item():>12.5f}")
    normed = thinker.norm(ours_norm)
    diff2 = (normed.float() - ref_hiddens[-1][0].float()).abs()
    print(f"{'norm_out':<10}{diff2.max().item():>12.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
