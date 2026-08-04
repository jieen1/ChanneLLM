#!/usr/bin/env python
"""P1 漂移留档 —— 自研 Thinker(bf16 原生)与官方 Qwen3(bf16)对照。

官方 MiniCPM-o 4.5 的 Thinker LLM 是标准 Qwen3ForCausalLM。权重原生
bf16,本脚本在**同精度**下对照自研与官方实现:
1. 从官方 checkpoint 流式装载两份权重(自研 Thinker + transformers 参考);
2. 同一 prompt 分别 prefill,比对 logits;
3. 各自贪心生成 n 个 token,记录首个分歧位置与一致率。

口径说明:bf16 深层残差流会放大 kernel 舍入顺序差,官方实现自身在
bf16/fp32 之间同样于个位数 token 内分歧(已实测),因此本脚本只做漂移
留档,不作为硬性门槛。结构正确性证据:fp32 时代曾逐 token 对齐官方
(48/48 与 121/121,见 git 历史);生产门禁是
``scripts/p1_graph_decode_check.py``(同 bf16 内核族 graph/eager 必须
逐 token 一致)与端到端 fixture 回放。

用法:
    python scripts/p1_thinker_parity.py [--tokens 128] [--prompt ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

SNAPSHOT_GLOB = (
    Path.home()
    / ".cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*"
)


def find_snapshot() -> Path:
    snaps = sorted(SNAPSHOT_GLOB.parent.glob("*/"))
    if not snaps:
        raise FileNotFoundError("未找到 MiniCPM-o 4.5 权重快照")
    return snaps[0]


def load_reference_qwen3(model_dir: Path, device: torch.device, dtype: torch.dtype):
    """transformers Qwen3ForCausalLM,直接在 GPU 上初始化(rotary 等 buffer
    随之就位),再逐 shard assign 权重;CPU RAM 不驻留全量权重。"""
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3ForCausalLM

    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = Qwen3Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw.get("head_dim"),
        intermediate_size=raw["intermediate_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw.get("rope_theta", 1e6),
        max_position_embeddings=raw.get("max_position_embeddings", 40960),
        attention_bias=raw.get("attention_bias", False),
        tie_word_embeddings=False,
        vocab_size=raw["vocab_size"],
        torch_dtype=dtype,
    )
    with torch.device(device):
        model = Qwen3ForCausalLM(config).to(dtype)

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    assigned = 0
    for shard, keys in _shard_groups(index["weight_map"]).items():
        with safe_open(str(model_dir / shard), framework="pt") as fh:
            for src in keys:
                if not src.startswith("llm."):
                    continue
                dst = src[len("llm."):]
                tensor = fh.get_tensor(src).to(device=device, dtype=dtype)
                _assign_param(model, dst, tensor)
                assigned += 1
                del tensor
    print(f"[reference] assigned {assigned} tensors")
    return model.eval()


def _shard_groups(weight_map: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        groups.setdefault(shard, []).append(key)
    return groups


def _assign_param(model: torch.nn.Module, dotted: str, tensor: torch.Tensor) -> None:
    parent_path, _, name = dotted.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    parent.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))


def ref_greedy(model, ids: torch.Tensor, n_new: int, eos_ids: set[int]) -> list[int]:
    """与自研 generate_greedy 同序:产出即 forward。"""
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        logits = model(
            input_ids=ids.unsqueeze(0), past_key_values=cache, use_cache=True
        ).logits[0]
        out: list[int] = []
        next_id = int(logits[-1].argmax())
        for _ in range(n_new):
            logits = model(
                input_ids=torch.tensor([[next_id]], device=ids.device),
                past_key_values=cache,
                use_cache=True,
            ).logits[0]
            out.append(next_id)
            if next_id in eos_ids:
                break
            next_id = int(logits[-1].argmax())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompt", default="请用三句话介绍一下杭州。")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument(
        "--kv-backend",
        choices=("sparkinfer", "torch"),
        default="sparkinfer",
        help="对齐时使用的 KV 后端;torch 用参考 list KV",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print("[mode] bf16-原生精度漂移留档")
    model_dir = find_snapshot()
    print(f"[setup] snapshot: {model_dir}")

    from transformers import AutoTokenizer

    from channellm.engine.thinker import ThinkerConfig, TorchListKV, load_thinker_weights

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    print(f"[setup] prompt {len(ids)} tokens: {args.prompt!r}")

    t0 = time.time()
    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] in-house Thinker {time.time() - t0:.1f}s")

    t0 = time.time()
    reference = load_reference_qwen3(model_dir, device, dtype)
    print(f"[load] reference Qwen3 {time.time() - t0:.1f}s")

    ids_cuda = ids.to(device)

    def make_kv():
        if args.kv_backend == "torch":
            return TorchListKV()
        from channellm.engine.thinker import SparkinferPagedKV
        from channellm.kernel.paged_kv import PagedKVPool
        from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn

        config = ThinkerConfig.from_official(model_dir / "config.json")
        pool = PagedKVPool(
            num_layers=config.num_hidden_layers,
            num_pages=512,
            page_size=args.page_size,
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
                page_size=args.page_size,
                dtype=dtype,
            ),
            device,
        )
        return SparkinferPagedKV(pool, attn)

    print(f"[backend] {args.kv_backend}")

    # --- prefill logits 对比 ---
    kv = make_kv()
    torch.cuda.synchronize()
    t0 = time.time()
    our_logits = thinker(ids_cuda, kv)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0

    from transformers import DynamicCache

    with torch.no_grad():
        ref_logits = reference(
            input_ids=ids_cuda.unsqueeze(0),
            past_key_values=DynamicCache(),
            use_cache=True,
        ).logits[0]

    diff = (our_logits.float() - ref_logits.float()).abs()
    max_diff = diff.max().item()
    our_tok = our_logits.argmax(-1)
    ref_tok = ref_logits.argmax(-1)
    tok_match = (our_tok == ref_tok).float().mean().item()
    print(
        f"[prefill] {len(ids)} tok in {prefill_s:.3f}s "
        f"({len(ids) / prefill_s:.0f} tok/s) | logits max|Δ|={max_diff:.6f} | "
        f"argmax 一致率 {tok_match * 100:.2f}%"
    )

    # --- 贪心生成对比(各自全新 KV) ---
    kv_gen = make_kv()
    torch.cuda.synchronize()
    t0 = time.time()
    our_seq = thinker.generate_greedy(ids.tolist(), args.tokens, kv_gen, eos_token_id=151645)
    torch.cuda.synchronize()
    gen_s = time.time() - t0

    ref_seq = ref_greedy(reference, ids_cuda, args.tokens, eos_ids={151645, 151643})

    min_len = min(len(our_seq), len(ref_seq))
    mismatch = next((i for i in range(min_len) if our_seq[i] != ref_seq[i]), None)
    match_count = min_len if mismatch is None else mismatch
    print(
        f"[decode] {len(our_seq)} tok in {gen_s:.3f}s "
        f"({len(our_seq) / gen_s:.1f} tok/s incl. 首个 token)"
    )
    print(f"[parity] token 一致 {match_count}/{min_len}")

    # bf16 同精度对照:漂移留档,不做硬性门槛(见 docstring 口径说明)
    rate = match_count / min_len * 100 if min_len else 0.0
    print(f"[parity] bf16 vs 官方 bf16: 首个分歧 @ {mismatch}, 一致率 {rate:.1f}%")
    if mismatch is not None:
        _show_divergence(tokenizer, our_seq, ref_seq, mismatch, min_len)
    return 0


def _show_divergence(tokenizer, our_seq, ref_seq, mismatch, min_len) -> None:
    if mismatch is None:
        if len(our_seq) != len(ref_seq):
            print(f"[parity] 长度分歧: our={len(our_seq)} ref={len(ref_seq)}")
        return
    lo = max(0, mismatch - 4)
    print(f"  our: {tokenizer.decode(our_seq[lo:mismatch + 4])!r}")
    print(f"  ref: {tokenizer.decode(ref_seq[lo:mismatch + 4])!r}")
    print(f"  len our={len(our_seq)} ref={len(ref_seq)} (共同前缀 {min_len})")


if __name__ == "__main__":
    raise SystemExit(main())
