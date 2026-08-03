"""sparkinfer paged attention 适配层 vs in-tree reference(GPU)。

对照 sparkinfer 自带的 paged_attention_reference(SGLang 口径),
覆盖 decode 与 extend 两种模式、GQA 32q/8kv/head_dim128/page64
(MiniCPM-o Thinker 的真实形状)。
"""

from __future__ import annotations

import pytest
import torch

from channellm.kernel.paged_kv import PagedKVPool, SeqKVState
from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")


def _importable() -> bool:
    try:
        import sparkinfer  # noqa: F401

        return True
    except Exception:
        return False


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def _fill_pool(pool, seq, n_tokens, dtype):
    """逐层写入随机 KV 并 advance(模拟真实模型用法)。"""
    gen = torch.Generator().manual_seed(41)
    shape = (n_tokens, pool.num_kv_heads, pool.head_dim)
    for layer in range(pool.num_layers):
        k = torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype)
        v = torch.randn(*shape, generator=gen, dtype=torch.float32).to(dtype)
        pool.append(layer, seq, k, v)
    seq.advance(n_tokens)


def _run_case(mode: str, q_seqlens: list[int], cache_seqlens: list[int]) -> None:
    from sparkinfer.attention.paged.reference import paged_attention_reference

    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = 64
    num_layers = 2

    max_len = max(cache_seqlens)
    pool = PagedKVPool(
        num_layers=num_layers,
        num_pages=64,
        page_size=page_size,
        num_kv_heads=8,
        head_dim=128,
        dtype=dtype,
        device=device,
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=32, num_kv_heads=8, head_dim=128, page_size=page_size, dtype=dtype
        ),
        device,
    )

    seqs = []
    for length in cache_seqlens:
        seq = SeqKVState()
        _fill_pool(pool, seq, length, dtype)
        seqs.append(seq)

    width = (max_len + page_size - 1) // page_size
    page_table = pool.page_table(seqs, width=width)
    seqlens_t = pool.cache_seqlens(seqs, device=device)

    gen = torch.Generator(device="cuda").manual_seed(7)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, 32, 128, generator=gen, device=device, dtype=dtype)
    offsets = torch.tensor(q_seqlens).cumsum(0).tolist()
    cu = torch.tensor([0, *offsets], dtype=torch.int32, device=device)

    layer = 0
    out = attn(
        q,
        pool.k_pages[layer],
        pool.v_pages[layer],
        page_table,
        seqlens_t,
        cu,
        mode=mode,
    )
    ref, _ = paged_attention_reference(
        q, pool.k_pages[layer], pool.v_pages[layer], page_table, seqlens_t, cu, causal=True
    )
    torch.cuda.synchronize()
    assert (out - ref).abs().max().item() <= 0.02
    assert _cosine(out, ref) >= 0.99999


def test_decode_matches_reference() -> None:
    if not _importable():
        pytest.skip("sparkinfer 未安装")
    _run_case("decode", q_seqlens=[1], cache_seqlens=[128])


def test_decode_batch_matches_reference() -> None:
    if not _importable():
        pytest.skip("sparkinfer 未安装")
    _run_case("decode", q_seqlens=[1, 1], cache_seqlens=[64, 200])


def test_extend_prefill_matches_reference() -> None:
    if not _importable():
        pytest.skip("sparkinfer 未安装")
    # 纯 prefill:cache 为空,q 即全部 token
    _run_case("extend", q_seqlens=[37], cache_seqlens=[37])


def test_extend_chunked_prefill_matches_reference() -> None:
    if not _importable():
        pytest.skip("sparkinfer 未安装")
    # 已有 128 前缀,再 extend 16 个新 token(cache 已含新 token)
    _run_case("extend", q_seqlens=[16], cache_seqlens=[144])


def test_decode_rebinds_metadata_each_step() -> None:
    if not _importable():
        pytest.skip("sparkinfer 未安装")

    from sparkinfer.attention.paged.reference import paged_attention_reference

    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = 64
    pool = PagedKVPool(
        num_layers=1,
        num_pages=64,
        page_size=page_size,
        num_kv_heads=8,
        head_dim=128,
        dtype=dtype,
        device=device,
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=32, num_kv_heads=8, head_dim=128, page_size=page_size, dtype=dtype
        ),
        device,
    )
    seq = SeqKVState()
    _fill_pool(pool, seq, 48, dtype)

    q_gen = torch.Generator(device="cuda").manual_seed(17)
    kv_gen = torch.Generator(device="cuda").manual_seed(23)

    for _ in range(6):
        slot = pool.slot_for(seq, 1)
        k = torch.randn(1, 8, 128, generator=kv_gen, device=device, dtype=torch.float32).to(dtype)
        v = torch.randn(1, 8, 128, generator=kv_gen, device=device, dtype=torch.float32).to(dtype)
        pool.append(0, seq, k, v, slot=slot)
        page_table = pool.page_table([seq])
        cache_seqlens = torch.tensor([seq.length + 1], dtype=torch.int32, device=device)
        cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
        q = torch.randn(1, 32, 128, generator=q_gen, device=device, dtype=torch.float32).to(dtype)

        out = attn(
            q,
            pool.k_pages[0],
            pool.v_pages[0],
            page_table,
            cache_seqlens,
            cu,
            mode="decode",
        )
        ref, _ = paged_attention_reference(
            q, pool.k_pages[0], pool.v_pages[0], page_table, cache_seqlens, cu, causal=True
        )
        torch.cuda.synchronize()
        assert (out - ref).abs().max().item() <= 0.02
        assert _cosine(out, ref) >= 0.99999
        seq.advance(1)
