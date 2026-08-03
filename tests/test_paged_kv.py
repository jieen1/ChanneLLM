"""paged_kv 页池与分配器的 CPU 契约测试。"""

from __future__ import annotations

import pytest
import torch

from channellm.kernel.paged_kv import PagedKVPool, SeqKVState


def make_pool(num_pages: int = 8, page_size: int = 4) -> PagedKVPool:
    return PagedKVPool(
        num_layers=2,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=3,
        head_dim=8,
        dtype=torch.float32,
    )


def test_append_then_materialize_roundtrip() -> None:
    pool = make_pool()
    seq = SeqKVState()
    k = torch.arange(6 * 3 * 8, dtype=torch.float32).reshape(6, 3, 8)
    v = k + 1000
    pool.append(0, seq, k, v)
    seq.advance(6)

    assert seq.length == 6
    assert len(seq.pages) == 2  # page_size=4,6 个 token 要 2 页
    got_k, got_v = pool.materialize(0, seq)
    torch.testing.assert_close(got_k, k)
    torch.testing.assert_close(got_v, v)


def test_append_grows_incrementally_and_keeps_order() -> None:
    pool = make_pool()
    seq = SeqKVState()
    chunks = [torch.randn(3, 3, 8) for _ in range(4)]
    for chunk in chunks:
        pool.append(1, seq, chunk, chunk * 2)
        seq.advance(3)
    got_k, got_v = pool.materialize(1, seq)
    torch.testing.assert_close(got_k, torch.cat(chunks))
    torch.testing.assert_close(got_v, torch.cat(chunks) * 2)


def test_layers_are_isolated() -> None:
    pool = make_pool()
    seq = SeqKVState()
    k0 = torch.ones(5, 3, 8)
    k1 = torch.full((5, 3, 8), 7.0)
    # 真实用法:同一批 token 对每层各 append 一次,length 跨层一致
    pool.append(0, seq, k0, k0)
    pool.append(1, seq, k1, k1)
    seq.advance(5)
    assert seq.length == 5
    got0, _ = pool.materialize(0, seq)
    got1, _ = pool.materialize(1, seq)
    torch.testing.assert_close(got0, k0)
    torch.testing.assert_close(got1, k1)


def test_allocator_exhaustion_raises() -> None:
    pool = make_pool(num_pages=2, page_size=4)
    seq = SeqKVState()
    with pytest.raises(MemoryError):
        pool.append(0, seq, torch.randn(9, 3, 8), torch.randn(9, 3, 8))


def test_advance_rejects_negative() -> None:
    seq = SeqKVState()
    with pytest.raises(ValueError):
        seq.advance(-1)


def test_free_seq_returns_pages() -> None:
    pool = make_pool(num_pages=4)
    before = pool.allocator.free_count
    seq = SeqKVState()
    pool.append(0, seq, torch.randn(8, 3, 8), torch.randn(8, 3, 8))
    assert pool.allocator.free_count == before - 2
    pool.free_seq(seq)
    assert pool.allocator.free_count == before
    assert seq.length == 0


def test_page_table_and_seqlens_shapes() -> None:
    pool = make_pool()
    a, b = SeqKVState(), SeqKVState()
    pool.append(0, a, torch.randn(5, 3, 8), torch.randn(5, 3, 8))
    a.advance(5)
    pool.append(0, b, torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    b.advance(2)
    table = pool.page_table([a, b])
    lens = pool.cache_seqlens([a, b])
    assert table.shape == (2, 2)
    assert table.dtype == torch.int32
    assert (table[0] >= 0).all()
    assert table[1, 1].item() == -1  # b 只有 1 页,补 -1
    assert lens.tolist() == [5, 2]
