"""Paged KV cache —— 页池、分配器与写入(P1 内核面地基)。

契约与 sparkinfer.attention.paged 的 SGLang 口径一致:
- 页布局 ``[num_pages, page_size, num_kv_heads, head_dim]``(每层一个视图);
- 新 token 的 K/V **先写进页**再调 attention,``cache_seqlens`` 计入当前
  q(右对齐 causal),即"先 append 后 attend"。

实现纪律:
- 纯 torch,CPU 可测;GPU 路径不引入额外拷贝(advanced-indexing 原地写)。
- 页粒度取 64(sparkinfer decode/extend 测试覆盖的粒度,也是 fork 里
  Laguna 解析 decode 核的钉版粒度);KVCacheConfig.block_size 是预算口径,
  与 kernel 页粒度解耦。
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class SeqKVState:
    """单条序列的 KV 占用状态。pages 按序映射逻辑块 -> 物理页号。"""

    pages: list[int] = dataclasses.field(default_factory=list)
    length: int = 0

    def pages_needed(self, length: int, page_size: int) -> int:
        """容纳 length 个 token 需要的总页数。"""
        return (length + page_size - 1) // page_size

    def advance(self, n: int) -> None:
        """一个 token 步内对每层各 append 一次后,统一推进长度。"""
        if n < 0:
            raise ValueError("advance 不能为负")
        self.length += n


class PageAllocator:
    """物理页空闲栈。O(1) 分配/释放,不做合并(页是固定粒度)。"""

    def __init__(self, num_pages: int) -> None:
        self.num_pages = num_pages
        self._free: list[int] = list(range(num_pages - 1, -1, -1))

    @property
    def free_count(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise MemoryError(
                f"KV 页耗尽:要 {count} 页,剩 {len(self._free)} 页"
            )
        taken, self._free = self._free[-count:], self._free[:-count]
        return taken

    def free(self, pages: list[int]) -> None:
        self._free.extend(pages)


class PagedKVPool:
    """全部层共享页号的 K/V 页池。

    形状:[num_layers, num_pages, page_size, num_kv_heads, head_dim]。
    单层视图 ``k_pages[layer]`` 连续,可直接喂 sparkinfer bind。
    """

    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("num_pages/page_size 必须为正")
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k_pages = torch.zeros(
            (num_layers, num_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_pages = torch.zeros_like(self.k_pages)
        self.allocator = PageAllocator(num_pages)

    @property
    def device(self) -> torch.device:
        return self.k_pages.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k_pages.dtype

    def append(self, layer: int, seq: SeqKVState, k: torch.Tensor, v: torch.Tensor) -> None:
        """把 n 个新 token 的 K/V 写入 seq 在 ``[length, length+n)`` 的槽位。

        k/v: ``[n, num_kv_heads, head_dim]``。**不推进** ``seq.length`` ——
        一个 token 步要对所有层各 append 一次,步末调用一次 ``seq.advance(n)``。
        """
        n = k.shape[0]
        if n == 0:
            return
        if k.shape != v.shape or k.ndim != 3:
            raise ValueError(f"k/v 形状需一致且为 3 维,得到 {k.shape}/{v.shape}")
        if k.shape[1] != self.num_kv_heads or k.shape[2] != self.head_dim:
            raise ValueError("k/v 的 head 维与页池不匹配")

        start = seq.length
        end = start + n
        need = seq.pages_needed(end, self.page_size) - len(seq.pages)
        if need > 0:
            seq.pages.extend(self.allocator.allocate(need))

        idx = torch.arange(start, end, device=self.device)
        page_idx = idx // self.page_size
        offset = idx % self.page_size
        phys = torch.tensor(
            [seq.pages[int(i)] for i in page_idx], dtype=torch.long, device=self.device
        )
        self.k_pages[layer][phys, offset] = k.to(self.dtype)
        self.v_pages[layer][phys, offset] = v.to(self.dtype)

    def free_seq(self, seq: SeqKVState) -> None:
        self.allocator.free(seq.pages)
        seq.pages = []
        seq.length = 0

    def page_table(self, seqs: list[SeqKVState], width: int | None = None) -> torch.Tensor:
        """[batch, width] int32 页表;不足补 -1(sparkinfer 侧由 cache_seqlens 界内)。"""
        max_pages = max((len(s.pages) for s in seqs), default=0)
        width = width or max_pages
        if width < max_pages:
            raise ValueError(f"page_table 宽度 {width} 不够容纳 {max_pages} 页")
        table = torch.full((len(seqs), max(width, 1)), -1, dtype=torch.int32, device=self.device)
        for row, seq in enumerate(seqs):
            if seq.pages:
                table[row, : len(seq.pages)] = torch.tensor(seq.pages, dtype=torch.int32)
        return table

    @staticmethod
    def cache_seqlens(seqs: list[SeqKVState], device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor([s.length for s in seqs], dtype=torch.int32, device=device)

    def materialize(self, layer: int, seq: SeqKVState) -> tuple[torch.Tensor, torch.Tensor]:
        """按 seq 顺序 gather 出连续 K/V ``[length, heads, dim]``(对齐验证用)。"""
        if seq.length == 0:
            empty = self.k_pages.new_zeros((0, self.num_kv_heads, self.head_dim))
            return empty, empty.clone()
        idx = torch.arange(seq.length, device=self.device)
        phys = torch.tensor(
            [seq.pages[int(i // self.page_size)] for i in idx],
            dtype=torch.long,
            device=self.device,
        )
        offset = idx % self.page_size
        return (
            self.k_pages[layer][phys, offset].clone(),
            self.v_pages[layer][phys, offset].clone(),
        )
