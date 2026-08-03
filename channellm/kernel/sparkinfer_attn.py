"""sparkinfer.attention.paged 适配层 —— plan/bind/run 生命周期封装。

sparkinfer 的 paged attention 是 planned API:
``plan(Caps)`` 定容量 -> ``bind`` 零拷贝绑 tensor -> ``run`` 发射。
本适配层负责:

1. 按 (mode, 容量档) 缓存 Plan 与 scratch,避免每步重 plan;
2. 请求超出容量档时自动升档重 plan(容量只增不减);
3. 对上层暴露单函数 ``forward``,契约与 SGLang 口径一致
   (先 append KV 后 attend,cache_seqlens 计入当前 q)。

单流假设:每个 Plan 只配一份 scratch,不支持同 plan 并发 run
(P1 单会话串行;连续批处理上线时按 request 池化 scratch)。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

# 与 sparkinfer in-tree 测试一致的预算上限
_MAX_WORK_ITEMS = 1024
_MAX_PARTIAL_ROWS = 16384


@dataclasses.dataclass(frozen=True)
class PagedAttnConfig:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int = 64
    dtype: torch.dtype = torch.bfloat16
    max_batch: int = 1
    max_seq_len: int = 40960  # 与 MiniCPM-o max_position_embeddings 对齐


class SparkinferPagedAttn:
    """decode/extend 两种模式的 paged attention 封装。"""

    def __init__(self, config: PagedAttnConfig, device: torch.device | str) -> None:
        from sparkinfer.attention import paged  # GPU 面,延迟导入

        self._paged = paged
        self.config = config
        self.device = torch.device(device)
        # (mode, max_total_q, max_batch, max_width) -> (plan, scratch)
        self._plans: dict[tuple[str, int, int, int], tuple[Any, torch.Tensor]] = {}

    def _page_table_width(self, max_seq_len: int) -> int:
        return (max_seq_len + self.config.page_size - 1) // self.config.page_size

    def _get_plan(
        self, mode: str, max_total_q: int, max_batch: int, max_width: int, num_pages: int
    ):
        key = (mode, max_total_q, max_batch, max_width)
        cached = self._plans.get(key)
        if cached is not None:
            return cached
        paged = self._paged
        plan = paged.plan(
            paged.Caps(
                device=self.device,
                mode=mode,
                dtype=self.config.dtype,
                kv_dtype=self.config.dtype,
                num_q_heads=self.config.num_q_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim_qk=self.config.head_dim,
                head_dim_vo=self.config.head_dim,
                page_size=self.config.page_size,
                max_total_q=max_total_q,
                max_batch=max_batch,
                max_page_table_width=max_width,
                max_work_items=_MAX_WORK_ITEMS,
                max_partial_rows=_MAX_PARTIAL_ROWS,
                num_cache_pages=num_pages,
                use_cuda_graph=False,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
        self._plans[key] = (plan, scratch)
        return plan, scratch

    def _required_capacity(
        self, mode: str, total_q: int, batch: int, width: int
    ) -> tuple[int, int, int]:
        """容量只增不减:请求超出当前档时升到 2 的幂档。"""
        q_cap = max(1, 1 << (total_q - 1).bit_length())
        if mode == "decode":
            q_cap = max(q_cap, batch)  # decode 每请求 1 个 q
        return q_cap, batch, width

    def forward(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        mode: str,
    ) -> torch.Tensor:
        """q: [total_q, num_q_heads, head_dim] -> out: 同形。"""
        if mode not in ("decode", "extend"):
            raise ValueError(f"mode 必须是 decode/extend,得到 {mode}")
        total_q = int(q.shape[0])
        batch = int(page_table.shape[0])
        width = int(page_table.shape[1])
        num_pages = int(k_pages.shape[0])

        q_cap, b_cap, w_cap = self._required_capacity(mode, total_q, batch, width)
        plan, scratch = self._get_plan(mode, q_cap, b_cap, w_cap, num_pages)

        output = torch.empty(
            (total_q, self.config.num_q_heads, self.config.head_dim),
            dtype=self.config.dtype,
            device=self.device,
        )
        binding = self._paged.bind(
            plan,
            scratch=scratch,
            q=q,
            k_cache=k_pages,
            v_cache=v_pages,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=total_q,
        )
        out, _lse = self._paged.run(binding=binding)
        return out

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)
