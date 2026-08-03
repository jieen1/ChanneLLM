"""sparkinfer.attention.paged 适配层 —— plan/bind/run 生命周期封装。

sparkinfer 的 paged attention 是 planned API:
``plan(Caps)`` 定容量 -> ``bind`` 零拷贝绑 tensor -> ``run`` 发射。
实测 bind 约 2.7ms/次而 run 仅 0.6ms/次,逐层重 bind 会把 decode 拖到
~120ms/token。因此本适配层:

1. 按 (mode, 容量档) 缓存 Plan 与 scratch(容量只增不减);
2. 按 (mode, 形状, 层) 缓存 **静态 binding**:q/output/page_table/
   cache_seqlens/cu 全用固定缓冲,调用时 in-place 拷贝后直接 run,
   bind 成本在 decode 热路径摊薄为零;
3. 同一 plan 的多个 binding 共享一份 scratch(顺序执行,无并发)。

返回张量是静态输出缓冲的视图,**在下一次同键 forward 之前有效**
(模型逐层同步消费,契约安全)。

单流假设:每个 Plan 只配一份 scratch,不支持同 plan 并发 run
(P1 单会话串行;连续批处理上线时按 request 池化 scratch)。
"""

from __future__ import annotations

import dataclasses

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


@dataclasses.dataclass
class _StaticRun:
    binding: object
    q_buf: torch.Tensor
    out_buf: torch.Tensor
    pt_buf: torch.Tensor
    cs_buf: torch.Tensor
    cu_buf: torch.Tensor


class SparkinferPagedAttn:
    """decode/extend 两种模式的 paged attention 封装(binding 缓存)。"""

    def __init__(self, config: PagedAttnConfig, device: torch.device | str) -> None:
        from sparkinfer.attention import paged  # GPU 面,延迟导入

        self._paged = paged
        self.config = config
        self.device = torch.device(device)
        # (mode, q_cap, batch, width) -> (plan, scratch)
        self._plans: dict[tuple[int, int, int, int], tuple[object, torch.Tensor]] = {}
        # (mode, total_q, batch, width, k_pages 数据地址) -> _StaticRun
        self._runs: dict[tuple[int, int, int, int, int], _StaticRun] = {}

    def _get_plan(
        self, mode: str, max_total_q: int, max_batch: int, max_width: int, num_pages: int
    ):
        q_cap = max(1, 1 << (max_total_q - 1).bit_length())
        if mode == "decode":
            q_cap = max(q_cap, max_batch)
        key = (mode, q_cap, max_batch, max_width)
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
                max_total_q=q_cap,
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
        """q: [total_q, num_q_heads, head_dim] -> out: 同形(静态缓冲视图)。"""
        if mode not in ("decode", "extend"):
            raise ValueError(f"mode 必须是 decode/extend,得到 {mode}")
        total_q = int(q.shape[0])
        batch = int(page_table.shape[0])
        width = int(page_table.shape[1])
        num_pages = int(k_pages.shape[0])

        run_key = (mode, total_q, batch, width, k_pages.data_ptr())
        run = self._runs.get(run_key)
        if run is None:
            plan, scratch = self._get_plan(mode, total_q, batch, width, num_pages)
            q_buf = torch.empty_like(q)
            out_buf = torch.empty(
                (total_q, self.config.num_q_heads, self.config.head_dim),
                dtype=self.config.dtype,
                device=self.device,
            )
            pt_buf = page_table.contiguous().clone()
            cs_buf = cache_seqlens.contiguous().clone()
            cu_buf = cu_seqlens_q.contiguous().clone()
            q_buf.copy_(q)
            binding = self._paged.bind(
                plan,
                scratch=scratch,
                q=q_buf,
                k_cache=k_pages,
                v_cache=v_pages,
                output=out_buf,
                page_table=pt_buf,
                cache_seqlens=cs_buf,
                cu_seqlens_q=cu_buf,
                active_total_q=total_q,
            )
            run = _StaticRun(binding, q_buf, out_buf, pt_buf, cs_buf, cu_buf)
            self._runs[run_key] = run
        else:
            run.q_buf.copy_(q)
            run.pt_buf.copy_(page_table)
            run.cs_buf.copy_(cache_seqlens)
            run.cu_buf.copy_(cu_seqlens_q)

        self._paged.run(binding=run.binding)
        return run.out_buf

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)
