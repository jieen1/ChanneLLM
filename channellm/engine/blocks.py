"""自研引擎共享构件 —— norm/rope/MLP/KV 后端(P1)。

Thinker(Qwen3 骨干)与 Talker(Llama 骨干)共用的最小构件集。
KV 后端契约(begin_step/append_layer/attend/commit + prefix_len/length)
是模型代码与 attention 实现之间唯一的接口。
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """与 transformers Qwen3RotaryEmbedding 逐位一致:fp32 频率、
    cat(freqs, freqs)、cos/sin  cast 回模型 dtype。"""

    def __init__(self, head_dim: int, theta: float, max_pos: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        positions = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_table", emb.cos(), persistent=False)
        self.register_buffer("sin_table", emb.sin(), persistent=False)

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = positions.long()
        return self.cos_table[idx].to(dtype), self.sin_table[idx].to(dtype)


class KVBackend(Protocol):
    """模型对 KV/attention 的全部要求:四件事 + 两个长度。"""

    prefix_len: int  # 本步开始前的缓存长度
    length: int  # 当前缓存总长度

    def begin_step(self, n_new: int) -> None: ...
    def append_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None: ...
    def attend(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor: ...
    def commit(self) -> None: ...


class TorchListKV:
    """参考后端:每层 list 存连续 K/V,SDPA 计算。CPU/GPU 通用。"""

    def __init__(self) -> None:
        self.k: list[torch.Tensor | None] = []
        self.v: list[torch.Tensor | None] = []
        self.prefix_len = 0
        self._n_new = 0

    @property
    def length(self) -> int:
        return len(self.k[0]) if self.k and self.k[0] is not None else 0

    def begin_step(self, n_new: int) -> None:
        self._n_new = n_new
        self.prefix_len = len(self.k[0]) if self.k and self.k[0] is not None else 0

    def append_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        while len(self.k) <= layer_idx:
            self.k.append(None)
            self.v.append(None)
        self.k[layer_idx] = k if self.k[layer_idx] is None else torch.cat([self.k[layer_idx], k])
        self.v[layer_idx] = v if self.v[layer_idx] is None else torch.cat([self.v[layer_idx], v])

    def attend(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        k, v = self.k[layer_idx], self.v[layer_idx]
        n_new = self._n_new
        q_t = q.transpose(0, 1).unsqueeze(0)  # [1, heads, S, dim]
        k_t = k.transpose(0, 1).unsqueeze(0)
        v_t = v.transpose(0, 1).unsqueeze(0)
        if n_new == k.shape[0]:  # 纯 prefill 且无 GQA 展开需求以外的通用路径
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, enable_gqa=True)
        else:
            seqlen_q, seqlen_k = n_new, k.shape[0]
            q_idx = torch.arange(seqlen_q, device=q.device).view(-1, 1)
            k_idx = torch.arange(seqlen_k, device=q.device).view(1, -1)
            mask = (k_idx > q_idx + seqlen_k - seqlen_q).view(1, 1, seqlen_q, seqlen_k)
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=~mask, enable_gqa=True
            )
        return out.squeeze(0).transpose(0, 1)  # [S, heads, dim]

    def commit(self) -> None:
        self._n_new = 0


class TorchStaticKV:
    """连续预分配 KV 后端：保留 Torch SDPA 语义，避免 decode 时逐层 ``cat``。

    Talker 的 12 heads / head_dim 64 目前不满足 sparkinfer paged kernel 的
    可用 trait，因此不能把 Thinker 的 paged 后端硬套进来。此后端为单会话
    单序列预留固定容量，每步只做 slice + 原地写入；满容量时显式失败，而非
    静默扩容或截断历史。
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if min(num_layers, max_seq_len, num_kv_heads, head_dim) <= 0:
            raise ValueError("static KV dimensions must be positive")
        shape = (num_layers, max_seq_len, num_kv_heads, head_dim)
        self.k_pages = torch.empty(shape, device=device, dtype=dtype)
        self.v_pages = torch.empty_like(self.k_pages)
        self.max_seq_len = max_seq_len
        self.prefix_len = 0
        self.length = 0
        self._n_new = 0

    def begin_step(self, n_new: int) -> None:
        if n_new < 0:
            raise ValueError("n_new cannot be negative")
        if self.length + n_new > self.max_seq_len:
            raise MemoryError(
                f"static KV capacity exceeded: {self.length + n_new} > {self.max_seq_len}"
            )
        self.prefix_len = self.length
        self._n_new = n_new

    def append_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape or k.ndim != 3:
            raise ValueError("k/v must have matching [tokens, heads, dim] shape")
        if k.shape[0] != self._n_new:
            raise ValueError("k/v token count must match begin_step")
        if k.shape[1:] != self.k_pages.shape[2:]:
            raise ValueError("k/v head shape does not match static KV")
        start = self.prefix_len
        end = start + self._n_new
        self.k_pages[layer_idx, start:end].copy_(k)
        self.v_pages[layer_idx, start:end].copy_(v)

    def attend(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        cache_len = self.prefix_len + self._n_new
        q_t = q.transpose(0, 1).unsqueeze(0)
        k_t = self.k_pages[layer_idx, :cache_len].transpose(0, 1).unsqueeze(0)
        v_t = self.v_pages[layer_idx, :cache_len].transpose(0, 1).unsqueeze(0)
        if self._n_new == cache_len:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, enable_gqa=True
            )
        elif self._n_new == 1:
            # 单 token decode 位于当前 cache 的最后，所有 cached KV 都可见。
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, enable_gqa=True)
        else:
            q_idx = torch.arange(self._n_new, device=q.device).view(-1, 1)
            k_idx = torch.arange(cache_len, device=q.device).view(1, -1)
            mask = (k_idx > q_idx + cache_len - self._n_new).view(
                1, 1, self._n_new, cache_len
            )
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=~mask, enable_gqa=True
            )
        return out.squeeze(0).transpose(0, 1)

    def commit(self) -> None:
        self.length += self._n_new
        self._n_new = 0

    def reset(self) -> None:
        """逻辑复位；无需清零，因为 length 是唯一可见边界。"""
        self.prefix_len = 0
        self.length = 0
        self._n_new = 0


class SparkinferPagedKV:
    """生产后端:PagedKVPool + sparkinfer paged attention,单序列。

    decode 热路径:每步在 begin_step 一次性算好写槽 slot,36 层 append
    复用索引,不再逐层重建。
    """

    def __init__(self, pool, attn, seq=None) -> None:
        from channellm.kernel.paged_kv import SeqKVState

        self.pool = pool
        self.attn = attn
        self.seq = seq or SeqKVState()
        self.prefix_len = 0
        self._n_new = 0
        self._slot = None
        self._page_table: torch.Tensor | None = None
        self._cache_seqlens: torch.Tensor | None = None
        self._cu_seqlens_q: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return self.seq.length

    def begin_step(self, n_new: int) -> None:
        self._n_new = n_new
        self.prefix_len = self.seq.length
        self._slot = self.pool.slot_for(self.seq, n_new) if n_new > 0 else None
        # 同一步的每个 Transformer 层读取相同的 paged-attention metadata。
        # 只在这里创建一次，既避免逐层的 GPU 小张量分配，也确保所有层严格
        # 使用同一份「先 append、后 attend」长度视图。
        self._page_table = self.pool.page_table([self.seq])
        self._cache_seqlens = torch.tensor(
            [self.seq.length + n_new], dtype=torch.int32, device=self.pool.device
        )
        self._cu_seqlens_q = torch.tensor(
            [0, n_new], dtype=torch.int32, device=self.pool.device
        )

    def append_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.pool.append(layer_idx, self.seq, k, v, slot=self._slot)

    def attend(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        if self._page_table is None or self._cache_seqlens is None or self._cu_seqlens_q is None:
            raise RuntimeError("attend 必须在 begin_step 之后调用")
        mode = "decode" if self._n_new == 1 else "extend"
        out = self.attn(
            q,
            self.pool.k_pages[layer_idx],
            self.pool.v_pages[layer_idx],
            self._page_table,
            self._cache_seqlens,
            self._cu_seqlens_q,
            mode=mode,
        )
        return out

    def commit(self) -> None:
        self.seq.advance(self._n_new)
        self._n_new = 0
        self._slot = None
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None




class MLP(nn.Module):
    """SwiGLU MLP(gate/up/down),Thinker 与 Talker 共用。"""

    def __init__(self, hidden_size: int, intermediate_size: int, device=None, dtype=None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
