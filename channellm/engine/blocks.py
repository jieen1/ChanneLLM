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


class SparkinferPagedKV:
    """生产后端:PagedKVPool + sparkinfer paged attention,单序列。"""

    def __init__(self, pool, attn, seq=None) -> None:
        from channellm.kernel.paged_kv import SeqKVState

        self.pool = pool
        self.attn = attn
        self.seq = seq or SeqKVState()
        self.prefix_len = 0
        self._n_new = 0

    @property
    def length(self) -> int:
        return self.seq.length

    def begin_step(self, n_new: int) -> None:
        self._n_new = n_new
        self.prefix_len = self.seq.length

    def append_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.pool.append(layer_idx, self.seq, k, v)

    def attend(self, layer_idx: int, q: torch.Tensor) -> torch.Tensor:
        n_new = self._n_new
        cache_len = self.seq.length + n_new  # SGLang 口径:含当前 q
        page_table = self.pool.page_table([self.seq])
        cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=self.pool.device)
        cu = torch.tensor([0, n_new], dtype=torch.int32, device=self.pool.device)
        mode = "decode" if n_new == 1 else "extend"
        out = self.attn(
            q,
            self.pool.k_pages[layer_idx],
            self.pool.v_pages[layer_idx],
            page_table,
            cache_seqlens,
            cu,
            mode=mode,
        )
        return out

    def commit(self) -> None:
        self.seq.advance(self._n_new)
        self._n_new = 0




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
