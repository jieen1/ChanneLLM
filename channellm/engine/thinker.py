"""Thinker 自研前向 —— Qwen3 骨干 + 可换 attention 后端(P1 核心)。

官方 MiniCPM-o 4.5 的 Thinker LLM 就是 ``Qwen3ForCausalLM``(权重内
modeling 代码第 113 行 ``self.llm = Qwen3ForCausalLM(config)``),36 层、
hidden 4096、GQA 32q/8kv、head_dim 128、rope_theta 1e6、纯 full
attention。本模块按同一结构自建,权重从官方 safetensors 的 ``llm.*``
键流式装入,逐 token 对齐官方输出是 P1 验收门槛。

后端纪律(设计文档 §P1):
- ``TorchListKV``:参考/兜底路径,CPU 可跑,对齐用;
- ``SparkinferPagedKV``:生产路径,KV 先写页再 attend(SGLang 口径)。
模型代码对后端无知,只走 begin_step/append_layer/attend/commit 四件事。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class ThinkerConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    vocab_size: int = 151748
    max_position_embeddings: int = 40960

    @classmethod
    def from_official(cls, config_path: str | Path) -> ThinkerConfig:
        """MiniCPM-o 顶层 config.json 直接带 LLM 字段。"""
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_q_heads=raw["num_attention_heads"],
            num_kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            intermediate_size=raw["intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta", 1e6),
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw.get("max_position_embeddings", 40960),
        )


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


class ThinkerAttention(nn.Module):
    def __init__(self, config: ThinkerConfig, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_q_heads * config.head_dim, bias=False,
            device=device, dtype=dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False,
            device=device, dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False,
            device=device, dtype=dtype,
        )
        self.o_proj = nn.Linear(
            config.num_q_heads * config.head_dim, config.hidden_size, bias=False,
            device=device, dtype=dtype,
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: KVBackend,
        layer_idx: int,
    ) -> torch.Tensor:
        seq_len = hidden.shape[0]
        cfg = self.config
        q = self.q_norm(self.q_proj(hidden).view(seq_len, cfg.num_q_heads, cfg.head_dim))
        k = self.k_norm(self.k_proj(hidden).view(seq_len, cfg.num_kv_heads, cfg.head_dim))
        v = self.v_proj(hidden).view(seq_len, cfg.num_kv_heads, cfg.head_dim)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        kv.append_layer(layer_idx, k, v)
        out = kv.attend(layer_idx, q)
        return self.o_proj(out.reshape(seq_len, cfg.num_q_heads * cfg.head_dim))


class ThinkerMLP(nn.Module):
    def __init__(self, config: ThinkerConfig, device=None, dtype=None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ThinkerLayer(nn.Module):
    def __init__(self, config: ThinkerConfig, device=None, dtype=None) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.self_attn = ThinkerAttention(config, device, dtype)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device, dtype
        )
        self.mlp = ThinkerMLP(config, device, dtype)

    def forward(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        kv: KVBackend, layer_idx: int,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, kv, layer_idx)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Thinker(nn.Module):
    def __init__(
        self,
        config: ThinkerConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [ThinkerLayer(config, device, dtype) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        self.rotary = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )
        self.rotary = self.rotary.to(device)

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        kv: KVBackend,
        positions: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """token_ids: [S] -> logits [S, vocab]。KV 读写全走 kv 后端。

        output_hidden_states=True 时额外返回每层输出(含 embed,共 L+1 个),
        Talker 条件化与逐层对齐诊断都用它。
        """
        seq_len = token_ids.shape[0]
        kv.begin_step(seq_len)
        if positions is None:
            positions = torch.arange(
                kv.prefix_len, kv.prefix_len + seq_len, device=token_ids.device
            )
        cos, sin = self.rotary(positions, dtype=self.embed_tokens.weight.dtype)
        cos = cos.unsqueeze(1)  # [S, 1, dim] 广播 heads
        sin = sin.unsqueeze(1)

        hidden = self.embed_tokens(token_ids)
        hiddens: list[torch.Tensor] = [hidden] if output_hidden_states else []
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, kv, layer_idx)
            if output_hidden_states:
                hiddens.append(hidden)
        kv.commit()
        logits = self.lm_head(self.norm(hidden))
        if output_hidden_states:
            return logits, hiddens
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt_ids: list[int],
        n_new: int,
        kv: KVBackend,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """prefill + 逐 token decode,贪心。返回新产出的 token。"""
        device = self.embed_tokens.weight.device
        logits = self.forward(torch.tensor(prompt_ids, dtype=torch.long, device=device), kv)
        out: list[int] = []
        next_id = int(logits[-1].argmax())
        for _ in range(n_new):
            logits = self.forward(torch.tensor([next_id], dtype=torch.long, device=device), kv)
            out.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
            next_id = int(logits[-1].argmax())
        return out


# ---------------------------------------------------------------------------
# 权重装载
# ---------------------------------------------------------------------------

_KEY_MAP_STATIC = {
    "llm.model.embed_tokens.weight": "embed_tokens.weight",
    "llm.model.norm.weight": "norm.weight",
    "llm.lm_head.weight": "lm_head.weight",
}


def map_official_key(key: str) -> str | None:
    """官方 safetensors 键 -> 本模块参数路径;不属于 LLM 的键返回 None。"""
    if key in _KEY_MAP_STATIC:
        return _KEY_MAP_STATIC[key]
    if not key.startswith("llm.model.layers."):
        return None
    rest = key[len("llm.model.layers."):]
    layer_id, _, param = rest.partition(".")
    if param.startswith("self_attn."):
        return f"layers.{layer_id}.self_attn.{param[len('self_attn.'):]}"
    if param.startswith("mlp."):
        return f"layers.{layer_id}.mlp.{param[len('mlp.'):]}"
    if param in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return f"layers.{layer_id}.{param}"
    return None


def load_thinker_weights(
    model_dir: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    config: ThinkerConfig | None = None,
) -> Thinker:
    """从官方 MiniCPM-o checkpoint 流式装载 llm.* 权重(逐 shard,
    CPU RAM 只驻留当前 shard —— 23GB RAM 机器约束)。"""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    config = config or ThinkerConfig.from_official(model_dir / "config.json")
    model = Thinker(config, device=device, dtype=dtype)

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_keys: dict[str, list[tuple[str, str]]] = {}
    for key, shard in index["weight_map"].items():
        mapped = map_official_key(key)
        if mapped is not None:
            shard_keys.setdefault(shard, []).append((key, mapped))

    state = dict(model.named_parameters())
    loaded: set[str] = set()
    with torch.no_grad():
        for shard, pairs in shard_keys.items():
            with safe_open(str(model_dir / shard), framework="pt") as fh:
                for src, dst in pairs:
                    tensor = fh.get_tensor(src).to(device=device, dtype=dtype)
                    state[dst].copy_(tensor)
                    loaded.add(dst)
                    del tensor

    missing = set(state) - loaded
    if missing:
        raise RuntimeError(f"Thinker 权重未装齐,缺 {len(missing)} 项,如 {sorted(missing)[:3]}")
    return model.eval()
