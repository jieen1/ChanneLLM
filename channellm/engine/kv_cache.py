"""L1 KV cache 配置与显存预算(设计文档 §8)。

纯 full attention(无 hybrid),比 hybrid 简单。三阶段共驻显存基线参照
vllm-omni deploy/minicpmo_4_5.yaml:Thinker/Talker/Code2Wav ≈ 55%/15%/15%,
Talker KV 钉死 2 GiB,Code2Wav 非自回归不吃 KV。

禁止用权重加总估显存:运行峰值含 KV、激活、CUDA graph、workspace、
CUDA context 与 allocator cache(R7)。
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class KVCacheConfig:
    num_layers: int = 36
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 16  # token/block,paged 管理粒度
    bytes_per_element: int = 2  # bf16;FP8 时 = 1
    budget_bytes: int = 0  # 0 = 由 memory_fraction 推导

    def bytes_per_token(self) -> int:
        # K + V
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.bytes_per_element

    def max_tokens(self, budget_bytes: int) -> int:
        return budget_bytes // self.bytes_per_token()

    def num_blocks(self, budget_bytes: int) -> int:
        return self.max_tokens(budget_bytes) // self.block_size


@dataclasses.dataclass
class StageBudget:
    """三阶段显存切分(vllm-omni 基线,共驻测量矩阵后校准)。"""

    total_bytes: int
    thinker_frac: float = 0.55
    talker_frac: float = 0.15
    code2wav_frac: float = 0.15

    def thinker(self) -> int:
        return int(self.total_bytes * self.thinker_frac)

    def talker(self) -> int:
        return int(self.total_bytes * self.talker_frac)

    def code2wav(self) -> int:
        return int(self.total_bytes * self.code2wav_frac)
