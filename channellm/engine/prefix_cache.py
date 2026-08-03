"""L1 Prefix caching 骨架(P1 延迟杠杆 #2)。

语音场景命中率极高:系统提示固定 + 历史只追加。注意参考实现
(vllm-omni)默认 enable_prefix_caching=false —— 我们要显式打开并测量收益。

接口:radix 树按 token 前缀查 block 链;驱逐策略 LRU。
"""

from __future__ import annotations


class PrefixCache:
    """P1 实现。接口先行:lookup/insert 以 token tuple 为键。"""

    def __init__(self, capacity_blocks: int = 0) -> None:
        self.capacity_blocks = capacity_blocks
        self._store: dict[tuple[int, ...], list[int]] = {}

    def lookup(self, tokens: tuple[int, ...]) -> list[int] | None:
        return self._store.get(tokens)

    def insert(self, tokens: tuple[int, ...], blocks: list[int]) -> None:
        self._store[tokens] = list(blocks)

    def __len__(self) -> int:
        return len(self._store)
