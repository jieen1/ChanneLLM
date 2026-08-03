"""L1 ModelRunner 骨架(P1):CUDA graph 捕获 + forward 执行。

计划:
- decode 路径按 (batch, seq 桶) 捕获 CUDA graph;Code2Wav 按 (batch, frame 桶)
  (参考 vLLM 博客 2026-06-23-vllm-omni-tts 的 TTS 优化手法)。
- 三阶段 CUDA stream 分离 + 优先级(延迟杠杆 #6)。
- 参考实现 Code2Wav 是 enforce_eager=true 未捕获 —— 我们的机会点(杠杆 #4)。
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class GraphCapturePlan:
    stage: str
    batch_sizes: tuple[int, ...] = (1,)
    seq_buckets: tuple[int, ...] = ()


class ModelRunner:
    """P1 实现:load/forward/capture_graph/replay。"""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.graphs: dict[tuple[int, ...], Any] = {}

    def capture(self, plan: GraphCapturePlan) -> None:
        raise NotImplementedError("P1: CUDA graph capture")

    def forward(self, *args: object, **kwargs: object) -> Any:
        raise NotImplementedError("P1: forward")
