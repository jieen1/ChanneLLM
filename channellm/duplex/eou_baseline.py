"""SoulX-Duplug 独立 EOU 基准适配(设计文档 §5)。

用途只有两个:
1. 独立 EOU 第二意见 —— 没有它就无法客观测量 EOU_TO_FIRST_AUDIO
   (被测系统自己说自己何时说完不算数)。
2. 故障后备 —— duplex 不可用时退到 push-to-talk + transcript-only。

不参与说话决策。显存约 4GB,换可测量性。权重已缓存:
Soul-AILab/SoulX-Duplug-0.6B(24/24 文件 7.78GB)。参考仓库:
~/project/SoulX-Duplug。
"""

from __future__ import annotations

import dataclasses
from typing import Protocol


@dataclasses.dataclass
class EOUDecision:
    is_end_of_utterance: bool
    confidence: float
    ts_ns: int = 0


class EOUJudge(Protocol):
    """任何能提供独立 EOU 判定的实现(官方 duplex 之外)。"""

    def observe_chunk(self, waveform) -> EOUDecision:  # noqa: ANN001 - numpy array
        ...


class DuplugEOUBaseline:
    """P0/P3 实现:加载 SoulX-Duplug 0.6B,对 1s chunk 流给出 EOU 判定。"""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def load(self) -> None:
        raise NotImplementedError("P0 收尾/P3: 加载 SoulX-Duplug 作为独立 EOU 基准")

    def observe_chunk(self, waveform) -> EOUDecision:  # noqa: ANN001
        raise NotImplementedError("P0 收尾/P3")
