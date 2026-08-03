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
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np


@dataclasses.dataclass
class EOUDecision:
    is_end_of_utterance: bool
    # SoulX-Duplug 官方流式 server 只公开离散 state，不公开置信度；不得编造。
    confidence: float | None = None
    ts_ns: int = 0


class EOUJudge(Protocol):
    """任何能提供独立 EOU 判定的实现(官方 duplex 之外)。"""

    def observe_chunk(self, waveform) -> EOUDecision:  # noqa: ANN001 - numpy array
        ...


class _DuplugEngine(Protocol):
    """SoulX 官方 ``TurnTakingEngine`` 的最小稳定调用面。"""

    def process(self, audio: np.ndarray) -> Mapping[str, Any]:
        ...


EngineLoader = Callable[[str | None], _DuplugEngine]


class DuplugEOUBaseline:
    """SoulX-Duplug 的独立 EOU 观测适配器。

    SoulX 必须在它自己的官方依赖环境中加载，再通过 ``engine_loader`` 注入这里；
    这避免把它的训练/ASR 依赖混进实时 MiniCPM-o runtime，也不会让它取得说话权。
    官方 ``TurnTakingEngine.process`` 返回的 ``state == \"speak\"`` 是唯一被映射为
    EOU 的信号；模型未公开 confidence，因此结果保持 ``None``。
    """

    def __init__(
        self,
        model_path: str | None = None,
        *,
        engine_loader: EngineLoader | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.model_path = model_path
        self._engine_loader = engine_loader
        self._clock_ns = clock_ns
        self._engine: _DuplugEngine | None = None

    def load(self) -> None:
        """加载已独立部署的官方引擎，不修改当前项目的依赖闭包。"""
        if self._engine is not None:
            return
        if self._engine_loader is None:
            raise RuntimeError(
                "SoulX-Duplug baseline requires an independently provisioned official "
                "TurnTakingEngine; pass engine_loader instead of adding SoulX runtime "
                "dependencies to ChanneLLM."
            )
        engine = self._engine_loader(self.model_path)
        if not callable(getattr(engine, "process", None)):
            raise TypeError("SoulX engine must provide process(np.float32 waveform)")
        self._engine = engine

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def observe_chunk(self, waveform: np.ndarray) -> EOUDecision:
        """观察一个 16kHz float PCM 流片段；结果只能用于测量或故障后备。"""
        if self._engine is None:
            self.load()
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not len(samples):
            raise ValueError("SoulX EOU observation requires a non-empty waveform")
        if not np.isfinite(samples).all():
            raise ValueError("SoulX EOU observation rejects non-finite samples")
        state = self._engine.process(samples)
        if not isinstance(state, Mapping):
            raise TypeError("SoulX engine process() must return a state mapping")
        state_name = state.get("state")
        if not isinstance(state_name, str):
            raise ValueError("SoulX state mapping is missing string field 'state'")
        return EOUDecision(
            is_end_of_utterance=state_name == "speak",
            confidence=None,
            ts_ns=self._clock_ns(),
        )
