from __future__ import annotations

import numpy as np
import pytest

from channellm.duplex.eou_baseline import DuplugEOUBaseline


class FakeDuplugEngine:
    def __init__(self, state: str) -> None:
        self.state = state
        self.observed: list[np.ndarray] = []

    def process(self, audio: np.ndarray) -> dict[str, str]:
        self.observed.append(audio)
        return {"state": self.state}


def test_duplug_baseline_maps_only_official_speak_state_to_eou() -> None:
    engine = FakeDuplugEngine("speak")
    loaded_paths: list[str | None] = []
    baseline = DuplugEOUBaseline(
        "weights-dir",
        engine_loader=lambda path: loaded_paths.append(path) or engine,
        clock_ns=lambda: 42,
    )

    decision = baseline.observe_chunk(np.array([0.2, -0.2], dtype=np.float64))

    assert baseline.loaded
    assert loaded_paths == ["weights-dir"]
    assert decision.is_end_of_utterance
    assert decision.confidence is None
    assert decision.ts_ns == 42
    assert engine.observed[0].dtype == np.float32


def test_duplug_baseline_keeps_non_speak_states_as_non_eou() -> None:
    baseline = DuplugEOUBaseline(engine_loader=lambda _path: FakeDuplugEngine("nonidle"))

    assert not baseline.observe_chunk(np.ones(4, dtype=np.float32)).is_end_of_utterance


def test_duplug_baseline_requires_independent_engine_and_valid_audio() -> None:
    with pytest.raises(RuntimeError, match="independently provisioned"):
        DuplugEOUBaseline().load()

    baseline = DuplugEOUBaseline(engine_loader=lambda _path: FakeDuplugEngine("idle"))
    with pytest.raises(ValueError, match="non-empty"):
        baseline.observe_chunk(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        baseline.observe_chunk(np.array([np.nan], dtype=np.float32))
