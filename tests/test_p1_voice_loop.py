from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "p1_voice_loop.py"
_SPEC = importlib.util.spec_from_file_location("p1_voice_loop", _SCRIPT)
assert _SPEC and _SPEC.loader
voice_loop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(voice_loop)


def test_text_sampling_penalizes_a_repeated_positive_logit() -> None:
    token = voice_loop.sample_text_token(
        torch.tensor([0.09, 1.0]),
        [1],
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=20.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert token == 0


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p", "repetition_penalty"),
    [
        (0.0, 1, 1.0, 1.0),
        (1.0, -1, 1.0, 1.0),
        (1.0, 1, 0.0, 1.0),
        (1.0, 1, 1.0, 0.0),
    ],
)
def test_text_sampling_rejects_invalid_parameters(
    temperature: float, top_k: int, top_p: float, repetition_penalty: float,
) -> None:
    with pytest.raises(ValueError, match="invalid text sampling parameters"):
        voice_loop.sample_text_token(
            torch.tensor([0.0, 1.0]),
            [],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=torch.Generator().manual_seed(1),
        )
