from __future__ import annotations

import numpy as np
import pytest

from channellm.duplex.epoch import EpochTag
from channellm.duplex.ingress import PcmIngress


class _Submitter:
    def __init__(self) -> None:
        self.turn = 0
        self.chunks: list[tuple[EpochTag, np.ndarray]] = []
        self.eou: list[EpochTag] = []

    def begin_turn(self, speech_id: str = "") -> EpochTag:
        self.turn += 1
        return EpochTag(self.turn, speech_id)

    def submit_audio(self, tag: EpochTag, pcm: np.ndarray) -> bool:
        self.chunks.append((tag, pcm))
        return True

    def on_eou(self, tag: EpochTag) -> bool:
        self.eou.append(tag)
        return True


def test_ingress_assembles_stereo_pcm16_and_pads_only_the_final_tail() -> None:
    submitter = _Submitter()
    ingress = PcmIngress(submitter)
    tag = ingress.begin_speech("speech")

    assert ingress.push_frame(
        np.full(8000 * 2, 16384, dtype=np.int16), sample_rate=16_000, channels=2
    ) == 0
    assert ingress.push_frame(
        np.full((8000, 2), 16384, dtype=np.int16), sample_rate=16_000, channels=2
    ) == 1
    assert ingress.end_speech()

    assert [item_tag for item_tag, _chunk in submitter.chunks] == [tag]
    assert len(submitter.chunks[0][1]) == 16_000
    assert np.allclose(submitter.chunks[0][1], 0.5, atol=1e-4)
    assert submitter.eou == [tag]


def test_ingress_rejects_implicit_resampling_and_nonfinite_input() -> None:
    ingress = PcmIngress(_Submitter())
    ingress.begin_speech()

    with pytest.raises(ValueError, match="explicitly resampled"):
        ingress.push_frame(np.zeros(480, dtype=np.float32), sample_rate=48_000)
    with pytest.raises(ValueError, match="non-finite"):
        ingress.push_frame(np.array([np.nan], dtype=np.float32), sample_rate=16_000)


def test_new_speech_discards_an_unfinished_previous_input_tail() -> None:
    submitter = _Submitter()
    ingress = PcmIngress(submitter)
    old_tag = ingress.begin_speech("old")
    ingress.push_frame(np.ones(8_000, dtype=np.float32), sample_rate=16_000)
    new_tag = ingress.begin_speech("new")
    ingress.push_frame(np.ones(16_000, dtype=np.float32), sample_rate=16_000)

    assert old_tag.turn_epoch + 1 == new_tag.turn_epoch
    assert [tag for tag, _chunk in submitter.chunks] == [new_tag]
