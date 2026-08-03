from __future__ import annotations

from dataclasses import dataclass

import torch

from channellm.duplex.driver import DuplexPipelineDriver
from channellm.duplex.runtime import RealtimeRuntime
from channellm.pipeline.orchestrator import Orchestrator
from channellm.pipeline.stages import PipelineChunk, StageId
from channellm.tracing import Anchor, TraceRecorder, load_records


@dataclass
class FakeDecision:
    is_listen: bool
    end_of_turn: bool = False


class FakeDuplexSession:
    def __init__(self, decision: FakeDecision) -> None:
        self.decision = decision
        self.calls: list[bytes] = []

    def on_chunk(self, pcm: bytes) -> FakeDecision:
        self.calls.append(pcm)
        return self.decision

    def latest_unit_conditioning(self):
        return torch.tensor([3]), torch.zeros((1, 4))


class FakeTalkerStream:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.push_calls: list[tuple[list[int], bool]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def push(self, token_ids, _hidden_states, *, end_of_turn: bool):
        self.push_calls.append((token_ids.tolist(), end_of_turn))
        return [11, 12]


class FakeCode2Wav:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.calls: list[tuple[list[int], bool]] = []

    def stream_reset(self) -> None:
        self.reset_calls += 1

    def stream_chunk(self, tokens: list[int], last_chunk: bool = False) -> bytes:
        self.calls.append((tokens, last_chunk))
        return bytes(tokens)


class FakeSink:
    def __init__(self) -> None:
        self.muted = 0
        self.published: list[bytes] = []

    def mute(self) -> None:
        self.muted += 1

    def publish(self, pcm: bytes, _tag) -> None:
        self.published.append(pcm)


def test_driver_runs_real_stage_contract_and_flushes_eou_tail(tmp_path) -> None:
    trace_path = tmp_path / "driver.jsonl"
    sink = FakeSink()
    session = FakeDuplexSession(FakeDecision(is_listen=False, end_of_turn=True))
    talker = FakeTalkerStream()
    code2wav = FakeCode2Wav()
    with TraceRecorder(trace_path) as recorder:
        runtime = RealtimeRuntime(
            Orchestrator(codec_chunk_frames=2, codec_left_context_frames=0), sink, recorder
        )
        driver = DuplexPipelineDriver(
            runtime,
            session,
            talker,
            code2wav,
        )
        tag = driver.begin_turn("speech-1")
        assert driver.on_eou(tag)
        decision = driver.process_audio_chunk(tag, b"input")

    assert decision is session.decision
    assert talker.push_calls == [([3], True)]
    assert code2wav.calls == [([11, 12], True)]
    assert sink.published == [b"\x0b\x0c"]
    assert driver.runtime.active_tag is None
    anchors = [record.anchor for record in load_records(trace_path)]
    assert Anchor.CODE2WAV_FIRST_PCM in anchors
    assert Anchor.PUBLISHED in anchors


def test_driver_drops_stale_audio_before_running_the_thinker() -> None:
    sink = FakeSink()
    session = FakeDuplexSession(FakeDecision(is_listen=False))
    talker = FakeTalkerStream()
    code2wav = FakeCode2Wav()
    driver = DuplexPipelineDriver(
        RealtimeRuntime(Orchestrator(codec_chunk_frames=2, codec_left_context_frames=0), sink),
        session,
        talker,
        code2wav,
    )

    old_tag = driver.begin_turn("speech-old")
    driver.begin_turn("speech-new")

    assert driver.process_audio_chunk(old_tag, b"stale") is None
    assert session.calls == []
    assert talker.push_calls == []
    assert code2wav.calls == []


def test_driver_barge_in_purges_stale_interstage_queue_before_it_can_run() -> None:
    sink = FakeSink()
    session = FakeDuplexSession(FakeDecision(is_listen=False))
    talker = FakeTalkerStream()
    code2wav = FakeCode2Wav()
    driver = DuplexPipelineDriver(
        RealtimeRuntime(Orchestrator(), sink), session, talker, code2wav
    )
    old_tag = driver.begin_turn("speech-old")
    driver.thinker_to_talker.put_nowait(
        PipelineChunk(
            stage=StageId.TALKER,
            source=StageId.THINKER,
            payload="old-work",
            turn_epoch=old_tag.turn_epoch,
            speech_id=old_tag.speech_id,
        )
    )

    driver.begin_turn("speech-new")

    assert driver.thinker_to_talker.qsize() == 0
