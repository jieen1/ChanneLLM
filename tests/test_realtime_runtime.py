from __future__ import annotations

from dataclasses import dataclass

from channellm.app.event_store import EventKind, EventStore
from channellm.duplex.epoch import EpochTag
from channellm.duplex.runtime import RealtimeRuntime
from channellm.duplex.session import TurnPhase
from channellm.pipeline.orchestrator import Orchestrator
from channellm.pipeline.stages import StageId, StageRequestState
from channellm.tracing import Anchor, TraceRecorder, load_records


@dataclass(eq=True)
class FakePipelineChunk:
    stage: StageId
    payload: bytes
    tag: EpochTag
    final: bool = False


class FakeOrchestrator:
    def __init__(self) -> None:
        self.initial_calls: list[tuple[str, int, str]] = []
        self.update_calls: list[tuple[str, StageId, object]] = []
        self.cancel_calls: list[str] = []
        self.cleanup_calls: list[str] = []
        self.responses: dict[tuple[str, StageId], list[FakePipelineChunk]] = {}

    def submit_initial(
        self, request_id: str, turn_epoch: int, speech_id: str = ""
    ) -> StageRequestState:
        self.initial_calls.append((request_id, turn_epoch, speech_id))
        return StageRequestState(
            request_id=request_id,
            turn_epoch=turn_epoch,
            speech_id=speech_id,
        )

    def submit_update(
        self,
        request_id: str,
        stage: StageId,
        chunk: object,
    ) -> list[FakePipelineChunk]:
        self.update_calls.append((request_id, stage, chunk))
        return list(self.responses.get((request_id, stage), ()))

    def cancel(self, request_id: str) -> None:
        self.cancel_calls.append(request_id)

    def cleanup(self, request_id: str) -> None:
        self.cleanup_calls.append(request_id)


class FakeSink:
    def __init__(self) -> None:
        self.mute_calls = 0
        self.published: list[tuple[bytes, EpochTag]] = []

    def mute(self) -> None:
        self.mute_calls += 1

    def publish(self, pcm: bytes, tag: EpochTag) -> None:
        self.published.append((pcm, tag))


def test_runtime_publishes_fresh_audio_and_records_trace_and_events(tmp_path):
    trace_path = tmp_path / "runtime.jsonl"
    event_path = tmp_path / "events.sqlite"
    orchestrator = FakeOrchestrator()
    sink = FakeSink()
    with TraceRecorder(trace_path, session_id="rt-1") as recorder, EventStore(event_path) as store:
        runtime = RealtimeRuntime(
            orchestrator=orchestrator,
            sink=sink,
            trace_recorder=recorder,
            event_store=store,
        )

        tag = runtime.begin_turn("speech-1")
        runtime.on_eou()
        request_id = orchestrator.initial_calls[0][0]
        chunk = FakePipelineChunk(stage=StageId.CODE2WAV, payload=b"pcm-1", tag=tag, final=False)
        orchestrator.responses[(request_id, StageId.CODE2WAV)] = [chunk]

        emitted = runtime.submit_stage_output(tag, StageId.CODE2WAV, {"codec": [1, 2, 3]})
        runtime.finish_turn(tag)

    assert emitted == [chunk]
    assert sink.published == [(b"pcm-1", tag)]
    assert runtime.state_machine.phase is TurnPhase.LISTENING
    assert orchestrator.cleanup_calls == [request_id]

    with EventStore(event_path) as store:
        events = list(store.iterate())
    assert [event.kind for event in events] == [
        EventKind.AGENT_SPEECH_PLANNED.value,
        EventKind.AGENT_SPEECH_ACTUALLY_PLAYED.value,
    ]
    assert all(event.speech_id == "speech-1" for event in events)
    assert all(event.turn_id == request_id for event in events)

    anchors = [record.anchor for record in load_records(trace_path)]
    assert Anchor.EOU_DETECTED in anchors
    assert Anchor.CODE2WAV_FIRST_PCM in anchors
    assert Anchor.PUBLISHED in anchors
    records = load_records(trace_path)
    assert {record.trace_id for record in records} == {records[0].trace_id}
    assert records[0].trace_id


def test_runtime_barge_in_cancels_without_cleanup_and_drops_stale_audio(tmp_path):
    trace_path = tmp_path / "runtime.jsonl"
    orchestrator = FakeOrchestrator()
    sink = FakeSink()
    with TraceRecorder(trace_path, session_id="rt-2") as recorder:
        runtime = RealtimeRuntime(orchestrator=orchestrator, sink=sink, trace_recorder=recorder)

        old_tag = runtime.begin_turn("speech-old")
        old_request_id = orchestrator.initial_calls[0][0]
        new_tag = runtime.begin_turn("speech-new")
        stale_chunk = FakePipelineChunk(
            stage=StageId.CODE2WAV,
            payload=b"stale-pcm",
            tag=old_tag,
            final=False,
        )
        orchestrator.responses[(old_request_id, StageId.CODE2WAV)] = [stale_chunk]

        emitted = runtime.submit_stage_output(old_tag, StageId.CODE2WAV, {"codec": [9]})

    assert new_tag.turn_epoch == old_tag.turn_epoch + 1
    assert sink.mute_calls == 1
    assert sink.published == []
    assert orchestrator.cancel_calls == [old_request_id]
    assert orchestrator.cleanup_calls == []
    assert emitted == []
    assert runtime.state_machine.phase is TurnPhase.INTERRUPTED
    assert runtime.reap_cancelled() == 1
    assert orchestrator.cleanup_calls == [old_request_id]

    anchors = [record.anchor for record in load_records(trace_path)]
    assert Anchor.BARGE_IN_DETECTED in anchors
    assert Anchor.PLAYOUT_MUTED in anchors


def test_runtime_finish_turn_resets_terminal_no_output_state(tmp_path):
    orchestrator = FakeOrchestrator()
    sink = FakeSink()
    with EventStore(tmp_path / "events.sqlite") as store:
        runtime = RealtimeRuntime(orchestrator=orchestrator, sink=sink, event_store=store)

        tag = runtime.begin_turn("speech-silent")
        runtime.on_eou()
        request_id = orchestrator.initial_calls[0][0]

        emitted = runtime.submit_stage_output(tag, StageId.THINKER, {"text": "..."}, final=True)
        runtime.finish_turn(tag)
        events = list(store.iterate())

    assert emitted == []
    assert sink.published == []
    assert runtime.state_machine.phase is TurnPhase.LISTENING
    assert orchestrator.cleanup_calls == [request_id]
    assert events == []


def test_runtime_only_publishes_code2wav_pcm_not_interstage_codec(tmp_path):
    orchestrator = Orchestrator(codec_chunk_frames=2, codec_left_context_frames=0)
    sink = FakeSink()
    trace_path = tmp_path / "runtime.jsonl"
    with TraceRecorder(trace_path) as recorder:
        runtime = RealtimeRuntime(
            orchestrator=orchestrator,
            sink=sink,
            trace_recorder=recorder,
        )

        tag = runtime.begin_turn("speech-stream")
        runtime.on_eou(tag)
        talker_input = runtime.submit_stage_output(tag, StageId.THINKER, {"hidden": "h"})
        assert talker_input[0].stage is StageId.TALKER

        codec_input = runtime.submit_stage_output(tag, StageId.TALKER, [1, 2])
        assert codec_input[0].stage is StageId.CODE2WAV
        assert sink.published == []

        pcm = runtime.submit_stage_output(tag, StageId.CODE2WAV, b"real-pcm")
        assert pcm[0].source is StageId.CODE2WAV
        assert sink.published == [(b"real-pcm", tag)]

    anchors = [record.anchor for record in load_records(trace_path)]
    assert Anchor.SPEAK_DECISION in anchors
    assert Anchor.TALKER_CHUNK_READY in anchors
