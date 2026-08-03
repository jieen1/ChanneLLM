from __future__ import annotations

from dataclasses import dataclass

from channellm.app.event_store import EventKind, EventStore
from channellm.duplex.epoch import EpochTag
from channellm.duplex.playback import BufferedPlaybackSink
from channellm.duplex.runtime import RealtimeRuntime
from channellm.duplex.session import TurnPhase
from channellm.metrics.latency import waterfall
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


def test_runtime_routes_three_stages_and_records_a_single_first_pcm_anchor(tmp_path):
    trace_path = tmp_path / "runtime.jsonl"
    sink = FakeSink()
    with TraceRecorder(trace_path, session_id="rt-three-stage") as recorder:
        runtime = RealtimeRuntime(
            orchestrator=Orchestrator(codec_chunk_frames=2, codec_left_context_frames=1),
            sink=sink,
            trace_recorder=recorder,
        )
        tag = runtime.begin_turn("speech-three-stage")
        assert runtime.on_eou(tag)

        talker_input = runtime.submit_stage_output(tag, StageId.THINKER, {"hidden": "h"})
        assert len(talker_input) == 1
        assert talker_input[0].stage is StageId.TALKER

        assert runtime.submit_stage_output(tag, StageId.TALKER, [1, 2]) == []
        code2wav_input = runtime.submit_stage_output(tag, StageId.TALKER, [3])
        assert len(code2wav_input) == 1
        assert code2wav_input[0].stage is StageId.CODE2WAV
        assert code2wav_input[0].payload == (1, 2, 3)
        assert sink.published == []

        runtime.submit_stage_output(tag, StageId.CODE2WAV, b"pcm-1")
        runtime.submit_stage_output(tag, StageId.CODE2WAV, b"pcm-2")

    assert sink.published == [(b"pcm-1", tag), (b"pcm-2", tag)]
    assert runtime.state_machine.phase is TurnPhase.SPEAKING
    records = load_records(trace_path)
    assert all(record.turn_epoch == tag.turn_epoch for record in records)
    assert all(record.speech_id == tag.speech_id for record in records)
    assert len({record.trace_id for record in records}) == 1
    anchors = [record.anchor for record in records]
    assert anchors.count(Anchor.SPEAK_DECISION) == 1
    assert anchors.count(Anchor.TALKER_CHUNK_READY) == 1
    assert anchors.count(Anchor.CODE2WAV_FIRST_PCM) == 1
    assert anchors.count(Anchor.PUBLISHED) == 2

    report = waterfall(records)[()]
    assert report["eou_to_speak_decision"]["n"] == 1
    assert report["eou_to_first_pcm_local"]["n"] == 1
    assert report["speak_decision_to_first_pcm"]["n"] == 1


def test_runtime_barge_in_drops_stale_thinker_and_talker_before_downstream_work():
    orchestrator = FakeOrchestrator()
    sink = FakeSink()
    runtime = RealtimeRuntime(orchestrator=orchestrator, sink=sink)

    old_tag = runtime.begin_turn("speech-old")
    runtime.begin_turn("speech-new")
    assert runtime.submit_stage_output(old_tag, StageId.THINKER, {"hidden": "old"}) == []
    assert runtime.submit_stage_output(old_tag, StageId.TALKER, [1, 2, 3]) == []

    assert orchestrator.update_calls == []
    assert sink.published == []


def test_runtime_rejects_mismatched_speech_id_pcm_even_if_epoch_matches():
    orchestrator = FakeOrchestrator()
    sink = FakeSink()
    runtime = RealtimeRuntime(orchestrator=orchestrator, sink=sink)

    tag = runtime.begin_turn("speech-current")
    request_id = orchestrator.initial_calls[0][0]
    stale_speech = EpochTag(turn_epoch=tag.turn_epoch, speech_id="speech-other")
    emitted_chunk = FakePipelineChunk(
        stage=StageId.CODE2WAV,
        payload=b"wrong-speech",
        tag=stale_speech,
    )
    orchestrator.responses[(request_id, StageId.CODE2WAV)] = [emitted_chunk]

    assert runtime.submit_stage_output(tag, StageId.CODE2WAV, b"unused") == [emitted_chunk]
    assert sink.published == []


def test_trace_pairing_does_not_cross_turns_after_barge_in(tmp_path):
    trace_path = tmp_path / "runtime.jsonl"
    sink = FakeSink()
    with TraceRecorder(trace_path, session_id="rt-barge-pairing") as recorder:
        runtime = RealtimeRuntime(
            orchestrator=Orchestrator(codec_chunk_frames=1, codec_left_context_frames=0),
            sink=sink,
            trace_recorder=recorder,
        )
        old_tag = runtime.begin_turn("speech-old")
        assert runtime.on_eou(old_tag)
        new_tag = runtime.begin_turn("speech-new")
        assert runtime.on_eou(new_tag)
        runtime.submit_stage_output(new_tag, StageId.THINKER, {"hidden": "new"})
        runtime.submit_stage_output(new_tag, StageId.TALKER, [7])
        runtime.submit_stage_output(new_tag, StageId.CODE2WAV, b"new-pcm")

    records = load_records(trace_path)
    report = waterfall(records)[()]
    assert report["eou_to_first_pcm_local"]["n"] == 1
    assert report["barge_in_to_silence"]["n"] == 1
    pcm_records = [record for record in records if record.anchor == Anchor.CODE2WAV_FIRST_PCM]
    assert [(record.turn_epoch, record.speech_id) for record in pcm_records] == [
        (new_tag.turn_epoch, new_tag.speech_id)
    ]


def test_media_playout_anchor_survives_reply_cleanup_but_not_barge_in(tmp_path):
    trace_path = tmp_path / "runtime.jsonl"
    sink = BufferedPlaybackSink()
    with TraceRecorder(trace_path) as recorder:
        runtime = RealtimeRuntime(Orchestrator(), sink, trace_recorder=recorder)
        tag = runtime.begin_turn("speech-finished")
        runtime.submit_stage_output(tag, StageId.CODE2WAV, b"pcm")
        assert runtime.finish_turn(tag)
        assert runtime.on_device_playout_start(tag)
        assert sink.drain() == [(b"pcm", tag)]

        old_tag = runtime.begin_turn("speech-old")
        runtime.submit_stage_output(old_tag, StageId.CODE2WAV, b"old-pcm")
        runtime.begin_turn("speech-new")
        assert not runtime.on_device_playout_start(old_tag)

    anchors = [record.anchor for record in load_records(trace_path)]
    assert anchors.count(Anchor.DEVICE_PLAYOUT_START) == 1
    assert sink.muted_items == 1


def test_new_input_mutes_buffered_reply_after_model_generation_has_finished(tmp_path):
    trace_path = tmp_path / "runtime.jsonl"
    sink = BufferedPlaybackSink()
    with TraceRecorder(trace_path) as recorder:
        runtime = RealtimeRuntime(Orchestrator(), sink, trace_recorder=recorder)
        old_tag = runtime.begin_turn("speech-old")
        runtime.submit_stage_output(old_tag, StageId.CODE2WAV, b"old-pcm")
        assert runtime.finish_turn(old_tag)
        assert sink.qsize() == 1

        runtime.begin_turn("speech-new")

    assert sink.qsize() == 0
    assert sink.muted_items == 1
    anchors = [record.anchor for record in load_records(trace_path)]
    assert anchors.count(Anchor.BARGE_IN_DETECTED) == 1
    assert anchors.count(Anchor.PLAYOUT_MUTED) == 1


def test_finished_playout_is_not_muted_by_the_next_input() -> None:
    sink = BufferedPlaybackSink()
    runtime = RealtimeRuntime(Orchestrator(), sink)
    tag = runtime.begin_turn("speech-finished")
    runtime.submit_stage_output(tag, StageId.CODE2WAV, b"pcm")
    assert runtime.finish_turn(tag)
    assert sink.drain() == [(b"pcm", tag)]
    assert runtime.on_device_playout_finished(tag)

    runtime.begin_turn("speech-next")

    assert sink.muted_items == 0
