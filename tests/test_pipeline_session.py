from channellm.duplex.session import SessionStateMachine, TurnPhase
from channellm.models.minicpmo import FACTS, find_weights
from channellm.pipeline.orchestrator import Orchestrator, new_request_id
from channellm.pipeline.stages import PipelineChunk, StageId, StageRequestState


def test_orchestrator_identity_and_cancel():
    orch = Orchestrator()
    request_id = new_request_id()
    state = orch.submit_initial(request_id, turn_epoch=1, speech_id="s1")
    assert isinstance(state, StageRequestState)
    assert state.request_id == request_id
    assert state.stage_request_ids[StageId.THINKER] == f"{request_id}:thinker"
    orch.cancel(request_id)
    assert orch._requests[request_id].cancelled
    orch.cleanup(request_id)
    assert request_id not in orch._requests


def test_orchestrator_rejects_duplicate():
    orch = Orchestrator()
    orch.submit_initial("r1", turn_epoch=1)
    try:
        orch.submit_initial("r1", turn_epoch=2)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_session_phases():
    session = SessionStateMachine()
    assert session.phase is TurnPhase.LISTENING
    session.on_eou()
    assert session.phase is TurnPhase.THINKING
    session.on_speak_start()
    assert session.phase is TurnPhase.SPEAKING
    session.on_barge_in()
    assert session.phase is TurnPhase.INTERRUPTED
    session.on_reply_done()
    assert session.phase is TurnPhase.LISTENING


def test_model_facts_match_design():
    # 与设计文档 §1 已核实事实一致
    assert FACTS.num_layers == 36
    assert FACTS.num_kv_heads == 8
    assert FACTS.head_dim == 128
    assert FACTS.audio_chunk_length == 1.0
    assert FACTS.audio_pool_step == 5


def test_find_weights_locates_cached_snapshot():
    path = find_weights()
    # 本机已下载校验 54/54;若环境缺失只允许返回 None,不允许抛错
    if path is not None:
        assert (path / "modeling_minicpmo.py").exists()
        assert (path / "config.json").exists()


def test_stage_pipeline_order():
    from channellm.pipeline.stages import PIPELINE_ORDER

    assert PIPELINE_ORDER == (StageId.THINKER, StageId.TALKER, StageId.CODE2WAV)


def test_orchestrator_incrementally_routes_and_flushes_codec_tail():
    orch = Orchestrator(codec_chunk_frames=2, codec_left_context_frames=1)
    orch.submit_initial("r1", turn_epoch=4, speech_id="s4")

    thinker = orch.submit_update("r1", StageId.THINKER, {"hidden": "h"})
    assert thinker == [
        PipelineChunk(
            stage=StageId.TALKER,
            source=StageId.THINKER,
            payload={"hidden": "h"},
            turn_epoch=4,
            speech_id="s4",
        )
    ]

    assert orch.submit_update("r1", StageId.TALKER, [1, 2]) == []
    codec = orch.submit_update("r1", StageId.TALKER, [3])
    assert codec[0].stage is StageId.CODE2WAV
    assert codec[0].source is StageId.TALKER
    assert codec[0].payload == (1, 2, 3)

    tail = orch.submit_update("r1", StageId.TALKER, [4], final=True)
    assert [chunk.payload for chunk in tail] == [(3, 4), None]
    assert tail[-1].final

    pcm = orch.submit_update("r1", StageId.CODE2WAV, b"pcm")
    assert pcm[0].source is StageId.CODE2WAV
    assert pcm[0].payload == b"pcm"
    terminal = orch.submit_update("r1", StageId.CODE2WAV, final=True)
    assert terminal == [
        PipelineChunk(
            stage=StageId.CODE2WAV,
            source=StageId.CODE2WAV,
            turn_epoch=4,
            speech_id="s4",
            final=True,
        )
    ]
    assert orch.submit_update("r1", StageId.CODE2WAV, final=True) == []


def test_orchestrator_prewarm_once_and_cancel_drops_pending_codec():
    orch = Orchestrator()
    seen: list[StageId] = []
    orch.prewarm(seen.append)
    orch.prewarm(seen.append)
    assert seen == [StageId.TALKER, StageId.CODE2WAV]

    state = orch.submit_initial("r1", turn_epoch=1)
    orch.submit_update("r1", StageId.TALKER, [1, 2])
    assert state.codec_buffer == [1, 2]
    orch.cancel("r1")
    assert state.codec_buffer == []
    assert orch.submit_update("r1", StageId.TALKER, [3]) == []


def test_orchestrator_flushes_terminal_once_and_ignores_post_final_updates():
    orch = Orchestrator(codec_chunk_frames=3, codec_left_context_frames=1)
    state = orch.submit_initial("r1", turn_epoch=1)

    emitted = orch.submit_update("r1", StageId.TALKER, [1, 2, 3, 4], final=True)

    # 正常块消费 3 帧后，最后 1 帧作为上下文保留给尾包；不能提前丢掉，
    # 否则 Code2Wav 在收尾处会断音。
    assert [chunk.payload for chunk in emitted] == [(1, 2, 3, 4), (4,), None]
    assert emitted[-1].final
    assert StageId.TALKER in state.finished_stages
    assert orch.submit_update("r1", StageId.TALKER, [5]) == []
    assert orch.submit_update("r1", StageId.TALKER, final=True) == []
