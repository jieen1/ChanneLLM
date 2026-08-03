from channellm.duplex.session import SessionStateMachine, TurnPhase
from channellm.models.minicpmo import FACTS, find_weights
from channellm.pipeline.orchestrator import Orchestrator, new_request_id
from channellm.pipeline.stages import StageId, StageRequestState


def test_orchestrator_identity_and_cancel():
    orch = Orchestrator()
    request_id = new_request_id()
    state = orch.submit_initial(request_id, turn_epoch=1, speech_id="s1")
    assert isinstance(state, StageRequestState)
    assert state.request_id == request_id
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
