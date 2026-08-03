from channellm.tracing import Anchor, TraceRecorder, load_records


def test_recorder_roundtrip(tmp_path):
    path = tmp_path / "run.jsonl"
    with TraceRecorder(path, session_id="s1", tags={"loc": "local"}) as rec:
        trace_id = rec.new_trace()
        rec.anchor(Anchor.EOU_DETECTED, trace_id=trace_id, turn_epoch=2)
        rec.anchor(
            Anchor.CODE2WAV_FIRST_PCM,
            trace_id=trace_id,
            turn_epoch=2,
            extra_note="synthetic",
        )
    records = load_records(path)
    assert len(records) == 2
    assert records[0].anchor == Anchor.EOU_DETECTED
    assert records[0].tags == {"loc": "local"}
    assert records[0].session_id == "s1"
    assert records[1].extra == {"extra_note": "synthetic"}
    assert records[0].ts_ns <= records[1].ts_ns


def test_recorder_can_replace_prior_experiment_records(tmp_path):
    path = tmp_path / "trace.jsonl"
    with TraceRecorder(path, session_id="old") as recorder:
        recorder.anchor(Anchor.EOU_DETECTED, trace_id="old")

    with TraceRecorder(path, session_id="new", append=False) as recorder:
        recorder.anchor(Anchor.CODE2WAV_FIRST_PCM, trace_id="new")

    records = load_records(path)
    assert [(record.session_id, record.trace_id, record.anchor) for record in records] == [
        ("new", "new", Anchor.CODE2WAV_FIRST_PCM)
    ]
