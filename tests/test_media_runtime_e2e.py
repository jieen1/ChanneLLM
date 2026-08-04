from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from channellm.app.event_store import EventStore
from channellm.duplex.driver import DuplexPipelineDriver
from channellm.duplex.ingress import PcmIngress
from channellm.duplex.playback import BufferedPlaybackSink, PcmPlayoutPump
from channellm.duplex.queued_runtime import QueuedDuplexRuntime
from channellm.duplex.runtime import RealtimeRuntime
from channellm.pipeline.orchestrator import Orchestrator
from channellm.tracing import TraceRecorder, load_records


@dataclass
class _Decision:
    is_listen: bool = False
    end_of_turn: bool = True
    prefill_start_ns: int = 0
    prefill_done_ns: int = 0
    first_token_decoded_ns: int = 0


class _Session:
    def on_chunk(self, _pcm: np.ndarray) -> _Decision:
        return _Decision()

    def latest_unit_conditioning(self):
        return torch.tensor([1]), torch.zeros((1, 1))


class _Talker:
    def reset(self) -> None:
        pass

    def push_streaming(self, _token_ids, _hidden_states, *, end_of_turn: bool):
        assert end_of_turn
        yield [7, 8], True


class _Code2Wav:
    def reset(self) -> None:
        pass

    def stream_reset(self) -> None:
        pass

    def stream_chunk(self, tokens: list[int], last_chunk: bool = False) -> bytes:
        assert tokens == [7, 8]
        assert last_chunk
        return b"pcm"


def test_local_media_path_reaches_played_fact_and_recovery_context(tmp_path) -> None:
    trace_path = tmp_path / "media.jsonl"
    with EventStore(tmp_path / "events.sqlite", session_epoch=7) as store:
        with TraceRecorder(trace_path) as recorder:
            sink = BufferedPlaybackSink()
            runtime = RealtimeRuntime(
                Orchestrator(codec_chunk_frames=2, codec_left_context_frames=0),
                sink,
                trace_recorder=recorder,
                event_store=store,
            )
            driver = DuplexPipelineDriver(
                runtime,
                _Session(),
                _Talker(),
                _Code2Wav(),
                response_text=lambda: "端到端回答",
            )
            queued = QueuedDuplexRuntime(driver)
            ingress = PcmIngress(queued)
            try:
                tag = ingress.begin_speech("speech")
                assert ingress.push_frame(
                    np.zeros(16_000, dtype=np.float32), sample_rate=16_000
                ) == 1
                assert ingress.end_speech()
                assert queued.wait_idle(1.0)
                assert queued.failures == []
            finally:
                assert queued.close()

            written: list[tuple[bytes, object]] = []
            pump = PcmPlayoutPump(
                sink,
                runtime,
                lambda pcm, item_tag: written.append((pcm, item_tag)),
            )
            assert pump.pump() == 1
            state = store.recovery_state(budget_tokens=10, token_counter=lambda _text: 1)

    assert written == [(b"pcm", tag)]
    assert state.context.text == "助手: 端到端回答"
    anchors = [record.anchor for record in load_records(trace_path)]
    assert "eou_detected" in anchors
    assert "code2wav_first_pcm" in anchors
    assert "device_playout_start" in anchors
