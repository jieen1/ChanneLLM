"""真实模型三阶段驱动。

把 L1 的 ``DuplexSession`` / ``TalkerStream`` / ``Code2Wav`` 接到 L2/L3；
调用者只需在每个输入音频 chunk 到达时调用 ``process_audio_chunk``。该驱动
不引入后台 worker 或网络抽象，所有 stage 之间仍是进程内 ``PipelineChunk``。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from channellm.duplex.epoch import EpochTag
from channellm.duplex.runtime import RealtimeRuntime
from channellm.pipeline.stages import PipelineChunk, StageId
from channellm.pipeline.transport import ChunkChannel


@dataclasses.dataclass(frozen=True)
class ThinkerUnit:
    """一个 duplex unit 可交给 Talker 的语义条件。"""

    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    end_of_turn: bool


class DuplexPipelineDriver:
    """同步推进真实 Thinker → Talker → Code2Wav 的单回合路径。

    Talker/Code2Wav 必须以当前 ``EpochTag`` 进入 ``RealtimeRuntime``，因此
    打断后的旧音频在调用模型前就被拒绝，且即使 race 到合成完成也无法发布。
    """

    def __init__(
        self,
        runtime: RealtimeRuntime,
        duplex_session: Any,
        talker_stream: Any,
        code2wav: Any,
        thinker_to_talker: ChunkChannel[PipelineChunk] | None = None,
        talker_to_code2wav: ChunkChannel[PipelineChunk] | None = None,
    ) -> None:
        self.runtime = runtime
        self.duplex_session = duplex_session
        self.talker_stream = talker_stream
        self.code2wav = code2wav
        self.thinker_to_talker = thinker_to_talker or ChunkChannel("thinker->talker")
        self.talker_to_code2wav = talker_to_code2wav or ChunkChannel("talker->code2wav")

    def begin_turn(self, speech_id: str = "") -> EpochTag:
        """开始输入回合，并同步清除上一回合的 TTS/vocoder 状态。"""
        self.talker_stream.reset()
        self.code2wav.stream_reset()
        tag = self.runtime.begin_turn(speech_id)
        self._discard_stale_queued_work(tag)
        return tag

    def on_eou(self, tag: EpochTag) -> bool:
        """记录外部 EOU 观测；说话权仍由 duplex 模型输出决定。"""
        return self.runtime.on_eou(tag)

    def process_audio_chunk(self, tag: EpochTag, pcm: Any) -> Any | None:
        """同步处理一个 16kHz 输入 chunk，并立即发布可得的 24kHz PCM。"""
        if self.runtime.active_tag != tag:
            return None

        decision = self.duplex_session.on_chunk(pcm)
        if decision.is_listen or self.runtime.active_tag != tag:
            return decision

        token_ids, hidden_states = self.duplex_session.latest_unit_conditioning()
        unit = ThinkerUnit(token_ids, hidden_states, decision.end_of_turn)
        talker_inputs = self.runtime.submit_stage_output(tag, StageId.THINKER, unit)
        for talker_input in talker_inputs:
            self.thinker_to_talker.put_nowait(talker_input)
        while (talker_input := self.thinker_to_talker.get_nowait()) is not None:
            if self.runtime.active_tag != tag:
                return decision
            if talker_input.stage is not StageId.TALKER:
                continue
            condition = talker_input.payload
            if not isinstance(condition, ThinkerUnit):
                raise TypeError("Thinker stage must emit ThinkerUnit")
            codec_tokens = self.talker_stream.push(
                condition.token_ids,
                condition.hidden_states,
                end_of_turn=condition.end_of_turn,
            )
            code2wav_inputs = self.runtime.submit_stage_output(
                tag,
                StageId.TALKER,
                codec_tokens,
                final=condition.end_of_turn,
            )
            for code2wav_input in code2wav_inputs:
                self.talker_to_code2wav.put_nowait(code2wav_input)
        self._synthesize(tag, self._drain(self.talker_to_code2wav))
        return decision

    @staticmethod
    def _drain(channel: ChunkChannel[PipelineChunk]) -> list[PipelineChunk]:
        chunks: list[PipelineChunk] = []
        while (chunk := channel.get_nowait()) is not None:
            chunks.append(chunk)
        return chunks

    def _discard_stale_queued_work(self, tag: EpochTag) -> None:
        for channel in (self.thinker_to_talker, self.talker_to_code2wav):
            channel.discard(
                lambda chunk: (chunk.turn_epoch, chunk.speech_id)
                != (tag.turn_epoch, tag.speech_id)
            )

    def _synthesize(self, tag: EpochTag, inputs: list[PipelineChunk]) -> None:
        for index, chunk in enumerate(inputs):
            if self.runtime.active_tag != tag:
                return
            if chunk.source is StageId.TALKER and chunk.payload is not None:
                next_chunk = inputs[index + 1] if index + 1 < len(inputs) else None
                is_last_audio_chunk = bool(next_chunk is not None and next_chunk.final)
                wave = self.code2wav.stream_chunk(
                    list(chunk.payload), last_chunk=is_last_audio_chunk
                )
                if _has_samples(wave):
                    self.runtime.submit_stage_output(tag, StageId.CODE2WAV, wave)
            elif chunk.source is StageId.TALKER and chunk.final:
                self.runtime.submit_stage_output(tag, StageId.CODE2WAV, final=True)
                self.runtime.finish_turn(tag)


def _has_samples(wave: Any) -> bool:
    size = getattr(wave, "size", None)
    if size is not None:
        return bool(size)
    try:
        return bool(len(wave))
    except TypeError:
        return wave is not None
