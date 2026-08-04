"""L2 三阶段编排骨架(P2)—— 这一层就是首包延迟本身(设计文档 §4)。

九件事清单(缺一即退化为串行或泄漏):

#1 三引擎身份对齐        -> StageRequestState(pipeline/stages.py)
#2 增量提交              -> submit_initial / submit_update(本文件)
#3 下游 resumable        -> 下游生成完不关闭,挂等更多上游输入
#4 攒够单元再转发        -> hold 直到够 CODEC_CHUNK_FRAMES,切太碎/等太久都不行
#5 下游预热              -> Stage0 启动即预热 Stage1/2,消首包冷启动
#6 跨阶段 tokenizer      -> Stage0 独占 tokenizer,借给下游输入处理器
#7 错误三边清理          -> 任一 stage 失败,三处请求全清,防泄漏
#8 终止输出合成          -> 上游结束下游无产出时凭空造终止输出,防客户端挂起
#9 打断时三阶段同时取消  -> Thinker/Talker/Code2Wav/传输队列四处齐停(见 duplex.epoch)

参考实现(vllm_omni/engine/orchestrator.py):_build_terminal_empty_output、
_cleanup_request_ids、_prewarm_async_chunk_stages、_forward_to_next_stage、
_orchestration_loop、_route_output。
明确不做:PD 分离、CFG companion、collective RPC、分布式 KV transfer、TP/PP。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from typing import Any

from channellm.pipeline.stages import (
    CODEC_CHUNK_FRAMES,
    CODEC_INITIAL_MIN_AUDIO_FRAMES,
    CODEC_LEFT_CONTEXT_FRAMES,
    CODEC_STREAM_SILENCE_TOKEN,
    PIPELINE_ORDER,
    PipelineChunk,
    StageId,
    StageRequestState,
)


class Orchestrator:
    """单进程三阶段编排的生命周期与增量路由。

    引擎仍由调用方拥有：本类只把上游刚产出的数据变为下游立即可消费的
    ``PipelineChunk``，没有 RPC、后台 worker 或分布式状态。这是把实时节拍
    与模型实现解耦的最小边界。
    """

    def __init__(
        self,
        codec_chunk_frames: int = CODEC_CHUNK_FRAMES,
        codec_left_context_frames: int = CODEC_LEFT_CONTEXT_FRAMES,
        codec_silence_token: int = CODEC_STREAM_SILENCE_TOKEN,
        codec_initial_min_audio_frames: int = CODEC_INITIAL_MIN_AUDIO_FRAMES,
    ) -> None:
        if codec_chunk_frames <= 0:
            raise ValueError("codec_chunk_frames must be positive")
        if codec_left_context_frames < 0:
            raise ValueError("codec_left_context_frames must be non-negative")
        if codec_initial_min_audio_frames <= 0:
            raise ValueError("codec_initial_min_audio_frames must be positive")
        if not isinstance(codec_silence_token, int):
            raise TypeError("codec_silence_token must be an int")
        self._requests: dict[str, StageRequestState] = {}
        self.codec_chunk_frames = codec_chunk_frames
        self.codec_left_context_frames = codec_left_context_frames
        self.codec_silence_token = codec_silence_token
        self.codec_initial_min_audio_frames = codec_initial_min_audio_frames
        self._prewarmed: set[StageId] = set()

    def submit_initial(
        self, request_id: str, turn_epoch: int, speech_id: str = ""
    ) -> StageRequestState:
        """首次提交(九件事 #2 的前半)。"""
        if request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        state = StageRequestState(request_id=request_id, turn_epoch=turn_epoch, speech_id=speech_id)
        state.stage_request_ids = {
            stage: f"{request_id}:{stage.value}" for stage in PIPELINE_ORDER
        }
        self._requests[request_id] = state
        return state

    def prewarm(self, callback: Callable[[StageId], None] | None = None) -> None:
        """预热下游 stage（#5）。

        真正的权重加载由引擎所有者注入 callback；编排器只保证在首请求之前
        按 Talker→Code2Wav 的顺序执行一次，避免把加载策略硬编码到 L2。
        """
        for stage in (StageId.TALKER, StageId.CODE2WAV):
            if stage not in self._prewarmed:
                if callback is not None:
                    callback(stage)
                self._prewarmed.add(stage)

    def submit_update(
        self,
        request_id: str,
        stage: StageId,
        chunk: Any = None,
        *,
        final: bool = False,
    ) -> list[PipelineChunk]:
        """提交一个 stage 的增量输出并返回立即可推进的下游输入。

        Thinker 输出逐块转发给 Talker；Talker 的 codec token 只在达到
        ``chunk_frames + left_context`` 时转发给 Code2Wav。``final`` 会冲刷
        尾部并发出终止输入，因此下游可保持 resumable，而不会因上游先结束而
        永久等待（#2/#3/#4/#8）。
        """
        state = self._requests.get(request_id)
        if state is None or state.cancelled:
            return []
        if stage in state.finished_stages:
            return []

        if stage is StageId.THINKER:
            return self._route_thinker(state, chunk, final)
        if stage is StageId.TALKER:
            return self._route_talker(state, chunk, final)
        if stage is StageId.CODE2WAV:
            return self._route_code2wav(state, chunk, final)
        raise ValueError(f"unknown stage: {stage}")

    def _route_thinker(
        self, state: StageRequestState, chunk: Any, final: bool
    ) -> list[PipelineChunk]:
        if final:
            state.finished_stages.add(StageId.THINKER)
        if chunk is None and not final:
            return []
        return [
            self._chunk(
                state, StageId.TALKER, chunk, source=StageId.THINKER, final=final
            )
        ]

    def _route_talker(
        self, state: StageRequestState, chunk: Any, final: bool
    ) -> list[PipelineChunk]:
        first_talker_delta = False
        if chunk is not None:
            frames = _as_frames(chunk)
            if frames:
                first_talker_delta = not state.codec_prefix_seeded
                self._seed_codec_prefix(state)
                state.codec_buffer.extend(frames)

        emitted: list[PipelineChunk] = []
        if first_talker_delta:
            # MiniCPM-o 的首个 TTS 调用使用 force_flush：不必等待 25 帧，只要
            # 有 3 帧前瞻和至少 5 个 codec 帧就调用同一 Token2Wav stream。该
            # 尝试只属于首个 delta；若它本身不足阈值，后续仍走普通 25 帧节拍，
            # 与官方 `force_flush` 的单次调用语义一致。
            state.initial_codec_flush_attempted = True
            initial_audio_frames = min(
                self.codec_initial_min_audio_frames, self.codec_chunk_frames
            )
            initial_required = self.codec_left_context_frames + initial_audio_frames
            if len(state.codec_buffer) >= initial_required:
                initial_size = min(
                    self.codec_chunk_frames + self.codec_left_context_frames,
                    len(state.codec_buffer),
                )
                emitted.append(
                    self._chunk(
                        state,
                        StageId.CODE2WAV,
                        tuple(state.codec_buffer[:initial_size]),
                        source=StageId.TALKER,
                    )
                )
                consumed = min(
                    self.codec_chunk_frames,
                    initial_size - self.codec_left_context_frames,
                )
                del state.codec_buffer[:consumed]

        required = self.codec_chunk_frames + self.codec_left_context_frames
        while len(state.codec_buffer) >= required:
            emitted.append(
                self._chunk(
                    state,
                    StageId.CODE2WAV,
                    tuple(state.codec_buffer[:required]),
                    source=StageId.TALKER,
                )
            )
            del state.codec_buffer[: self.codec_chunk_frames]

        if final:
            state.finished_stages.add(StageId.TALKER)
            if state.codec_buffer:
                emitted.append(
                    self._chunk(
                        state,
                        StageId.CODE2WAV,
                        tuple(state.codec_buffer),
                        source=StageId.TALKER,
                    )
                )
                state.codec_buffer.clear()
            emitted.append(
                self._chunk(
                    state, StageId.CODE2WAV, None, source=StageId.TALKER, final=True
                )
            )
        return emitted

    def _seed_codec_prefix(self, state: StageRequestState) -> None:
        """只在首个真实 codec delta 前插入官方 S3 静音前瞻码。

        首块须为 ``[4218] * 3 + 25 个模型帧``，这样 Code2Wav 不必等待第二个
        Talker phrase。若一个 turn 从未产生 codec，绝不因该前缀合成虚假静音。
        """
        if state.codec_prefix_seeded:
            return
        state.codec_prefix_seeded = True
        state.codec_buffer.extend(
            [self.codec_silence_token] * self.codec_left_context_frames
        )

    def _route_code2wav(
        self, state: StageRequestState, chunk: Any, final: bool
    ) -> list[PipelineChunk]:
        if final:
            state.finished_stages.add(StageId.CODE2WAV)
            if state.terminal_emitted:
                return []
            state.terminal_emitted = True
        if chunk is None and not final:
            return []
        return [
            self._chunk(
                state, StageId.CODE2WAV, chunk, source=StageId.CODE2WAV, final=final
            )
        ]

    @staticmethod
    def _chunk(
        state: StageRequestState,
        stage: StageId,
        payload: Any,
        source: StageId,
        final: bool = False,
    ) -> PipelineChunk:
        return PipelineChunk(
            stage=stage,
            source=source,
            payload=payload,
            turn_epoch=state.turn_epoch,
            speech_id=state.speech_id,
            final=final,
        )

    def cancel(self, request_id: str) -> None:
        """九件事 #9:三阶段 + 传输队列四处齐停,旧 epoch 无条件丢弃。"""
        state = self._requests.get(request_id)
        if state is not None:
            state.cancelled = True
            state.codec_buffer.clear()
            state.finished_stages.update(PIPELINE_ORDER)

    def cleanup(self, request_id: str) -> None:
        """九件事 #7:三边清理,释放显存与队列。"""
        self._requests.pop(request_id, None)

    def active_requests(self) -> Iterable[StageRequestState]:
        return tuple(self._requests.values())


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _as_frames(chunk: Any) -> tuple[Any, ...]:
    """把单个 codec token 或一次产出的一组 token 规范化为帧序列。"""
    if isinstance(chunk, (str, bytes)):
        return (chunk,)
    try:
        return tuple(chunk)
    except TypeError:
        return (chunk,)
