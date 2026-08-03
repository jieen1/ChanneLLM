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

参考实现(vllm_omni/engine/orchestrator.py):
_forward_to_next_stage:1728、_orchestration_loop、_route_output、
_prewarm_async_chunk_stages、_cleanup_request_ids、_build_terminal_empty_output。
明确不做:PD 分离、CFG companion、collective RPC、分布式 KV transfer、TP/PP。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from channellm.pipeline.stages import StageId, StageRequestState


class Orchestrator:
    """P2 实现。先落增量提交与生命周期接口,禁止提前引入分布式抽象。"""

    def __init__(self) -> None:
        self._requests: dict[str, StageRequestState] = {}

    def submit_initial(
        self, request_id: str, turn_epoch: int, speech_id: str = ""
    ) -> StageRequestState:
        """首次提交(九件事 #2 的前半)。"""
        if request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request_id}")
        state = StageRequestState(request_id=request_id, turn_epoch=turn_epoch, speech_id=speech_id)
        self._requests[request_id] = state
        return state

    def submit_update(self, request_id: str, stage: StageId, chunk: Any) -> None:
        """增量追加(九件事 #2 的后半):上游每产出一块就喂下游,不等整句。"""
        state = self._requests.get(request_id)
        if state is None or state.cancelled:
            return
        raise NotImplementedError("P2: route chunk to next stage when a unit is ready (#4)")

    def cancel(self, request_id: str) -> None:
        """九件事 #9:三阶段 + 传输队列四处齐停,旧 epoch 无条件丢弃。"""
        state = self._requests.get(request_id)
        if state is not None:
            state.cancelled = True

    def cleanup(self, request_id: str) -> None:
        """九件事 #7:三边清理,释放显存与队列。"""
        self._requests.pop(request_id, None)

    def active_requests(self) -> Iterable[StageRequestState]:
        return tuple(self._requests.values())


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
