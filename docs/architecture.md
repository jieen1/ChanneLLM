# 架构总览

源:设计文档 v3 §3。自研边界 L0–L4;不自研:LiveKit(媒体传输)、外部大模型(任务执行)。

```
App(LiveKit Client SDK,内置 AEC)
  │ WebRTC / Opus
LiveKit SFU(公网 VPS,唯一外部依赖)
  │ worker 主动外连注册
┌────────────── 本项目 @ SM120 96GB ──────────────┐
│ L4 app        事件存储 / 播报仲裁 / Router / 任务 │
│ L3 duplex     epoch 取消 / 状态机 / EOU 基准      │
│ L2 pipeline   Thinker → Talker → Code2Wav 编排    │
│ L1 engine     KV cache / 调度 / graph / prefix    │
│ L0 kernel     sparkinfer(SM120 CuTe DSL)        │
└──────────────────────────────────────────────────┘
  │ HTTP 异步,不阻塞语音
外部大模型(任务层)
```

## 双时钟不变量(L4 与实时平面的隔离)

1. 会话主循环不 await 外部模型 / HTTP callback / 长事务。
2. task enqueue 落盘即返回,网络发送由独立 worker 负责。
3. task result 只进通知队列,播不播由仲裁器决定。
4. task worker 崩溃不影响实时媒体;实时模型崩溃不丢任务。
5. GPU 只属于实时平面。

## L2 九件事(编排骨架见 `channellm/pipeline/orchestrator.py`)

| # | 机制 | 参考(vllm-omni) |
|---|---|---|
| 1 | 三引擎身份对齐 | `orchestrator.py: OrchestratorRequestState` |
| 2 | 增量提交 | `_forward_to_next_stage` |
| 3 | 下游 resumable | `build_engine_core_request_from_tokens(resumable=)` |
| 4 | 攒够单元再转发 | `if not next_inputs: return` |
| 5 | 下游预热 | `_prewarm_async_chunk_stages` |
| 6 | 跨阶段 tokenizer | `streaming.source_token_decoder` |
| 7 | 错误三边清理 | `_cleanup_request_ids` |
| 8 | 终止输出合成 | `_build_terminal_empty_output` |
| 9 | 打断四处齐停 | `experimental/fullduplex/core/runtime.py` |

## epoch 端到端取消(L3)

所有 token/codec/audio chunk 携带 `(turn_epoch, speech_id)`;旧 epoch 无条件丢弃。
四个独立状态域:Input / Reply / Notification / Task。
实现面:`channellm/duplex/epoch.py`(判定),`pipeline/orchestrator.py:cancel`(执行)。
