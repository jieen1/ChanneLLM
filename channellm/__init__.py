"""ChanneLLM — 单卡 SM120 自研全双工语音 runtime。

分层(设计文档 voice-runtime-design-v3.md §3):
- L0 kernel        channellm.kernel     sparkinfer SM120 内核适配
- L1 engine        channellm.engine     KV cache / 调度 / CUDA graph / prefix cache
- L2 orchestration channellm.pipeline   三阶段编排(Thinker→Talker→Code2Wav)
- L3 duplex        channellm.duplex     双工会话控制 / epoch 取消 / EOU
- L4 app           channellm.app        事件存储 / 播报仲裁 / Router / 任务派发

测量与音频基础设施(横切,P0 先行):
- channellm.tracing  延迟 trace schema 与 JSONL 记录器
- channellm.metrics  p50/p95/p99 分段统计与 waterfall
- channellm.audio    16kHz 分块 / PCM 工具
"""

__version__ = "0.1.0"
