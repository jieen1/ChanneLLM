# ChanneLLM

单张 NVIDIA SM120(Blackwell 96GB)上的**自研全双工语音 runtime**。
模型:MiniCPM-o 4.5(端到端全双工)。内核:sparkinfer(SM120 CuTe DSL,深度优化 fork)。

> 随时可以说话的思考伙伴:实时接话、完整记录、异步派活。
> 设计全文见 [`docs/design/voice-runtime-design-v3.md`](docs/design/voice-runtime-design-v3.md)。

## 为什么自研

- 中文场景没有"又端到端又低延迟"的开源选项(Moshi/PersonaPlex 均纯英语)。
- 不依赖 vLLM / vllm-omni —— 仅作参考实现;延迟深度优化必须握在自己手里。
- SM120(消费级 Blackwell)有专门的 kernel 面(sparkinfer),值得吃透。

## 分层

| 层 | 目录 | 职责 |
|---|---|---|
| L0 kernel | `channellm/kernel/` | sparkinfer 适配(attention.paged/varlen、FP8 KV、CUDA graph) |
| L1 engine | `channellm/engine/` | KV cache、连续批处理、prefix cache、CUDA graph |
| L2 orchestration | `channellm/pipeline/` | 三阶段编排 Thinker→Talker→Code2Wav(九件事) |
| L3 duplex | `channellm/duplex/` | epoch 端到端取消、会话状态机、独立 EOU 基准 |
| L4 app | `channellm/app/` | 事件存储(SQLite WAL)、播报仲裁、多标签 Router、任务派发 |
| 横切 | `channellm/tracing/` `metrics/` `audio/` | 延迟 trace、p50/p95/p99、16kHz 分块 |

## 快速开始

```bash
./scripts/setup_env.sh            # uv venv + 依赖 + sparkinfer fork + preflight
source .venv/bin/activate

python scripts/preflight.py       # 基础环境检查
python scripts/preflight.py --gpu # GPU 全量检查(torch/cuda/sm120/sparkinfer)

# P0:官方 duplex 串行基线回放(需 GPU + 权重)
python scripts/p0_run_official_duplex.py --manifest data/audio_set/manifest.yaml
python scripts/p0_waterfall.py traces/*.jsonl --report artifacts/waterfall.md

make test && make lint            # 日常验证
```

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 官方单进程跑通 + 串行基线 waterfall | **进行中** |
| P1 | 推理内核(三子模型加载、paged attention、调度、CUDA graph、prefix cache) | 未开始 |
| P2 | 三阶段编排(九件事) | 未开始 |
| P3 | 双工会话控制(epoch 四处齐停) | 未开始 |
| P4 | 应用层(事件存储/仲裁/Router/任务) | 未开始 |
| P5 | LiveKit 远程 + iOS 真机 AEC 矩阵 | 未开始 |
| P6 | 性能长尾 | 持续 |

## 参考实现(不作为运行时依赖)

均在 `~/project/`:`vllm-omni`(三阶段编排参考)、`MiniCPM-o`(官方仓库)、
`MiniCPM-o-Demo`(WebRTC demo 参考)、`SoulX-Duplug`(独立 EOU 基准)、
`sparkinfer`(内核 fork,`third_party/` submodule 钉版)。
详见 [`docs/references.md`](docs/references.md)。
