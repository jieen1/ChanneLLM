# 参考实现地图

Apache 2.0,仅参考,**不作为运行时依赖**。本地路径均在 `~/project/`。

| 要实现的 | 参考位置 |
|---|---|
| 模型前向 / 双工接口 | 权重内 `modeling_minicpmo.py`(`MiniCPMODuplex:2438`、`streaming_prefill:2777`、`streaming_generate:3151`) |
| 输入处理 / 音频分块 | 权重内 `processing_minicpmo.py`、`utils.py` |
| 三阶段编排 | `~/project/vllm-omni/vllm_omni/engine/orchestrator.py` |
| stage 启动与配置 | `~/project/vllm-omni/vllm_omni/engine/stage_engine_startup.py` |
| 跨阶段连接器 | `~/project/vllm-omni/vllm_omni/worker/omni_connector_model_runner_mixin.py` |
| 拓扑与阶段定义 | `~/project/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py` |
| 三阶段模型实现 | `~/project/vllm-omni/vllm_omni/model_executor/models/minicpmo_4_5/` |
| 双工会话控制 | `~/project/vllm-omni/vllm_omni/experimental/fullduplex/minicpmo45/`(stage0/data_plane/adapter;注意上游在快速演进,R5) |
| epoch 取消最小范式 | `~/project/vllm-omni/vllm_omni/experimental/fullduplex/core/runtime.py`(cancel-not-await 注释) |
| 显存/chunk 参数基线 | `~/project/vllm-omni/vllm_omni/deploy/minicpmo_4_5*.yaml` |
| WebRTC / 媒体面参考 | `~/project/MiniCPM-o-Demo`(gateway/worker/vad) |
| 独立 EOU 基准 | `~/project/SoulX-Duplug`(Soul-AILab) |
| 官方仓库 | `~/project/MiniCPM-o`(OpenBMB) |
| SM120 内核 | `~/project/sparkinfer`(jieen1/sparkinfer fork;本仓库 `third_party/sparkinfer` 钉版) |

**sparkinfer 双轨关系:** `third_party/sparkinfer` submodule 钉 fork 远程的
已推送状态(可复现口径);日常开发装的是 editable 的 `~/project/sparkinfer`
(`scripts/setup_env.sh` 默认),即 fork 的最新本地状态。两者分叉时运行时以
editable 为准,验证通过后推送 fork 并更新 submodule 钉版。
| SM120 环境经验 | `~/project/qwen-sm120-runtime`(BlackweLLM;无代码复用,仅环境契约) |

**明确不参考**:PD 分离、CFG companion、collective RPC、分布式 KV transfer、
TP/PP、diffusion、其余模型。

## 链接

MiniCPM-o <https://github.com/OpenBMB/MiniCPM-o> ·
MiniCPM-o-Demo <https://github.com/OpenBMB/MiniCPM-o-Demo> ·
vllm-omni <https://github.com/vllm-project/vllm-omni> ·
SoulX-Duplug <https://github.com/Soul-AILab/SoulX-Duplug> ·
sparkinfer <https://github.com/local-inference-lab/sparkinfer> ·
LiveKit <https://docs.livekit.io/agents/> ·
TELEVAL <https://github.com/Tele-AI/TELEVAL> ·
Full-Duplex-Bench <https://github.com/DanielLin94144/Full-Duplex-Bench>
