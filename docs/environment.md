# SM120 环境契约

目标机:NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition,96GB,
driver 610.47,SM120(CC 12.0)。

## 版本钉版(pyproject.toml `[project.optional-dependencies].cuda`)

| 包 | 版本 | 说明 |
|---|---|---|
| torch | ==2.13.0 | 与 BlackweLLM(qwen-sm120-runtime)同一契约;参考环境跑自编译 2.13.0a0+cu13.3,PyPI wheel 亦满足 |
| triton | >=3.6,<4 | 验证过 3.6.0 |
| transformers | >=5,<6 | 实测 5.14.1;官方 P0 路径经 `channellm/models/minicpmo_compat.py` 六个垫片兼容,不降级 |
| huggingface_hub | >=1,<2 | 验证过 1.26.0 |
| safetensors | >=0.7,<1 | 验证过 0.8.0 |
| sparkinfer | editable fork | torch>=2.12、nvidia-cutlass-dsl==4.6.0(JIT,无构建步骤) |
| minicpmo-utils[all] | >=1.0.5,--no-deps | stepaudio2 Token2wav vocoder;元数据钉 transformers<4.53/librosa==0.9.0 是保守钉子,`--no-deps` 绕过并手动补真实依赖 |
| setuptools | >=75,<81 | 81+ 移除了 pkg_resources,stepaudio2/librosa 仍依赖它(P1 换自研路径后可解除) |
| torchaudio | 随 torch 2.13(2.11.0) | 2.11 默认 torchcodec 后端需系统 ffmpeg;本机无 ffmpeg,`patch_torchaudio_load` 对 wav 走 soundfile 兜底 |

装不上的钉版逐个解(R6):独立干净 venv,不混用其它项目环境。

## 版本策略(不降级原则)

- **主栈永远用最新已验证版本**:torch/transformers/numpy/triton 等核心件跟随
  上游主线,新版本带来的性能与硬件支持收益(如 Blackwell kernel 改进)属于本项目。
- **参考项目的版本钉子一律不继承**:minicpmo-utils 元数据钉 transformers<4.53、
  librosa==0.9.0,MiniCPM-o-Demo 钉 transformers==4.51.0 —— 全部用 `--no-deps` +
  兼容垫片(`channellm/models/minicpmo_compat.py`)绕过,而不是把主栈拖回旧版本。
- **升级走验证制**:任何版本变更必须在本机(SM120)跑通 preflight + P0 回放才入库;
  「最新」不是信仰,验证过的最新才是契约。
- 唯一允许的旧版本钉子,是官方权重内嵌 legacy 代码强依赖的面(pkg_resources),
  且随 P1 自研内核替换官方路径后逐个解除。

## 搭建

```bash
./scripts/setup_env.sh          # 全量
./scripts/setup_env.sh --base   # 仅 CPU 可测试面(tracing/事件存储等)
SPARKINFER_PATH=~/project/sparkinfer ./scripts/setup_env.sh
```

## 权重

| 资产 | HF id | 状态 |
|---|---|---|
| MiniCPM-o 4.5(20.05GB) | `openbmb/MiniCPM-o-4_5` | 已下载校验 54/54 |
| SoulX-Duplug 0.6B(7.78GB) | `Soul-AILab/SoulX-Duplug-0.6B` | 已下载校验 24/24 |

**下载注意:** `HF_ENDPOINT=https://hf-mirror.com` 对 Xet 后端仓库不可用
(308 跳回源站且丢失 `x-linked-etag`/`x-linked-size`,报 LocalEntryNotFoundError);
需 `env -u HF_ENDPOINT` 直连(实测 23 MB/s)。

## 显存口径(设计文档 §8)

不使用权重加总。官方 GPU demo 要求 >28GB,初始化后约 21.5GB(运行口径)。
共驻测量矩阵 6 场景与门槛(峰值 <85GB;稳态留 ≥10GB 或 12%;warm 后
10 分钟增长 <1GB)在 P2 执行。

## 实测事实(P0,2026-08-03,run3)

- **transformers 5 兼容达成**:官方 duplex 在 5.14.1 下完整跑通,六垫片见
  `channellm/models/minicpmo_compat.py`(config 注入 / all_tied_weights_keys /
  DynamicCache 列表视图 / Whisper attention 2→3 元组 / EncoderDecoderCache
  legacy 下标 / torchaudio.load 兜底)。
- **显存**:权重加载后 17.57GB;`duplex.prepare`(含 token2wav 初始化)后 20.83GB,
  与设计「约 21.5GB」口径吻合。
- **延迟首批数**(详见 `artifacts/p0/waterfall.md`,n=3):
  - 串行 EOU→首包(实时等效):0.98 / 1.49 / 1.74s(min/median/max)
  - streaming_prefill warm(chunk≥1):p50 57.7ms,p95 83.1ms(n=25)
  - token2wav 后续调用:p50 125ms;首次调用含 vocoder 冷启动,最高 0.78s
  - 模型在「话音结束 + 同 chunk 静音」内即决定开口,无整 chunk 级额外等待
- **模型加载**:进程内首次 114.2s,同进程复载 49.8s(23GB RAM 机器必须走
  meta device 路径)。
