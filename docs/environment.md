# SM120 环境契约

目标机:NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition,96GB,
driver 610.47,SM120(CC 12.0)。

## 版本钉版(pyproject.toml `[project.optional-dependencies].cuda`)

| 包 | 版本 | 说明 |
|---|---|---|
| torch | ==2.13.0 | 与 BlackweLLM(qwen-sm120-runtime)同一契约;参考环境跑自编译 2.13.0a0+cu13.3,PyPI wheel 亦满足 |
| triton | >=3.6,<4 | 验证过 3.6.0 |
| transformers | >=4.52,<6 | minicpmo-utils 钉 <5,实际解析 4.52.4;官方 P0 路径为一等公民 |
| huggingface_hub | >=1,<2 | 验证过 1.13.0 |
| safetensors | >=0.7,<1 | 验证过 0.7.0 |
| sparkinfer | editable fork | torch>=2.12、nvidia-cutlass-dsl==4.6.0(JIT,无构建步骤) |
| minicpmo-utils[all] | >=1.0.5 | stepaudio2 Token2wav vocoder;会连带 torchaudio、降 tokenizers |
| setuptools | >=75,<81 | 81+ 移除了 pkg_resources,stepaudio2/librosa 仍依赖它 |

装不上的钉版逐个解(R6):独立干净 venv,不混用其它项目环境。

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
