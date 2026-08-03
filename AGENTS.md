# ChanneLLM 项目工作契约

本仓库是自研全双工语音 runtime。任何 agent 在本仓库工作必须遵守以下契约。

## 证据纪律(最高优先级)

沿用设计文档 §1 口径:**[事实]**(官方源码/文档直接支持或本机复现)、
**[推断]**(需实测)、**[未知]**(不写进承诺)。
- 延迟数字一律来自实测 trace,禁止 nominal 相加(R2)。
- 报告 p50/p95/p99,local/remote、cold/warm 分开;禁止只报均值。
- 改动 `channellm/models/minicpmo.py` 的 ModelFacts 前必须重新对照
  权重内 `modeling_minicpmo.py` 验证。

## 架构红线

- **不依赖 vLLM / vllm-omni 运行时代码**;它们与 MiniCPM-o 官方代码仅作参考
  (路径见 `docs/references.md`)。
- 唯一说话权 = MiniCPM-o duplex;SoulX-Duplug 只做 EOU 基准与后备,不参与决策。
- 三阶段传输用进程内队列;不引入分布式抽象(单进程单卡)。
- 事件日志(SQLite WAL 单写者)是 L4 唯一事实源;transcript/markdown 只是投影。
- 旧 epoch 产物无条件丢弃;barge-in 先 cancel 不 await。

## 分层与写范围

L0 `channellm/kernel/` · L1 `channellm/engine/` · L2 `channellm/pipeline/` ·
L3 `channellm/duplex/` · L4 `channellm/app/` · 横切 `tracing/metrics/audio` ·
脚本 `scripts/` · 测试 `tests/`。跨层改动需在 PR/commit 说明理由。

## 环境

- 独立干净 venv(R6):`./scripts/setup_env.sh`,不要复用其它项目环境。
- 版本契约在 `pyproject.toml`(torch==2.13.0 钉版;sparkinfer 从 fork editable)。
- sparkinfer 是共同深度优化的内核面:fork 在 `~/project/sparkinfer`
  (origin=jieen1/sparkinfer),本仓库 `third_party/sparkinfer` submodule 钉版。
- HF 下载:Xet 仓库禁用 `HF_ENDPOINT=hf-mirror`(308 丢头),直连即可。

## 验证命令

```bash
.venv/bin/python -m pytest -q     # 必须全绿
.venv/bin/python -m ruff check .  # 必须干净
python scripts/preflight.py       # 环境检查
```

GPU 相关改动:先 `preflight.py --gpu`,再跑受影响的 P0 脚本对照 trace。

## 提交

遵循 Lore commit protocol(见 `docs/conventions.md`:意图行在前,trailer
记录约束/拒绝/置信度/验证缺口)。小步提交,每步验证。
