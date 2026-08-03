# P0 Runbook —— 串行基线 waterfall

目标:拿到官方单进程路径的真实 `EOU → 首个 PCM` 分段数据。这个数字决定
P2(三阶段编排)的优先级和投入 —— 它就是流水线相对串行的真实收益。

## 0. 环境

```bash
./scripts/setup_env.sh
python scripts/preflight.py --gpu   # 全 PASS 才继续
```

## 1. 准备回放集

按 `data/audio_set/README.md` 录/集四类语料(16kHz 单声道,尾部留 ≥1s 静音),
填 `data/audio_set/manifest.yaml`,**人工标注 `eou_offset_s`**(说完时刻)。

## 2. 跑串行基线

```bash
python scripts/p0_run_official_duplex.py \
    --manifest data/audio_set/manifest.yaml \
    --out traces/p0_serial_$(date +%Y%m%d).jsonl \
    --artifact-dir artifacts/p0/
```

产出:trace JSONL + 每条回复的 wav(人工耳检质量)。
首次运行 = cold;重复跑同 manifest 得 warm 样本。

## 3. 聚合 waterfall

```bash
python scripts/p0_waterfall.py traces/p0_serial_*.jsonl \
    --group-by temp,loc,category --report artifacts/p0/waterfall.md
```

## 4. 验收(设计文档 §P0)

- [ ] 官方 `MiniCPMODuplex` 单进程跑通(SM120 依赖已解)
- [ ] latency trace schema 覆盖 §7 全部段
- [ ] 固定音频回放集四类就位
- [ ] **拿到 EOU → 首个 PCM 的真实分段数据(p50/p95/p99,cold/warm 分开)**

## 已知陷阱

- **transformers 版本冲突:** 官方 modeling 代码按 transformers==4.51.0 开发
  (MiniCPM-o-Demo requirements);本仓库主契约是 >=5(BlackweLLM 谱系)。
  若官方代码在 transformers 5 下加载失败:为 P0 单独建 venv 钉 4.51.0
  (R6 逐个解),并在本文档记录结论;不要在主 venv 里降级。
- attn 默认 `sdpa`;`flash_attention_2` 在 SM120 需单独构建验证。
- 官方 `streaming_generate` 自带 `cost_llm/cost_tts/cost_token2wav` 分段耗时,
  已随锚点落库 —— 与本仓库锚点交叉验证,不要互相覆盖结论。
- Xet 仓库下载禁用 hf-mirror(见 `docs/environment.md`)。
