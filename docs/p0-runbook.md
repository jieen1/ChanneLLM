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

串行回放以机器速度跑,锚点 ts 差不是产品延迟。诚实口径按 chunk 语义重建:

```bash
# 诚实串行口径(P0 验收用它):流等待 + prefill + 决策算力 + 尾部等待
python scripts/p0_serial_report.py traces/p0_serial_*.jsonl \
    --out artifacts/p0/waterfall.md

# 锚点对口径:串行回放下是假象(0.1ms),仅 P2+ 实时 trace 才有效
python scripts/p0_waterfall.py traces/p0_serial_*.jsonl \
    --group-by temp,loc,category
```

## 4. 验收(设计文档 §P0)

- [x] 官方 `MiniCPMODuplex` 单进程跑通(SM120 依赖已解;transformers 5.14.1
      下经六垫片兼容,未降级主栈)
- [x] latency trace schema 覆盖 §7 全部段
- [x] 固定音频回放集三类就位(manifest 三条;第四类 barge-in 属 P3 面,四类语料
      待扩)
- [x] **拿到 EOU → 首个 PCM 的真实分段数据**(诚实串行口径:0.98/1.49/1.74s
      min/median/max,n=3;样本不足,p50 有效、p95/p99 待扩样本;cold/warm 已在
      prefill/token2wav 分段拆开)

## 已知陷阱

- **transformers 版本冲突(已解决,2026-08-03):** 官方 modeling 代码按
  transformers==4.51.0 开发;本仓库主契约 >=5,**不降级**。实际在 5.14.1 下经
  `channellm/models/minicpmo_compat.py` 六个幂等垫片跑通(细节见
  `docs/environment.md` 实测事实)。动态模块必须经
  `config.auto_map["AutoModel"]` + `get_class_from_dynamic_module` 加载,
  不能 sys.path 直接 import(相对导入)。minicpmo-utils 用 `--no-deps` 装,
  其 transformers<4.53 元数据钉子不继承。
- attn 默认 `sdpa`;`flash_attention_2` 在 SM120 需单独构建验证。
- 官方 `streaming_generate` 自带 `cost_llm/cost_tts/cost_token2wav` 分段耗时,
  已随锚点落库 —— 与本仓库锚点交叉验证,不要互相覆盖结论。
- Xet 仓库下载禁用 hf-mirror(见 `docs/environment.md`)。
