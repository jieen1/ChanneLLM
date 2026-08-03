# P0 串行回放 waterfall(诚实口径)

- 生成时间:2026-08-03 18:08 CST
- trace 来源:traces/p0_serial_run3.jsonl
- 环境:torch 2.13.0+cu130 · NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition · transformers 5.14.1
- 模型:MiniCPM-o 4.5 官方 duplex 路径 + ChanneLLM transformers 5 兼容垫片
- chunk 时长:1.0s(16kHz mono)

## 口径说明

回放以机器速度串行喂 chunk,`eou_detected` 与 `code2wav_first_pcm` 锚点 ts 差
只反映脚本内部先后,不是产品延迟。真实全双工系统里,音频按 1× 实时流入,
EOU→首包 = 静音对齐等待 + 决策 chunk 算力(+ 首包静音尾巴等待),
即下表 serial_total。

## 逐条明细

| category | eou_offset_s | decision_chunk | first_pcm_chunk | 流等待 | prefill | 决策算力 (llm+tts+t2w) | 尾部等待 | **serial_total** | 回放锚点差(留档) |
|---|---|---|---|---|---|---|---|---|---|
| names-numbers | 3.6 (chunk 3) | 3 | 3 | 0.400s | 0.085s | 1.254s (0.17+0.30+0.78) | 0.000s | **1.739s** | 0.1ms |
| pause-think | 11.2 (chunk 11) | 11 | 11 | 0.800s | 0.067s | 0.622s (0.22+0.23+0.17) | 0.000s | **1.488s** | 0.1ms |
| backchannel | 1.5 (chunk 1) | 1 | 1 | 0.500s | 0.052s | 0.424s (0.19+0.07+0.16) | 0.000s | **0.976s** | 0.1ms |

## 汇总

- **EOU→首包(serial_total)**:min 0.976 / median 1.488 / max 1.739 (n=3,样本不足,不报 p95/p99)
- 流等待(EOU→决策 chunk 流完):min 0.400 / median 0.500 / max 0.800 (n=3,样本不足,不报 p95/p99)
- 决策 chunk 算力(llm+tts+t2w):min 0.424 / median 0.622 / max 1.254 (n=3,样本不足,不报 p95/p99)
- streaming_prefill warm(chunk≥1):p50 57.701 / p95 83.113 / p99 87.095 (n=25) ms
- streaming_prefill cold(chunk 0):min 53.228 / median 83.303 / max 131.305 (n=3,样本不足,不报 p95/p99) ms
- token2wav 首次调用(含冷启动):min 0.162 / median 0.171 / max 0.781 (n=3,样本不足,不报 p95/p99)
- token2wav 后续调用:p50 0.125 / p95 0.271 / p99 0.354 (n=10)

## 解读

- **流等待是 chunk 粒度的税**:决策发生在 EOU 所在 chunk 内(模型反应快),
  但该 chunk 必须完整流完才能处理,贡献 0.4–0.8s。P2 编排若把决策 chunk 粒度
  减半,这部分直接减半。
- **决策算力 warm 约 0.4–0.6s**:token2wav 首次调用含 vocoder 冷启动(最高
  0.78s),会话内预热(如 prepare 时跑一次 ref audio)可消掉。
- **eou_offset_s 为机器估算**(末帧有声 +0.1s),正式报告前建议人工复核。
- **回放锚点差(亚毫秒)不是延迟**,仅作锚点链路完整性留档。
