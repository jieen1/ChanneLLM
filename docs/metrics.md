# SLO 指标与报告规则

源:设计文档 §2。延迟优先。

| 指标 | 定义 |
|---|---|
| `EOU_TO_FIRST_AUDIO` | 用户说完 → 客户端扬声器第一个 sample(产品级) |
| `EOU_TO_FIRST_PCM_LOCAL` | 用户说完 → 本机产出第一个 PCM sample(P0 口径) |
| `BARGE_IN_TO_SILENCE` | 用户开口 → 本地播放实际静音 |
| `ACK_AUDIO` | → backchannel 出声(**单独统计**) |
| `FIRST_MEANINGFUL_AUDIO` | → 有实质内容的音频 |

## 报告规则(强制)

- p50/p95/p99;**禁止只报均值**。
- local/remote 分开、cold/warm 分开(`channellm.metrics.latency.waterfall(group_by=...)`)。
- 禁止把各段 nominal 值相加当总延迟(R2);分段 waterfall 是唯一口径。

## 锚点与分段

锚点常量在 `channellm/tracing/schema.py:Anchor`,分段定义在同文件 `Segment`。
配对规则:同一 `(trace_id, turn_epoch)` 内第一个 start 锚点与之后第一个
end 锚点成样;end 早于 start 丢弃。

## 对话质量门槛(非延迟)

- 停顿思考不被抢话。
- 头脑风暴 10 分钟,主动插话 ≤3 次。
- backchannel 与真实状态一致(无任务时不得说"我去办")。
