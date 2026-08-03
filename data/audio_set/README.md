# P0 固定回放集

串行基线 waterfall 的输入(设计文档 §P0)。一律 **16kHz 单声道 wav**;
禁止 8kHz 窄带源(恶化中文擦音/塞擦音)。

## 必备类别

| 类别 | 目的 |
|---|---|
| `pause-think` | 停顿思考不被抢话:句中 1–2s 停顿后继续 |
| `backchannel` | 附和语料(嗯/对/然后呢),测误触发 |
| `barge-in` | 打断:模型开口后用户插入 |
| `names-numbers` | 人名/数字/专名(R11 低置信确认) |

## manifest.yaml 条目字段

- `path`:相对本目录的 wav 路径
- `category`:上表之一
- `eou_offset_s`:**人工标注的用户说完时刻(秒)**,EOU 权威口径;缺省时
  脚本退回模型 is_listen 翻转近似(打 `eou_source=approx` 标签)
- `transcript`:文本真值(后续 ASR 修订/对齐用)

录音时保留 ≥1s 尾部静音,便于 EOU 标注与播放判定。
