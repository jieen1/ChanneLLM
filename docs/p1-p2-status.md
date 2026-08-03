# P1–P2 阶段状态:自研引擎的听-想-说闭环(2026-08-03)

## 一句话

MiniCPM-o 4.5 的全双工语音闭环(音频流入 → listen/speak 决策 → 回复语音)
已在**自研引擎**上端到端跑通并验证;sparkinfer fork 作为内核面接入,
decode 热路径的适配层开销已定位并消除,下一步是 CUDA graph 与实时传输。

## 已建成的组件

| 层 | 组件 | 状态 |
|---|---|---|
| L0 内核 | `kernel/paged_kv.py` 页池/分配器/slot 复用 | ✅ CPU+GPU 测试 |
| L0 内核 | `kernel/sparkinfer_attn.py` plan/bind/run + **binding 缓存** | ✅ GPU parity 对齐 in-tree reference |
| L1 引擎 | `engine/thinker.py` 自研 Thinker(Qwen3 骨干,forward/forward_embeds) | ✅ fp32 逐 token 对齐官方 |
| L1 引擎 | `engine/talker.py` 自研 Talker(Llama 骨干,hidden_text_merge) | ✅ 官方采样契约 |
| L1 引擎 | `engine/code2wav.py` Token2wav 封装 + StreamingSynth 分块流式 | ✅ 批量+流式双路径 |
| L1 引擎 | `engine/audio_front.py` 官方流式 whisper 编码器混合封装 | ✅ 音频理解验证 |
| L2 编排 | `engine/duplex_session.py` listen/speak 决策环 | ✅ 闭环验证 |

## 验证证据

1. **Thinker 结构对齐**:`scripts/p1_thinker_parity.py --fp32` 64/64 token
   与官方 Qwen3 逐 token 一致(logits max|Δ|=4e-5);399/399 权重逐位一致;
   bf16 分歧为 ULP 舍入累积(层间 cos≥0.9997),非结构缺陷。
2. **语音输出**:`artifacts/p1/voice_loop_reply.wav` 文本→15.9s 连续语音。
3. **语音输入**:`scripts/p1_audio_in.py` 流式音频→自研 Thinker→准确复述
   fixture 内容(植物大战僵尸)。
4. **全双工闭环**:`artifacts/p2/duplex_reply.wav` fixture 回放:
   LISTEN×3(用户说话中)→ EOU 当个 chunk 即 SPEAK → 回复语音
   (2.0–4.4s,随采样种子)。

## 实时预算(实测,fixture 回放)

| 段 | 耗时 | 1s 预算 |
|---|---|---|
| 音频编码 + chunk prefill | 300–470ms | ✅ |
| LISTEN 决策 | ~200ms | ✅ |
| SPEAK 决策(每 token) | ~200ms/tok | ❌ 待优化 |
| Talker codec 生成 | ~60–90 tok/s | ✅ |
| Code2Wav 流式首块 | 1157ms | ⚠️ 流式化已通,延迟随 turn 末触发 |

## decode 性能优化(本阶段成果)

- **定位**:隔离剖析显示 KV 后端占 decode 大头,其中 attend(bind+run)87%。
  微基准(GPU 空闲窗口):bind 2.72ms/次、run 仅 0.63ms/次 —— 36 层逐层
  重 bind ≈ 117ms/token 的纯适配层开销。
- **已实施**:
  1. `sparkinfer_attn` 静态缓冲 + 按 (mode,形状,层) 缓存 binding,
     bind 成本在热路径摊薄为零(run 0.63ms/层);
  2. `paged_kv.slot_for` 每步一次算槽,36 层 append 复用索引;
  3. SparkinferPagedKV 在 begin_step 提升 slot。
- **测量说明**:全模型对比数字受共享 GPU 上不可见并发负载污染
  (nvidia-smi 可见 60%+ 利用率但无进程归属),最终全模型数字待
  GPU 空闲窗口复测;微基准数字为 GPU 空闲时测得,可信。

## 下一步

1. **CUDA graph 捕获 decode 步**(sparkinfer decode 原生支持 graph replay
   + on-device metadata 重建)——目标 <30ms/tok;
2. **Talker 流式化**:官方 prefill_text 交错路径,边说边合成,
   首包延迟从"轮末"提前到"首个 EOT";
3. **实时会话循环**:chunk 节拍、barge-in/epoch 打断(P3 语义)、
   麦克风/扬声器或 LiveKit 传输(P5)。
