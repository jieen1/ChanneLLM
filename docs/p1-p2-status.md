# P1–P2 阶段状态:自研引擎的听-想-说闭环(2026-08-04)

## 一句话

MiniCPM-o 4.5 的全双工语音闭环(音频流入 → listen/speak 决策 → 回复语音)
已在**自研单进程 runtime**按 unit 增量跑通：Thinker 每次开口即续写 Talker
KV，25 帧 phrase 经有界进程内队列送入 Code2Wav，再由 epoch-guarded L3
runtime 发布本地 PCM。当前证据仅覆盖 GPU→本地 PCM；LiveKit、AEC、设备播放
及统计意义上的 SLO 仍未完成。

## 已建成的组件

| 层 | 组件 | 状态 |
|---|---|---|
| L0 内核 | `kernel/paged_kv.py` 页池/分配器/slot 复用 | ✅ CPU+GPU 测试 |
| L0 内核 | `kernel/sparkinfer_attn.py` plan/bind/run + **binding 缓存** | ✅ GPU parity 对齐 in-tree reference |
| L1 引擎 | `engine/thinker.py` 自研 Thinker(Qwen3 骨干,forward/forward_embeds) | ✅ fp32 逐 token 对齐官方 |
| L1 引擎 | `engine/talker.py` 自研 Talker(Llama 骨干,hidden_text_merge) | ✅ unit 级 KV 续写 + 25 帧 phrase |
| L1 引擎 | `engine/code2wav.py` Token2wav 封装 + StreamingSynth 分块流式 | ✅ 批量+流式双路径 |
| L1 引擎 | `engine/audio_front.py` 官方流式 whisper 编码器混合封装 | ✅ 音频理解验证 |
| L2 编排 | `pipeline/orchestrator.py` + `pipeline/transport.py` | ✅ 增量路由、有界队列、旧 epoch 清理 |
| L3 控制 | `duplex/runtime.py` + `duplex/driver.py` | ✅ 真实三阶段驱动、cancel-not-await、仅发布当前 epoch PCM |

## 验证证据

1. **Thinker 结构对齐**:`scripts/p1_thinker_parity.py --fp32` 64/64 token
   与官方 Qwen3 逐 token 一致(logits max|Δ|=4e-5);399/399 权重逐位一致;
   bf16 分歧为 ULP 舍入累积(层间 cos≥0.9997),非结构缺陷。
2. **语音输出**:`artifacts/p1/voice_loop_reply.wav` 文本→15.9s 连续语音。
3. **语音输入**:`scripts/p1_audio_in.py` 流式音频→自研 Thinker→准确复述
   fixture 内容(植物大战僵尸)。
4. **真实三阶段本地回放**:`scripts/p1_duplex_loop.py` fixture 回放产生
   `EOU → SPEAK_DECISION → TALKER_CHUNK_READY → CODE2WAV_FIRST_PCM → PUBLISHED`
   完整 trace。最新一次 GPU 运行输出 24kHz / 2.000s，RMS 0.092007、峰值
   0.678711、零削波；该门禁只证明信号完整性，不证明可懂度或主观自然度。
5. **epoch 与队列回归**:旧 Thinker/Talker 输出在进入下游前被拒绝；新 epoch
   会清除两个 interstage 队列中的旧 tag，防止旧任务继续占用 GPU。

## 实时预算(单个本地 fixture 回放，非 SLO)

| 段 | 耗时 | 1s 预算 |
|---|---|---|
| EOU → speak decision | 2845.7ms | ⚠️ 单样本，含仍待优化的 Thinker 决策 |
| speak decision → 首 PCM | 294.5ms | ✅ 单样本本地 GPU 路径 |
| EOU → 首 PCM | 3140.2ms | ⚠️ 仅本地 PCM，不含网络/设备播放 |

这些是 trace 实测，不可与其它轮次拼接，也不能代替 cold/warm、p50/p95/p99 报告。

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

1. **P3 实测打断**:接入真实输入/播放适配器，记录 barge-in→静音 trace，验证
   已在 GPU 中的旧请求不会泄漏到媒体端；
2. **性能批测**:区分 cold/warm、本地/远端，采集足够 trace 后报告 p50/p95/p99；
3. **CUDA graph 捕获 decode 步**(sparkinfer decode 原生支持 graph replay
   + on-device metadata 重建)；
4. **P5 媒体接入**:LiveKit/AEC/设备播放，补齐真实客户端扬声器口径与主观试听。
