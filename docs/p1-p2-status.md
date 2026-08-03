# P1–P2 阶段状态:自研引擎的听-想-说闭环(2026-08-04)

## 一句话

MiniCPM-o 4.5 的全双工语音闭环(音频流入 → listen/speak 决策 → 回复语音)
已在**自研单进程 runtime**按 unit 增量跑通：Thinker 每次开口即续写 Talker
KV，25 帧 phrase 经有界进程内队列送入 Code2Wav，再由 epoch-guarded L3
runtime 发布本地 PCM，并经可取消待播缓冲交给本地媒体 writer。当前证据仅覆盖
GPU→本地缓冲取出；LiveKit、AEC、物理设备播放及统计意义上的 SLO 仍未完成。

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
| L3 控制 | `duplex/runtime.py` + `duplex/driver.py` + `duplex/queued_runtime.py` + `duplex/playback.py` | ✅ 真实三阶段驱动、单 GPU 有界 worker、cancel-not-await、待播 PCM 静音、writer-handoff 播放生命周期 |
| L3 EOU 基准 | `duplex/eou_baseline.py` | ✅ 可注入 SoulX 官方状态流；⚠️ 当前 venv 未混入 SoulX 依赖，待独立官方服务/引擎部署实测 |

## 验证证据

1. **Thinker 结构对齐**:`scripts/p1_thinker_parity.py --fp32` 64/64 token
   与官方 Qwen3 逐 token 一致(logits max|Δ|=4e-5);399/399 权重逐位一致;
   bf16 分歧为 ULP 舍入累积(层间 cos≥0.9997),非结构缺陷。
2. **语音输出**:`artifacts/p1/voice_loop_reply.wav` 文本→15.9s 连续语音。
3. **语音输入**:`scripts/p1_audio_in.py` 流式音频→自研 Thinker→准确复述
   fixture 内容(植物大战僵尸)。
4. **真实三阶段本地回放**:`scripts/p1_duplex_loop.py` fixture 回放产生
   `EOU → SPEAK_DECISION → TALKER_CHUNK_READY → CODE2WAV_FIRST_PCM → PUBLISHED
   → DEVICE_PLAYOUT_START` 完整 trace。`--repeat 5` 现以一次模型加载后的首轮为
   cold、其余为 warm，并为每轮创建独立 artifact，避免既有 JSONL 污染分位数。
   单轮指定的 trace 路径也会由该脚本覆盖写入；P0 等需要累积样本的 recorder
   调用仍显式保持追加语义。
   最新一批的五条 24kHz WAV 均未触发非有限值/削波/直流偏置/采样突变门禁，RMS 为
   0.0741–0.1517、峰值为 0.4089–0.9900、最大相邻采样步长为 0.12866–0.59177。
   第五条有 6 个 sample ≥0.98（但没有 sample ≥0.999），因接近满幅而必须保留
   该 artifact 做人工试听。L1 Code2Wav 现在还会在 PCM 进入媒体层前拒绝非 24kHz
   WAV bytes 与非有限 sample；门禁只证明信号完整性，不证明可懂度或主观自然度。
   随后的质量优先五轮复测（相同驻留模型进程）五条均通过硬门禁和人工复核门槛：
   RMS 0.0621–0.1275、peak 0.3801–0.8851、最大相邻采样步长 0.18384–0.34772，
   削波比均为零；此前的 near-full-scale artifact 仍保留，不能被这批结果抹除。
5. **epoch 与队列回归**:旧 Thinker/Talker 输出在进入下游前被拒绝；新 epoch
   会清除两个 interstage 队列中的旧 tag，防止旧任务继续占用 GPU。
6. **待播残音回归**:即使模型请求已完成，新输入也会清空尚未由媒体 writer
   取走的旧 PCM；只有 writer 明确报告播放结束后，下一输入才不会触发 mute。
   `AgentSpeechActuallyPlayed` 也只在 writer 首次取走 PCM 时写入 SQLite，绝不把
   “仅发布到待播缓冲、随后被 barge-in 丢弃”的音频伪装成已播放事实。
7. **输入线程不等待旧模型**:新的 `QueuedDuplexRuntime` 把 GPU 模型调用移入单
   worker 有界队列；新输入会先推进 epoch/cancel/mute，已在运行的旧调用可自然返回，
   其输出因旧 tag 被 runtime 拒绝。测试以阻塞的旧模型调用验证新回合控制面在 50ms
   内返回，并验证未启动旧音频和队满时最旧输入均不进入模型。真实权重
   `p1_duplex_loop.py --queued-runtime` 也已产生完整 PCM/trace 并通过信号门禁；
   该脚本以机器速度灌入输入，故它的 2439.3ms EOU→首 PCM 只证明队列正确性，
   不可作为实时 SLO。
8. **可审计的首回复分段**:说话 chunk 现在写入模型内部单调时钟采集的
   `STREAMING_PREFILL_START → THINKER_PREFILL_DONE → FIRST_TOKEN_DECODED`，并且
   只在实际进入说话分支时落锚，避免把先前的 listen chunk 错配进回复延迟。
   最新真实权重 queued 回放的该段为 174.2ms prefill、3.1ms 首 token、
   616.1ms 首 token→首 Talker phrase、304.5ms 首 Code2Wav；输出为 24kHz、
   RMS 0.0788、peak 0.3842，无信号完整性失败或人工复核警告。该 run 同样是
   机器速度入队的功能/测量链路验证，不是实时 SLO 或主观音质结论。
9. **独立 EOU 适配**:SoulX-Duplug 只把官方 `state == "speak"` 映射为独立 EOU
   观测，且官方状态流未提供置信度时保留为未知；该观测不进入 MiniCPM-o 的说话
   决策。当前项目 venv 缺少 SoulX 官方推理依赖，因而仅验证注入契约，未伪称模型
   已共驻运行。
10. **媒体 handoff 边界**:`PcmPlayoutPump` 逐帧把有界 PCM 缓冲交给注入的
    LiveKit/声卡 writer，并在首个 handoff 与生成终止后的缓冲排空处驱动播放
    生命周期。打断发生在 handoff 前时，旧 PCM 会被同步清空且 pump 不会写出它。
    真实权重回放通过该路径产生 24kHz WAV（RMS 0.0623、peak 0.4304），未触发
    信号失败或人工复核；它是 writer 边界，不是物理 DAC 或远端客户端证据。
11. **可恢复会话事实**:首次 writer handoff 才写 `AgentSpeechActuallyPlayed`；如果
    首帧先于最终文本生成，最终文本以 append-only supersede 修订该事实。重启恢复只
    读取当前 session epoch 的有效 `ContextSnapshot` 与未收到 `TaskResultReady` 的
    任务，绝不恢复未播放 PCM 或 `AgentSpeechPlanned`。这完成本地 crash-recovery
    的事实边界，不等价于恢复 GPU KV、设备缓冲或远端媒体会话。

## 实时预算(质量优先五轮同进程本地 fixture 回放，非 SLO)

| 段 | cold n=1 p50 | warm n=4 p50/p95/p99 | 说明 |
|---|---|---|---|
| EOU → speak decision | 490.7ms | 237.0 / 324.8 / 324.8ms | ⚠️ 真实 Thinker 决策锚点 |
| speak decision → 首 PCM | 1557.5ms | 1255.0 / 2001.0 / 2001.0ms | ⚠️ 含 Talker 首块与首段 Code2Wav |
| EOU → 首 PCM | 2048.2ms | 1492.0 / 2325.8 / 2325.8ms | ⚠️ 未达 1s，且仅本地 PCM |
| Code2Wav 首块 | 264.9ms | 115.9 / 169.8 / 169.8ms | ⚠️ 不等于端到端客户体验 |

`speak decision`、`Thinker prefill`、首 Thinker token 与 `talker chunk ready` 已分离
记录。该批次的 cold/warm 标签仅指
模型已经驻留后的第 1/后续实际推理轮次，**不包含权重加载**；每组样本远不足以形成
统计意义的 SLO，且共享 GPU 调度会造成明显波动。它是按强制格式报告的可审计批次，
不是性能承诺。

## decode 性能优化(本阶段成果)

- **定位**:隔离剖析显示 KV 后端占 decode 大头,其中 attend(bind+run)87%。
  微基准(GPU 空闲窗口):bind 2.72ms/次、run 仅 0.63ms/次 —— 36 层逐层
  重 bind ≈ 117ms/token 的纯适配层开销。
- **已实施**:
  1. `sparkinfer_attn` 静态缓冲 + 按 (mode,形状,层) 缓存 binding,
     bind 成本在热路径摊薄为零(run 0.63ms/层);
  2. `paged_kv.slot_for` 每步一次算槽,36 层 append 复用索引;
  3. `SparkinferPagedKV` 在 `begin_step` 提升 slot，并在一个 token 的
     全部层间复用同一份页表、cache length 与 cumulative-q metadata；commit 后
     无条件释放，跨页时重新生成。
- **Talker 首块**:
  1. 首个 codec phrase 按官方 S3 流式口径插入 3 个 `4218` 静音前瞻码，令
     25 个新帧立刻形成首个 28 帧 Code2Wav 窗口；空回复不会合成伪静音；
  2. Talker 默认改用连续预分配 KV + Torch SDPA，消除每层、每 token 的
     `torch.cat`。真实权重连续 12 个贪心 codec token 与参考 `TorchListKV`
     完全一致；隔离 decode 微基准为 28.24→9.33ms/token。sparkinfer paged
     内核不支持 Talker 的 12 heads / head_dim 64，故未强行接入。
- **受控 graph 原型验证**:必须在真实 prefill 前 capture，随后释放 warmup/capture
  的 dummy KV 页并同步静态页表；否则第二个 token 起会与 eager 分歧。在该生命周期
  下，真实权重的连续 8 个贪心 token 与 eager 完全相同；隔离基准为 eager
  37.29ms/token（26.82 tok/s）、graph 19.39ms/token（51.58 tok/s）。该原型尚未
  进入受控生产热路径，不能据此声称端到端提速或扩大质量承诺。
- **测量说明**:全模型对比数字受共享 GPU 上不可见并发负载污染
  (nvidia-smi 可见 60%+ 利用率但无进程归属),最终全模型数字待
  GPU 空闲窗口复测;微基准数字为 GPU 空闲时测得,可信。

## 下一步

1. **P3 实测打断**:接入真实输入/物理播放适配器，记录 barge-in→静音 trace，
   验证已在 GPU 中的旧请求不会泄漏到媒体端；
2. **性能批测**:区分 cold/warm、本地/远端，采集足够 trace 后报告 p50/p95/p99；
3. **CUDA graph 捕获 decode 步**(sparkinfer decode 原生支持 graph replay
   + on-device metadata 重建)；
4. **P5 媒体接入**:LiveKit/AEC/设备播放，补齐真实客户端扬声器口径与主观试听。
