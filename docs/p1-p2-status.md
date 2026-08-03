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

1. **Thinker 结构与精度边界**:`scripts/p1_thinker_parity.py --fp32 --tokens 48`
   在“请用一句话介绍杭州。”上与官方 Qwen3 48/48 token 逐 token 一致
   (logits max|Δ|=0.000067);399/399 权重已赋值。相同输入的 bf16
   `sparkinfer` 路径在第 10 token 分歧，bf16 `TorchListKV` 路径在第 7 token
   分歧（prefill max|Δ| 分别为 0.875/1.039）。因此这不是可忽略的质量误差：
   bf16 只能用于性能诊断，不能作为语义质量通过或线上默认路径。P1/P2 回归
   默认使用 fp32 + Torch SDPA 语义 KV，直至 bf16 长序列 parity 有新的实测证据。
2. **语音输出**:`artifacts/p1/voice_loop_reply.wav` 文本→15.9s 连续语音。
   最近的完整自研三引擎回放(`post-runtime-voice-loop.wav`，提示“请用一句话
   介绍杭州”)也生成 12.0s/24kHz PCM；RMS=0.08664、peak=0.66208、削波比例为零、
   DC=-0.00002、最大采样步长=0.30704，无完整性失败或复核警告。该样本证明
   Thinker→Talker→Code2Wav 共同加载和输出门禁，不证明语义与官方逐波形相同。
   最新 fp32 质量模式样本`quality-priority-fp32-voice-loop.wav`输出“杭州是
   浙江省省会，因西湖美景而闻名，是一座融合了自然风光与现代都市魅力的历史文化
   名城。”，时长 8.12s，RMS=0.08041、peak=0.71237、削波比例为零、
   DC=-0.00002、最大采样步长=0.29922；硬门与复核门均通过。此前 bf16 样本的
   重复文本不能再作为质量证据。
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
   该 artifact 做人工试听。L1 Code2Wav 会在 PCM 进入媒体层前拒绝非 24kHz
   WAV bytes、非有限值、削波、过大直流偏置与采样突变；门禁只证明信号完整性，
   不证明可懂度或主观自然度。
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
    SQLite WAL 连接允许 GPU worker 与媒体线程使用，但全部 connection 操作经同一
    re-entrant 锁串行；集成回归覆盖 16k ingress → queue → 三阶段 → PCM handoff
    → 已播事实 → 恢复上下文的完整本地路径。
12. **媒体输入质量边界**:`PcmIngress` 只接受已由媒体适配器显式转换到 16kHz 的
    int16/float PCM；它确定性下混、按官方 1s unit 组装，并拒绝隐式线性重采样、
    非有限值与错位 interleaved frame。新的输入会丢弃未凑满的旧输入尾帧再推进
    epoch，避免跨用户说话片段混合。LiveKit 48kHz → 16kHz 的高保真重采样仍须由
    具体 SDK/设备适配器显式提供，未伪称已接入。
13. **P5 部署预检**:`scripts/media_preflight.py` 分别检查 LiveKit SDK、四项
    连接配置存在性和本地 PCM 设备，不读取或输出 secret。当前环境四项远端前提及
    本地 PCM 设备均缺失，因此 P5 仍是明确的外部环境阻塞，而非“测试通过”的假象。
    `duplex/livekit.py` 已提供可选 SDK 适配：远端输入显式请求 16kHz 单声道，
    下行把已通过质量门禁的 24kHz PCM 以 `AudioSource.capture_frame()` 异步背压
    交给 LiveKit；barge-in 同时清空本地与 SDK 下行队列。适配器用 fake SDK 覆盖，
    但当前环境没有实际 SDK/房间/设备，不能作为远端或真机证据。
    `duplex/aec_policy.py` 已把客户端上报的 AEC 健康状态转为明确媒体降级：仅健康
    时允许扬声器全双工；失效或未知时优先要求耳机，无耳机则 push-to-talk。该策略
    不参与 MiniCPM-o 的说话决策；iOS 扬声器/听筒/蓝牙/耳机/后台恢复矩阵仍须真机验证。
14. **输出质量故障闭环**:后续十轮真实权重回放发现两轮产生 peak=1.0 的削波，
    且最大相邻采样步长分别为 0.83481 与 0.84146；这两轮按硬门禁失败，不能作为
    质量通过证据。现在每个 Token2wav 流式块在 publish 前执行同一套硬检查；命中后
    runtime 立即 mute、cancel-not-await，并将 `AgentSpeechRejected` 与
    `PCM_QUALITY_REJECTED` 写入事实/trace。已经由设备取走的健康前缀仍保留为事实，
    但待播及后续异常 PCM 不会交付媒体层。无削波的轻微满幅（0.98<peak<0.999）
    先整体缩放到 0.97，保留波形形状并留出播放 headroom；削波、直流偏置、或缩放后
    仍超阈值的突变才硬性拒绝。离线报告仍保留 peak≥0.98 的复核标记，便于发现
    接近阈值的趋势。
    启用严格阈值后的首个十轮批测中，3 轮产生并通过硬检查，7 轮在媒体边界前被
    拒绝（不会产生可播放 WAV）；这证明安全边界生效，但拒绝率本身是待改善的生成
    稳定性风险。紧接的三轮复测均安全完成（peak 0.628–0.887，最大采样步长
    0.257–0.329），说明风险随输入/采样条件波动，不能据此宣称已根除。
    应用保形归一化后的十轮真实权重复测中，9 轮产生 PCM，全部通过硬门禁与复核
    （RMS 0.0718–0.1093、peak 0.4612–0.9424、最大采样步长 0.1632–0.7209），
    没有 `PCM_QUALITY_REJECTED`；第 4 轮在模型已作出 speak decision 后没有产生
    codec/PCM，属于独立的 Talker 空输出可用性风险，不能归为音频质量通过。该状态
    现在会落为 `AgentSpeechNotProduced`（原因 `speak_decision_without_pcm`）而非伪造
    `AgentSpeechActuallyPlayed`，以便上层选择安全重试或文字/通知降级。
   最近五轮真实权重回放（`post-runtime-quality.batch-18c860dd30eb65a4`）均产生
    可播放 PCM 并通过完整性检查：RMS 0.07021–0.12222、peak 0.42169–0.96997、
    最大采样步长 0.15103–0.72485、削波比例均为零；没有
   `PCM_QUALITY_REJECTED` 或 peak 复核警告。这只证明这五个 fixture 回放样本的
   信号完整性，不等价于主观自然度、远端设备播放或长期稳定性。
15. **质量优先的完整全双工重放**:`p1_duplex_loop.py`现默认 fp32/Torch 质量
    模式。真实 16kHz fixture 经音频前端、MiniCPM-o duplex 决策、Talker、
    Code2Wav、L3 runtime 与本地 writer 后输出“好的，没问题。”；产物
    `quality-priority-fp32-duplex.wav`为 1.68s/24kHz，RMS=0.06320、
    peak=0.42627、削波比例为零、DC=-0.00004、最大采样步长=0.12674，且 trace
    中有完整 EOU→SPEAK_DECISION→TALKER_CHUNK_READY→CODE2WAV_FIRST_PCM→
    PUBLISHED 链。这是本地单样本功能及信号完整性证据，不是主观评测、远端播放或
    统计意义 SLO；该进程全部模型驻留后占用 35.62GiB、该轮峰值 36.60GiB。

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
     最新 `scripts/p1_talker_bench.py --warmup 2 --repeat 10` 固定 25 帧 phrase
     为 p50/p95/p99 217.3/277.5/277.5ms（p50 8.69ms/token）；这是同一已加载模型
     的隔离热路径样本，不能与共享 GPU 的端到端 trace 直接比较或声明 SLO。
- **受控 graph 原型验证**:必须在真实 prefill 前 capture，随后释放 warmup/capture
  的 dummy KV 页并同步静态页表；否则第二个 token 起会与 eager 分歧。在该生命周期
  下，真实权重的连续 8 个贪心 token 与 eager 完全相同；隔离基准为 eager
  37.29ms/token（26.82 tok/s）、graph 19.39ms/token（51.58 tok/s）。该原型尚未
  进入受控生产热路径，不能据此声称端到端提速或扩大质量承诺。
- **测量说明**:全模型对比数字受共享 GPU 上不可见并发负载污染
  (nvidia-smi 可见 60%+ 利用率但无进程归属),最终全模型数字待
  GPU 空闲窗口复测;微基准数字为 GPU 空闲时测得,可信。

## 共驻显存初始证据(非完整矩阵)

`p1_duplex_loop.py` 现会报告**本进程** CUDA allocator 的 allocated/reserved，不能
推断其他进程、驱动工作区或全卡峰值。最新真实三阶段单回合回放在全部模型加载后为
19.78/19.80GiB，回合峰值为 26.32/26.72GiB，且输出通过信号完整性与人工复核门槛。
它只覆盖设计 §8 的“全部 load 后未 forward”和“一轮真实 forward”初始证据；重叠、
barge-in、10/30/60 分钟 soak、stage crash/restart 仍未测，不能据此宣称 85GB 门槛或
长会话稳定性已验收。

## 下一步

1. **P3 实测打断**:接入真实输入/物理播放适配器，记录 barge-in→静音 trace，
   验证已在 GPU 中的旧请求不会泄漏到媒体端；
2. **性能批测**:区分 cold/warm、本地/远端，采集足够 trace 后报告 p50/p95/p99；
3. **CUDA graph 捕获 decode 步**(sparkinfer decode 原生支持 graph replay
   + on-device metadata 重建)；
4. **P5 媒体接入**:LiveKit/AEC/设备播放，补齐真实客户端扬声器口径与主观试听。
5. **bf16 数值修复**:定位并消除自研 Thinker bf16 长序列与官方 Qwen3 的 token
   分歧；在同一语义质量回归通过前，不得把 sparkinfer bf16 设为默认质量路径。
