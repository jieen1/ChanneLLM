# P1–P2 阶段状态:自研引擎的听-想-说闭环(2026-08-04)

## 一句话

MiniCPM-o 4.5 的全双工语音闭环(音频流入 → listen/speak 决策 → 回复语音)
已在**自研单进程 runtime**按 unit 增量跑通：Thinker 每次开口即续写 Talker
KV，25 帧 phrase 经有界进程内队列送入 Code2Wav，再由 epoch-guarded L3
runtime 发布本地 PCM，并经可取消待播缓冲交给本地媒体 writer。当前证据仅覆盖
GPU→本地缓冲取出；LiveKit、AEC、物理设备播放及统计意义上的 SLO 仍未完成。

**性能核心对照**是 vLLM-omni 的 MiniCPM-o 三阶段实现；本项目不链接或调用其
runtime，而以同权重、同输入、同 trace 口径复现并逐项验证其可迁移机制。下文明确
区分“vLLM-omni async bridge 的同进程协议对照”和“实际 vLLM-omni runtime 基准”，
前者不能冒充后者。

## 当前验收快照与后续规划（2026-08-04）

- **[事实] P1–P4 本地链路已闭环**：16kHz 输入经 bf16 原生 Thinker、MiniCPM-o
  duplex 决策、Talker、Code2Wav、epoch-guarded L3 runtime 到 24kHz 本地 PCM；
  首包、信号完整性、barge-in cancel-not-await、SQLite 播放事实均有回归覆盖。它
  不是 LiveKit/物理设备端到端验收。
- **[事实] runtime 全面 bf16 原生，不做反量化**：权重原生 bf16，vllm-omni 参考
  实现同为 bf16 口径。实测官方 Qwen3 自身 fp32/bf16 贪心序列在 0–8 token 内即
  分歧（dtype 固有，任何 bf16 runtime 都一样），故"对齐 fp32"不是有效质量门禁；
  现行门禁是 bf16 同内核族 graph/eager 逐 token 一致 + 端到端 fixture 信号门禁。
  fp32 路径及其专用图捕获模块已从 runtime 删除。Token2Wav 每块在发布前通过
  非有限值、削波、直流偏置和采样突变门禁。
- **[事实] vLLM-omni 是性能参考基线**：已逐段参考其三阶段、async bridge、Stage2
  首块预热实现；当前环境没有可运行的 `vllm`/`vllm_omni` 安装，因此还没有真实跨
  runtime 的公平基准。bridge 协议实验不得改写为 vLLM runtime 成绩。
- **[事实] bf16 CUDA graph decode 是 Thinker 唯一 decode 路径**：sparkinfer
  paged plan/bind 开销曾使 bf16 eager decode 在 duplex 环内高达 ~275ms/token；
  `GraphDecodeSession`（paged KV + on-device metadata replay）把它降到
  ~21ms/token，`p1_graph_decode_check.py` 61/61 贪心 token 与 eager 逐位一致。
  生命周期修正已入库：capture 后释放 dummy 页并同步静态页表；step() 用
  `_expected_length` 检测 eager prefill 的外部推进并重新同步页表。
- **[事实] Talker 首 phrase 提前交接已落地**：官方 force_flush 的 5 帧首块语义
  由 `push_streaming` 惰性 generator 实现,采样/RNG/KV 次序不变并由固定 seed
  对照与契约测试锁定;短回复 fixture 收益被 Code2Wav 固定成本掩盖,长回复待量化。
- **[事实] Code2Wav flow-matching 已按人工试听决策降为 6 步**：同 codec 对照
  批次(10/6/5 步)信号门禁全过,人工确认 10 步与 6 步音质无实质差异、5 步劣化,
  故默认 `n_timesteps=6` 生效;fp16 组合因官方流式路径 dtype 不完整而不可用
  (留档于脚本)。端到端回归:回复与门禁不变,warm 首 PCM p50 272.8→187.6ms。
- **[下一步，按质量优先]**：先做真实输入/物理播放的 `barge-in → 静音` trace；随后在
  独立可审计 GPU 环境运行 vLLM-omni 与 ChanneLLM 的同权重、同 fixture、冷/热
  p50/p95/p99 对照；性能面剩余大头是 Code2Wav flow-matching 固定成本（需质量
  评审取舍）与音频 chunk prefill 的 sparkinfer extend 开销。P5 的 LiveKit、
  真机 AEC 和远端设备播放仍是外部环境验收项。

## 已建成的组件

| 层 | 组件 | 状态 |
|---|---|---|
| L0 内核 | `kernel/paged_kv.py` 页池/分配器/slot 复用 | ✅ CPU+GPU 测试 |
| L0 内核 | `kernel/sparkinfer_attn.py` plan/bind/run + **binding 缓存** | ✅ GPU parity 对齐 in-tree reference |
| L1 引擎 | `engine/thinker.py` 自研 Thinker(Qwen3 骨干,forward/forward_embeds) | ✅ bf16 原生;结构正确性存档见 git 历史 fp32 parity |
| L1 引擎 | `engine/talker.py` 自研 Talker(Llama 骨干,hidden_text_merge) | ✅ unit 级 KV 续写 + 25 帧 phrase |
| L1 引擎 | `engine/code2wav.py` Token2wav 封装 + StreamingSynth 分块流式 | ✅ 批量+流式双路径 |
| L1 引擎 | `engine/audio_front.py` 官方流式 whisper 编码器混合封装 | ✅ 音频理解验证 |
| L2 编排 | `pipeline/orchestrator.py` + `pipeline/transport.py` | ✅ 增量路由、有界队列、旧 epoch 清理 |
| L3 控制 | `duplex/runtime.py` + `duplex/driver.py` + `duplex/queued_runtime.py` + `duplex/playback.py` | ✅ 真实三阶段驱动、单 GPU 有界 worker、cancel-not-await、待播 PCM 静音、writer-handoff 播放生命周期 |
| L3 EOU 基准 | `duplex/eou_baseline.py` | ✅ 可注入 SoulX 官方状态流；⚠️ 当前 venv 未混入 SoulX 依赖，待独立官方服务/引擎部署实测 |

## 验证证据

1. **Thinker 结构与精度边界**:`scripts/p1_thinker_parity.py --fp32 --kv-backend static --tokens 48`
   在“请用一句话介绍杭州。”上与官方 Qwen3 48/48 token 逐 token 一致
   (logits max|Δ|=0.000067);399/399 权重已赋值。相同输入的 bf16
   `sparkinfer` 路径在第 10 token 分歧，bf16 `TorchListKV` 路径在第 7 token
   分歧（prefill max|Δ| 分别为 0.875/1.039）。因此这不是可忽略的质量误差：
   bf16 只能用于性能诊断，不能作为语义质量通过或线上默认路径。P1/P2 回归
   默认使用 fp32 + 预分配 Torch SDPA 语义 KV，直至 bf16 长序列 parity 有新的实测证据。
   相同 48-token 实测中，该静态 KV 路径也为 48/48 一致（prefill
   max|Δ|=0.000067，decode 16.3 tok/s）；它只消除了逐层 `torch.cat`，不改变
   注意力或采样语义。
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
   为防止机器速度回放伪造实时延迟，该脚本还支持
   `--realtime-input --eou-offset-s <标注秒数>`：它按 16kHz 输入时钟投递，并在
   标注 EOU 后记录本地首 PCM。最近一次共享 GPU cold 样本为 1236.5ms
   EOU→首 PCM、1903.4ms speak decision→首 PCM，音频 RMS=0.0679、peak=0.4355、
   无质量告警。该样本明确反映当时 GPU 争用，不能与隔离热路径、远端或 SLO 混用。
   2026-08-04 的三轮质量模式 paced 复测均输出“好的，没问题。”，信号门禁均通过
   （RMS=0.0606、peak=0.4243–0.4245、无削波）；warm EOU→首 PCM 为 669.1/737.5ms。
   但该 fixture 的 MiniCPM-o 在标注 EOU 前已作出 speak decision，因此该 EOU 数字
   不能代表“决定开口后的首包”优化成果：对应 warm speak decision→首 PCM 为
   1354.8/1451.9ms，Talker 25-frame 可路由块为 1500.6/1525.1ms，Code2Wav 首块为
   103.7/128.0ms。该现象暴露首个短 codec delta 尚未凑齐 25+3 帧时必须等待下一
   duplex unit 的结构性延迟；不得用较小的 EOU 数字掩盖它。
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
15. **质量优先的完整全双工重放**:`p1_duplex_loop.py`现默认 fp32/Torch-static 质量
    模式。真实 16kHz fixture 经音频前端、MiniCPM-o duplex 决策、Talker、
    Code2Wav、L3 runtime 与本地 writer 后输出“好的，没问题。”；产物
    `quality-priority-fp32-duplex.wav`为 1.68s/24kHz，RMS=0.06320、
    peak=0.42627、削波比例为零、DC=-0.00004、最大采样步长=0.12674，且 trace
    中有完整 EOU→SPEAK_DECISION→TALKER_CHUNK_READY→CODE2WAV_FIRST_PCM→
    PUBLISHED 链。这是本地单样本功能及信号完整性证据，不是主观评测、远端播放或
   统计意义 SLO。静态 KV 接入后的最近单轮产物
   `quality-priority-static-fp32-duplex.wav`仍输出“好的，没问题。”，RMS=0.0668、
   peak=0.3817、削波比例为零、DC=-0.00002、最大采样步长=0.10106；EOU 至首
   PCM 为 787.7ms、speak decision 至首 PCM 为 1018.6ms（均为 cold n=1）。该
   进程模型驻留为 35.62GiB，静态 KV 使本轮峰值达到 47.83GiB；这是明确的质量
   保真/显存取舍，未将其描述为 SLO 或长期稳定性证据。
16. **流式 vocoder 跨回合隔离**:十轮 static-fp32 回放曾暴露相同 codec 序列的
   波形逐轮增益累积（隔离复现 RMS 从 0.04365 升至 0.15864，峰值被归一化到
   0.97）。根因是 stepaudio2 Flow 在模块级 ``*_cache_buffer`` 中原地写入状态，
   原有 `stream_cache` clone 未覆盖该状态。`Code2Wav.stream_reset()` 现在同时
   恢复四个 decoder cache buffer 的初始化快照；固定 codec 十轮复验 RMS 跨轮跨度
   为 0.000038、峰值跨度为 0.00723。真实 fixture 十轮均产生“好的，没问题。”，
   RMS=0.0596–0.0653、peak=0.3455–0.4206、最大采样步长=0.10116–0.12963，
   无削波、质量拒绝或 review warning。该保护使峰值显存变为 50.50GiB，仍低于
   85GiB 共驻门槛；cold n=1 EOU→首 PCM=830.2ms，warm n=9 为
   p50/p95/p99=673.3/690.9/690.9ms。它证明该 fixture 的回合隔离与信号完整性，
   仍不等价于主观听感、远端播放或长会话 soak 验收。
17. **vLLM-omni 核心对照（同环境 bridge）**:查阅本地 vLLM-omni 的
   `tts2code2wav_async_chunk` 与其 24/25/26 帧回归后，确认其 async bridge 的
   首块门槛是 25 个新 codec frame（加 3 个左上下文），而 MiniCPM-o 官方 duplex
   首次 TTS 调用的 `force_flush` 允许 5 个新 frame。`p1_duplex_loop.py` 因此提供
   `--vllm-omni-codec-bridge`，以同一已加载模型、同一 16kHz fixture、相同 paced
   输入和 trace 锚点，将首块门槛切为 25；默认仍是官方 5-frame 语义。2026-08-04
   两个各三轮批次的输出均为“好的，没问题。”，且所有 WAV 通过信号完整性门禁。
   `official-first-flush.batch-18c870bdfbbd71d0` 的 warm n=2
   speak-decision→first-PCM 为 p50/p95/p99 `258.3/289.3/289.3ms`，Talker 首块为
   `322.2/345.5/345.5ms`；25-frame 对照
   `vllm-omni-bridge.batch-18c870fc42c06aa3` 对应为
   `1269.2/1355.3/1355.3ms` 与 `1296.3/1356.0/1356.0ms`。故本轮定位的主因是
   首个 codec block 的等待，不是 Code2Wav（分别为 `131.8/156.4/156.4ms` 与
   `164.4/204.1/204.1ms`）。这是可重复的**协议敏感性实验**，不表示实际
   vLLM-omni runtime 的绝对性能。检查显示本项目 `.venv` 与系统 Python 均未安装
   `vllm`/`vllm_omni`，所以真实跨 runtime benchmark 尚未执行；不得把该缺口写成
   vLLM-omni 已跑通或据此宣布 SLO。
18. **从 vLLM-omni 迁入的 Stage2 首块预热**:其 `BatchedToken2Wav` 在构造期以
   固定 50-mel 首块触发 HiFT 的 CUDA/cuDNN 初始化。自研 `Code2Wav.prewarm_stream()`
   改用 Token2wav 的公开流式接口送入 `3 + 25` 个 silence codec，完成后无条件复位
   flow/HiFT/module cache；单元测试锁定“预热块不发布、两次 reset、下一回合基线干净”。
   三轮 paced 回放 `official-first-flush-prewarmed.batch-18c871b63bfde54e` 均输出
   “好的，没问题。”且信号门禁全过（RMS=0.0669、peak=0.4213–0.4217、无削波）。
   cold Code2Wav 首块为 207.1ms（此前未预热批次为 290.1ms），但 warm n=2 为
   p50/p95/p99 `175.2/188.9/188.9ms`（此前为 `131.8/156.4/156.4ms`）；共享 GPU
   噪声下不能从这一批推出稳定加速，保留实现和 trace，待独占 GPU 扩样验证。

19. **fp32 CUDA graph decode 过门禁并接入默认路径**：背景是 sparkinfer paged
    planner 拒绝 fp32 q dtype，而 tensor core 没有真 fp32 MMA（tf32 截断尾数等价
    引入 bf16 级误差），故不改 sparkinfer，改为在质量路径自身上做图捕获。
    `channellm/engine/static_graph_decode.py` 把单 token 前向（embed→36 层→
    lm_head）捕获为按宽度分桶的图：attention 用显式 IEEE-fp32 bmm+加性 mask+
    softmax+bmm（padding 位 -inf，`exp(-inf)=0`、`x+0=x` 精确成立，mask 不改变
    有效位置数值），KV 写位置/有效长度经设备端索引缓冲在 replay 时按值读取。
    调试中发现本机（Blackwell SM120 + torch 2.13 cuBLAS）**捕获期执行的 GEMM
    结果与 eager 分歧、replay 逐位一致**（最小复现：单 Linear 捕获期 diff≈2.2、
    replay diff=0.0），因此每个桶捕获后立即 replay 一次，用正确结果覆写捕获期写
    入 KV 的垃圾。门禁：`scripts/p1_graph_decode_check.py --dtype fp32
    --kv-backend static --tokens 120` 输出 parity PASS（121/121 token 一致），
    分离计时 graph 段 27.8ms/token vs eager 对照 35.9ms/token（1.29x，含逐 token
    同步）；eager 对照本身快于早期 61ms/token 口径，因测量循环不同，不做跨口径
    比较。`DuplexSession` 注入该会话后单 token decode 走 replay、多 token 音频
    prefill 仍 eager；hidden 是复用缓冲，入 unit 条件化前克隆。短回复 fixture
    （"好的，没问题。"）realtime x3 A/B：有/无 graph 均通过开口与信号完整性门禁、
    文本一致，端到端延迟差在该 fixture 的短 unit 下不显著（p95 分别 404.8 /
    399.7ms）；收益随 unit 长度线性放大，受控证据以门禁脚本为准。
    `--no-graph-decode` 保留 eager 路径供后续 A/B。

20. **Talker 首 phrase 提前交接(官方 force_flush 落地)**:`TalkerStream.push`
    此前总生成满 25 帧才返回,使 L2 早已实现的 5 帧首块窗口一直等整段 phrase。
    现改为 `push_streaming` 惰性 generator:回合首个 phrase 在确认越过提前阈值
    (第 early+1 帧生成且未 EOS)时先 yield 前 early 帧,driver 逐段编排+合成,
    尾段带 is_last 使 final 仍附在最后一个音频块。采样顺序/RNG/KV 次序不变,
    flatten 与 push 逐位相等(固定 seed 对照),新增 4 个 streaming 契约测试。
    短回复 fixture realtime x3:信号门禁与回复文本一致,talker_chunk_ready 收敛
    到 129–144ms,first-PCM p95 由 405ms 收紧到 296ms;但 p50 收益被 Code2Wav
    固定成本掩盖(见下),长回复收益待独立量化。`early_first_frames` 与
    `codec_initial_min_audio_frames` 同源,vllm-omni 25 帧桥接模式自动关闭提前交接。

21. **精度口径决定：全面 bf16 原生，删除 fp32 路径**。证据链：
    (a) 官方 Qwen3 自身 fp32 vs bf16 贪心对照在 3 个 prompt 上分别于第
    8/0/无分歧（裸 prompt 贪心落入退化区，两者输出都是乱码且互不相同）——
    bf16 漂移是 dtype 固有，vllm-omni 在内的任何 bf16 runtime 都无法"对齐
    fp32"；(b) 权重原生 bf16，fp32 运行只是把 bf16 权重零填充到 32 位，不
    恢复任何信息，却使权重流量与 KV 显存翻倍；(c) bf16 duplex 端到端回放
    与 fp32 同样输出"好的，没问题。"且信号门禁全过（RMS/peak/削波/DC/步长
    同量级）；(d) bf16 graph decode 对 bf16 eager 61/61 token 逐位一致。
    据此删除：`channellm/engine/static_graph_decode.py`（fp32 专用分桶图捕获）、
    各脚本 `--thinker-dtype`/`--no-graph-decode`/`--fp32` 开关、Thinker/Talker/
    TorchStaticKV 的 fp32 默认值。`p1_graph_decode_check.py` 收敛为 bf16
    graph/eager parity 单一门禁；`p1_thinker_parity.py` 收敛为 bf16 同精度
    漂移留档。bf16 原生端到端（realtime x3）：回复与门禁全过，峰值显存
    28.3GiB（fp32 为 50.5GiB），warm speak_decision→first_PCM p50/p95
    272.8/275.6ms，chunk 决策 19–175ms（bf16 eager 时代为 188–1375ms），
    talker_first_chunk p50 211ms。历史 fp32 parity 证据（48/48、121/121）
    保留在 git 历史作为结构正确性存档，不再作为运行时口径。

24. **决策采样向量化 + Code2Wav 深挖结论**:
    (a) `DuplexSession._decode_step` 的重复惩罚原为窗口内逐 token GPU 标量
    索引(512 窗口实测 2.8–4.4ms/步),改为一次 gather/scatter,与标量版
    **逐位一致**(torch.equal 验证),降到 0.7ms/步;决策采样总开销 ~19ms/步,
    其余为 softmax/multinomial/同步等不可语义变更部分。
    (b) Code2Wav 深挖:torch.profiler 实测整段合成 wall 686.7ms 中 kernel
    总和仅 136ms —— **80% 是 launch/主机侧开销**(每段 ~2500 个小 kernel,
    含 2188 次 cudnn nchw→nhwc 转置)。但两条消开销路径均被实测阻断:
    CUDA graph 捕获要求静态形状,而官方 conformer attention cache 每 chunk
    `torch.cat` 增长(动态形状,除非重写官方内核为预分配最大缓存+索引写入);
    torch.compile(dynamic=True)在官方 flow_matching 代码上触发 PyTorch
    PythonDispatcher 内部断言失败。结论:vocoder 开销消除属于"重写官方
    vocoder 内核"级别的独立工程,不在本轮范围;6 步 fp32 维持现状。
23. **Talker decode 图捕获过门禁并接入默认路径**:Talker(20 层 Llama,
    12 头 MHA/head_dim 64)逐帧 decode 是 launch 受限负载,
    `channellm/engine/talker_graph_decode.py` 把单帧步捕获为按宽度分桶的图:
    显式 bf16 bmm + 加性 mask(mask cast 到模型 dtype,避免 softmax 升型与
    bf16 V 矩阵乘冲突),连续静态 KV 无页表同步问题,条件化 prefill 仍 eager。
    门禁 `scripts/p1_talker_graph_check.py`:4 unit × 25 帧贪心 codec 流
    eager/graph **100/100 逐帧一致**,受控计时 20.62→3.58ms/帧(5.76x),
    捕获一次性开销 23ms。`TalkerStream.graph` 注入后 duplex 默认启用。
    短回复 fixture realtime x3:回复与信号门禁不变,warm
    speak_decision→first_PCM p50 156.7ms;该 fixture 单元短、决策循环长度
    随机,端到端差被噪声掩盖,机制收益以受控门禁为准。
22. **音频 prefill 改走 SDPA,单 chunk 处理 ~390ms→~105ms**:分阶段计时定位到
    每个 1s 音频 chunk 的 Thinker extend prefill(仅 11 个 q token)要 ~200ms,
    其中 sparkinfer 逐层 bind ~2.9ms×36≈105ms、run ~30ms、其余为主机侧
    metadata。音频 chunk 是带宽受限的小负载,plan/bind 开销远超内核本身,
    故 `SparkinferPagedKV` 的多 token prefill 改为 SDPA:把当前序列已写入的
    页 gather 成连续缓冲(MB 级拷贝)后走 Torch SDPA 因果 attention——与
    fp32 parity 时代验证过的内核族同源;decode 单 token 仍走 sparkinfer
    graph replay。门禁:同 prompt 贪心 61/61 token 与纯 sparkinfer 路径完全
    一致(prefill logits max|Δ|=0.625 为 bf16 内核族间正常漂移,argmax
    100% 一致);prefill 计时 864→71ms(含双侧首次调用预热)。端到端
    realtime x3:回复与信号门禁不变,chunk prefill ~68-73ms(原 185-257ms),
    warm speak_decision→first_PCM p50 156.1ms(6 步 vocoder 时代 187.6ms)。
    `prefill_backend="sparkinfer"` 保留旧路径供 A/B。

## Code2Wav flow-matching 步数决策(2026-08-04 人工试听确认)

分阶段计时显示 `code2wav_first` 主要来自 `flow.inference_chunk` 的 flow-matching
步数(fp32 autocast),且对 8/28 帧窗口近似相同(固定开销主导);参考音频条件
(spk_emb/prompt_mels)已在 `self.cache` 一次性预编码。`scripts/p1_code2wav_quality_ab.py`
以同一固定 seed 的 100 帧 codec 在生产同款流式路径下对照:

| 配置 | 首块 | 全段 | 信号门禁 | 人工试听 |
|---|---|---|---|---|
| 10 步(官方默认) | 135.2ms | 600.8ms | PASS | 好 |
| 6 步 | 89.8ms | 488.3ms | PASS | 好(与 10 步无实质差异) |
| 5 步 | 90.0ms | 364.6ms | PASS | 差一点 |

据此默认 `n_timesteps=6`。fp16 选项实测不可用:官方 Token2wav 只对 flow 做
half(),`set_stream_cache` 的 spk 仿射层与 hift cache 仍 fp32,直接 dtype 冲突;
如需 fp16/bf16 vocoder 须改造官方实现内部,另行立项。端到端回归(realtime x3):
回复"好的，没问题。"不变、门禁全过,warm code2wav_first p50 114.4ms、
speak_decision→first_PCM p50 187.6ms(10 步时代为 272.8ms)。

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
  1. `sparkinfer_attn` 复用 plan/scratch 与 q/output 静态缓冲；不能复用
     `binding` 本身：六个连续 decode step 的 GPU-reference 回归证明 binding 会
     固化 page table/cache length/cumulative-q metadata，复用会在后续 token 发生
     语义发散。因此当前每步按最新 metadata rebind，先保证数值正确，不再把
     “binding 成本为零”作为性能承诺;
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
  (nvidia-smi 可见 60%+ 利用率但无进程归属)；即使采样时显示 4% 利用率且没有
  可见 compute process，连续三次相同 Talker 20-repeat 微基准仍得到 p50
  196.2/259.9/277.3ms、第三次 p95=1029.7ms。故“看起来空闲”不足以证明独占，
  当前任何全模型数字均非 SLO，必须在可验证的独占窗口重新采集足够样本。

## 共驻显存初始证据(非完整矩阵)

`p1_duplex_loop.py` 现会报告**本进程** CUDA allocator 的 allocated/reserved，不能
推断其他进程、驱动工作区或全卡峰值。最新真实三阶段单回合回放在全部模型加载后为
19.78/19.80GiB，回合峰值为 26.32/26.72GiB，且输出通过信号完整性与人工复核门槛。
它只覆盖设计 §8 的“全部 load 后未 forward”和“一轮真实 forward”初始证据；重叠、
barge-in、10/30/60 分钟 soak、stage crash/restart 仍未测，不能据此宣称 85GB 门槛或
长会话稳定性已验收。

## 下一步

1. **实际 vLLM-omni 基准**:在独立、可审计的 vLLM-omni 环境中，以固定权重、
   固定 fixture、冷/热分组及同一 trace 锚点执行真实三阶段回放；记录 GPU 独占与
   驱动状态，不能用同环境 bridge 替代；
2. **P3 实测打断**:接入真实输入/物理播放适配器，记录 barge-in→静音 trace，
   验证已在 GPU 中的旧请求不会泄漏到媒体端；
3. **性能批测**:区分 cold/warm、本地/远端，采集足够 trace 后报告 p50/p95/p99；
4. **prefill 与 Code2Wav**:decode 已是 bf16 graph replay(~21ms/token);剩余
   Thinker 侧开销在音频 chunk 的 sparkinfer extend/prefill(~200ms/chunk,
   plan 开销待剖析);Code2Wav flow-matching 固定成本需质量评审后决策；
5. **P5 媒体接入**:LiveKit/AEC/设备播放，补齐真实客户端扬声器口径与主观试听。
6. ~~bf16 数值修复~~ **已撤销**:原口径要求 bf16 对齐官方 fp32,实测官方自身
   fp32/bf16 亦在个位数 token 内分歧(证据 21),该门禁不成立;runtime 已全面
   bf16 原生,质量门禁改为同精度 graph/eager parity + 端到端信号门禁。
