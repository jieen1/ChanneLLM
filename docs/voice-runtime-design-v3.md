# 全双工语音 Runtime 设计 v3(自研独立项目)

> 日期:2026-08-03
> 目标硬件:单张 NVIDIA SM120 (Blackwell) 96GB
> 模型:MiniCPM-o 4.5(端到端全双工,已定案)
> **技术路线:从零自研,不依赖 vLLM / vllm-omni,仅作参考实现**
> 与既有仓库关系:独立项目,与 `qwen-sm120-runtime` 无代码复用(架构不通),仅共享 SM120 环境经验

## 决策记录

| 决策 | 结论 | 说明 |
|---|---|---|
| 端到端 vs 级联 | **端到端** | 保证最完整交互体验;不用小模型级联换延迟 |
| 延迟策略 | **深度优化** | 不通过降级模型规避 |
| 依赖 vllm-omni | **不依赖** | 独立项目,vllm-omni 与 MiniCPM-o 官方代码仅作参考 |
| 说话权 | **唯一**:MiniCPM-o duplex | 禁止多控制器并存 |
| 任务层 | 先走外部 API | 不是主要风险 |

---

## 1. 证据口径

**[事实]** 官方文档/源码直接支持,或本机已复现测量 · **[推断]** 需实测验证 · **[未知]** 无可靠数据,不写进承诺

### 已核实

- **[事实]** MiniCPM-o 4.5 权重内自带 **10,537 行官方推理代码**,含 `class MiniCPMODuplex`(`modeling_minicpmo.py:2438`)、`streaming_prefill()`、`streaming_generate()`、`get_session_schema()`。配置 `stream_input: True`、`audio_chunk_length: 1.0`、`audio_pool_step: 5`、`listen_speak_type: asr`。**单进程即可跑通全双工。**
- **[事实]** 模型结构:36 层、hidden 4096、32 heads / 8 KV heads、head_dim 128、rope_theta 1e6、上下文 40960、**纯 full attention**(无 `layer_types`,无 linear attention,无 MTP)。骨干继承 `Qwen3PreTrainedModel`。TTS backbone = llama,audio tokenizer = s3tokenizer @16kHz,输出 24kHz。
- **[事实]** 官方 GPU demo 要求 **>28GB 显存**,初始化后约 **21.5GB**。model card 的 19GB/11GB 是权重口径,非运行口径。
- **[事实]** vllm-omni 三阶段拓扑:Stage0 Thinker(`LLM_AR`)→ Stage1 Talker(`LLM_AR`)→ Stage2 Code2Wav(`LLM_GENERATION`,非自回归)。单卡显存预算 55%/15%/15%,Stage1 KV 钉死 2 GiB,Stage2 `enforce_eager: true`。
- **[事实]** vllm-omni 默认 `enable_prefix_caching: false`;跨阶段 `codec_chunk_frames: 25`、`codec_left_context_frames: 3`。
- **[事实]** 中文场景无"又端到端又低延迟"的开源选项:Moshi(~160ms)、PersonaPlex(0.170s)均纯英语。
- **[事实]** 权重已下载校验:`openbmb/MiniCPM-o-4_5` 54/54 文件 20.05GB;`Soul-AILab/SoulX-Duplug-0.6B` 24/24 文件 7.78GB。

### 未核实

- **[未知]** MiniCPM-o 在 SM120 上的实际 `EOU → 首个 PCM`
- **[未知]** 官方单进程(串行)与三阶段流水线的首包延迟差值 —— **这是本项目最重要的待测数字**
- **[未知]** "官方 demo 明示 AEC 影响打断成功率" —— 两个 README 均未检索到
- **[未知]** 三阶段共驻的真实显存峰值

---

## 2. 产品目标与 SLO

### 目标

随时可以说话的思考伙伴:实时接话、完整记录、异步派活。

### 明确不做

- 不让小模型干需要智商的活
- 不做 PSTN(8kHz 窄带恶化中文擦音塞擦音)
- 不混合多套 turn authority

### 指标(延迟优先)

| 指标 | 定义 |
|---|---|
| `EOU_TO_FIRST_AUDIO` | 用户说完 → **客户端扬声器第一个 sample** |
| `BARGE_IN_TO_SILENCE` | 用户开口 → 本地播放实际静音 |
| `ACK_AUDIO` | → backchannel 出声(**单独统计**) |
| `FIRST_MEANINGFUL_AUDIO` | → 有实质内容的音频 |

报告规则:p50/p95/p99;本地/远程、冷/热分开;禁止只报均值。

### 对话质量门槛

- 停顿思考不被抢话
- 头脑风暴 10 分钟,主动插话 ≤3 次
- backchannel 与真实状态一致(无任务时不得说"我去办")

---

## 3. 总体架构

```
┌──── App(LiveKit Client SDK,内置 AEC)────┐
└────────────────┬───────────────────────────┘
                 │ WebRTC / Opus
┌────────────────▼───────────────────────────┐
│  媒体层(LiveKit SFU,公网 VPS)            │  ← 唯一外部依赖,不自研
└────────────────┬───────────────────────────┘
                 │ worker 主动外连注册
┌════════════════▼═══════════════════════════════════════┐
║  本项目 @ SM120 96GB                                    ║
║                                                          ║
║  ┌── L4 应用层 ────────────────────────────────────┐   ║
║  │  事件存储 / 播报仲裁 / Router / 任务派发         │   ║
║  └────────────────┬─────────────────────────────────┘   ║
║  ┌── L3 双工会话控制 ──────────────────────────────┐   ║
║  │  说话权 / turn epoch / 端到端取消 / 会话生命周期 │   ║
║  └────────────────┬─────────────────────────────────┘   ║
║  ┌── L2 三阶段编排 ────────────────────────────────┐   ║
║  │  增量提交 / resumable / 攒单元 / 预热 / 三边清理 │   ║
║  └────────────────┬─────────────────────────────────┘   ║
║  ┌── L1 推理内核 ──────────────────────────────────┐   ║
║  │  paged attn / 连续批处理 / CUDA graph / prefix   │   ║
║  └────────────────┬─────────────────────────────────┘   ║
║  ┌── L0 模型 ──────────────────────────────────────┐   ║
║  │  Thinker(Qwen3-8B+Whisper+SigLIP)              │   ║
║  │  Talker(MiniCPMTTS/llama+s3tokenizer)          │   ║
║  │  Code2Wav(vocoder → 24kHz)                     │   ║
║  └──────────────────────────────────────────────────┘   ║
╚═════════════════┬════════════════════════════════════════╝
                  │ HTTP,异步,不阻塞语音
        ┌─────────▼──────────┐
        │ 外部大模型(任务层)│  ← 不自研
        └────────────────────┘
```

**自研边界:L0–L4。不自研:LiveKit(媒体传输)、外部大模型(任务执行)。**

### 双时钟不变量

1. 会话主循环不 await 外部模型 / HTTP callback / 长事务
2. task enqueue 落盘即返回,网络发送由独立 worker 负责
3. task result 只能进通知队列,播不播由仲裁器决定
4. task worker 崩溃不影响实时媒体;实时模型崩溃不丢任务
5. GPU 只属于实时平面

---

## 4. L2 三阶段编排:必须实现的九件事

**这一层就是首包延迟本身。** 去掉它 = 串行 = 首包等整句生成完。

### 串行 vs 流水线

```
串行:   Thinker[整段] ──> Talker[整段] ──> Code2Wav[整段] ──> 出声
        首包延迟 = A + B + C

流水线: Thinker  [ph1][ph2][ph3][ph4]...
                   ↓    ↓    ↓
        Talker    [25帧][25帧][25帧]...
                     ↓     ↓     ↓
        Code2Wav   [音频][音频][音频]...
                      ↓
                    出声   ← 首包 ≈ 一个 phrase
```

### 九件事清单

| # | 机制 | 不做会怎样 | 参考实现 |
|---|---|---|---|
| 1 | **三引擎身份对齐** — 一个请求在三个引擎各有 request_id/replica_id,需统一状态对象 | 请求丢失、错乱 | `orchestrator.py: OrchestratorRequestState` |
| 2 | **增量提交** — 首次 `submit_initial`,后续每块新输出 `submit_update` 追加 | **退化成串行,首包几秒** | `orchestrator.py:_forward_to_next_stage` |
| 3 | **下游 resumable** — 下游请求生成完不能关闭,挂着等更多上游输入 | 请求提前关闭,后续内容丢失 | `build_engine_core_request_from_tokens(resumable=)` |
| 4 | **攒够单元再转发** — 上游产出不够下游用一次时 hold | 切太碎(开销)或等太久(延迟) | `if not next_inputs: return` |
| 5 | **下游预热** — Stage0 启动即预热 Stage1/2 | 首包多一次冷启动 | `_prewarm_async_chunk_stages` |
| 6 | **跨阶段 tokenizer** — Stage0 独占 tokenizer,需借给下游输入处理器 | 下游无法处理上游输出 | `streaming.source_token_decoder` |
| 7 | **错误三边清理** — 任一 stage 失败,三处请求全清 | 请求泄漏、显存不释放 | `_cleanup_request_ids` |
| 8 | **终止输出合成** — 上游结束下游无产出时凭空造终止输出 | 客户端永久挂起 | `_build_terminal_empty_output` |
| 9 | **打断时三阶段同时取消** — Thinker/Talker/Code2Wav/传输队列四处齐停 | **听到上一轮残留音频** | 见 §5 epoch |

> **不需要实现的**(vllm-omni 有但本项目用不上):PD prefill-decode 分离、CFG companion(diffusion)、collective RPC(多 worker)、分布式 KV transfer、TP/PP。

### 跨阶段传输

单进程单卡,不要分布式那套。共享内存或进程内队列传 tensor,必须有:

- 有界队列 + 背压(上游快于下游时不能无限堆)
- 首块与后续块的不同超时(参考:首块 3000ms,后续 300ms)
- chunk 粒度可配(参考:25 帧 + 3 帧左上下文)—— **这是首包延迟主旋钮**

---

## 5. L3 双工会话控制

### 唯一说话权

MiniCPM-o duplex 是唯一 turn/speech authority。**SoulX-Duplug 只作两件事:**

1. **独立 EOU 基准** — 提供与被测系统无关的"用户何时说完"第二意见。没有它无法客观测量 `EOU_TO_FIRST_AUDIO`。
2. **故障后备** — duplex 不可用时退到 `push-to-talk + transcript-only`。

**不参与说话决策。** 显存约 4GB,换可测量性。

### epoch 端到端取消

- 所有 LLM token、codec chunk、audio chunk 携带 `turn_epoch + speech_id`
- **旧 epoch 无条件丢弃**,覆盖:已生成未播、已在传输队列、已在客户端 jitter buffer
- 新 response 到来时**先 barge-in + cancel 旧的,不能 await 旧的完成** —— 否则输入循环被阻塞,无法及时响应打断
- 四个独立状态域:**Input / Reply / Notification / Task**

### 状态机

MiniCPM-o 的内部判定是 **observation**,不是产品状态机。产品层需自己维护 turn 状态、播放游标、待播队列。

---

## 6. L4 应用层

### 事实源:事件日志,不是 transcript

markdown 不能当权威:附和可能被计入、打断后有文本从未播放、ASR 后续会修订、append 无事务与顺序保证。

- **authority**:SQLite WAL append-only,单写者
- 事件带 `session_epoch / seq / turn_id / speech_id / task_id / supersedes`
- 关键事件:`UserSpeechFinal`、`UserBackchannelObserved`、`AgentSpeechPlanned`、**`AgentSpeechActuallyPlayed`(与 planned 分开)**、`TaskEnqueued`、`TaskResultReady`、`Superseded`
- markdown = 投影视图,可随时重建
- 上下文用 token-budgeted `ContextSnapshot`,不是"硬截断最近 N 轮"

### Router 不是三选一

`CHAT/NOTE/TASK` 不互斥(「查一下 X,然后提醒我」两者都要)。改**多标签 proposal**:全部记录,task candidate 单独确认。低置信人名/数字必须确认后才触发有副作用的外部动作。

### 播报仲裁

用户语音永远优先。任务结果进通知队列,等 idle 窗口播报,幂等、可合并。

### 头脑风暴模式

默认只 backchannel,攒够一段(~30s 或话题段落)才给一次有质量回应。**语音助手最大的毛病是话太多。**

---

## 7. 延迟预算(分段,不做单一数字承诺)

**禁止把各段 nominal 值直接相加。**

| 段 | 串行? | 当前依据 |
|---|---|---|
| capture + Opus packetization | 与采集重叠 | **[未知]** |
| uplink + SFU + jitter/decode | 串行 | **[未知],自适应** |
| 重采样 + chunk 对齐 | 部分串行 | **[未知]** |
| 音频 encoder(Whisper,1.0s chunk) | **是否随音频增量?** | **[未知] ← 关键** |
| Thinker prefill | EOU 后还剩多少? | **[未知] ← 关键** |
| "决定开口"判定 | 串行 | **[未知]** |
| first token decode | 串行 | **[未知]** |
| first token → first phrase | 串行 | 攒几个 token |
| Talker → 25 帧 codec | 串行 | 可调 |
| Code2Wav 首个 PCM | 串行 | **[未知]** |
| publish + downlink + client jitter | 串行 | **[未知],自适应** |
| device playout | 串行终点 | **[未知]** |

### 可流水线重叠

- ASR partial 与 turn observation 随采集持续运行
- EOU 前 speculative prefill(错猜丢弃,**不得写成 final**)
- 事件写入、context snapshot、reply 调度可并行
- 后续 token / codec / 音频 / 传输 / 播放构成流水线
- task dispatch 与整条对话链并行

### 延迟杠杆(按预期收益)

1. **音频 encoder 流式增量化** — 若 EOU 后仍需重算,这是最大的一刀
2. **Prefix caching** — 系统提示固定 + 历史只追加,语音场景命中率极高(参考实现默认**关着**)
3. **Speculative prefill** — EOU 前预填
4. **Code2Wav CUDA graph** — 参考实现是 `enforce_eager: true`,未捕获
5. **codec chunk 粒度下调** — 首包主旋钮
6. **三阶段 CUDA stream 分离 + 优先级**
7. **backchannel 预录接场** — 消掉生成时间,但**不是 0ms**,网络与播放缓冲仍在

---

## 8. 显存与调度

**不使用权重加总。** 运行峰值含 KV cache、激活、CUDA graph、cuDNN/CUTLASS workspace(HiFi-GAN vocoder 尤其吃)、每进程 CUDA context 与 allocator cache、首次 forward。

### 共驻测量矩阵(6 场景)

1. 全部 load 后未 forward
2. 每个 stage 首次真实 forward 后
3. Thinker 生成 + Talker + Code2Wav 三者重叠
4. 播放中 barge-in,取消并立即开始下一轮
5. 10 / 30 / 60 分钟 soak
6. 单个 stage crash/restart 后的显存回收

**建议门槛:** 峰值 <85GB;稳态留 ≥10GB 或 12%(取大);warm 后 10 分钟增长 <1GB。

### 计算 QoS

三个子模型抢同一张卡。Talker 是延迟敏感的单 token 解码,Code2Wav 是吞吐敏感的大批量前向——**必须防止 Code2Wav 把 Talker 的节拍挤掉**。手段:CUDA stream 分离 + 优先级、准入控制、有界并发。

---

## 9. 实施阶段

### P0 — 基线与测量地基(1–3 天)

**在写任何自研代码之前完成。**

- [ ] 官方 `MiniCPMODuplex` 单进程跑通(transformers,SM120 上解依赖)
- [ ] latency trace schema:`trace_id / turn_epoch / speech_id`,锚点覆盖 §7 全部段
- [ ] 固定音频回放集(中文:停顿思考、附和、打断、人名数字专名)
- [ ] **产出串行基线 waterfall** ← 后续所有优化的对照系

**验收:** 拿到官方单进程路径的真实 `EOU → 首个 PCM` 分段数据。

> P0 的 waterfall 直接决定 P1 的价值。若串行首包已可接受,三阶段编排的优先级可下调。

### P1 — 推理内核(4–6 周)

- [ ] 三个子模型加载与前向(参照官方 10,537 行 + vllm-omni 实现)
- [ ] Paged attention / KV block 管理(**纯 full attention,比 hybrid 简单**)
- [ ] 连续批处理调度器
- [ ] CUDA graph 捕获
- [ ] Prefix caching

**验收:** 单 stage 吞吐与显存达标,对齐官方数值输出(逐 token 比对)。

### P2 — 三阶段编排(3–5 周)

- [ ] §4 九件事全部实现
- [ ] 跨阶段有界队列 + 背压 + 超时
- [ ] chunk 粒度可配
- [ ] 三阶段共驻测量矩阵

**验收:** 流水线首包显著优于 P0 串行基线;数值输出与串行一致。

### P3 — 双工会话控制(2–3 周)

- [ ] epoch 端到端取消(四处齐停)
- [ ] 四状态域
- [ ] 会话生命周期、断连恢复

**验收:** 打断后听不到旧音频;crash 重启会话可续。

### P4 — 应用层(2–4 周)

- [ ] 事件存储(SQLite WAL 单写者)
- [ ] 播报仲裁、多标签 Router、任务派发(MCP)
- [ ] 头脑风暴模式、预录 backchannel

### P5 — 远程与真机(2–3 周)

- [ ] LiveKit 接入,worker 主动外连(家里不开公网端口)
- [ ] **iOS 真机 AEC 矩阵**:扬声器/听筒/蓝牙/耳机/后台恢复
- [ ] AEC 失败降级到耳机或 push-to-talk

### P6 — 性能长尾(持续)

CUDA graph 覆盖率、prefix 命中调优、批处理参数、异常路径、长会话稳定性。**这部分是无底洞,按收益停。**

### 时间汇总

| 阶段 | 估算 |
|---|---|
| P0 | 1–3 天 |
| P1–P3(核心跑通) | **9–14 周** |
| P4–P5 | 4–7 周 |
| P6 | 持续 |

> 估算前提:已有 paged attention / 批处理 / CUDA graph 的工程经验。**最大不确定项是 P1 的模型适配**——MiniCPM-o 是三个异构子模型(Whisper encoder + Qwen3 backbone + llama TTS + vocoder),不是单一 transformer。

---

## 10. 参考实现地图

**Apache 2.0,可直接参考。不作为运行时依赖。**

| 要实现的 | 参考位置 | 行数 |
|---|---|---|
| 模型前向 / 双工接口 | MiniCPM-o 权重内 `modeling_minicpmo.py`(`class MiniCPMODuplex:2438`、`streaming_prefill:2777`、`streaming_generate:3151`) | 5,094 |
| 输入处理 / 音频分块 | 权重内 `processing_minicpmo.py`、`utils.py` | 4,082 |
| 视觉编码 | 权重内 `modeling_navit_siglip.py` | 981 |
| **三阶段编排** | `vllm_omni/engine/orchestrator.py`(`_forward_to_next_stage:1728`、`_orchestration_loop`、`_route_output`) | 核心 ~1,200–1,500 |
| stage 启动与配置 | `vllm_omni/engine/stage_engine_startup.py` | 1,587 |
| 跨阶段连接器 | `vllm_omni/worker/omni_connector_model_runner_mixin.py` | 2,309 |
| 拓扑与阶段定义 | `vllm_omni/model_executor/models/minicpmo_4_5/pipeline.py` | 89 |
| 阶段间输入转换 | `stage_input_processors/minicpmo_4_5_omni.py`(`llm2tts`、`tts2code2wav_*`) | — |
| 三阶段模型实现 | `models/minicpmo_4_5/`(`minicpmo_4_5_omni_llm.py` 5,008 / `code2wav.py` 862 / `omni_tts.py` 846 / `batched_token2wav.py` 522) | 8,831 |
| **双工会话控制** | `experimental/fullduplex/minicpmo45/`(`stage0.py` 983 / `data_plane.py` 898 / `runtime.py` 332 / `policy.py` 151) | 3,344 |
| epoch 取消最小范式 | `experimental/fullduplex/core/runtime.py`(`_start_response` 注释解释了为何必须 cancel 而非 await) | 101 |
| 显存/chunk 参数基线 | `vllm_omni/deploy/minicpmo_4_5.yaml`、`minicpmo_4_5_duplex.yaml` | — |
| TTS 优化手法 | vLLM 博客 `2026-06-23-vllm-omni-tts`(chunk 解耦、torch.compile、CUDA graph 按 batch+frame 捕获) | — |

**明确不参考**(用不到):PD 分离、CFG companion、collective RPC、分布式 KV transfer、TP/PP、diffusion、其余 40 个模型。

---

## 11. 风险登记册

| # | 严重度 | 风险 | 缓解 |
|---|---|---|---|
| R1 | **Blocker** | P1 模型适配受阻——三个异构子模型(Whisper/Qwen3/llama TTS/vocoder)非单一 transformer | P0 先跑官方实现摸清结构;逐 token 对齐验证 |
| R2 | **Critical** | 延迟预算把 chunk 粒度当总延迟 | 全链路 waterfall,禁止 nominal 相加 |
| R3 | **Critical** | barge-in 只停一处,旧音频穿越 | epoch 四处齐停 + 客户端缓冲丢弃 |
| R4 | **Critical** | 事件乱序/丢失/修订覆盖 | SQLite WAL 单写者 + revision |
| R5 | High | **自研落后于上游** — vllm-omni 每日推进,MiniCPM-o 会出新版,duplex 仍是 experimental(33 pipeline 仅 1 个在用,还会大改) | 定期 diff 参考实现;锁定模型版本 |
| R6 | High | SM120 依赖钉版装不上 | 独立干净 venv,逐个解 |
| R7 | High | 三阶段共驻 OOM / 计算争抢 | 85GB 门 + stream 优先级 + 准入控制 |
| R8 | High | AEC 设备相关,自我打断循环 | 真机矩阵 + 耳机/PTT 降级 |
| R9 | High | 冷启动 / compile 分钟级 | warm readiness 探针 |
| R10 | High | 无限反压,延迟随会话时长漂移 | 有界队列 + overrun 策略 |
| R11 | High | ASR 人名/数字错误触发错误外部动作 | 低置信确认 + 禁副作用自动执行 |
| R12 | High | 上下文管理致遗忘或串 session | ContextSnapshot + session epoch |
| R13 | Medium | 音频/transcript 隐私外发 | consent + 加密 + 保留策略 |
| R14 | Medium | 单卡单机故障 | 事件与任务 durability + 快速重启 |

---

## 12. 待决策

| # | 问题 | 建议 |
|---|---|---|
| Q1 | 任务层 API vs 本地 | **先 API** |
| Q2 | app 先 Web 还是原生 | 先 Web 验协议;**iOS 真机测试不得后置** |
| Q3 | 延迟目标数值 | **待 P0 waterfall 出数后设定**,在此之前不承诺 |
| Q4 | 项目名与仓库位置 | 待定 |

---

## 13. 资产

| 资产 | 状态 |
|---|---|
| `openbmb/MiniCPM-o-4_5` 20.05GB | ✅ 已下载校验 54/54 |
| `Soul-AILab/SoulX-Duplug-0.6B` 7.78GB | ✅ 已下载校验 24/24 |
| vllm-omni 源码(参考用) | ✅ 已 clone 至 scratchpad |

> **下载注意:** `HF_ENDPOINT=https://hf-mirror.com` 对 **Xet 后端仓库不可用**(308 跳回源站且丢失 `x-linked-etag`/`x-linked-size`,报 `LocalEntryNotFoundError`)。需 `env -u HF_ENDPOINT` 直连,实测 23 MB/s。

### 链接

MiniCPM-o <https://github.com/OpenBMB/MiniCPM-o> · Demo <https://github.com/OpenBMB/MiniCPM-o-Demo> ·
vllm-omni <https://github.com/vllm-project/vllm-omni> · SoulX-Duplug <https://github.com/Soul-AILab/SoulX-Duplug> ·
LiveKit <https://docs.livekit.io/agents/> · TELEVAL <https://github.com/Tele-AI/TELEVAL> ·
Full-Duplex-Bench <https://github.com/DanielLin94144/Full-Duplex-Bench>

---

## 14. 下一步

**P0,1–3 天,不依赖任何架构决策。**

产出串行基线 waterfall。这个数字决定 P2(三阶段编排)的优先级和投入——**它就是流水线相对串行的真实收益**。
