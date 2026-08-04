"""Talker 自研前向 —— Llama 骨干 codec 生成器(P1)。

结构对照 vllm-omni ``MiniCPMO45OmniTTSForConditionalGeneration`` 与官方
``MiniCPMTTS``(权重内 modeling 4204 行):20 层 Llama(hidden 768、12 头
MHA、head_dim 64、rope_theta 1e4),条件化走 hidden_text_merge ——
``emb_text(ids) + L2norm(projector_semantic(thinker_hidden))``,尾部拼
``<audio_bos>``(duplex 流式口径);AR 采样 codec token(temp 0.8 / top_k 25 /
top_p 0.85 / rep_penalty 1.05,前 50 步屏蔽 EOS=625,seed 42)。

head_code 的 checkpoint 是 torch parametrizations weight_norm 的 g/v 两半,
装载时用 ``torch._weight_norm(v, g, dim=0)`` 融合(vllm-omni 同款)。
``projector_spk`` 参考音频路在 Code2Wav 侧处理,本模块不装载。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from channellm.engine.blocks import (
    MLP,
    KVBackend,
    RMSNorm,
    RotaryEmbedding,
    TorchStaticKV,
    rotate_half,
)


@dataclasses.dataclass
class TalkerConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 20
    num_heads: int = 12
    num_kv_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e4
    max_position_embeddings: int = 4096
    num_text_tokens: int = 152064
    num_audio_tokens: int = 6562
    llm_dim: int = 4096
    normalize_projected_hidden: bool = True
    audio_bos_token_id: int = 151687
    text_eos_token_id: int = 151692
    codec_eos_token_id: int = 6561  # 非流式 = num_audio_tokens-1;625 是 duplex 流式专用
    temperature: float = 0.8
    top_k: int = 25
    top_p: float = 0.85
    repetition_penalty: float = 1.05
    min_new_tokens: int = 50
    max_new_tokens: int = 2048
    seed: int = 42

    @classmethod
    def from_official(cls, config_path: str | Path) -> TalkerConfig:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        tts = raw["tts_config"]
        return cls(
            hidden_size=tts["hidden_size"],
            num_hidden_layers=tts["num_hidden_layers"],
            num_heads=tts["num_attention_heads"],
            num_kv_heads=tts["num_key_value_heads"],
            head_dim=tts["hidden_size"] // tts["num_attention_heads"],
            intermediate_size=tts["intermediate_size"],
            max_position_embeddings=tts["max_position_embeddings"],
            num_text_tokens=tts["num_text_tokens"],
            num_audio_tokens=tts["num_audio_tokens"],
            codec_eos_token_id=tts["num_audio_tokens"] - 1,
            llm_dim=tts["llm_dim"],
            normalize_projected_hidden=tts.get("normalize_projected_hidden", True),
            audio_bos_token_id=tts["audio_bos_token_id"],
            text_eos_token_id=tts["text_eos_token_id"],
            temperature=tts.get("temperature", 0.8),
            top_k=tts.get("top_k", 25),
            top_p=tts.get("top_p", 0.85),
            repetition_penalty=tts.get("repetition_penalty", 1.05),
            min_new_tokens=tts.get("min_new_tokens", 50),
        )


class TalkerAttention(nn.Module):
    """Llama 风格 GQA attention(无 q/k norm)。"""

    def __init__(self, config: TalkerConfig, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_heads * config.head_dim, bias=False,
            device=device, dtype=dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False,
            device=device, dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_kv_heads * config.head_dim, bias=False,
            device=device, dtype=dtype,
        )
        self.o_proj = nn.Linear(
            config.num_heads * config.head_dim, config.hidden_size, bias=False,
            device=device, dtype=dtype,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: KVBackend,
        layer_idx: int,
    ) -> torch.Tensor:
        seq_len = hidden.shape[0]
        cfg = self.config
        q = self.q_proj(hidden).view(seq_len, cfg.num_heads, cfg.head_dim)
        k = self.k_proj(hidden).view(seq_len, cfg.num_kv_heads, cfg.head_dim)
        v = self.v_proj(hidden).view(seq_len, cfg.num_kv_heads, cfg.head_dim)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        kv.append_layer(layer_idx, k, v)
        out = kv.attend(layer_idx, q)
        return self.o_proj(out.reshape(seq_len, cfg.num_heads * cfg.head_dim))


class TTSProjector(nn.Module):
    """官方 MultiModalProjector 同构:linear1 -> ReLU -> linear2。"""

    def __init__(self, in_dim: int, out_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class TalkerLayer(nn.Module):
    def __init__(self, config: TalkerConfig, device=None, dtype=None) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.self_attn = TalkerAttention(config, device, dtype)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device, dtype
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size, device, dtype)

    def forward(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        kv: KVBackend, layer_idx: int,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, kv, layer_idx)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Talker(nn.Module):
    def __init__(
        self,
        config: TalkerConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [TalkerLayer(config, device, dtype) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.emb_text = nn.Embedding(
            config.num_text_tokens, config.hidden_size, device=device, dtype=dtype
        )
        self.projector_semantic = TTSProjector(config.llm_dim, config.hidden_size, device, dtype)
        self.emb_code = nn.Embedding(
            config.num_audio_tokens, config.hidden_size, device=device, dtype=dtype
        )
        self.head_code = nn.Linear(
            config.hidden_size, config.num_audio_tokens, bias=False, device=device, dtype=dtype
        )
        # checkpoint 里的 tts.model.embed_tokens(LLama 自带,codec 路径不用,
        # 装载权重保完整性)
        self.model_embed = nn.Embedding(32000, config.hidden_size, device=device, dtype=dtype)
        self.rotary = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        ).to(device)

    def build_condition(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        duplex: bool = True,
    ) -> torch.Tensor:
        """hidden_text_merge 条件化(vllm-omni 口径)。

        token_ids: [N] 文本 token;hidden_states: [N, llm_dim] thinker 隐层。
        duplex=True 尾部拼 <audio_bos>;否则拼 <text_eos><audio_bos> 边界。
        """
        cfg = self.config
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        ids = token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = hidden_states.to(device=device, dtype=dtype)
        if ids.numel() == 0 or hidden.numel() == 0:
            # 对齐官方 MiniCPMODuplex._convert_results_to_tts_input：没有
            # thinker 语义 token 时只能以 audio_bos 作为条件。text_eos 属于
            # 非流式「有文本条件」的收束边界，不能凭空插入，否则会改变首帧
            # codec 分布并污染静音/短回复回退路径。
            return self.emb_text(torch.tensor([cfg.audio_bos_token_id], device=device))
        text_embeds = self.emb_text(ids)
        hidden_embeds = self.projector_semantic(hidden)
        if cfg.normalize_projected_hidden:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        condition = text_embeds + hidden_embeds
        tail = (
            [cfg.audio_bos_token_id]
            if duplex
            else [cfg.text_eos_token_id, cfg.audio_bos_token_id]
        )
        tail_embeds = self.emb_text(torch.tensor(tail, device=device))
        return torch.cat([condition, tail_embeds], dim=0)

    @torch.no_grad()
    def forward_embeds(
        self,
        embeds: torch.Tensor,
        kv: KVBackend,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """embeds: [S, hidden] -> 末层归一化隐层 [S, hidden]。"""
        seq_len = embeds.shape[0]
        kv.begin_step(seq_len)
        if positions is None:
            positions = torch.arange(
                kv.prefix_len, kv.prefix_len + seq_len, device=embeds.device
            )
        cos, sin = self.rotary(positions, dtype=self.emb_text.weight.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        hidden = embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, kv, layer_idx)
        kv.commit()
        return self.norm(hidden)

    def _get_warpers(self):
        if getattr(self, "_warper_top_p", None) is None:
            from transformers.generation.logits_process import (
                TopKLogitsWarper,
                TopPLogitsWarper,
            )

            self._warper_top_p = TopPLogitsWarper(self.config.top_p, min_tokens_to_keep=3)
            self._warper_top_k = TopKLogitsWarper(self.config.top_k, min_tokens_to_keep=3)
        return self._warper_top_p, self._warper_top_k

    def _sample_codec(
        self,
        logits: torch.Tensor,
        generated: list[int],
        generator: torch.Generator,
        min_new_tokens: int | None = None,
    ) -> int:
        """官方非流式采样序列:temperature → 窗口 16 频次式重复惩罚 →
        top_p → top_k → 前 min_new_tokens 步屏蔽 EOS → softmax → multinomial。"""
        cfg = self.config
        scores = logits.float().unsqueeze(0)
        scores = scores / cfg.temperature
        if generated:
            input_ids = torch.tensor([generated], dtype=torch.long, device=scores.device)
            if input_ids.size(1) > 16:
                input_ids = input_ids.narrow(1, -16, 16)
            freq = F.one_hot(input_ids, scores.size(1)).sum(1)
            alpha = torch.pow(torch.tensor(cfg.repetition_penalty, device=scores.device), freq)
            inp = scores * alpha
            oth = scores / alpha
            scores = torch.where(scores < 0, inp, oth)
            top_p, top_k = self._get_warpers()
            scores = top_p(input_ids, scores)
            scores = top_k(input_ids, scores)
        minimum = cfg.min_new_tokens if min_new_tokens is None else min_new_tokens
        if len(generated) < minimum:
            scores[:, cfg.codec_eos_token_id] = float("-inf")
        probs = F.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).view(-1).item())

    @torch.no_grad()
    def generate_codec_tokens(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv: KVBackend,
        max_new_tokens: int | None = None,
        duplex: bool = True,
    ) -> list[int]:
        """条件化 + AR 采样 codec token 序列(含 EOS 则不含 EOS)。"""
        cfg = self.config
        device = self.emb_text.weight.device
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        cond = self.build_condition(token_ids, hidden_states, duplex=duplex)
        hidden = self.forward_embeds(cond, kv)
        logits = self.head_code(hidden[-1])

        out: list[int] = []
        limit = max_new_tokens or cfg.max_new_tokens
        next_id = self._sample_codec(logits, out, generator)
        for _ in range(limit):
            out.append(next_id)
            if next_id == cfg.codec_eos_token_id:
                break
            embeds = self.emb_code(torch.tensor([next_id], device=device)).squeeze(0).unsqueeze(0)
            hidden = self.forward_embeds(embeds, kv)
            logits = self.head_code(hidden[-1])
            next_id = self._sample_codec(logits, out, generator)
        if out and out[-1] == cfg.codec_eos_token_id:
            out.pop()
        return out


class TalkerStream:
    """以官方 ``MiniCPMTTS.generate_chunk`` 口径续写单回合 codec。

    每个 Thinker unit 都把新语义条件和 ``audio_bos`` 追加到同一 KV，再生成
    一个可交给 Code2Wav 的 phrase。正常 unit 强制 25 帧，首 unit 和 EOU
    unit 允许不足 25 帧，以避免首包和收尾被无意义填充拖慢。

    ``kv_factory`` 是显式注入的：不同回合绝不复用 TTS KV，barge-in 时调用
    ``reset`` 即可同步丢弃旧回合状态，而不需要等待 GPU 旧请求完成。
    """

    def __init__(self, talker: Talker, kv_factory=None, early_first_frames: int = 0) -> None:
        """early_first_frames>0 时,回合首个 phrase 生成到该帧数即先 yield 一次
        (官方首次 TTS force_flush 的单次提前语义),其余帧在 phrase 结束时
        yield;其他 unit 仍整 phrase 一次 yield。0 = 关闭。"""
        if early_first_frames < 0:
            raise ValueError("early_first_frames must be non-negative")
        self.talker = talker
        self._kv_factory = kv_factory or _static_kv_factory(talker)
        self._kv: KVBackend
        self._generator: torch.Generator
        self._codec_token_input = torch.empty(
            1,
            dtype=torch.long,
            device=talker.emb_text.weight.device,
        )
        self._early_first_frames = int(early_first_frames)
        self._started = False
        self.reset()

    def reset(self) -> None:
        previous = getattr(self, "_kv", None)
        reset = getattr(previous, "reset", None)
        if callable(reset):
            reset()
        self._kv = self._kv_factory()
        self._generator = torch.Generator(device=self.talker.emb_text.weight.device)
        self._generator.manual_seed(self.talker.config.seed)
        self._started = False

    @torch.no_grad()
    def push(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        end_of_turn: bool = False,
    ) -> list[int]:
        """生成本 unit 的 codec 帧；EOU 后立即释放该回合 KV。

        与 ``push_streaming`` 的逐位关系:flatten 后完全一致(同一生成核、
        同一 RNG/KV 次序)。
        """
        frames: list[int] = []
        for part, _is_last in self.push_streaming(
            token_ids, hidden_states, end_of_turn=end_of_turn
        ):
            frames.extend(part)
        return frames

    @torch.no_grad()
    def push_streaming(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        end_of_turn: bool = False,
    ):
        """增量生成本 unit 的 codec 帧(惰性 generator)。

        yield ``(frames, is_last)`` 对。回合首个 phrase(官方首次 TTS
        force_flush 语义)在确认 phrase 越过 ``early_first_frames`` 阈值时
        (即第 early+1 帧生成后)先 yield 前 ``early_first_frames`` 帧,调用方
        可立刻交给 Code2Wav,其余帧在 generator 恢复时继续生成并在 phrase
        结束(EOS 或满 25 帧)时 yield 尾段(is_last=True);其余 unit 整
        phrase 一次 yield。延迟一帧确认保证尾段永不为空、final 总能附在
        最后一个音频块上。提前交接不改变采样顺序、generator 推进与 KV
        写入次序。调用方必须把 generator 跑到耗尽(或随后调用 reset),
        否则回合状态(_started/EOT 复位)不落地。
        """
        cfg = self.talker.config
        condition = self.talker.build_condition(token_ids, hidden_states, duplex=True)
        hidden = self.talker.forward_embeds(condition, self._kv)
        logits = self.talker.head_code(hidden[-1])

        # 官方输入 max_new_token=26，返回前 25 个 token，同时用第 26 次
        # forward 把第 25 帧写进 KV。这里直接执行等价的 25 次“采样→写 KV”。
        target_frames = 25
        min_frames = 0 if (not self._started or end_of_turn) else target_frames
        early_at = (
            self._early_first_frames
            if not self._started and 0 < self._early_first_frames < target_frames
            else 0
        )
        generated: list[int] = []
        emitted_early = False
        completed_phrase = True
        for _ in range(target_frames):
            token = self.talker._sample_codec(
                logits,
                generated,
                self._generator,
                min_new_tokens=min_frames,
            )
            if token == cfg.codec_eos_token_id:
                completed_phrase = False
                break
            generated.append(token)
            # 25-frame phrase 的每一步都需要把采样 token 喂回 Talker。复用这个
            # 单元素设备 buffer，避免每步分配一个新的 CUDA tensor；采样顺序、
            # generator 与 KV 写入次序保持不变。
            self._codec_token_input.fill_(token)
            token_embed = self.talker.emb_code(self._codec_token_input)
            hidden = self.talker.forward_embeds(token_embed, self._kv)
            logits = self.talker.head_code(hidden[-1])
            if early_at and not emitted_early and len(generated) == early_at + 1:
                emitted_early = True
                yield list(generated[:early_at]), False

        if completed_phrase:
            # 官方在第 26 次 decode 时会采样一个不返回的 lookahead token。
            # 它不写 KV，却会推进 RNG；保留这一步使下一 unit 的采样序列与
            # 官方 generate_chunk 的状态机一致。
            self.talker._sample_codec(
                logits,
                generated,
                self._generator,
                min_new_tokens=min_frames,
            )

        emitted = early_at if emitted_early else 0
        yield list(generated[emitted:]), True

        self._started = True
        if end_of_turn:
            self.reset()


def _static_kv_factory(talker: Talker):
    """为单个 Talker stream 持有一个可逻辑复位的连续 KV 缓冲。"""
    cfg = talker.config
    kv: TorchStaticKV | None = None

    def acquire() -> TorchStaticKV:
        nonlocal kv
        if kv is None:
            kv = TorchStaticKV(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
                cfg.num_kv_heads,
                cfg.head_dim,
                device=talker.emb_text.weight.device,
                dtype=talker.emb_text.weight.dtype,
            )
        return kv

    return acquire


# ---------------------------------------------------------------------------
# 权重装载
# ---------------------------------------------------------------------------


def map_tts_key(key: str) -> str | None:
    """官方 safetensors tts.* 键 -> 本模块参数路径;不装载的键返回 None。"""
    if not key.startswith("tts."):
        return None
    rest = key[len("tts."):]
    if rest.startswith("projector_spk."):
        return None  # 参考音频路由 Code2Wav 处理,本模块不装载
    if rest == "emb_code.0.weight":
        return "emb_code.weight"  # num_vq=1,解包 ModuleList 索引
    if rest == "head_code.0.parametrizations.weight.original0":
        return "head_code.__weight_norm_g"
    if rest == "head_code.0.parametrizations.weight.original1":
        return "head_code.__weight_norm_v"
    if rest.startswith("model."):
        inner = rest[len("model."):]
        if inner == "embed_tokens.weight":
            return "model_embed.weight"
        return inner
    return rest


def load_talker_weights(
    model_dir: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    config: TalkerConfig | None = None,
) -> Talker:
    """从官方 checkpoint 流式装载 tts.* 权重;head_code 融合 weight_norm。"""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    config = config or TalkerConfig.from_official(model_dir / "config.json")
    model = Talker(config, device=device, dtype=dtype)

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard_keys: dict[str, list[tuple[str, str]]] = {}
    for key, shard in index["weight_map"].items():
        mapped = map_tts_key(key)
        if mapped is not None:
            shard_keys.setdefault(shard, []).append((key, mapped))

    state = dict(model.named_parameters())
    pending_g: torch.Tensor | None = None
    pending_v: torch.Tensor | None = None
    loaded = 0
    with torch.no_grad():
        for shard, pairs in shard_keys.items():
            with safe_open(str(model_dir / shard), framework="pt") as fh:
                for src, dst in pairs:
                    tensor = fh.get_tensor(src).to(device=device, dtype=dtype)
                    if dst == "head_code.__weight_norm_g":
                        pending_g = tensor
                    elif dst == "head_code.__weight_norm_v":
                        pending_v = tensor
                    else:
                        state[dst].copy_(tensor)
                        loaded += 1
                    del tensor
    if pending_g is None or pending_v is None:
        raise RuntimeError("head_code weight_norm g/v 未装齐")
    with torch.no_grad():
        fused = torch._weight_norm(pending_v, pending_g, dim=0)
        model.head_code.weight.copy_(fused.to(dtype))
    loaded += 1

    expected = len(state)
    if loaded != expected:
        raise RuntimeError(f"Talker 权重未装齐:装 {loaded} 项,应 {expected} 项")
    return model.eval()
