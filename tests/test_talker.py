"""Talker CPU 冒烟:随机小模型,验证条件化/前向/采样回路(不载真实权重)。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from channellm.engine.blocks import TorchListKV, TorchStaticKV
from channellm.engine.talker import Talker, TalkerConfig, TalkerStream, map_tts_key


def tiny_config() -> TalkerConfig:
    return TalkerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=4,
        head_dim=16,
        intermediate_size=128,
        num_text_tokens=512,
        num_audio_tokens=256,
        llm_dim=96,
        max_position_embeddings=512,
        min_new_tokens=2,
        max_new_tokens=12,
        codec_eos_token_id=7,
        audio_bos_token_id=200,
        text_eos_token_id=201,
    )


def test_build_condition_shapes() -> None:
    torch.manual_seed(3)
    talker = Talker(tiny_config())
    ids = torch.tensor([5, 9, 11])
    hidden = torch.randn(3, 96)

    duplex = talker.build_condition(ids, hidden, duplex=True)
    assert duplex.shape == (4, 64)  # 3 条件 + audio_bos

    full = talker.build_condition(ids, hidden, duplex=False)
    assert full.shape == (5, 64)  # 3 条件 + text_eos + audio_bos

    empty = talker.build_condition(torch.tensor([]), torch.zeros(0, 96))
    assert empty.shape == (1, 64)  # 官方 duplex 回退:仅 audio_bos


def test_generate_codec_loop_terminates() -> None:
    torch.manual_seed(11)
    talker = Talker(tiny_config())
    kv = TorchListKV()
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)
    tokens = talker.generate_codec_tokens(ids, hidden, kv, max_new_tokens=8)
    assert 1 <= len(tokens) <= 8
    assert all(0 <= t < 256 for t in tokens)
    assert talker.config.codec_eos_token_id not in tokens  # EOS 不入序列


def test_generate_is_seeded_deterministic() -> None:
    torch.manual_seed(5)
    talker = Talker(tiny_config())
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)
    a = talker.generate_codec_tokens(ids, hidden, TorchListKV(), max_new_tokens=6)
    b = talker.generate_codec_tokens(ids, hidden, TorchListKV(), max_new_tokens=6)
    assert a == b  # seed 42 固定,两次生成一致


def test_static_kv_matches_list_kv_for_prefill_and_decode() -> None:
    torch.manual_seed(23)
    config = tiny_config()
    talker = Talker(config)
    embeds = torch.randn(3, config.hidden_size)

    list_kv = TorchListKV()
    static_kv = TorchStaticKV(
        config.num_hidden_layers,
        max_seq_len=16,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    )
    torch.testing.assert_close(
        talker.forward_embeds(embeds, list_kv),
        talker.forward_embeds(embeds, static_kv),
    )
    step = torch.randn(1, config.hidden_size)
    torch.testing.assert_close(
        talker.forward_embeds(step, list_kv),
        talker.forward_embeds(step, static_kv),
    )
    assert static_kv.length == list_kv.length == 4


def test_stream_keeps_kv_between_units_and_resets_after_eou() -> None:
    torch.manual_seed(13)
    talker = Talker(tiny_config())
    stream = TalkerStream(talker, TorchListKV)
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)

    first = stream.push(ids, hidden)
    assert 1 <= len(first) <= 25  # 首 unit 可提前停止
    assert stream._kv.length > len(first)

    next_chunk = stream.push(ids, hidden)
    assert len(next_chunk) == 25  # 非末 unit 必须形成完整 phrase
    stream.push(ids, hidden, end_of_turn=True)
    assert stream._kv.length == 0
    assert not stream._started


def test_stream_defaults_to_reusable_static_kv() -> None:
    stream = TalkerStream(Talker(tiny_config()))
    initial = stream._kv

    stream.reset()

    assert isinstance(stream._kv, TorchStaticKV)
    assert stream._kv is initial
    assert stream._kv.length == 0


def test_stream_reused_token_buffer_matches_legacy_multi_unit_sampling() -> None:
    """热循环的 buffer 复用不得改变固定 seed 下的 codec 序列或 KV 演进。"""
    torch.manual_seed(31)
    talker = Talker(tiny_config())
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)
    stream = TalkerStream(talker, TorchListKV)
    legacy_kv = TorchListKV()
    legacy_generator = torch.Generator().manual_seed(talker.config.seed)

    expected_first = _legacy_stream_push(
        talker, legacy_kv, legacy_generator, ids, hidden, started=False
    )
    actual_first = stream.push(ids, hidden)
    expected_second = _legacy_stream_push(
        talker, legacy_kv, legacy_generator, ids, hidden, started=True
    )
    actual_second = stream.push(ids, hidden)
    expected_last = _legacy_stream_push(
        talker, legacy_kv, legacy_generator, ids, hidden, started=True, end_of_turn=True
    )
    actual_last = stream.push(ids, hidden, end_of_turn=True)

    assert (actual_first, actual_second, actual_last) == (
        expected_first,
        expected_second,
        expected_last,
    )
    assert stream._kv.length == 0
    assert legacy_kv.length > 0


def test_stream_reuses_same_device_token_buffer_for_each_codec_step() -> None:
    torch.manual_seed(37)
    talker = Talker(tiny_config())
    stream = TalkerStream(talker, TorchListKV)
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)
    original_buffer = stream._codec_token_input

    class _SpyEmbedding(nn.Module):
        def __init__(self, inner: nn.Embedding) -> None:
            super().__init__()
            self.inner = inner
            self.ptrs: list[int] = []

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            self.ptrs.append(token_ids.data_ptr())
            return self.inner(token_ids)

    talker.emb_code = _SpyEmbedding(talker.emb_code)
    out = stream.push(ids, hidden)

    assert out
    assert talker.emb_code.ptrs
    assert set(talker.emb_code.ptrs) == {original_buffer.data_ptr()}
    assert stream._codec_token_input.data_ptr() == original_buffer.data_ptr()


def _legacy_sample_codec(
    talker: Talker,
    logits: torch.Tensor,
    generated: list[int],
    generator: torch.Generator,
    *,
    min_new_tokens: int | None = None,
) -> int:
    """优化前的官方采样步骤，作为固定种子精确回归基线。"""
    cfg = talker.config
    scores = logits.float().unsqueeze(0)
    scores = scores / cfg.temperature
    if generated:
        input_ids = torch.tensor([generated], dtype=torch.long, device=scores.device)
        if input_ids.size(1) > 16:
            input_ids = input_ids.narrow(1, -16, 16)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        alpha = torch.pow(torch.tensor(cfg.repetition_penalty, device=scores.device), freq)
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)
        top_p, top_k = talker._get_warpers()
        scores = top_p(input_ids, scores)
        scores = top_k(input_ids, scores)
    minimum = cfg.min_new_tokens if min_new_tokens is None else min_new_tokens
    if len(generated) < minimum:
        scores[:, cfg.codec_eos_token_id] = float("-inf")
    probs = F.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).view(-1).item())


@torch.no_grad()
def _legacy_stream_push(
    talker: Talker,
    kv: TorchListKV,
    generator: torch.Generator,
    token_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    started: bool,
    end_of_turn: bool = False,
) -> list[int]:
    """buffer 优化前的逐 token 输入路径，仅作固定种子等价基线。"""
    cfg = talker.config
    condition = talker.build_condition(token_ids, hidden_states, duplex=True)
    hidden = talker.forward_embeds(condition, kv)
    logits = talker.head_code(hidden[-1])
    min_frames = 0 if (not started or end_of_turn) else 25
    generated: list[int] = []
    completed_phrase = True
    for _ in range(25):
        token = _legacy_sample_codec(
            talker,
            logits, generated, generator, min_new_tokens=min_frames
        )
        if token == cfg.codec_eos_token_id:
            completed_phrase = False
            break
        generated.append(token)
        token_embed = talker.emb_code(torch.tensor([token], dtype=torch.long))
        hidden = talker.forward_embeds(token_embed, kv)
        logits = talker.head_code(hidden[-1])
    if completed_phrase:
        _legacy_sample_codec(talker, logits, generated, generator, min_new_tokens=min_frames)
    return generated


def test_map_tts_key() -> None:
    assert map_tts_key("tts.emb_text.weight") == "emb_text.weight"
    assert map_tts_key("tts.model.layers.3.self_attn.q_proj.weight") == (
        "layers.3.self_attn.q_proj.weight"
    )
    assert map_tts_key("tts.model.embed_tokens.weight") == "model_embed.weight"
    assert map_tts_key("tts.emb_code.0.weight") == "emb_code.weight"
    assert map_tts_key("tts.head_code.0.parametrizations.weight.original0") == (
        "head_code.__weight_norm_g"
    )
    assert map_tts_key("tts.projector_spk.linear1.weight") is None
    assert map_tts_key("llm.lm_head.weight") is None


def _fixed_sampler(talker: Talker, tokens: list[int]) -> None:
    """用固定序列替换采样,隔离 streaming 切分语义(不受随机 EOS 干扰)。"""
    seq = iter(tokens)
    talker._sample_codec = (
        lambda _logits, _generated, _generator, min_new_tokens=None: next(seq)
    )


def test_push_streaming_yields_early_first_phrase_segment() -> None:
    torch.manual_seed(41)
    talker = Talker(tiny_config())
    stream = TalkerStream(talker, TorchListKV, early_first_frames=5)
    _fixed_sampler(talker, [10 + (i % 50) for i in range(40)])
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)

    emissions = list(stream.push_streaming(ids, hidden))

    assert [len(part) for part, _last in emissions] == [5, 20]
    assert [last for _part, last in emissions] == [False, True]


def test_push_streaming_second_unit_is_single_emission() -> None:
    torch.manual_seed(41)
    talker = Talker(tiny_config())
    stream = TalkerStream(talker, TorchListKV, early_first_frames=5)
    _fixed_sampler(talker, [10 + (i % 50) for i in range(80)])
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)

    first = list(stream.push_streaming(ids, hidden))
    second = list(stream.push_streaming(ids, hidden))

    assert [len(part) for part, _last in first] == [5, 20]
    assert [(len(part), last) for part, last in second] == [(25, True)]


def test_push_streaming_short_phrase_stays_single_emission() -> None:
    torch.manual_seed(41)
    config = tiny_config()
    talker = Talker(config)
    stream = TalkerStream(talker, TorchListKV, early_first_frames=5)
    # 第 3 帧即 EOS(7):未越过提前阈值,必须整段一次交出。
    _fixed_sampler(talker, [10, 11, config.codec_eos_token_id])
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)

    emissions = list(stream.push_streaming(ids, hidden, end_of_turn=True))

    assert emissions == [([10, 11], True)]


def test_push_matches_push_streaming_flatten_under_fixed_seed() -> None:
    torch.manual_seed(43)
    talker = Talker(tiny_config())
    ids = torch.tensor([5, 9])
    hidden = torch.randn(2, 96)
    early = TalkerStream(talker, TorchListKV, early_first_frames=5)
    plain = TalkerStream(talker, TorchListKV)

    assert early.push(ids, hidden) == plain.push(ids, hidden)
    assert early.push(ids, hidden, end_of_turn=True) == plain.push(
        ids, hidden, end_of_turn=True
    )
