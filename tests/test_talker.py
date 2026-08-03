"""Talker CPU 冒烟:随机小模型,验证条件化/前向/采样回路(不载真实权重)。"""

from __future__ import annotations

import torch

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
