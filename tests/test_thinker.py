"""Thinker CPU 冒烟:随机小模型,验证前向/生成/KV 契约(不载真实权重)。"""

from __future__ import annotations

import torch

from channellm.engine.thinker import Thinker, ThinkerConfig, TorchListKV, map_official_key


def tiny_config() -> ThinkerConfig:
    return ThinkerConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
    )


def test_forward_shapes_and_determinism() -> None:
    torch.manual_seed(3)
    model = Thinker(tiny_config())
    kv = TorchListKV()
    ids = torch.tensor([1, 5, 7, 9])
    logits1 = model(ids, kv)
    assert logits1.shape == (4, 256)
    assert kv.length == 4


def test_generate_greedy_respects_eos() -> None:
    torch.manual_seed(11)
    model = Thinker(tiny_config())
    kv = TorchListKV()
    out = model.generate_greedy([2, 3], n_new=8, kv=kv, eos_token_id=-1)
    assert len(out) == 8
    assert kv.length == 2 + 8  # 产出 token 全部 forward 过,KV 状态完整


def test_prefill_then_decode_matches_full_prefill() -> None:
    """[a,b,c] 整体 prefill 与 [a,b] prefill + [c] decode 的末位 logits 一致。"""
    torch.manual_seed(5)
    config = tiny_config()
    model = Thinker(config)

    kv_full = TorchListKV()
    logits_full = model(torch.tensor([10, 20, 30]), kv_full)

    kv_split = TorchListKV()
    model(torch.tensor([10, 20]), kv_split)
    logits_step = model(torch.tensor([30]), kv_split)

    torch.testing.assert_close(logits_full[-1], logits_step[-1], atol=1e-4, rtol=1e-4)


def test_map_official_key() -> None:
    assert map_official_key("llm.lm_head.weight") == "lm_head.weight"
    assert (
        map_official_key("llm.model.layers.3.self_attn.q_norm.weight")
        == "layers.3.self_attn.q_norm.weight"
    )
    assert (
        map_official_key("llm.model.layers.35.mlp.down_proj.weight")
        == "layers.35.mlp.down_proj.weight"
    )
    assert map_official_key("llm.model.layers.0.input_layernorm.weight") == (
        "layers.0.input_layernorm.weight"
    )
    assert map_official_key("apm.encoder.layers.0.self_attn.q_proj.weight") is None
    assert map_official_key("tts.whatever") is None
