"""compat 垫片单测:CPU 即可,不需要 GPU。"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from channellm.models.minicpmo_compat import (  # noqa: E402
    TTS_CONFIG_DEFAULTS,
    patch_config,
    patch_dynamic_cache,
    patch_encoder_decoder_cache,
    patch_whisper_attention,
)


class FakeTTSConfig:
    pass


class FakeConfig:
    def __init__(self) -> None:
        self.tts_config = FakeTTSConfig()


def test_patch_config_injects_missing_only():
    config = FakeConfig()
    config.tts_config.top_p = 0.5  # 已存在的值不得覆盖
    patch_config(config)
    assert config.tts_config.top_p == 0.5
    assert config.tts_config.top_k == TTS_CONFIG_DEFAULTS["top_k"]
    assert config.tts_config.temperature == TTS_CONFIG_DEFAULTS["temperature"]


def test_dynamic_cache_list_view_roundtrip():
    patch_dynamic_cache()
    patch_dynamic_cache()  # 幂等
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    if hasattr(cache, "key_cache") and not hasattr(DynamicCache, "_channellm_compat"):
        pytest.skip("transformers 4.x 原生接口")
    cache.key_cache = [torch.randn(1, 2, 3, 8) for _ in range(4)]
    cache.value_cache = [torch.randn(1, 2, 3, 8) for _ in range(4)]
    assert len(cache.key_cache) == 4
    assert cache.key_cache[0].numel() > 0
    cache.key_cache[1] = torch.randn(1, 2, 5, 8)
    assert cache.key_cache[1].shape[-2] == 5
    assert [t.shape[-2] for t in cache.value_cache] == [3, 3, 3, 3]
    # 原生 update 路径与视图共存
    cache.update(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8), layer_idx=3)
    assert cache.key_cache[3].shape[-2] == 2


def test_dynamic_cache_get_usable_length():
    patch_dynamic_cache()
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8), layer_idx=0)
    assert cache.get_usable_length(10) == 3


def test_encoder_decoder_cache_legacy_subscript():
    patch_encoder_decoder_cache()
    patch_encoder_decoder_cache()  # 幂等
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    edc = EncoderDecoderCache(DynamicCache(), DynamicCache())
    # 空缓存:legacy 语义 = 无历史,shape[2] 读 0
    assert edc[0][0].shape[2] == 0
    edc.self_attention_cache.update(torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8), layer_idx=0)
    edc.cross_attention_cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), layer_idx=0)
    assert edc[0][0].shape[2] == 7  # self-attn key seq len
    assert edc[0][1].shape[2] == 7
    assert edc[0][2].shape[2] == 5  # cross-attn key seq len


def test_whisper_attention_patch_idempotent():
    patch_whisper_attention()
    patch_whisper_attention()
    from transformers.models.whisper import modeling_whisper

    cls = modeling_whisper.WhisperAttention
    assert getattr(cls, "_channellm_compat", False)
