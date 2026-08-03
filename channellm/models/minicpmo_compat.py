"""官方 MiniCPM-o 代码与 transformers 5 的兼容垫片(P0)。

事实:权重内 configuration_minicpmo.py 的 MiniCPMTTSConfig 未声明
top_p/top_k/temperature/repetition_penalty,config.json 的 tts_config
也没有这些键;modeling_minicpmo.py:4225 起却直接读取它们。
transformers 4.x 对缺失属性宽容,5.x 的 PretrainedConfig.__getattribute__
会显式抛 AttributeError —— 这是官方代码按 transformers==4.51.0 开发留下
的口子(设计文档 R5/风险登记:官方代码与上游版本漂移)。

策略:加载前把缺失键以官方 streaming_generate 默认值注入 tts_config,
不改动权重目录内任何文件。每发现一个新的不兼容点都登记在这里;
若垫片数量失控,按 docs/p0-runbook.md 退回 transformers 4.51 独立 venv。
"""

from __future__ import annotations

from typing import Any

import torch

# 默认值取自官方 streaming_generate 签名(modeling_minicpmo.py:3151)
TTS_CONFIG_DEFAULTS: dict[str, Any] = {
    "top_p": 0.8,
    "top_k": 100,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
}


def patch_config(config: Any) -> Any:
    """就地补全 tts_config 缺失字段;其它字段一律不动。"""
    tts_config = getattr(config, "tts_config", None)
    if tts_config is None:
        return config
    for key, value in TTS_CONFIG_DEFAULTS.items():
        if not hasattr(tts_config, key):
            setattr(tts_config, key, value)
    return config


def patch_model_class(model_cls: type) -> type:
    """MiniCPMO.__init__ 不调用 post_init();transformers 5 依赖 post_init
    注册的 all_tied_weights_keys 做加载收尾。这里在 __init__ 之后补注册,
    不触发权重初始化(权重随后被 checkpoint 覆盖)。"""
    original_init = model_cls.__init__

    def patched_init(self: Any, config: Any) -> None:
        original_init(self, config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(
                all_submodels=False
            )

    model_cls.__init__ = patched_init
    return model_cls

# ---------------------------------------------------------------------------
# DynamicCache.key_cache / value_cache 兼容层
#
# 官方 utils.py/modeling_minicpmo.py 共 38 处按 transformers 4.x 的
# DynamicCache 接口读写 key_cache/value_cache(list of per-layer tensors);
# transformers 5 改成 .layers: list[DynamicLayer](layer.keys/.values)。
# 官方重赋值语义经核实均为"全层替换、两列表等长"(窗口截取/rotary 重对齐/
# 拼接),因此 setter 可安全地按给定列表重建 layers。
# ---------------------------------------------------------------------------


class _CacheTensorListView:
    """让 cache.key_cache / cache.value_cache 继续按 4.x 的列表语义使用。"""

    def __init__(self, cache: Any, which: str) -> None:
        self._cache = cache
        self._which = which  # 'keys' or 'values'

    def __len__(self) -> int:
        return len(self._cache.layers)

    def __iter__(self):
        for layer in self._cache.layers:
            yield getattr(layer, self._which)

    def __getitem__(self, idx):
        return getattr(self._cache.layers[idx], self._which)

    def __setitem__(self, idx, tensor) -> None:
        setattr(self._cache.layers[idx], self._which, tensor)

    def __bool__(self) -> bool:
        return len(self._cache.layers) > 0


def _rebuild_layers(cache: Any, tensors, which: str) -> None:
    """全层替换。官方重赋值语义经核实均为等长列表(窗口截取/rotary 重对齐/拼接)。"""
    from transformers.cache_utils import DynamicLayer

    tensors = list(tensors)
    layers = cache.layers
    while len(layers) < len(tensors):
        layers.append(DynamicLayer())
    del layers[len(tensors):]
    for layer, tensor in zip(layers, tensors):
        setattr(layer, which, tensor)


def patch_dynamic_cache() -> None:
    """给 transformers 5 的 DynamicCache 补回 4.x 的 key/value_cache 接口。

    transformers 4.x 上实例自带 key_cache 属性,检测到即跳过,保持幂等。
    """
    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache(), "key_cache"):
        return  # transformers 4.x:原生接口
    if getattr(DynamicCache, "_channellm_compat", False):
        return

    def key_cache_getter(self):
        return _CacheTensorListView(self, "keys")

    def key_cache_setter(self, tensors):
        _rebuild_layers(self, tensors, "keys")

    def value_cache_getter(self):
        return _CacheTensorListView(self, "values")

    def value_cache_setter(self, tensors):
        _rebuild_layers(self, tensors, "values")

    DynamicCache.key_cache = property(key_cache_getter, key_cache_setter)
    DynamicCache.value_cache = property(value_cache_getter, value_cache_setter)
    DynamicCache._channellm_compat = True

    if not hasattr(DynamicCache, "get_usable_length"):
        # 4.x 语义:无 sliding window 时即该层已缓存长度(Whisper encoder 用)
        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length


def patch_whisper_attention() -> None:
    """transformers 5 的 WhisperAttention.forward 返回 2 元组(不再带 cache);
    官方 audio encoder 按 4.x 解包 3 元组。5.x 的 cache 是原地更新,
    把传入的 past_key_value 补回第三位即可,幂等。"""
    try:
        from transformers.models.whisper import modeling_whisper
    except ImportError:
        return

    classes: list[type] = []
    mapping = getattr(modeling_whisper, "WHISPER_ATTENTION_CLASSES", None)
    if isinstance(mapping, dict):
        classes.extend(mapping.values())
    for name in ("WhisperAttention", "WhisperFlashAttention2", "WhisperSdpaAttention"):
        cls = getattr(modeling_whisper, name, None)
        if cls is not None:
            classes.append(cls)

    seen: set[type] = set()
    for cls in classes:
        if cls in seen or getattr(cls, "_channellm_compat", False):
            continue
        seen.add(cls)
        original_forward = cls.forward

        def make_wrapped(orig):
            def wrapped(self, *args, **kwargs):
                result = orig(self, *args, **kwargs)
                if isinstance(result, tuple) and len(result) == 2:
                    return (result[0], result[1], kwargs.get("past_key_value"))
                return result

            return wrapped

        cls.forward = make_wrapped(original_forward)
        cls._channellm_compat = True


class _LegacyEncoderDecoderLayerView:
    """legacy 4.x 布局按层索引:cache[layer_idx] ->
    (self_k, self_v, cross_k, cross_v)。"""

    def __init__(self, self_attn: Any, cross_attn: Any, layer_idx: int) -> None:
        self._self_attn = self_attn
        self._cross_attn = cross_attn
        self._layer_idx = layer_idx

    def _layer_tensor(self, cache: Any, which: str):
        layers = cache.layers
        if self._layer_idx >= len(layers):
            return torch.empty((0, 0, 0, 0))  # 空缓存 = 无历史,长度读 0
        tensor = getattr(layers[self._layer_idx], which, None)
        return tensor if tensor is not None else torch.empty((0, 0, 0, 0))

    def __getitem__(self, idx: int):
        if idx == 0:
            return self._layer_tensor(self._self_attn, "keys")
        if idx == 1:
            return self._layer_tensor(self._self_attn, "values")
        if idx == 2:
            return self._layer_tensor(self._cross_attn, "keys")
        if idx == 3:
            return self._layer_tensor(self._cross_attn, "values")
        raise IndexError(idx)


def patch_encoder_decoder_cache() -> None:
    """官方 get_audio_embedding_streaming 用 legacy 下标读缓存长度
    (audio_past_key_values[0][0].shape[2]);transformers 5 的
    EncoderDecoderCache 不可下标。补只读视图,4.x 已可下标则跳过,幂等。"""
    from transformers.cache_utils import EncoderDecoderCache

    probe = EncoderDecoderCache.__new__(EncoderDecoderCache)
    try:
        _ = probe[0]  # noqa: F841 - 探测下标能力
        return
    except (TypeError, IndexError, AttributeError):
        pass
    if getattr(EncoderDecoderCache, "_channellm_compat", False):
        return

    def getitem(self, idx: int):
        return _LegacyEncoderDecoderLayerView(
            self.self_attention_cache, self.cross_attention_cache, idx
        )

    EncoderDecoderCache.__getitem__ = getitem
    EncoderDecoderCache._channellm_compat = True


def patch_torchaudio_load() -> None:
    """torchaudio>=2.11 默认走 torchcodec 后端,需要系统 ffmpeg;SM120 机器
    没有 ffmpeg 库。wav 文件用 soundfile 兜底即可覆盖 P0 全部路径。"""
    try:
        import torchaudio
    except ImportError:
        return
    if getattr(torchaudio, "_channellm_compat", False):
        return
    original_load = torchaudio.load

    def patched_load(file, *args, **kwargs):
        try:
            return original_load(file, *args, **kwargs)
        except (ImportError, OSError):
            if str(file).lower().endswith(".wav"):
                import soundfile as sf

                data, sample_rate = sf.read(file, dtype="float32", always_2d=True)
                return torch.from_numpy(data.T), sample_rate
            raise

    torchaudio.load = patched_load
    torchaudio._channellm_compat = True


def patch_torchaudio_save() -> None:
    """torchaudio>=2.11 的 save 同样默认 torchcodec(需系统 ffmpeg)。
    wav 写出用 soundfile 兜底:支持路径与 BytesIO,覆盖 Token2wav 的
    ``torchaudio.save(output, wav, 24000, format='wav')`` 调用。"""
    try:
        import torchaudio
    except ImportError:
        return
    if getattr(torchaudio, "_channellm_compat_save", False):
        return
    original_save = torchaudio.save

    def patched_save(filepath_or_obj, src, sample_rate, *args, **kwargs):
        fmt = kwargs.get("format")
        target = str(filepath_or_obj).lower() if not hasattr(filepath_or_obj, "write") else ""
        if fmt in (None, "wav") or target.endswith(".wav"):
            import soundfile as sf

            data = src.detach().float().cpu().numpy()
            if data.ndim == 2:
                data = data.T  # [channels, samples] -> [samples, channels]
            sf.write(filepath_or_obj, data, int(sample_rate), format="WAV")
            return
        return original_save(filepath_or_obj, src, sample_rate, *args, **kwargs)

    torchaudio.save = patched_save
    torchaudio._channellm_compat_save = True
