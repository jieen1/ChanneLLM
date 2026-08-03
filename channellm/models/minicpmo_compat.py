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
