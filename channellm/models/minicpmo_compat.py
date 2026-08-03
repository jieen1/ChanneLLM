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
