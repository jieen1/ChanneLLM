"""音频前端 —— 官方流式音频编码器混合封装(P1 语音输入)。

策略与 vllm-omni stage0 一致:音频编码器(Whisper 系 apm + projection +
avgpool)不是逐 token 热路径,直接复用官方 ``MiniCPMO.get_audio_embedding_streaming``
与 processor 的 ``process_audio_streaming``,只装载音频相关权重
(``apm.*`` 367 键 + ``audio_projection_layer.*`` 4 键),LLM/vision/tts 不装载。

每 1s chunk 产出若干 4096 维 audio embedding,由自研 Thinker 的
``forward_embeds`` 喂入 paged KV —— duplex 协议里每个 chunk 前还要拼一个
``<unit>`` token embedding(官方 streaming_prefill 同款)。

cnn_redundancy_ms 默认 20 对齐 MiniCPMODuplex 参数表(官方模型属性是 0,
duplex 包装层覆盖为 20)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class AudioFront:
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cnn_redundancy_ms: int = 20,
    ) -> None:
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        from channellm.models.minicpmo_compat import (
            patch_config,
            patch_dynamic_cache,
            patch_encoder_decoder_cache,
            patch_model_class,
            patch_torchaudio_load,
            patch_whisper_attention,
        )

        model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype

        patch_dynamic_cache()
        patch_whisper_attention()
        patch_encoder_decoder_cache()
        patch_torchaudio_load()

        config = patch_config(
            AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        )
        # 只建音频面:LLM 在 meta 上零开销占位,永不触碰
        config.init_vision = False
        config.init_tts = False
        config.init_audio = True

        model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], str(model_dir))
        patch_model_class(model_cls)
        with torch.device("meta"):
            self.model = model_cls(config)

        self._load_audio_weights(model_dir)
        self.model.audio_encoder_layer = -1
        self.model.audio_past_key_values = None
        self.model.eval()

        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model.prepare_processor(processor=processor, tokenizer=processor.tokenizer)
        # duplex 默认 cnn_redundancy_ms=20(MiniCPMODuplex 参数表),
        # init_streaming_processor 读模型属性,先对齐再初始化。
        self.model.CNN_REDUNDANCY_MS = cnn_redundancy_ms
        self.model.init_streaming_processor()
        self.tokenizer = processor.tokenizer
        self.unit_token_id = self.tokenizer.convert_tokens_to_ids("<unit>")
        self._chunk_idx = 0

    def _load_audio_weights(self, model_dir: Path) -> None:
        from safetensors import safe_open

        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        shards: dict[str, list[str]] = {}
        for key, shard in index["weight_map"].items():
            if key.startswith(("apm.", "audio_projection_layer.")):
                shards.setdefault(shard, []).append(key)

        loaded = 0
        with torch.no_grad():
            for shard, keys in shards.items():
                with safe_open(str(model_dir / shard), framework="pt") as fh:
                    for key in keys:
                        tensor = fh.get_tensor(key).to(device=self.device, dtype=self.dtype)
                        parent_path, _, name = key.rpartition(".")
                        parent = self.model.get_submodule(parent_path)
                        # meta 模型必须 assign 替换,copy_ 不落地
                        parent.register_parameter(
                            name, torch.nn.Parameter(tensor, requires_grad=False)
                        )
                        del tensor
                        loaded += 1
        if loaded == 0:
            raise RuntimeError("音频权重未装载")

    def reset(self) -> None:
        self.model.audio_past_key_values = None
        if hasattr(self.model.processor, "reset_streaming"):
            self.model.processor.reset_streaming()
        self._chunk_idx = 0

    @torch.no_grad()
    def feed_chunk(self, pcm_16k: np.ndarray) -> torch.Tensor:
        """喂一个 16kHz chunk,返回 [n_tokens, 4096] audio embedding。"""
        batch_feature = self.model.processor.process_audio_streaming(
            pcm_16k, reset=False, return_batch_feature=True
        )
        if hasattr(batch_feature, "to"):
            batch_feature = batch_feature.to(self.device)
        prefix_frames = 0 if self._chunk_idx == 0 else 2
        embeds_nested = self.model.get_audio_embedding_streaming(
            batch_feature,
            use_extra_context=True,
            prefix_extra_frames=prefix_frames,
            suffix_extra_frames=2,
        )
        self._chunk_idx += 1
        tensors = []
        for group in embeds_nested:
            for item in group:
                tensors.append(item)
        if not tensors:
            return torch.zeros(
                (0, self.model.config.hidden_size), device=self.device, dtype=self.dtype
            )
        return torch.cat(tensors, dim=0).to(self.dtype)

    # unit embedding 由自研 Thinker 的 embed_tokens 产出(llm 权重在本体,
    # 此处仅暴露 token id)
