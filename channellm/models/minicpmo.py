"""MiniCPM-o 4.5 模型事实与官方 duplex 入口(设计文档 §1 已核实部分)。

这些是 P0/P1 的基准事实,改动前必须重新对照权重内代码验证:

- 权重内自带 10,537 行官方推理代码;class MiniCPMODuplex 在
  modeling_minicpmo.py:2438,含 streaming_prefill():2777、
  streaming_generate():3151、get_session_schema():3407。
- 结构:36 层、hidden 4096、32 heads / 8 KV heads、head_dim 128、
  rope_theta 1e6、上下文 40960、纯 full attention。骨干继承
  Qwen3PreTrainedModel。TTS backbone = llama,audio tokenizer =
  s3tokenizer @16kHz,输出 24kHz。
- 配置:stream_input=True、audio_chunk_length=1.0、audio_pool_step=5、
  listen_speak_type=asr。单进程即可跑通全双工。
- 显存:官方 GPU demo 要求 >28GB,初始化后约 21.5GB(运行口径,
  不是 model card 的权重口径)。

官方 duplex 调用面(from_existing_model 包装 MiniCPMO 实例):
prepare(prefix_system_prompt, ref_audio, prompt_wav_path)
streaming_prefill(audio_waveform=..., frame_list=..., text_list=...)
streaming_generate(...) -> dict(is_listen, text, audio_waveform, end_of_turn,
    cost_llm, cost_tts_prep, cost_tts, cost_token2wav, cost_all, n_tokens)
set_break_event / clear_break_event / set_session_stop / is_break_set
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

HF_REPO_ID = "openbmb/MiniCPM-o-4_5"

DEFAULT_SYSTEM_PROMPT = "Streaming Omni Conversation."


@dataclasses.dataclass(frozen=True)
class ModelFacts:
    """已核实的模型结构事实 —— 供 kernel/engine 配置推导,禁止凭空改。"""

    num_layers: int = 36
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1e6
    max_context: int = 40960
    audio_chunk_length: float = 1.0
    audio_pool_step: int = 5
    audio_input_rate: int = 16000
    audio_output_rate: int = 24000


FACTS = ModelFacts()


def find_weights(hf_home: str | None = None) -> Path | None:
    """定位 HF cache 中的 MiniCPM-o 4.5 snapshot(trust_remote_code 直接用)。"""
    hf_home = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
    hub = hf_home / "hub" / "models--openbmb--MiniCPM-o-4_5/snapshots"
    if not hub.is_dir():
        return None
    for snapshot in sorted(hub.iterdir()):
        if (snapshot / "config.json").exists():
            return snapshot
    return None
