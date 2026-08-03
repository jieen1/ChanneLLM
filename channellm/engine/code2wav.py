"""Code2Wav —— stepaudio2 Token2wav 封装(P1 合成段)。

官方 checkpoint 自带 vocoder 资产(flow.pt/hift.pt/speech_tokenizer/
campplus),``Token2wav(model_path)`` 原地加载。两条路径:

- ``synthesize``:整段 codec token 一次出 24kHz wav(首环用,已在 P0 验证);
- ``stream_*``:官方 ``set_stream_cache``/``stream`` 的分块流式路径,
  每轮开始前用 base cache 克隆复位(官方 duplex 同款,epoch 间复用 base)。

参考:vllm-omni ``minicpmo_4_5_code2wav.py`` 的 prompt cache 生命周期与
官方 modeling 2647 行附近的 stream_cache 复位逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class Code2Wav:
    def __init__(
        self,
        model_dir: str | Path,
        ref_wav_path: str | Path,
        float16: bool = False,
        n_timesteps: int = 10,
    ) -> None:
        from stepaudio2.token2wav import Token2wav

        self.model_dir = Path(model_dir)
        # vocoder 资产在 checkpoint 的 assets/token2wav/ 下(flow.pt/hift.pt/
        # speech_tokenizer/campplus),Token2wav 按目录平铺约定加载。
        assets_dir = self.model_dir / "assets" / "token2wav"
        if not (assets_dir / "flow.pt").exists():
            raise FileNotFoundError(f"Token2wav 资产缺失: {assets_dir}")
        self.ref_wav_path = str(ref_wav_path)
        self.t2w = Token2wav(str(assets_dir), float16=float16, n_timesteps=n_timesteps)
        self._stream_base: tuple[Any, Any] | None = None

    @torch.no_grad()
    def synthesize(self, codec_tokens: list[int]) -> np.ndarray:
        """整段合成,返回 24kHz float32 单声道波形。"""
        if not codec_tokens:
            return np.zeros(0, dtype=np.float32)
        import io

        import soundfile as sf

        wav_bytes = self.t2w(list(codec_tokens), self.ref_wav_path)
        data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if sample_rate != 24000:
            raise RuntimeError(f"Token2wav 应输出 24kHz,得到 {sample_rate}")
        return np.asarray(data).reshape(-1)

    @torch.no_grad()
    def stream_reset(self) -> None:
        """(重新)建立 base cache 并复位流式状态到轮首。"""
        if self._stream_base is None:
            flow_cache, hift_cache = self.t2w.set_stream_cache(self.ref_wav_path)
            self._stream_base = (flow_cache, hift_cache)
        flow_base, hift_base = self._stream_base
        self.t2w.stream_cache = _clone_recursive(flow_base)
        self.t2w.hift_cache_dict = _clone_recursive(hift_base)

    @torch.no_grad()
    def stream_chunk(self, codec_tokens: list[int], last_chunk: bool = False) -> np.ndarray:
        """流式合成一块 codec token,返回 24kHz float32 波形块。"""
        if self.t2w.stream_cache is None:
            raise RuntimeError("先调用 stream_reset()")
        import io

        import soundfile as sf

        out = self.t2w.stream(
            list(codec_tokens), self.ref_wav_path, last_chunk=last_chunk, return_waveform=True
        )
        if isinstance(out, (bytes, bytearray)):
            data, _ = sf.read(io.BytesIO(bytes(out)), dtype="float32")
            return np.asarray(data).reshape(-1)
        wav = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        return wav.detach().float().cpu().numpy().reshape(-1)


def _clone_recursive(obj: Any) -> Any:
    """官方 torch_clone_recursive 的最小等价:tensor 克隆,容器深拷贝。"""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_recursive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cloned = [_clone_recursive(v) for v in obj]
        return type(obj)(cloned)
    return obj


class StreamingSynth:
    """流式分块合成(官方 _generate_waveform_from_tokens 契约)。

    缓冲 codec token:满 CHUNK_SIZE+pre_lookahead(25+3)即送一块给
    Token2wav.stream,buffer 前进 CHUNK_SIZE(前瞻 3 帧留作下一块左上下文);
    flush 时余量带 last_chunk 收尾。
    """

    def __init__(self, code2wav, chunk_size=25, pre_lookahead=3) -> None:
        self.code2wav = code2wav
        self.chunk_size = chunk_size
        self.pre_lookahead = pre_lookahead
        self.buffer: list[int] = []
        self.n_chunks = 0
        code2wav.stream_reset()

    def push(self, tokens, flush=False):
        """喂入新 codec token,返回本步产出的 24kHz float32 波形(可能为 None)。"""
        self.buffer.extend(int(t) for t in tokens)
        parts = []
        need = self.chunk_size + self.pre_lookahead
        while len(self.buffer) >= need:
            wav = self.code2wav.stream_chunk(self.buffer[:need])
            parts.append(wav)
            self.buffer = self.buffer[self.chunk_size:]
            self.n_chunks += 1
        if flush and self.buffer:
            parts.append(self.code2wav.stream_chunk(self.buffer, last_chunk=True))
            self.buffer = []
            self.n_chunks += 1
        if not parts:
            return None
        import numpy as np

        return np.concatenate(parts)
