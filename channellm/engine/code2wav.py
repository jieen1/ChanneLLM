"""Code2Wav —— stepaudio2 Token2wav 封装(P1 合成段)。

官方 checkpoint 自带 vocoder 资产(flow.pt/hift.pt/speech_tokenizer/
campplus),``Token2wav(model_path)`` 原地加载。两条路径:

- ``synthesize``:整段 codec token 一次出 24kHz wav(首环用,已在 P0 验证);
- ``stream_*``:官方 ``set_stream_cache``/``stream`` 的分块流式路径,
  每轮开始前用 base cache 克隆复位(官方 duplex 同款,epoch 间复用 base)。
  上游 Flow 还会原地写入 decoder 的非持久 cache buffer；它们也必须与
  ``stream_cache`` 一起复位，否则相同 codec 序列会跨回合漂移。

参考:vllm-omni ``minicpmo_4_5_code2wav.py`` 的 prompt cache 生命周期与
官方 modeling 2647 行附近的 stream_cache 复位逻辑。

flow-matching 步数默认 6(官方默认 10):经 2026-08-04 同 codec 对照批次
(scripts/p1_code2wav_quality_ab.py)人工试听确认 6 步与 10 步音质无实质
差异、5 步劣化,首块延迟 135.2→89.8ms;信号门禁三者均通过。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from channellm.audio.quality import inspect_signal


class PcmQualityError(RuntimeError):
    """Token2wav 产物违反播放前的硬完整性门禁。"""


_PLAYBACK_PEAK_CEILING = 0.98
_NORMALIZED_PEAK = 0.97


class Code2Wav:
    def __init__(
        self,
        model_dir: str | Path,
        ref_wav_path: str | Path,
        float16: bool = False,
        n_timesteps: int = 6,
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
        self._stream_module_cache_base: tuple[tuple[str, torch.Tensor], ...] = ()

    @torch.no_grad()
    def synthesize(self, codec_tokens: list[int]) -> np.ndarray:
        """整段合成,返回 24kHz float32 单声道波形。"""
        if not codec_tokens:
            return np.zeros(0, dtype=np.float32)
        import io

        import soundfile as sf

        wav_bytes = self.t2w(list(codec_tokens), self.ref_wav_path)
        data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return _validated_waveform(data, sample_rate)

    @torch.no_grad()
    def stream_reset(self) -> None:
        """(重新)建立 base cache 并复位流式状态到轮首。"""
        if self._stream_base is None:
            flow_cache, hift_cache = self.t2w.set_stream_cache(self.ref_wav_path)
            # ``inference_chunk`` 会直接写 ``flow.decoder`` 的 cache buffer。
            # 基线必须在首轮写入前脱离这些 alias；否则后续 clone 的仍是已污染状态。
            self._stream_base = (_clone_recursive(flow_cache), _clone_recursive(hift_cache))
            self._stream_module_cache_base = _snapshot_stream_module_cache_buffers(self.t2w)
        _restore_stream_module_cache_buffers(self.t2w, self._stream_module_cache_base)
        flow_base, hift_base = self._stream_base
        self.t2w.stream_cache = _clone_recursive(flow_base)
        self.t2w.hift_cache_dict = _clone_recursive(hift_base)

    def enable_stream_graphs(
        self,
        mel_sizes: tuple[int, ...] = (10, 56),
        max_size: int = 500,
    ) -> None:
        """启用官方 DiT 内置的 CUDA graph 流式路径,注册自研窗口尺寸。

        官方 ``DiT._init_cuda_graph_chunk`` 已实现"padding cache + padding
        mask + 静态缓冲 replay"的图机制(仅预置 30/48/96 三种 mel 尺寸);
        自研流式窗口的 mel 尺寸不同(首块 8 token→mel 10,常规 28 token→
        mel 56,实测为准),这里在封装层按同一机制注册
        自己的尺寸,不改 vendored 源码。同实例开关对照实测 corr=0.99998
        (与 eager 自身复现性基准相同),整段合成墙钟约 1.8x。

        捕获成本每尺寸一次(warmup+capture),应在服务就绪期(prewarm_stream
        之后)显式调用;重复调用幂等。未注册尺寸的窗口自动回落 eager。
        """
        decoder = self.t2w.flow.decoder.estimator
        if getattr(decoder, "use_cuda_graph", False) and all(
            s in decoder.graph_chunk for s in mel_sizes
        ):
            return
        dtype, device = decoder.cnn_cache_buffer.dtype, decoder.cnn_cache_buffer.device
        with torch.no_grad():
            for chunk_size in mel_sizes:
                decoder.max_size_chunk[chunk_size] = max_size
                static_x1 = torch.zeros((2, 320, chunk_size), dtype=dtype, device=device)
                static_t1 = torch.zeros((2, 1, 512), dtype=dtype, device=device)
                static_mask1 = torch.ones(
                    (2, chunk_size, max_size + chunk_size), dtype=torch.bool, device=device
                )
                static_att_cache = torch.zeros(
                    (16, 2, 8, max_size, 128), dtype=dtype, device=device
                )
                static_cnn_cache = torch.zeros((16, 2, 1024, 2), dtype=dtype, device=device)
                static_inputs1 = [
                    static_x1, static_t1, static_mask1, static_cnn_cache, static_att_cache,
                ]
                static_new_cnn_cache = torch.zeros((16, 2, 1024, 2), dtype=dtype, device=device)
                static_new_att_cache = torch.zeros(
                    (16, 2, 8, max_size + chunk_size, 128), dtype=dtype, device=device
                )
                decoder.blocks_forward_chunk(
                    static_x1, static_t1, static_mask1,
                    static_cnn_cache, static_att_cache,
                    static_new_cnn_cache, static_new_att_cache,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out1 = decoder.blocks_forward_chunk(
                        static_x1, static_t1, static_mask1,
                        static_cnn_cache, static_att_cache,
                        static_new_cnn_cache, static_new_att_cache,
                    )
                decoder.inference_buffers_chunk[chunk_size] = {
                    "static_inputs": static_inputs1,
                    "static_outputs": [static_out1, static_new_cnn_cache, static_new_att_cache],
                }
                decoder.graph_chunk[chunk_size] = graph
        decoder.use_cuda_graph = True

    @torch.no_grad()
    def prewarm_stream(
        self,
        *,
        codec_frames: int = 25,
        left_context_frames: int = 3,
        silence_token: int = 4218,
    ) -> None:
        """在服务就绪阶段预热首个流式 vocoder chunk，随后恢复干净回合基线。

        vLLM-omni 会在 Stage2 构造期用固定形状的首个 HiFT chunk 触发 CUDA/
        cuDNN 初始化；这里通过 Token2wav 的公开流式入口完成同一件事，避免
        依赖其内部模块结构。预热音频永不发布，且结束时重新复位所有流式 cache，
        因而不能污染第一位用户的声纹条件或波形状态。
        """
        if codec_frames <= 0:
            raise ValueError("codec_frames must be positive")
        if left_context_frames < 0:
            raise ValueError("left_context_frames must be non-negative")
        self.stream_reset()
        try:
            self.t2w.stream(
                [silence_token] * (codec_frames + left_context_frames),
                self.ref_wav_path,
                last_chunk=False,
                return_waveform=True,
            )
        finally:
            # 即使底层预热失败也恢复可诊断的干净状态；异常仍向调用者传播，不能
            # 静默把首次真实请求变成未预热路径。
            self.stream_reset()

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
            data, sample_rate = sf.read(io.BytesIO(bytes(out)), dtype="float32")
            return _validated_waveform(data, sample_rate)
        wav = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        return _validated_waveform(wav.detach().float().cpu().numpy())


def _validated_waveform(wave: Any, sample_rate: int | None = None) -> np.ndarray:
    """在 PCM 进入媒体层前执行无损的硬完整性门禁。

    流式块可以短于一段完整话语，也允许静音收尾，所以这里不对时长或 RMS
    做判断。非削波的轻微满幅只代表播放增益过高，先按固定目标峰值做保形缩放；
    削波、直流偏置和缩放后仍存在的突变则在任意块长度下都可能造成可听伪影，
    必须在 publish 前拒绝。整段可懂度/自然度仍由离线质检和人工回放负责。
    """
    if sample_rate is not None and sample_rate != 24_000:
        raise PcmQualityError(f"Token2wav 应输出 24kHz,得到 {sample_rate}")
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    quality = inspect_signal(samples, 24_000)
    # ``0.99`` 而非 ``1.0`` 的孤立峰值在真实 Token2wav 回放中反复出现，且
    # 没有 clipped sample。把整块缩到 -0.26dB 的 0.97 保留形状、同时给设备链
    # 路留出 headroom；不能对真正削波做同样处理，因为失真已经写入波形。
    if _PLAYBACK_PEAK_CEILING < quality.peak < 0.999:
        samples = samples * (_NORMALIZED_PEAK / quality.peak)
        quality = inspect_signal(samples, 24_000)
    failures = quality.failures(
        min_duration_s=0.0,
        min_rms=0.0,
        max_peak=_PLAYBACK_PEAK_CEILING,
    )
    if failures:
        raise PcmQualityError("Token2wav PCM 质量门禁拒绝: " + "; ".join(failures))
    return samples


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


def _snapshot_stream_module_cache_buffers(t2w: Any) -> tuple[tuple[str, torch.Tensor], ...]:
    """捕获 Token2wav Flow 中未包含在返回 cache 的可变 buffer 基线。

    stepaudio2 的 ``FlowMatching`` 在 ``forward_chunk`` 内写入
    ``*_cache_buffer``，并将这些 buffer 的 view 返回给调用方。仅克隆返回
    dict 不会清除模块自身的写入，因而会让下一回合从旧序列继续。
    """
    decoder = getattr(getattr(t2w, "flow", None), "decoder", None)
    named_buffers = getattr(decoder, "named_buffers", None)
    if not callable(named_buffers):
        return ()
    return tuple(
        (name, buffer.clone())
        for name, buffer in named_buffers()
        if name.endswith("_cache_buffer")
    )


def _restore_stream_module_cache_buffers(
    t2w: Any, baseline: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """将 Flow 的内部流式 cache 恢复到当前回合开始前的基线。"""
    if not baseline:
        return
    decoder = getattr(getattr(t2w, "flow", None), "decoder", None)
    named_buffers = getattr(decoder, "named_buffers", None)
    if not callable(named_buffers):
        raise RuntimeError("Token2wav Flow decoder cache buffer 不可访问")
    current = dict(named_buffers())
    for name, source in baseline:
        target = current.get(name)
        if target is None or target.shape != source.shape or target.dtype != source.dtype:
            raise RuntimeError(f"Token2wav Flow cache buffer 与初始化基线不兼容: {name}")
        target.copy_(source)


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
