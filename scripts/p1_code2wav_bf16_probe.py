#!/usr/bin/env python
"""Code2Wav bf16 探针 —— 自研改造官方 vocoder 的 bf16 流式路径。

官方 Token2wav 只有 fp16 旋钮且流式路径不完整(实测 dtype 冲突),bf16
根本不存在。本探针不改 vendored 代码,在封装层做三件事:
1. flow 与 hift 整体转 bf16;
2. set_stream_cache 在 bf16 autocast 下执行,并把全部 cache 张量转 bf16;
3. 以官方 stream() 同语义重实现 bf16 流式合成(autocast 包裹 + 收尾
   float()),经 Code2Wav 的质量门禁出波形。

使用与 A/B 批次完全相同的 codec(artifacts/p2/code2wav-ab/codec_frames.json),
输出供人工试听。探针只产证据,不进 runtime。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def _cast_recursive(obj, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)
    if isinstance(obj, dict):
        return {k: _cast_recursive(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        casted = [_cast_recursive(v, dtype) for v in obj]
        return type(obj)(casted)
    return obj


def make_bf16_vocoder(c2w) -> None:
    """把已加载的 Code2Wav 改造为 bf16 流式(封装层,不动 vendored 源码)。"""
    from stepaudio2.token2wav import fade_in_out

    t2w = c2w.t2w
    t2w.flow.to(torch.bfloat16)
    t2w.hift.to(torch.bfloat16)
    t2w.speech_window = t2w.speech_window.to(torch.bfloat16)

    orig_set_stream_cache = t2w.set_stream_cache

    def set_stream_cache_bf16(prompt_wav):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            flow_cache, hift_cache = orig_set_stream_cache(prompt_wav)
        flow_cache = _cast_recursive(flow_cache, torch.bfloat16)
        hift_cache = {k: v.to(torch.bfloat16) for k, v in hift_cache.items()}
        return flow_cache, hift_cache

    t2w.set_stream_cache = set_stream_cache_bf16

    def stream_bf16(generated_speech_tokens, prompt_wav, last_chunk=False,
                    return_waveform=False):
        """官方 stream() 同语义 + bf16 修正(autocast 与收尾 float)。"""
        if t2w.cache is None:
            t2w.cache = t2w._prepare_prompt(prompt_wav)
        _pt, _ptl, spk_emb, prompt_mels, _pml = t2w.cache

        tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device="cuda")
        if t2w.stream_cache is None:
            raise ValueError("stream_cache is not set")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            chunk_mel, t2w.stream_cache = t2w.flow.inference_chunk(
                token=tokens, spk=spk_emb, cache=t2w.stream_cache,
                last_chunk=last_chunk, n_timesteps=t2w.n_timesteps,
            )
            if t2w.stream_cache["estimator_att_cache"].shape[4] > (prompt_mels.shape[1] + 100):
                t2w.stream_cache["estimator_att_cache"] = torch.cat(
                    [
                        t2w.stream_cache["estimator_att_cache"][:, :, :, :, : prompt_mels.shape[1]],
                        t2w.stream_cache["estimator_att_cache"][:, :, :, :, -100:],
                    ],
                    dim=4,
                )
            if t2w.stream_cache["conformer_att_cache"].shape[3] > (prompt_mels.shape[1] + 100):
                t2w.stream_cache["conformer_att_cache"] = torch.cat(
                    [
                        t2w.stream_cache["conformer_att_cache"][:, :, :, : prompt_mels.shape[1], :],
                        t2w.stream_cache["conformer_att_cache"][:, :, :, -100:, :],
                    ],
                    dim=3,
                )

            hift_cache_mel = t2w.hift_cache_dict["mel"]
            hift_cache_source = t2w.hift_cache_dict["source"]
            hift_cache_speech = t2w.hift_cache_dict["speech"]
            mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)
            speech, source = t2w.hift(mel, hift_cache_source)
            if hift_cache_speech.shape[-1] > 0:
                speech = fade_in_out(speech, hift_cache_speech, t2w.speech_window)
            is_first_chunk = hift_cache_speech.shape[-1] == 0
            t2w.hift_cache_dict = dict(
                mel=mel[..., -t2w.mel_cache_len :].clone().detach(),
                source=source[:, :, -t2w.source_cache_len :].clone().detach(),
                speech=speech[:, -t2w.source_cache_len :].clone().detach(),
            )
            if not last_chunk:
                if is_first_chunk:
                    silence_padding = torch.zeros(
                        1, t2w.source_cache_len, device=speech.device, dtype=speech.dtype
                    )
                    speech = torch.cat(
                        [silence_padding, speech[:, : -t2w.source_cache_len]], dim=1
                    )
                else:
                    speech = speech[:, : -t2w.source_cache_len]
            speech = speech.float()

        wav_np = speech.cpu().numpy()
        if return_waveform:
            return wav_np
        raise NotImplementedError("bf16 探针只支持 return_waveform=True")

    t2w.stream = stream_bf16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec-json",
        type=Path,
        default=Path("artifacts/p2/code2wav-ab/codec_frames.json"),
    )
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/p2/code2wav-ab/F_bf16_6step.wav"))
    args = parser.parse_args()

    from channellm.audio.quality import inspect_signal
    from channellm.engine.code2wav import Code2Wav, StreamingSynth
    from channellm.models.minicpmo_compat import patch_torchaudio_load, patch_torchaudio_save

    patch_torchaudio_load()
    patch_torchaudio_save()
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from p1_voice_loop import REF_WAV_SUFFIX, find_snapshot

    model_dir = find_snapshot()
    data = json.loads(args.codec_json.read_text())
    frames = data["frames"]
    print(f"[setup] 复用 A/B 批次 codec: {len(frames)} 帧, reply={data['reply']!r}")

    c2w = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX, n_timesteps=6)
    make_bf16_vocoder(c2w)

    chunk_times: list[float] = []
    inner = c2w.stream_chunk

    def timed(tokens, last_chunk=False):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        wav = inner(tokens, last_chunk=last_chunk)
        torch.cuda.synchronize()
        chunk_times.append(time.monotonic() - t0)
        return wav

    c2w.stream_chunk = timed
    c2w.prewarm_stream()
    chunk_times.clear()

    synth = StreamingSynth(c2w, chunk_size=25, pre_lookahead=3)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    wav = synth.push(frames, flush=True)
    torch.cuda.synchronize()
    total_s = time.monotonic() - t0

    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
    q = inspect_signal(samples, 24_000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), samples, 24_000)
    failures = q.failures(min_duration_s=0.0, min_rms=0.0, max_peak=0.98)
    print(
        f"[bf16] first_chunk={chunk_times[0] * 1000:.1f}ms "
        f"chunk_ms={[round(t * 1000, 1) for t in chunk_times]} "
        f"total={total_s * 1000:.1f}ms dur={len(samples) / 24_000:.2f}s\n"
        f"[bf16] rms={q.rms:.5f} peak={q.peak:.5f} clip={q.clipped_ratio:.6f} "
        f"dc={q.dc_offset:.6f} max_step={q.max_step:.5f} "
        f"gate={'PASS' if not failures else failures}\n"
        f"[bf16] -> {args.out}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
