#!/usr/bin/env python
"""Code2Wav fp32 vs bf16 严谨对照 —— 排除实现质量干扰。

上一轮 bf16 探针用 autocast 包裹,可能引入 per-op 分派与 fp32↔bf16 cast
开销,不足以证明"bf16 本身慢"。本脚本:
1. F3 变体:只把迭代 flow matching 转 bf16,HiFT 保持 fp32(官方 fp16
   设计的正确化版本);纯 bf16 全量转换在 HiFT sine 激励路径必然 dtype 失败;
2. B(fp32)/F(全 bf16 autocast)/F3(bf16 flow)同进程交替测量,消除时序噪声;
3. 每轮完整合成(4 块窗口),报告 first_chunk 与 total 的 p50/min/max。

仍然只产证据,不进 runtime。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402
from p1_code2wav_bf16_probe import make_bf16_vocoder  # noqa: E402


def make_flow_bf16_vocoder(c2w) -> None:
    """F3:只有 flow matching 跑 bf16(官方 fp16 设计的正确化版本)。

    官方 float16 模式只 half flow、hift 保持 fp32,但其流式路径因 cache
    dtype 不完整而损坏。这里:flow 转 bf16 并在 autocast 下跑
    inference_chunk,chunk_mel 出 flow 即转回 fp32,hift 与其 cache 保持
    fp32 —— 隔离检验"bf16 迭代 flow 是否更快"这一个问题。
    """
    t2w = c2w.t2w
    t2w.flow.to(torch.bfloat16)

    orig_set = t2w.set_stream_cache

    def set_stream_cache_bf16flow(prompt_wav):
        from p1_code2wav_bf16_probe import _cast_recursive

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            flow_cache, hift_cache = orig_set(prompt_wav)
        flow_cache = _cast_recursive(flow_cache, torch.bfloat16)
        hift_cache = {k: v.to(torch.float32) for k, v in hift_cache.items()}
        pt, ptl, spk, mels, melsl = t2w.cache
        t2w.cache = (pt, ptl, spk.to(torch.bfloat16), mels, melsl)
        return flow_cache, hift_cache

    t2w.set_stream_cache = set_stream_cache_bf16flow

    def stream_flow_bf16(generated_speech_tokens, prompt_wav, last_chunk=False,
                         return_waveform=False):
        from stepaudio2.token2wav import fade_in_out

        if t2w.cache is None:
            set_stream_cache_bf16flow(prompt_wav)
        _pt, _ptl, spk_emb, prompt_mels, _pml = t2w.cache
        tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device="cuda")
        if t2w.stream_cache is None:
            raise ValueError("stream_cache is not set")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            chunk_mel, t2w.stream_cache = t2w.flow.inference_chunk(
                token=tokens, spk=spk_emb, cache=t2w.stream_cache,
                last_chunk=last_chunk, n_timesteps=t2w.n_timesteps,
            )
        chunk_mel = chunk_mel.float()
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
                speech = torch.cat([silence_padding, speech[:, : -t2w.source_cache_len]], dim=1)
            else:
                speech = speech[:, : -t2w.source_cache_len]
        wav_np = speech.cpu().numpy()
        if return_waveform:
            return wav_np
        raise NotImplementedError("race 探针只支持 return_waveform=True")

    t2w.stream = stream_flow_bf16


def install_timer(c2w) -> None:
    inner = c2w.stream_chunk
    c2w._race_times = []

    def timed(tokens, last_chunk=False):
        torch.cuda.synchronize()
        ts = time.monotonic()
        out = inner(tokens, last_chunk=last_chunk)
        torch.cuda.synchronize()
        c2w._race_times.append(time.monotonic() - ts)
        return out

    c2w.stream_chunk = timed


def synth_once(c2w, frames) -> float:
    from channellm.engine.code2wav import StreamingSynth

    c2w.stream_reset()
    c2w._race_times.clear()
    synth = StreamingSynth(c2w, chunk_size=25, pre_lookahead=3)
    torch.cuda.synchronize()
    t0 = time.monotonic()
    synth.push(frames, flush=True)
    torch.cuda.synchronize()
    return time.monotonic() - t0


def main() -> int:
    from channellm.engine.code2wav import Code2Wav
    from channellm.models.minicpmo_compat import patch_torchaudio_load, patch_torchaudio_save

    patch_torchaudio_load()
    patch_torchaudio_save()
    from p1_voice_loop import REF_WAV_SUFFIX, find_snapshot

    model_dir = find_snapshot()
    data = json.loads(Path("artifacts/p2/code2wav-ab/codec_frames.json").read_text())
    frames = data["frames"]
    print(f"[setup] codec {len(frames)} 帧")

    variants = {}
    variants["B_fp32"] = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX, n_timesteps=6)
    variants["F_bf16_autocast"] = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX, n_timesteps=6)
    make_bf16_vocoder(variants["F_bf16_autocast"])
    variants["F3_bf16flow_fp32hift"] = Code2Wav(
        model_dir, model_dir / REF_WAV_SUFFIX, n_timesteps=6
    )
    make_flow_bf16_vocoder(variants["F3_bf16flow_fp32hift"])
    for c2w in variants.values():
        install_timer(c2w)

    for name, c2w in variants.items():
        c2w.prewarm_stream()
        synth_once(c2w, frames)
        print(f"[warm] {name} 预热完成")

    N = 8
    results = {name: [] for name in variants}
    for i in range(N):
        line = []
        for name, c2w in variants.items():
            total = synth_once(c2w, frames)
            first = c2w._race_times[0] if c2w._race_times else None
            results[name].append((first, total))
            line.append(f"{name}: first={first * 1000:.0f} total={total * 1000:.0f}")
        print(f"[round {i + 1}/{N}] " + "  |  ".join(line))

    print(f"\n=== p50 汇总(交替测量 N={N})===")
    for name, rows in results.items():
        firsts = [r[0] * 1000 for r in rows if r[0] is not None]
        totals = [r[1] * 1000 for r in rows]
        print(
            f"{name}: first_chunk p50={statistics.median(firsts):.1f}ms "
            f"(min {min(firsts):.1f} / max {max(firsts):.1f}) | total p50="
            f"{statistics.median(totals):.1f}ms (min {min(totals):.1f} / max {max(totals):.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
