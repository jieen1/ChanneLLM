#!/usr/bin/env python
"""Code2Wav vocoder 质量/延迟对照批次。

同一固定 seed 生成同一段回复 codec,然后在官方 Token2wav 公开旋钮
(`n_timesteps` / `float16`)的不同组合下用**生产同款流式路径**合成,
报告首块延迟、总合成时间与信号指标,WAV 落盘供人工试听。这是质量取舍
的证据批次,不是结论:任何配置都不改变 runtime 默认,直到人工确认。

配置:
  A = n_timesteps=10, fp32(现行生产)
  B = n_timesteps=6,  fp32
  E = n_timesteps=5,  fp32

用法:
    python scripts/p1_code2wav_quality_ab.py [--prompt ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

# fp16 组合不可用:官方 Token2wav 只对 flow 做 half(),流式路径的
# set_stream_cache/hift cache 仍是 fp32,dtype 直接冲突(实测 RuntimeError)。
# 若试听后仍需要 fp16 选项,须先改造 vocoder 内部,另行立项。
CONFIGS = [
    ("A_baseline_10step_fp32", 10, False),
    ("B_6step_fp32", 6, False),
    ("E_5step_fp32", 5, False),
]


def generate_codec_frames(model_dir: Path, prompt: str) -> tuple[list[int], str]:
    """bf16 原生 Thinker(graph decode)+ Talker 生成回复 codec(固定 seed)。"""
    from p1_voice_loop import STOP_TOKEN_IDS, build_prompt_ids, sample_text_token
    from transformers import AutoTokenizer

    from channellm.engine.graph_decode import GraphDecodeSession
    from channellm.engine.talker import TalkerStream, load_talker_weights
    from channellm.engine.thinker import (
        SparkinferPagedKV,
        ThinkerConfig,
        load_thinker_weights,
    )
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    tconfig = ThinkerConfig.from_official(model_dir / "config.json")
    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype, config=tconfig)
    pool = PagedKVPool(
        tconfig.num_hidden_layers, 512, 64,
        tconfig.num_kv_heads, tconfig.head_dim, dtype=dtype, device=device,
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=tconfig.num_q_heads, num_kv_heads=tconfig.num_kv_heads,
            head_dim=tconfig.head_dim, page_size=64, dtype=dtype,
        ),
        device,
    )
    kv = SparkinferPagedKV(pool, attn)
    graph = GraphDecodeSession(thinker, kv)
    graph.capture()  # 必须在真实 prefill 之前(空 KV)捕获

    prompt_ids = build_prompt_ids(tokenizer, prompt)
    print(f"[thinker] prompt {len(prompt_ids)} tokens: {prompt!r}")
    ids_cuda = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    logits = thinker(ids_cuda, kv)

    text_tokens: list[int] = []
    text_hiddens: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(20_260_804)
    history = list(prompt_ids)
    next_id = sample_text_token(
        logits[-1], history, temperature=0.7, top_k=100, top_p=0.8,
        repetition_penalty=1.02, generator=generator,
    )
    for _ in range(256):
        _greedy, logits_row, hidden_row = graph.step(next_id)
        text_tokens.append(next_id)
        text_hiddens.append(hidden_row.clone())
        history.append(next_id)
        if next_id in STOP_TOKEN_IDS:
            break
        next_id = sample_text_token(
            logits_row, history, temperature=0.7, top_k=100, top_p=0.8,
            repetition_penalty=1.02, generator=generator,
        )
    reply = tokenizer.decode(text_tokens, skip_special_tokens=True)
    print(f"[thinker] reply: {reply!r}")

    talker = load_talker_weights(model_dir, device=device, dtype=dtype)
    stream = TalkerStream(talker)
    frames: list[int] = []
    # duplex 里每个 Thinker unit 条件化一个 phrase;这里按 ~12 token 切单元,
    # 逐单元 push,得到完整话语的多 phrase codec。
    unit = 12
    units = list(range(0, len(text_tokens), unit))
    for ui, start in enumerate(units):
        end = min(start + unit, len(text_tokens))
        last_unit = ui == len(units) - 1
        for part, _is_last in stream.push_streaming(
            torch.tensor(text_tokens[start:end], device=device),
            torch.stack(text_hiddens[start:end]).to(dtype),
            end_of_turn=last_unit,
        ):
            frames.extend(part)
    print(f"[talker] {len(units)} units -> {len(frames)} codec frames")
    return frames, reply


def synth_one(model_dir: Path, ref_wav: Path, frames: list[int], *,
              n_timesteps: int, float16: bool) -> dict:
    """生产同款流式合成(预热 + 25/3 分块),计时并返回信号指标。"""
    from channellm.audio.quality import inspect_signal
    from channellm.engine.code2wav import Code2Wav, StreamingSynth

    t0 = time.monotonic()
    c2w = Code2Wav(model_dir, ref_wav, float16=float16, n_timesteps=n_timesteps)
    load_s = time.monotonic() - t0

    chunk_times: list[float] = []
    inner = c2w.stream_chunk

    def timed(tokens, last_chunk=False):
        torch.cuda.synchronize()
        ts = time.monotonic()
        wav = inner(tokens, last_chunk=last_chunk)
        torch.cuda.synchronize()
        chunk_times.append(time.monotonic() - ts)
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
    stats = {
        "load_s": round(load_s, 2),
        "first_chunk_ms": round(chunk_times[0] * 1000, 1) if chunk_times else None,
        "chunk_ms": [round(t * 1000, 1) for t in chunk_times],
        "total_ms": round(total_s * 1000, 1),
        "duration_s": round(len(samples) / 24_000, 2),
        "rms": round(q.rms, 5),
        "peak": round(q.peak, 5),
        "clipped_ratio": round(q.clipped_ratio, 6),
        "dc_offset": round(q.dc_offset, 6),
        "max_step": round(q.max_step, 5),
        "finite": bool(q.finite),
        "gate_failures": q.failures(min_duration_s=0.0, min_rms=0.0, max_peak=0.98),
    }
    return stats, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="你好,请用两三句话介绍一下杭州。")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/p2/code2wav-ab"))
    args = parser.parse_args()

    from channellm.models.minicpmo_compat import patch_torchaudio_load, patch_torchaudio_save

    patch_torchaudio_load()
    patch_torchaudio_save()
    from p1_voice_loop import REF_WAV_SUFFIX, find_snapshot

    model_dir = find_snapshot()
    ref_wav = model_dir / REF_WAV_SUFFIX
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames, reply = generate_codec_frames(model_dir, args.prompt)
    (args.out_dir / "codec_frames.json").write_text(
        json.dumps({"prompt": args.prompt, "reply": reply, "frames": frames})
    )

    summary = {"prompt": args.prompt, "reply": reply, "n_frames": len(frames), "configs": {}}
    for name, n_steps, fp16 in CONFIGS:
        print(f"\n=== {name} (n_timesteps={n_steps}, float16={fp16})")
        stats, samples = synth_one(
            model_dir, ref_wav, frames, n_timesteps=n_steps, float16=fp16
        )
        summary["configs"][name] = stats
        out = args.out_dir / f"{name}.wav"
        sf.write(str(out), samples, 24_000)
        print(
            f"    first_chunk={stats['first_chunk_ms']}ms total={stats['total_ms']}ms "
            f"dur={stats['duration_s']}s rms={stats['rms']} peak={stats['peak']} "
            f"clip={stats['clipped_ratio']} dc={stats['dc_offset']} "
            f"max_step={stats['max_step']} "
            f"gate={stats['gate_failures'] or 'PASS'} -> {out}"
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[done] summary: {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
