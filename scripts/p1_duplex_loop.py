#!/usr/bin/env python
"""P2 里程碑 —— 自研引擎全双工闭环:语音进 -> 决策 -> 语音出。

音频按 duplex 协议流式喂入,每 chunk 跑官方口径的 listen/speak 决策;
模型决定开口后累积回复 token + 隐层,轮末经 Talker(hidden_text_merge)
与 Code2Wav 合成 24kHz 回复音频。这个脚本仍是回放验证，不能当作实时
端到端延迟证据。全程自研引擎(音频编码器除外,
策略同 vllm-omni:非热路径复用官方实现)。

用法:
    python scripts/p1_duplex_loop.py [--wav fixture.wav]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SNAPSHOT_GLOB = Path.home() / ".cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots"
DEFAULT_WAV = (
    Path.home()
    / "project/MiniCPM-o-Demo/tests/cases/common/user_audio/当出现植物大战僵尸的时候提醒我.wav"
)
REF_WAV_SUFFIX = Path("assets") / "HT_ref_audio.wav"


def find_snapshot() -> Path:
    snaps = sorted(SNAPSHOT_GLOB.glob("*/"))
    if not snaps:
        raise FileNotFoundError("未找到 MiniCPM-o 4.5 权重快照")
    return snaps[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--silence-tail-s", type=float, default=6.0)
    parser.add_argument("--max-codec-tokens", type=int, default=1500)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/p2/duplex_reply.wav")
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_dir = find_snapshot()
    print(f"[setup] snapshot: {model_dir}")
    print(f"[setup] wav: {args.wav}")

    from channellm.engine.audio_front import AudioFront
    from channellm.engine.code2wav import Code2Wav
    from channellm.models.minicpmo_compat import (
        patch_torchaudio_load,
        patch_torchaudio_save,
    )

    patch_torchaudio_load()
    patch_torchaudio_save()
    from channellm.engine.duplex_session import DuplexSession
    from channellm.engine.talker import load_talker_weights
    from channellm.engine.thinker import (
        SparkinferPagedKV,
        ThinkerConfig,
        load_thinker_weights,
    )
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import (
        PagedAttnConfig,
        SparkinferPagedAttn,
    )

    t0 = time.time()
    audio_front = AudioFront(model_dir, device=device, dtype=dtype)
    print(f"[load] AudioFront {time.time() - t0:.1f}s")

    t0 = time.time()
    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Thinker {time.time() - t0:.1f}s")

    t0 = time.time()
    talker = load_talker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Talker {time.time() - t0:.1f}s")

    t0 = time.time()
    code2wav = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX)
    print(f"[load] Code2Wav {time.time() - t0:.1f}s")

    tconfig = ThinkerConfig.from_official(model_dir / "config.json")
    pool = PagedKVPool(
        num_layers=tconfig.num_hidden_layers,
        num_pages=512,
        page_size=64,
        num_kv_heads=tconfig.num_kv_heads,
        head_dim=tconfig.head_dim,
        dtype=dtype,
        device=device,
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=tconfig.num_q_heads,
            num_kv_heads=tconfig.num_kv_heads,
            head_dim=tconfig.head_dim,
            page_size=64,
            dtype=dtype,
        ),
        device,
    )
    kv = SparkinferPagedKV(pool, attn)
    session = DuplexSession(thinker, kv, audio_front)
    session.prepare()
    print("[ctx] duplex session prepared")

    wave, sr = sf.read(str(args.wav), dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"需要 16kHz,得到 {sr}")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    tail = np.zeros(int(args.silence_tail_s * 16000), dtype=np.float32)
    stream = np.concatenate([wave, tail])

    proc = audio_front.model.processor
    pos, idx = 0, 0
    decisions = []
    torch.cuda.synchronize()
    t0 = time.time()
    while pos < len(stream):
        n = proc.get_streaming_chunk_size()
        piece = stream[pos : pos + n]
        if len(piece) < n:
            piece = np.concatenate(
                [piece, np.zeros(n - len(piece), dtype=np.float32)]
            )
        decision = session.on_chunk(piece)
        decisions.append(decision)
        tag = "LISTEN" if decision.is_listen else f"SPEAK+{decision.n_speak_tokens}"
        eot = " EOT" if decision.end_of_turn else ""
        print(
            f"[chunk{idx:02d}] {tag}{eot} "
            f"(embed {decision.cost_embed_ms:.0f}ms + 决策 {decision.cost_decision_ms:.0f}ms)"
        )
        pos += n
        idx += 1
    torch.cuda.synchronize()
    loop_s = time.time() - t0
    n_listen = sum(1 for d in decisions if d.is_listen)
    over = [i for i, d in enumerate(decisions)
            if d.cost_embed_ms + d.cost_decision_ms > 1000]
    print(f"[loop] {idx} chunks / {loop_s:.2f}s, listen={n_listen} speak={idx - n_listen}")
    print(f"[loop] 超 1s 实时预算的 chunk: {over if over else '无'}")

    tok = audio_front.tokenizer
    reply_text = tok.decode(session.res_ids, skip_special_tokens=True)
    print(f"[reply] {reply_text[:300]!r}")

    cond_ids, cond_hidden = session.collect_conditioning()
    if cond_ids is None:
        print("[verify] FAIL: 模型未开口,无回复可合成")
        return 1

    from channellm.engine.blocks import TorchListKV

    talker_kv = TorchListKV()
    torch.cuda.synchronize()
    t0 = time.time()
    codec_tokens = talker.generate_codec_tokens(
        cond_ids, cond_hidden, talker_kv,
        max_new_tokens=args.max_codec_tokens, duplex=True,
    )
    torch.cuda.synchronize()
    tts_s = time.time() - t0
    print(f"[talker] codec {len(codec_tokens)} tok / {tts_s:.2f}s")
    if not codec_tokens:
        print("[verify] FAIL: Talker 未产出 codec token")
        return 1

    from channellm.engine.code2wav import StreamingSynth

    synth = StreamingSynth(code2wav)
    torch.cuda.synchronize()
    t0 = time.time()
    ttfp = None
    parts = []
    piece_wav = synth.push(codec_tokens, flush=True)
    if piece_wav is not None:
        ttfp = time.time() - t0
        parts.append(piece_wav)
    torch.cuda.synchronize()
    synth_s = time.time() - t0
    wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    audio_s = len(wav) / 24000
    print(
        f"[code2wav-stream] {synth.n_chunks} 块 / {audio_s:.2f}s 音频 / "
        f"合成 {synth_s:.2f}s / 首块 {ttfp * 1000 if ttfp else 0:.0f}ms"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), wav, 24000)
    from channellm.audio.quality import inspect_signal

    quality = inspect_signal(wav, 24_000)
    failures = quality.failures()
    print(
        f"[done] 已写入 {args.out} (rms={quality.rms:.4f}, peak={quality.peak:.4f}, "
        f"clip={quality.clipped_ratio:.6f})"
    )
    spoke = bool(session.res_ids)
    ok = spoke and not failures
    detail = "; ".join(failures) if failures else "signal-integrity gate passed"
    print(f"[verify] {'PASS' if ok else 'FAIL'}: 开口={spoke}; {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
