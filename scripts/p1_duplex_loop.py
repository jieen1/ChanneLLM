#!/usr/bin/env python
"""P2 里程碑 —— 自研引擎全双工闭环:语音进 -> 决策 -> 语音出。

音频按 duplex 协议流式喂入,每 chunk 跑官方口径的 listen/speak 决策;
模型决定开口后立即以当前 unit 的 Thinker 隐层续写 Talker KV，codec phrase
经 L2 编排后流式送入 Code2Wav，再由 L3 runtime 发布 24kHz 回复音频。
这是本地回放的真实三阶段路径；不含 LiveKit 网络与设备播放，故不能把这里的
首 PCM 延迟报告为端到端客户体验延迟。全程自研引擎(音频编码器除外,
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
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/p2/duplex_reply.wav")
    )
    parser.add_argument(
        "--trace", type=Path, default=Path("artifacts/p2/duplex_trace.jsonl")
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
    from channellm.engine.talker import TalkerStream, load_talker_weights
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

    t0 = time.monotonic()
    audio_front = AudioFront(model_dir, device=device, dtype=dtype)
    print(f"[load] AudioFront {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Thinker {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    talker = load_talker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Talker {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    code2wav = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX)
    print(f"[load] Code2Wav {time.monotonic() - t0:.1f}s")

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

    from channellm.duplex.driver import DuplexPipelineDriver
    from channellm.duplex.playback import BufferedPlaybackSink
    from channellm.duplex.runtime import RealtimeRuntime
    from channellm.engine.blocks import TorchListKV
    from channellm.metrics.latency import format_waterfall, waterfall
    from channellm.pipeline.orchestrator import Orchestrator
    from channellm.tracing import TraceRecorder, load_records

    sink = BufferedPlaybackSink()
    talker_stream = TalkerStream(talker, TorchListKV)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    proc = audio_front.model.processor
    pos, idx = 0, 0
    decisions = []
    eou_marked = False
    torch.cuda.synchronize()
    t0 = time.monotonic()
    with TraceRecorder(args.trace, session_id="p2-local-replay") as recorder:
        runtime = RealtimeRuntime(Orchestrator(), sink, trace_recorder=recorder)
        driver = DuplexPipelineDriver(runtime, session, talker_stream, code2wav)
        tag = driver.begin_turn("local-replay")
        while pos < len(stream):
            if not eou_marked and pos >= len(wave):
                driver.on_eou(tag)
                eou_marked = True
            n = proc.get_streaming_chunk_size()
            piece = stream[pos : pos + n]
            if len(piece) < n:
                piece = np.concatenate(
                    [piece, np.zeros(n - len(piece), dtype=np.float32)]
                )
            decision = driver.process_audio_chunk(tag, piece)
            if decision is None:
                break
            decisions.append(decision)
            state = "LISTEN" if decision.is_listen else f"SPEAK+{decision.n_speak_tokens}"
            eot = " EOT" if decision.end_of_turn else ""
            print(
                f"[chunk{idx:02d}] {state}{eot} "
                f"(embed {decision.cost_embed_ms:.0f}ms + 决策 {decision.cost_decision_ms:.0f}ms)"
            )
            pos += n
            idx += 1
            if runtime.active_tag is None:
                break
        if not eou_marked:
            driver.on_eou(tag)
        played = sink.drain()
        if played:
            runtime.on_device_playout_start(played[0][1])
            runtime.on_device_playout_finished(played[-1][1])
    torch.cuda.synchronize()
    loop_s = time.monotonic() - t0
    n_listen = sum(1 for d in decisions if d.is_listen)
    over = [i for i, d in enumerate(decisions)
            if d.cost_embed_ms + d.cost_decision_ms > 1000]
    print(f"[loop] {idx} chunks / {loop_s:.2f}s, listen={n_listen} speak={idx - n_listen}")
    print(f"[loop] 超 1s 实时预算的 chunk: {over if over else '无'}")

    tok = audio_front.tokenizer
    reply_text = tok.decode(session.res_ids, skip_special_tokens=True)
    print(f"[reply] {reply_text[:300]!r}")

    wav = (
        np.concatenate([np.asarray(pcm, dtype=np.float32).reshape(-1) for pcm, _tag in played])
        if played
        else np.zeros(0, dtype=np.float32)
    )
    records = load_records(args.trace)
    print(format_waterfall(waterfall(records)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), wav, 24000)
    from channellm.audio.quality import inspect_signal

    quality = inspect_signal(wav, 24_000)
    failures = quality.failures()
    print(
        f"[done] 已写入 {args.out} (rms={quality.rms:.4f}, peak={quality.peak:.4f}, "
        f"clip={quality.clipped_ratio:.6f}, dc={quality.dc_offset:.5f}, "
        f"max-step={quality.max_step:.5f})"
    )
    spoke = bool(session.res_ids)
    ok = spoke and bool(played) and not failures
    detail = "; ".join(failures) if failures else "signal-integrity gate passed"
    print(f"[verify] {'PASS' if ok else 'FAIL'}: 开口={spoke}; {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
