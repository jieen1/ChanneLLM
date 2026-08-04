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


def batch_artifact_path(path: Path, batch_id: str, run_number: int) -> Path:
    """为多轮批测生成从不复用的输出文件名。"""
    return path.with_name(
        f"{path.stem}.batch-{batch_id}.run-{run_number}{path.suffix}"
    )


def cuda_memory_line(*, peak: bool = False) -> str:
    """当前进程的 allocator 指标；不用于推断其他进程或整卡峰值。"""
    if peak:
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        label = "peak"
    else:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        label = "resident"
    return (
        f"[memory] {label}: allocated={allocated / 2**30:.2f}GiB "
        f"reserved={reserved / 2**30:.2f}GiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--silence-tail-s", type=float, default=6.0)
    parser.add_argument(
        "--eou-offset-s",
        type=float,
        help="权威 EOU 标注秒数；默认在输入 WAV 结束处标记 EOU",
    )
    parser.add_argument(
        "--realtime-input",
        action="store_true",
        help="按 16kHz 输入时钟回放，供本地 EOU→首 PCM 延迟测量；默认机器速度回放",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/p2/duplex_reply.wav")
    )
    parser.add_argument(
        "--trace", type=Path, default=Path("artifacts/p2/duplex_trace.jsonl")
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "在同一已加载模型进程内回放次数；第 1 轮标为 cold，后续为 warm。"
            "多轮时为每轮生成唯一 trace/WAV，避免把旧样本混入统计。"
        ),
    )
    parser.add_argument(
        "--queued-runtime",
        action="store_true",
        help="用单 GPU worker 队列处理输入，验证 barge-in 不阻塞输入控制面。",
    )
    parser.add_argument(
        "--thinker-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="默认 fp32 质量模式；bf16 仅用于性能诊断，未通过长序列 parity",
    )
    parser.add_argument(
        "--vllm-omni-codec-bridge",
        action="store_true",
        help=(
            "使用 vLLM-omni async bridge 的 25-token 首块门槛，和默认的官方 "
            "MiniCPM-o 5-token first-force-flush 做同模型/同输入对照；这不是 "
            "vLLM-omni runtime 的端到端性能基准。"
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须至少为 1")

    device = torch.device("cuda")
    thinker_dtype = torch.float32 if args.thinker_dtype == "fp32" else torch.bfloat16
    talker_dtype = torch.bfloat16
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
    from channellm.engine.blocks import TorchStaticKV
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
    audio_front = AudioFront(model_dir, device=device, dtype=thinker_dtype)
    print(f"[load] AudioFront {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    thinker = load_thinker_weights(model_dir, device=device, dtype=thinker_dtype)
    print(f"[load] Thinker {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    talker = load_talker_weights(model_dir, device=device, dtype=talker_dtype)
    print(f"[load] Talker {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    code2wav = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX)
    print(f"[load] Code2Wav {time.monotonic() - t0:.1f}s")
    t0 = time.monotonic()
    code2wav.prewarm_stream()
    torch.cuda.synchronize()
    print(f"[warm] Code2Wav first-stream shape {time.monotonic() - t0:.1f}s")
    torch.cuda.synchronize()
    print(cuda_memory_line())

    tconfig = ThinkerConfig.from_official(model_dir / "config.json")
    if args.thinker_dtype == "fp32":
        # fp32 + Torch SDPA 是唯一通过长序列逐 token parity 的质量路径。
        # 静态缓存保留 SDPA 语义，避免逐 token/逐层 cat；每轮仅逻辑复位。
        kv = TorchStaticKV(
            tconfig.num_hidden_layers,
            tconfig.max_position_embeddings,
            tconfig.num_kv_heads,
            tconfig.head_dim,
            device=device,
            dtype=thinker_dtype,
        )
        kv_backend = "torch-static"
    else:
        pool = PagedKVPool(
            num_layers=tconfig.num_hidden_layers,
            num_pages=512,
            page_size=64,
            num_kv_heads=tconfig.num_kv_heads,
            head_dim=tconfig.head_dim,
            dtype=thinker_dtype,
            device=device,
        )
        attn = SparkinferPagedAttn(
            PagedAttnConfig(
                num_q_heads=tconfig.num_q_heads,
                num_kv_heads=tconfig.num_kv_heads,
                head_dim=tconfig.head_dim,
                page_size=64,
                dtype=thinker_dtype,
            ),
            device,
        )

        def make_kv():
            return SparkinferPagedKV(pool, attn)

        kv_backend = "sparkinfer"
    print(f"[thinker] mode={args.thinker_dtype}/{kv_backend}")

    wave, sr = sf.read(str(args.wav), dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"需要 16kHz,得到 {sr}")
    if args.eou_offset_s is not None and not 0.0 <= args.eou_offset_s <= len(wave) / sr:
        parser.error("--eou-offset-s 必须在输入 WAV 的时长范围内")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    tail = np.zeros(int(args.silence_tail_s * 16000), dtype=np.float32)
    stream = np.concatenate([wave, tail])
    eou_sample = (
        int(args.eou_offset_s * sr) if args.eou_offset_s is not None else len(wave)
    )

    from channellm.duplex.driver import DuplexPipelineDriver
    from channellm.duplex.playback import BufferedPlaybackSink, PcmPlayoutPump
    from channellm.duplex.queued_runtime import QueuedDuplexRuntime
    from channellm.duplex.runtime import RealtimeRuntime
    from channellm.metrics.latency import format_waterfall, waterfall
    from channellm.pipeline.orchestrator import Orchestrator
    from channellm.pipeline.stages import (
        CODEC_CHUNK_FRAMES,
        CODEC_INITIAL_MIN_AUDIO_FRAMES,
    )
    from channellm.tracing import Anchor, TraceRecorder, load_records

    codec_initial_min_audio_frames = (
        CODEC_CHUNK_FRAMES
        if args.vllm_omni_codec_bridge
        else CODEC_INITIAL_MIN_AUDIO_FRAMES
    )
    codec_bridge = (
        "vllm-omni-async-25" if args.vllm_omni_codec_bridge else "official-force-flush-5"
    )
    print(
        "[codec-bridge] "
        f"{codec_bridge}: initial-audio-frames={codec_initial_min_audio_frames}"
    )

    # 默认连续 KV 保持 Torch SDPA 数值语义，同时避免 Talker decode 的逐层 cat。
    talker_stream = TalkerStream(talker)
    proc = audio_front.model.processor
    from channellm.audio.quality import inspect_signal

    batch_id = f"{time.time_ns():x}" if args.repeat > 1 else ""
    all_records = []
    all_ok = True
    if args.thinker_dtype == "bf16":
        kv = None

    for run_index in range(args.repeat):
        run_number = run_index + 1
        temperature = "cold" if run_index == 0 else "warm"
        run_trace = (
            args.trace
            if args.repeat == 1
            else batch_artifact_path(args.trace, batch_id, run_number)
        )
        run_out = (
            args.out
            if args.repeat == 1
            else batch_artifact_path(args.out, batch_id, run_number)
        )
        print(f"[run {run_number}/{args.repeat}] {temperature}: trace={run_trace}")

        # 每轮必须回到同样的模型会话起点；模型权重和 CUDA allocator 保持驻留。
        if args.thinker_dtype == "bf16":
            if run_index:
                assert kv is not None
                pool.free_seq(kv.seq)
            kv = make_kv()
        elif run_index:
            kv.reset()
        audio_front.reset()
        session = DuplexSession(thinker, kv, audio_front)
        session.prepare()
        sink = BufferedPlaybackSink()
        pos, idx = 0, 0
        decisions = []
        eou_marked = False
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with TraceRecorder(
            run_trace,
            session_id=f"p2-local-replay-{batch_id or 'single'}-{run_number}",
            tags={
                "loc": "local",
                "temp": temperature,
                "run": str(run_number),
                "codec_bridge": codec_bridge,
            },
            append=False,
        ) as recorder:
            runtime = RealtimeRuntime(
                Orchestrator(
                    codec_initial_min_audio_frames=codec_initial_min_audio_frames
                ),
                sink,
                trace_recorder=recorder,
            )
            driver = DuplexPipelineDriver(
                runtime,
                session,
                talker_stream,
                code2wav,
                response_text=lambda: audio_front.tokenizer.decode(
                    session.res_ids, skip_special_tokens=True
                ),
            )
            queued = QueuedDuplexRuntime(driver) if args.queued_runtime else None
            tag = (
                queued.begin_turn("local-replay")
                if queued is not None
                else driver.begin_turn("local-replay")
            )
            input_started_at = time.monotonic()
            while pos < len(stream):
                if args.realtime_input:
                    deadline = input_started_at + pos / sr
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                if not eou_marked and pos >= eou_sample:
                    if queued is not None:
                        queued.on_eou(tag)
                    else:
                        driver.on_eou(tag)
                    eou_marked = True
                n = proc.get_streaming_chunk_size()
                piece = stream[pos : pos + n]
                if len(piece) < n:
                    piece = np.concatenate(
                        [piece, np.zeros(n - len(piece), dtype=np.float32)]
                    )
                if queued is not None:
                    if not queued.submit_audio(tag, piece):
                        break
                    pos += n
                    idx += 1
                    continue
                decision = driver.process_audio_chunk(tag, piece)
                if decision is None:
                    break
                decisions.append(decision)
                state = (
                    "LISTEN"
                    if decision.is_listen
                    else f"SPEAK+{decision.n_speak_tokens}"
                )
                eot = " EOT" if decision.end_of_turn else ""
                print(
                    f"[chunk{idx:02d}] {state}{eot} "
                    f"(embed {decision.cost_embed_ms:.0f}ms + "
                    f"决策 {decision.cost_decision_ms:.0f}ms)"
                )
                pos += n
                idx += 1
                if runtime.active_tag is None:
                    break
            if not eou_marked:
                if queued is not None:
                    queued.on_eou(tag)
                else:
                    driver.on_eou(tag)
            if queued is not None:
                if not queued.wait_idle(30.0):
                    queued.close()
                    raise TimeoutError("queued duplex runtime did not become idle in 30s")
                if queued.failures:
                    raise RuntimeError("queued duplex runtime worker failed") from (
                        queued.failures[0]
                    )
                if not queued.close():
                    raise RuntimeError("queued duplex runtime did not stop")
            played = []
            PcmPlayoutPump(sink, runtime, lambda pcm, tag: played.append((pcm, tag))).pump()
        torch.cuda.synchronize()
        print(cuda_memory_line(peak=True))

        loop_s = time.monotonic() - t0
        n_listen = sum(1 for decision in decisions if decision.is_listen)
        over = [
            i
            for i, decision in enumerate(decisions)
            if decision.cost_embed_ms + decision.cost_decision_ms > 1000
        ]
        if args.queued_runtime:
            print(f"[loop] {idx} queued chunks / {loop_s:.2f}s")
        else:
            print(
                f"[loop] {idx} chunks / {loop_s:.2f}s, "
                f"listen={n_listen} speak={idx - n_listen}"
            )
            print(f"[loop] 超 1s 实时预算的 chunk: {over if over else '无'}")

        reply_text = audio_front.tokenizer.decode(
            session.res_ids, skip_special_tokens=True
        )
        print(f"[reply] {reply_text[:300]!r}")
        wav = (
            np.concatenate(
                [np.asarray(pcm, dtype=np.float32).reshape(-1) for pcm, _tag in played]
            )
            if played
            else np.zeros(0, dtype=np.float32)
        )
        run_records = load_records(run_trace)
        all_records.extend(run_records)

        run_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(run_out), wav, 24000)
        quality = inspect_signal(wav, 24_000)
        failures = quality.failures()
        review_warnings = quality.review_warnings()
        rejection_reasons = [
            str(record.extra.get("reason", "unspecified PCM quality failure"))
            for record in run_records
            if record.anchor == Anchor.PCM_QUALITY_REJECTED
        ]
        print(
            f"[done] 已写入 {run_out} (rms={quality.rms:.4f}, "
            f"peak={quality.peak:.4f}, clip={quality.clipped_ratio:.6f}, "
            f"dc={quality.dc_offset:.5f}, max-step={quality.max_step:.5f})"
        )
        spoke = bool(session.res_ids)
        if rejection_reasons:
            ok = False
            print(f"[verify] REJECTED: {'; '.join(rejection_reasons)}")
        else:
            ok = spoke and bool(played) and not failures and not review_warnings
            integrity = "; ".join(failures) if failures else "signal-integrity gate passed"
            review = "; ".join(review_warnings) if review_warnings else "no review warning"
            print(
                f"[verify] {'PASS' if ok else 'REVIEW'}: 开口={spoke}; "
                f"{integrity}; {review}"
            )
        all_ok = all_ok and ok

    print(
        format_waterfall(
            waterfall(all_records, group_by=("temp", "loc")), ("temp", "loc")
        )
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
