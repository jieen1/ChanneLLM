#!/usr/bin/env python
"""P0 —— 官方 MiniCPMODuplex 单进程串行基线回放(设计文档 §P0)。

在写任何自研代码之前,先拿到官方路径的真实 EOU → 首个 PCM 分段数据。
本脚本是"固定音频回放"模式:不依赖麦克风,把标注好 EOU 时刻的 wav
逐个喂给官方 duplex,记录全链路锚点,输出串行基线 waterfall 的原料。

用法:
    python scripts/p0_run_official_duplex.py \
        --audio data/audio_set/xx.wav --eou-offset 6.2 --out traces/run1.jsonl

    # 多条目 manifest(见 data/audio_set/manifest.yaml)
    python scripts/p0_run_official_duplex.py --manifest data/audio_set/manifest.yaml

EOU 口径:
- manifest/eou-offset 提供人工标注的用户说完时刻(权威);
- 未标注时退回模型 is_listen 翻转近似,打 eou_source=approx 标签区分。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from channellm.audio.chunking import StreamChunker, load_wav_mono16k  # noqa: E402
from channellm.metrics.latency import format_waterfall, waterfall  # noqa: E402
from channellm.models.minicpmo import DEFAULT_SYSTEM_PROMPT, find_weights  # noqa: E402
from channellm.tracing.recorder import TraceRecorder, load_records  # noqa: E402
from channellm.tracing.schema import Anchor  # noqa: E402

AUDIO_SILENCE_THRESHOLD = 1e-3

# TTS 参考音色(官方 demo 自带,Apache 2.0,原地引用):
# prepare(prompt_wav_path=...) 触发 token2wav 初始化,缺它则只出 token 不出声。
DEFAULT_REF_AUDIO = (
    Path.home() / "project/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav"
)


def load_model(model_dir: Path, device: str, attn_implementation: str):
    import torch
    from transformers import AutoConfig, AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from channellm.models.minicpmo_compat import (
        patch_config,
        patch_dynamic_cache,
        patch_encoder_decoder_cache,
        patch_model_class,
        patch_whisper_attention,
    )

    # 权重目录的 modeling 代码是带相对导入的动态模块包,只能经
    # transformers 的 auto_map 机制加载,不能直接 sys.path import。
    config = patch_config(
        AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    )
    patch_dynamic_cache()
    patch_whisper_attention()
    patch_encoder_decoder_cache()
    model_cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], str(model_dir))
    patch_model_class(model_cls)
    model = AutoModel.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model = model.to(device).eval()
    return model


def build_duplex(model):
    import importlib

    module = importlib.import_module(model.__class__.__module__)
    MiniCPMODuplex = getattr(module, "MiniCPMODuplex")
    return MiniCPMODuplex.from_existing_model(model)


def run_one(
    duplex,
    recorder: TraceRecorder,
    wav_path: Path,
    eou_offset_s: float | None,
    system_prompt: str,
    out_wav: Path | None,
    tags: dict[str, str],
    silence_tail_s: float = 5.0,
    ref_audio_path: Path | None = None,
) -> dict:
    trace_id = recorder.new_trace()
    turn_epoch = int(time.time())  # 回放模式下每个文件一个 epoch
    wave = load_wav_mono16k(str(wav_path))
    chunker = StreamChunker()

    def anchor(name: str, **extra) -> None:
        recorder.anchor(name, trace_id=trace_id, turn_epoch=turn_epoch, tags=tags, **extra)

    prepare_kwargs: dict[str, Any] = {"prefix_system_prompt": system_prompt}
    if ref_audio_path is not None:
        prepare_kwargs["prompt_wav_path"] = str(ref_audio_path)
    duplex.prepare(**prepare_kwargs)
    anchor(Anchor.SESSION_PREPARE_DONE)

    total_s = len(wave) / chunker.sample_rate
    eou_anchored = False
    speak_anchored = False
    first_pcm_anchored = False
    elapsed_s = 0.0
    pcm_parts: list[np.ndarray] = []
    chunk_idx = 0

    # 模型靠"说话结束后的静音"判定 EOU;回放必须在 wav 之后继续喂静音,
    # 否则模型永远停在 listen(官方 demo 的麦克风流天然包含这一段)。
    tail_silence = np.zeros(
        int(chunker.sample_rate * silence_tail_s), dtype=np.float32
    )
    all_chunks = list(chunker.feed(wave)) + list(chunker.feed(tail_silence))
    tail = chunker.flush_tail()
    if tail is not None:
        all_chunks.append(tail)

    for chunk in all_chunks:
        elapsed_s += len(chunk) / chunker.sample_rate

        anchor(Anchor.CHUNK_ALIGNED, chunk_idx=chunk_idx)
        anchor(Anchor.STREAMING_PREFILL_START, chunk_idx=chunk_idx)
        prefill_result = duplex.streaming_prefill(audio_waveform=chunk)
        anchor(Anchor.STREAMING_PREFILL_DONE, chunk_idx=chunk_idx, **prefill_result)

        anchor(Anchor.STREAMING_GENERATE_START, chunk_idx=chunk_idx)
        gen = duplex.streaming_generate()
        anchor(
            Anchor.STREAMING_GENERATE_DONE,
            chunk_idx=chunk_idx,
            is_listen=gen.get("is_listen"),
            end_of_turn=gen.get("end_of_turn"),
            n_tokens=gen.get("n_tokens"),
            cost_llm=gen.get("cost_llm"),
            cost_tts=gen.get("cost_tts"),
            cost_token2wav=gen.get("cost_token2wav"),
        )

        if eou_offset_s is not None and not eou_anchored and elapsed_s >= eou_offset_s:
            anchor(Anchor.EOU_DETECTED, eou_source="manifest", eou_offset_s=eou_offset_s)
            eou_anchored = True

        if not gen.get("is_listen", True) and not speak_anchored:
            if not eou_anchored:
                anchor(Anchor.EOU_DETECTED, eou_source="approx")
                eou_anchored = True
            anchor(Anchor.SPEAK_DECISION)
            speak_anchored = True

        out_wave = gen.get("audio_waveform")
        if out_wave is not None and len(out_wave) > 0:
            pcm_parts.append(np.asarray(out_wave, dtype=np.float32))
            if not first_pcm_anchored and float(np.max(np.abs(out_wave))) > AUDIO_SILENCE_THRESHOLD:
                anchor(Anchor.CODE2WAV_FIRST_PCM, chunk_idx=chunk_idx)
                first_pcm_anchored = True

        chunk_idx += 1
        if gen.get("end_of_turn"):
            break

    if out_wav is not None and pcm_parts:
        import soundfile as sf

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), np.concatenate(pcm_parts), 24000)

    return {
        "trace_id": trace_id,
        "wav": str(wav_path),
        "total_s": round(total_s, 2),
        "chunks": chunk_idx,
        "spoke": speak_anchored,
        "first_pcm": first_pcm_anchored,
    }


def load_manifest(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("items", [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="单个 wav(16kHz 回放集)")
    parser.add_argument("--eou-offset", type=float, default=None, help="人工标注的说完时刻(秒)")
    parser.add_argument("--manifest", type=Path, help="多条目 manifest.yaml")
    parser.add_argument("--model-dir", type=Path, default=None, help="默认自动定位 HF cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn", default="sdpa", help="attn_implementation(sdpa/eager/flash_attention_2)"
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--out", type=Path, default=Path("traces/p0_serial.jsonl"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/p0"))
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=DEFAULT_REF_AUDIO,
        help="TTS 参考音色 wav;不给则 token2wav 不初始化(只出 token 不出声)",
    )
    args = parser.parse_args()

    if args.audio is None and args.manifest is None:
        parser.error("需要 --audio 或 --manifest")

    model_dir = args.model_dir or find_weights()
    if model_dir is None:
        print("ERROR: 未找到 MiniCPM-o 4.5 权重,先下载或用 --model-dir 指定", file=sys.stderr)
        return 2

    print(f"==> load model: {model_dir}")
    load_start = time.monotonic_ns()
    model = load_model(model_dir, args.device, args.attn)
    duplex = build_duplex(model)
    print(f"==> model ready in {(time.monotonic_ns() - load_start) / 1e9:.1f}s")

    items: list[dict] = []
    if args.manifest is not None:
        base = args.manifest.parent
        for entry in load_manifest(args.manifest):
            items.append(
                {
                    "path": (base / entry["path"]).resolve(),
                    "eou_offset_s": entry.get("eou_offset_s"),
                    "tags": {"category": entry.get("category", "unknown"), "loc": "local"},
                    "stem": Path(entry["path"]).stem,
                }
            )
    else:
        items.append(
            {
                "path": args.audio.resolve(),
                "eou_offset_s": args.eou_offset,
                "tags": {"category": "single", "loc": "local"},
                "stem": args.audio.stem,
            }
        )

    with TraceRecorder(args.out, session_id="p0-serial") as recorder:
        recorder.anchor(
            Anchor.LOAD_DONE,
            load_ms=(time.monotonic_ns() - load_start) / 1e6,
            tags={"loc": "local"},
        )
        summaries = []
        for item in items:
            print(f"==> replay {item['path']} (eou={item['eou_offset_s']})")
            summary = run_one(
                duplex,
                recorder,
                Path(item["path"]),
                item["eou_offset_s"],
                args.system_prompt,
                args.artifact_dir / f"{item['stem']}_reply.wav",
                item["tags"],
                ref_audio_path=args.ref_audio,
            )
            summaries.append(summary)
            print(
                f"    chunks={summary['chunks']} spoke={summary['spoke']} "
                f"first_pcm={summary['first_pcm']}"
            )

    records = load_records(args.out)
    report = waterfall(records, group_by=("loc",))
    print("\n==> waterfall (本次运行全量,样本少时仅供检查锚点)")
    print(format_waterfall(report, group_labels=("loc",)))
    print(f"\ntraces: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
