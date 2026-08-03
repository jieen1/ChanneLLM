#!/usr/bin/env python
"""P1 语音输入里程碑 —— 音频 chunk 流经自研 Thinker 并可被理解。

链路:16kHz PCM 按 1s chunk -> AudioFront(官方流式 whisper 编码器)
-> [<unit>] + audio embeds -> 自研 Thinker forward_embeds(paged KV)
-> 追加文本提问 -> 贪心解码。若回复命中音频内容(植物大战僵尸),
证明语音输入在自研引擎上贯通。

duplex 上下文按官方 prepare 口径:system 提示包裹 im_start/im_end。

用法:
    python scripts/p1_audio_in.py [--wav .../当出现植物大战僵尸的时候提醒我.wav]
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
SYSTEM_PROMPT = "Streaming Omni Conversation."
QUESTION = "用户刚才的语音里提到了什么游戏?只回答游戏名称。"


def find_snapshot() -> Path:
    snaps = sorted(SNAPSHOT_GLOB.glob("*/"))
    if not snaps:
        raise FileNotFoundError("未找到 MiniCPM-o 4.5 权重快照")
    return snaps[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--silence-tail-s", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model_dir = find_snapshot()
    print(f"[setup] snapshot: {model_dir}")
    print(f"[setup] wav: {args.wav}")

    from channellm.engine.audio_front import AudioFront
    from channellm.engine.thinker import (
        SparkinferPagedKV,
        ThinkerConfig,
        load_thinker_weights,
    )
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn

    t0 = time.time()
    audio_front = AudioFront(model_dir, device=device, dtype=dtype)
    print(f"[load] AudioFront {time.time() - t0:.1f}s (unit id={audio_front.unit_token_id})")

    t0 = time.time()
    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Thinker {time.time() - t0:.1f}s")

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
    tokenizer = audio_front.tokenizer

    def feed_ids(ids: list[int]) -> None:
        embeds = thinker.embed_tokens(torch.tensor(ids, dtype=torch.long, device=device))
        thinker.forward_embeds(embeds, kv)

    # ---- duplex 上下文(官方 prepare 口径) ----
    sys_ids = tokenizer.encode(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>", add_special_tokens=False
    )
    feed_ids(sys_ids)
    print(f"[ctx] system prompt {len(sys_ids)} tokens")

    # ---- 音频 chunk 流 ----
    wave, sr = sf.read(str(args.wav), dtype="float32")
    if sr != 16000:
        raise RuntimeError(f"需要 16kHz,得到 {sr}")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    tail = np.zeros(int(args.silence_tail_s * 16000), dtype=np.float32)
    stream = np.concatenate([wave, tail])
    # 按 processor 契约切块:首块 first_chunk_samples(1035ms 对齐),之后 1s
    chunks = []
    pos = 0
    while pos < len(stream):
        n = audio_front.model.processor.get_streaming_chunk_size()
        piece = stream[pos : pos + n]
        if len(piece) < n:
            piece = np.concatenate([piece, np.zeros(n - len(piece), dtype=np.float32)])
        chunks.append(piece)
        pos += n

    unit_id = audio_front.unit_token_id
    total_audio_tokens = 0
    torch.cuda.synchronize()
    t0 = time.time()
    for idx, chunk in enumerate(chunks):
        audio_embeds = audio_front.feed_chunk(chunk)
        unit_embed = thinker.embed_tokens(torch.tensor([unit_id], device=device))
        embeds = torch.cat([unit_embed, audio_embeds], dim=0)
        thinker.forward_embeds(embeds, kv)
        total_audio_tokens += embeds.shape[0]
    torch.cuda.synchronize()
    audio_s = time.time() - t0
    print(
        f"[audio] {len(chunks)} chunks ({len(stream) / 16000:.1f}s) -> "
        f"{total_audio_tokens} tokens / {audio_s:.2f}s"
    )

    # ---- 提问 + 贪心解码 ----
    q_ids = tokenizer.encode("\n" + QUESTION, add_special_tokens=False)
    q_embeds = thinker.embed_tokens(torch.tensor(q_ids, dtype=torch.long, device=device))
    torch.cuda.synchronize()
    t0 = time.time()
    logits = thinker.forward_embeds(q_embeds, kv)
    out: list[int] = []
    next_id = int(logits[-1].argmax())
    for _ in range(args.max_tokens):
        out.append(next_id)
        if next_id in (151643, 151645):
            break
        logits = thinker(torch.tensor([next_id], dtype=torch.long, device=device), kv)
        next_id = int(logits[-1].argmax())
    torch.cuda.synchronize()
    gen_s = time.time() - t0

    answer = tokenizer.decode(out, skip_special_tokens=True)
    print(f"[decode] {len(out)} tok / {gen_s:.2f}s")
    print(f"[answer] {answer!r}")
    hit = "植物大战僵尸" in answer
    print(f"[verify] 音频内容命中: {'PASS' if hit else 'FAIL'}")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
