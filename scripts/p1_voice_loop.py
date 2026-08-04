#!/usr/bin/env python
"""P1 里程碑 —— 自研引擎语音生成闭环(文本 → 语音)。

链路全部走自研引擎:
Thinker(fp32 + Torch SDPA 语义 KV)采样生成回复文本并收集隐层
  -> Talker(hidden_text_merge 条件化)AR 采样 codec token
  -> Code2Wav(stepaudio2 Token2wav)合成 24kHz wav

语音输入(音频编码器 + 流式 chunk)属 P2 编排面;本环验证的是生成链
在自研引擎上端到端跑通且语音可听。

用法:
    python scripts/p1_voice_loop.py [--prompt 你好,介绍一下杭州。]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

SNAPSHOT_GLOB = Path.home() / ".cache/huggingface/hub/models--openbmb--MiniCPM-o-4_5/snapshots"
# checkpoint 自带参考音色(vllm-omni deploy 的默认 prompt_cache_id 同款)
REF_WAV_SUFFIX = Path("assets") / "HT_ref_audio.wav"
SYSTEM_PROMPT = (
    "You are MiniCPM-o, a helpful multimodal assistant that can understand "
    "images, audio and video, and respond in text and speech."
)
STOP_TOKEN_IDS = {151643, 151645}


def sample_text_token(
    logits: torch.Tensor,
    history: list[int],
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generator: torch.Generator,
) -> int:
    """按权重官方 ``chat`` 默认值采样一个文本 token。

    P1 语音闭环是质量回归，不应使用 argmax 代替官方文本采样；贪心解码会
    系统性放大高频中文 token 的重复。重复惩罚遵循 Transformers 的语义：
    已出现 token 的负 score 乘 penalty，正 score 除 penalty。
    """
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
        raise ValueError("invalid text sampling parameters")

    scores = logits.float().clone()
    if history and repetition_penalty != 1.0:
        seen = torch.tensor(sorted(set(history)), device=scores.device)
        seen_scores = scores[seen]
        scores[seen] = torch.where(
            seen_scores < 0,
            seen_scores * repetition_penalty,
            seen_scores / repetition_penalty,
        )
    scores /= temperature
    if top_k:
        kth = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores.masked_fill_(scores < kth, float("-inf"))
    if top_p < 1:
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        remove = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1) > top_p
        remove[0] = False
        scores[sorted_ids[remove]] = float("-inf")
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator).item())


def find_snapshot() -> Path:
    snaps = sorted(SNAPSHOT_GLOB.glob("*/"))
    if not snaps:
        raise FileNotFoundError("未找到 MiniCPM-o 4.5 权重快照")
    return snaps[0]


def build_prompt_ids(tokenizer, user_text: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    # enable_thinking=False:模板预置空 think 块,模型直出答案,
    # Talker 条件化不再混入思考文本。
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="你好,请用两三句话介绍一下杭州。")
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--max-codec-tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.02)
    parser.add_argument("--seed", type=int, default=20_260_804)
    parser.add_argument("--out", type=Path, default=Path("artifacts/p1/voice_loop_reply.wav"))
    args = parser.parse_args()

    device = torch.device("cuda")
    # 权重原生 bf16,不做反量化;decode 走 sparkinfer paged CUDA graph。
    thinker_dtype = torch.bfloat16
    talker_dtype = torch.bfloat16
    model_dir = find_snapshot()
    print(f"[setup] snapshot: {model_dir}")

    import soundfile as sf
    from transformers import AutoTokenizer

    from channellm.engine.blocks import TorchStaticKV
    from channellm.engine.code2wav import Code2Wav
    from channellm.engine.graph_decode import GraphDecodeSession
    from channellm.engine.talker import load_talker_weights
    from channellm.engine.thinker import (
        SparkinferPagedKV,
        ThinkerConfig,
        load_thinker_weights,
    )
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn
    from channellm.models.minicpmo_compat import patch_torchaudio_load, patch_torchaudio_save

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    # stepaudio2/s3tokenizer 内部走 torchaudio.load;本机无 ffmpeg,
    # 用 P0 同款 soundfile 兜底垫片。
    patch_torchaudio_load()
    patch_torchaudio_save()

    # ---- 装载三段 ----
    t0 = time.time()
    thinker = load_thinker_weights(model_dir, device=device, dtype=thinker_dtype)
    print(f"[load] Thinker {time.time() - t0:.1f}s")

    t0 = time.time()
    talker = load_talker_weights(model_dir, device=device, dtype=talker_dtype)
    print(f"[load] Talker {time.time() - t0:.1f}s")

    t0 = time.time()
    code2wav = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX)
    print(f"[load] Code2Wav {time.time() - t0:.1f}s")

    # ---- Thinker:贪心生成回复文本 + 隐层 ----
    tconfig = ThinkerConfig.from_official(model_dir / "config.json")
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
    kv = SparkinferPagedKV(pool, attn)
    print("[thinker] mode=bf16/sparkinfer+graph")

    prompt_ids = build_prompt_ids(tokenizer, args.prompt)
    print(f"[thinker] prompt {len(prompt_ids)} tokens: {args.prompt!r}")

    def hit_stop(token_id: int) -> bool:
        return token_id in STOP_TOKEN_IDS

    torch.cuda.synchronize()
    t0 = time.time()
    # 自定义循环:需要在 stop token 集合上停,且保留 Talker 所需的逐 token 隐层。
    # 采样参数对齐权重内 MiniCPMO.chat 的默认 generation config。
    ids_cuda = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    logits = thinker(ids_cuda, kv)
    prefill_s = time.time() - t0
    text_tokens: list[int] = []
    text_hiddens: list[torch.Tensor] = []
    text_generator = torch.Generator(device=device).manual_seed(args.seed)
    history = list(prompt_ids)
    next_id = sample_text_token(
        logits[-1], history,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        generator=text_generator,
    )
    graph = GraphDecodeSession(thinker, kv)
    graph.capture()
    for _ in range(args.max_text_tokens):
        _greedy, logits_row, hidden_row = graph.step(next_id)
        text_tokens.append(next_id)
        text_hiddens.append(hidden_row.clone())
        history.append(next_id)
        if hit_stop(next_id):
            break
        next_id = sample_text_token(
            logits_row, history,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            generator=text_generator,
        )
    torch.cuda.synchronize()
    text_s = time.time() - t0 - prefill_s

    reply = tokenizer.decode(text_tokens, skip_special_tokens=True)
    print(f"[thinker] prefill {len(prompt_ids)} tok / {prefill_s:.2f}s; "
          f"decode {len(text_tokens)} tok / {text_s:.2f}s "
          f"({len(text_tokens) / max(text_s, 1e-6):.1f} tok/s)")
    print("[thinker] sampling "
          f"seed={args.seed}, temp={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}, repetition_penalty={args.repetition_penalty}")
    print(f"[thinker] 回复: {reply[:300]}")

    # ---- Talker:codec token ----
    talker_ids = torch.tensor(text_tokens, dtype=torch.long, device=device)
    talker_hidden = torch.stack(text_hiddens).to(talker_dtype)
    talker_kv = TorchStaticKV(
        talker.config.num_hidden_layers,
        talker.config.max_position_embeddings,
        talker.config.num_kv_heads,
        talker.config.head_dim,
        device=device,
        dtype=talker_dtype,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    codec_tokens = talker.generate_codec_tokens(
        talker_ids, talker_hidden, talker_kv, max_new_tokens=args.max_codec_tokens, duplex=False
    )
    torch.cuda.synchronize()
    codec_s = time.time() - t0
    print(f"[talker] codec {len(codec_tokens)} tok / {codec_s:.2f}s "
          f"({len(codec_tokens) / max(codec_s, 1e-6):.1f} tok/s)")
    if not codec_tokens:
        print("[talker] 未产出 codec token,退出")
        return 1

    # ---- Code2Wav:合成 ----
    t0 = time.time()
    wav = code2wav.synthesize(codec_tokens)
    torch.cuda.synchronize()
    synth_s = time.time() - t0
    audio_s = len(wav) / 24000
    print(f"[code2wav] {audio_s:.2f}s 音频 / 合成 {synth_s:.2f}s (RTF {synth_s / audio_s:.2f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), wav, 24000)
    print(f"[done] 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
