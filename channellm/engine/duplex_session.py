"""DuplexSession —— 自研引擎上的 listen/speak 决策环(P2 核心)。

逐 chunk 复刻官方 ``MiniCPMODuplex.streaming_generate`` 的决策语义:

1. chunk 音频 embedding 喂入后,末位 logits 是 pending logits;
2. 两阶段采样(官方 StreamDecoder.decode 口径):先按原始分布采一次,
   若命中 ``<|chunk_eos|>`` 直接尊重模型的停止决策;否则屏蔽禁用
   token、施加窗口 512 的重复惩罚,再按 temperature/top_k/top_p 采样;
3. 采到 ``<|listen|>`` 且本轮未结束 -> 强制 ``<|tts_bos|>`` 继续说;
   采到 chunk 终结符(listen/chunk_eos/chunk_tts_eos)结束本 chunk;
   其余 token 是回复内容,喂回 embedding 取下一 logits + 隐层;
4. 每 chunk 末喂 ``</unit>``;回复 token 与隐层按单元累积,
   供轮末 Talker 条件化(hidden_text_merge)。

与官方差异(首环取舍):TTS 用轮末批量合成(非流式分块),
sliding window 未实现(会话短于窗口时等价)。
"""

from __future__ import annotations

import dataclasses
import time

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class DuplexParams:
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.8
    text_repetition_penalty: float = 1.05
    text_repetition_window_size: int = 512
    listen_prob_scale: float = 1.0
    listen_top_k: int | None = None
    max_new_speak_tokens_per_chunk: int = 20
    force_listen_count: int = 0


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    """单行 logits [vocab] 的 top-k -> top-p 屏蔽。"""
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cutoff = probs.cumsum(0) > top_p
        cutoff[0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)
    return logits


@dataclasses.dataclass
class ChunkDecision:
    is_listen: bool
    unit_token_ids: list[int]
    end_of_turn: bool
    n_speak_tokens: int
    cost_embed_ms: float = 0.0  # 音频编码 + chunk prefill
    cost_decision_ms: float = 0.0  # listen/speak 决策 + 回复 token 生成
    prefill_start_ns: int = 0
    prefill_done_ns: int = 0
    first_token_decoded_ns: int = 0


class DuplexSession:
    def __init__(self, thinker, kv, audio_front, params=None) -> None:
        self.thinker = thinker
        self.kv = kv
        self.audio_front = audio_front
        self.params = params or DuplexParams()
        tok = audio_front.tokenizer
        self.device = thinker.embed_tokens.weight.device

        def tid(s):
            return tok.convert_tokens_to_ids(s)

        self.listen_id = tid("<|listen|>")
        self.speak_id = tid("<|speak|>")
        self.tts_bos_id = tid("<|tts_bos|>")
        self.tts_eos_id = tid("<|tts_eos|>")
        self.chunk_eos_id = tid("<|chunk_eos|>")
        self.chunk_tts_eos_id = tid("<|chunk_tts_eos|>")
        self.turn_eos_id = tid("<|turn_eos|>")
        self.unit_end_id = tid("</unit>")
        self.tts_pad_id = tid("<|tts_pad|>")
        self.chunk_terminator_ids = {self.listen_id, self.chunk_eos_id, self.chunk_tts_eos_id}
        self.turn_terminator_ids = {self.turn_eos_id}
        self.special_ids = set(getattr(tok, "all_special_ids", [])) | {
            self.listen_id, self.speak_id, self.tts_bos_id, self.tts_eos_id,
            self.chunk_eos_id, self.chunk_tts_eos_id, self.turn_eos_id,
            self.unit_end_id, self.tts_pad_id,
        }
        self.forbidden_ids = [self.tts_pad_id] + list(getattr(tok, "bad_token_ids", []))

        self.current_turn_ended = True
        self.generated_tokens = []
        self.res_ids = []
        self.unit_records = []
        self._generate_count = 0

    @torch.no_grad()
    def prepare(self, system_prompt="Streaming Omni Conversation."):
        tok = self.audio_front.tokenizer
        im_start = tok.convert_tokens_to_ids("<|im_start|>")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        sys_word = tok.encode("system", add_special_tokens=False)
        prompt_ids = tok.encode(system_prompt, add_special_tokens=False)
        newline = tok.encode("\n", add_special_tokens=False)
        ids = [im_start] + sys_word + newline + prompt_ids + [im_end]
        self._feed_ids(ids)

    def _embed_ids(self, ids):
        return self.thinker.embed_tokens(
            torch.tensor(ids, dtype=torch.long, device=self.device)
        )

    @torch.no_grad()
    def _feed_ids(self, ids):
        return self.thinker.forward_embeds(self._embed_ids(ids), self.kv)

    @torch.no_grad()
    def _feed_token(self, token_id):
        return self.thinker.forward_embeds(self._embed_ids([token_id]), self.kv)

    def _decode_step(self, logits):
        """官方 StreamDecoder.decode 两阶段采样,返回 token id。"""
        p = self.params
        logits = logits.clone()
        probs0 = F.softmax(logits, dim=-1)
        sampled = int(torch.multinomial(probs0, 1).item())
        if sampled == self.chunk_eos_id:
            return self.chunk_eos_id
        if self.forbidden_ids:
            logits[self.forbidden_ids] = float("-inf")
        if p.text_repetition_penalty != 1.0 and self.generated_tokens:
            window = self.generated_tokens[-p.text_repetition_window_size:]
            for token_id in set(window):
                if token_id < logits.shape[0]:
                    if p.text_repetition_penalty > 1.0:
                        logits[token_id] /= p.text_repetition_penalty
                    else:
                        logits[token_id] *= 1.0 / p.text_repetition_penalty
        if p.listen_prob_scale != 1.0:
            logits[self.listen_id] *= p.listen_prob_scale
        if p.listen_top_k is not None:
            listen_rank = int((logits > logits[self.listen_id]).sum().item())
            if listen_rank < p.listen_top_k:
                return self.listen_id
        logits = logits / p.temperature
        logits = top_k_top_p_filtering(logits, top_k=p.top_k, top_p=p.top_p)
        probs = F.softmax(logits, dim=-1)
        token = int(torch.multinomial(probs, 1).item())
        if token not in self.special_ids:
            self.generated_tokens.append(token)
        return token

    @torch.no_grad()
    def on_chunk(self, pcm) -> ChunkDecision:
        """喂一个音频 chunk,执行 listen/speak 决策,返回本 chunk 结果。"""
        p = self.params
        torch.cuda.synchronize()
        # 性能数字属于单机耗时，必须使用单调时钟；wall clock 的 NTP 校时会
        # 产生负延迟并污染 trace/质量报告。
        t_embed = time.monotonic()
        prefill_start_ns = time.monotonic_ns()
        audio_embeds = self.audio_front.feed_chunk(pcm)
        unit_id = self.audio_front.unit_token_id
        embeds = torch.cat([self._embed_ids([unit_id]), audio_embeds], dim=0)
        logits = self.thinker.forward_embeds(embeds, self.kv)[-1].float()
        torch.cuda.synchronize()
        prefill_done_ns = time.monotonic_ns()
        cost_embed_ms = (time.monotonic() - t_embed) * 1000
        t_dec = time.monotonic()

        force_listen = self._generate_count < p.force_listen_count
        self._generate_count += 1

        unit_records = []
        unit_ids = []
        is_listen = False
        end_of_turn = False
        n_speak = 0
        first_token_decoded_ns = 0

        for j in range(p.max_new_speak_tokens_per_chunk):
            if j == p.max_new_speak_tokens_per_chunk - 1:
                self._feed_token(self.chunk_eos_id)  # explicit 模式收尾
                break
            if force_listen:
                token = self.listen_id
            else:
                token = self._decode_step(logits)
                if token == self.listen_id and not self.current_turn_ended:
                    token = self.tts_bos_id
            if j == 0:
                first_token_decoded_ns = time.monotonic_ns()
            is_listen = token == self.listen_id
            if token in self.chunk_terminator_ids:
                break
            self.current_turn_ended = False
            if token != self.speak_id:
                self.res_ids.append(token)
                n_speak += 1
            out = self.thinker.forward_embeds(
                self._embed_ids([token]), self.kv, output_hidden_states=True
            )
            logits_t, hiddens = out
            logits = logits_t[-1].float()
            hidden = hiddens[-1][-1]
            end_of_turn = token in self.turn_terminator_ids
            if end_of_turn:
                self.current_turn_ended = True
            if j != 0:
                unit_records.append((token, hidden, end_of_turn))
                unit_ids.append(token)

        self._feed_token(self.unit_end_id)
        self.unit_records.append(unit_records)
        torch.cuda.synchronize()
        cost_decision_ms = (time.monotonic() - t_dec) * 1000
        return ChunkDecision(
            is_listen, unit_ids, end_of_turn, n_speak,
            cost_embed_ms, cost_decision_ms, prefill_start_ns, prefill_done_ns,
            first_token_decoded_ns,
        )

    def collect_conditioning(self):
        """拍平全部单元的 (token_id, hidden, end_of_turn) 给 Talker。"""
        token_ids = []
        hiddens = []
        for unit in self.unit_records:
            for token_id, hidden, _eot in unit:
                token_ids.append(token_id)
                hiddens.append(hidden)
        if not hiddens:
            return None, None
        return (
            torch.tensor(token_ids, dtype=torch.long, device=self.device),
            torch.stack(hiddens).to(self.thinker.embed_tokens.weight.dtype),
        )

    def latest_unit_conditioning(self):
        """返回刚完成 unit 的语义条件，供实时 Talker 续写。

        官方 ``streaming_generate`` 每次只将当前 ``total_hidden_in_unit``
        交给 TTS；不能把整回合重放给已有 KV 的 Talker，否则条件与 codec
        会被重复编码，导致边界处音质和节奏退化。
        """
        records = self.unit_records[-1] if self.unit_records else []
        if not records:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(
                    (0, self.thinker.config.hidden_size),
                    dtype=self.thinker.embed_tokens.weight.dtype,
                    device=self.device,
                ),
            )
        return (
            torch.tensor([token_id for token_id, _hidden, _eot in records], device=self.device),
            torch.stack([hidden for _token_id, hidden, _eot in records]).to(
                self.thinker.embed_tokens.weight.dtype
            ),
        )
