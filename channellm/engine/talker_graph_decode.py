"""TalkerGraphDecodeSession —— Talker decode 步的 CUDA graph 捕获(bf16 原生)。

Talker(20 层 Llama,12 头 MHA/head_dim 64)在 TorchStaticKV + SDPA 下逐帧
decode 是 launch 受限负载(实测 ~5-9ms/帧)。本会话把单帧 decode 步捕获为
按宽度分桶的图,机制与 Thinker paged graph 同源但更简单——连续静态 KV 没有
页表,只有 kv.length 一个状态:

- attention 用显式 bf16 bmm + 加性 mask(padding 位 -inf;``exp(-inf)=0``、
  ``x+0=x`` 精确成立),12 头无 GQA 分组;
- KV 写位置(pos_buf)与有效长度(len_buf)是设备端缓冲,replay 按值读取;
  mask 每步由 ``arange < len_buf`` 生成,36→20 层共用;
- 条件化 prefill(每 unit 一次,多 token)仍走 eager forward_embeds,decode
  帧走 replay;宿主侧按 kv.length 填缓冲,无需任何外部同步协议;
- 每个桶的首个 token 以 eager(同一显式 attention 代码)执行并作为该宽度
  的预热,第二个 token 捕获(捕获后立即 replay 一次,覆写捕获期副作用)。

质量门禁:``scripts/p1_talker_graph_check.py`` 贪心 codec 流必须与 eager
SDPA 路径一致;通过后才能接入 TalkerStream 默认路径。
"""

from __future__ import annotations

import dataclasses
import time

import torch

from channellm.engine.blocks import TorchStaticKV, rotate_half

DEFAULT_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096)


def select_bucket(length: int, buckets: tuple[int, ...]) -> int:
    """返回容纳 ``length`` 个有效 token 的最小桶宽。"""
    if length <= 0:
        raise ValueError("length must be positive")
    for width in buckets:
        if width >= length:
            return width
    raise MemoryError(f"talker graph decode 宽度不足: {length} > {buckets[-1]}")


@dataclasses.dataclass
class _BucketEntry:
    width: int
    primed: bool = False
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None


class TalkerGraphDecodeSession:
    """把 Talker 单帧 decode 捕获为按宽度分桶的 CUDA graph。"""

    def __init__(
        self,
        talker,
        kv: TorchStaticKV,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    ) -> None:
        if not isinstance(kv, TorchStaticKV):
            raise TypeError("TalkerGraphDecodeSession 只支持 TorchStaticKV")
        cfg = talker.config
        buckets = tuple(sorted(set(int(b) for b in buckets)))
        if not buckets or buckets[-1] > kv.max_seq_len:
            raise ValueError(f"buckets 必须在 (0, {kv.max_seq_len}] 内,得到 {buckets}")
        self.talker = talker
        self.kv = kv
        self.buckets = buckets
        self.device = talker.emb_code.weight.device
        self.dtype = talker.emb_code.weight.dtype
        self.scale = cfg.head_dim**-0.5

        self.token_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.pos_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.len_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self._arange = torch.arange(kv.max_seq_len, device=self.device)
        self._neg_inf = torch.tensor(float("-inf"), dtype=self.dtype, device=self.device)
        self._entries: dict[int, _BucketEntry] = {}
        self._logits_buf: torch.Tensor | None = None
        self.capture_count = 0
        self.capture_ms = 0.0

    @torch.no_grad()
    def _decode_step(self, width: int) -> None:
        talker = self.talker
        cfg = talker.config
        kv = self.kv

        hidden = talker.emb_code(self.token_buf)
        cos, sin = talker.rotary(self.pos_buf, dtype=self.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # 替换式 mask:padding 位的分数被 -inf 直接替换,而非加 -inf。
        # padding 槽位可能含陈旧数据(跨会话复用的显存位模式可含 NaN/Inf),
        # 加性 mask 会遇到 Inf+(-Inf)=NaN / NaN 传播;替换式与 padding 内容
        # 完全解耦。
        valid = (self._arange[:width] < self.len_buf).view(1, 1, width)

        for li, layer in enumerate(talker.layers):
            h = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_proj(h).view(1, cfg.num_heads, cfg.head_dim)
            k = attn.k_proj(h).view(1, cfg.num_kv_heads, cfg.head_dim)
            v = attn.v_proj(h).view(1, cfg.num_kv_heads, cfg.head_dim)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            kv.k_pages[li].index_copy_(0, self.pos_buf, k)
            kv.v_pages[li].index_copy_(0, self.pos_buf, v)

            k_slice = kv.k_pages[li, :width].permute(1, 0, 2)
            v_slice = kv.v_pages[li, :width].permute(1, 0, 2)
            scores = torch.matmul(
                q.squeeze(0).unsqueeze(1), k_slice.transpose(1, 2)
            )
            scores = torch.where(valid, scores * self.scale, self._neg_inf)
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v_slice)
            out = out.reshape(1, cfg.num_heads * cfg.head_dim)

            hidden = hidden + attn.o_proj(out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        self._logits_buf = talker.head_code(talker.norm(hidden))

    @torch.no_grad()
    def step(self, token_id: int) -> tuple[int, torch.Tensor]:
        """replay/eager/capture 一帧 decode,返回 ``(argmax_token, logits)``。

        logits 是会话复用缓冲的视图,下一次 step 会覆写。
        """
        length = self.kv.length
        if length >= self.kv.max_seq_len:
            raise MemoryError(f"talker KV 容量已满: {length} >= {self.kv.max_seq_len}")
        width = select_bucket(length + 1, self.buckets)
        self.token_buf.fill_(token_id)
        self.pos_buf.fill_(length)
        self.len_buf.fill_(length + 1)

        entry = self._entries.get(width)
        if entry is None:
            entry = _BucketEntry(width=width)
            self._entries[width] = entry

        if entry.graph is not None:
            entry.graph.replay()
        elif not entry.primed:
            self._decode_step(width)
            entry.primed = True
            entry.logits = self._logits_buf
        else:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.cuda.graph(graph):
                self._decode_step(width)
            entry.graph = graph
            entry.logits = self._logits_buf
            entry.graph.replay()
            torch.cuda.synchronize()
            self.capture_count += 1
            self.capture_ms += (time.monotonic() - t0) * 1000

        self.kv.length = length + 1
        assert entry.logits is not None
        return int(entry.logits[0].argmax().item()), entry.logits[0]
