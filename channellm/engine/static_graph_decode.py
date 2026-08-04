"""StaticGraphDecodeSession —— fp32 TorchStaticKV 上的 CUDA graph decode(P1 性能面)。

设计动机(质量优先):默认质量路径是 fp32 + Torch SDPA,逐 token 对齐官方;
sparkinfer paged 内核是 tensor-core 路线,fp32 精度要么不可得(tf32 截断尾数,
等价于引入 bf16 级误差),要么需要全新 CUDA-core 内核。单 token decode 的
attention 是带宽_bound(batch=1 GEMV),tensor core 并非必需,因此本会话直接
在 fp32 质量路径上做图捕获,精度语义与 eager 完全同源:

- 每个 decode token 的完整前向(embed -> 36 层 -> lm_head)捕获成一张图,
  消除 36 层 x 每层 ~10 个 kernel 的 launch 与 Python dispatch 开销;
- attention 用显式 fp32 实现(bmm + 加性 mask + softmax + bmm),GQA 用
  广播 bmm,不做 KV head 展开拷贝;不经过 SDPA 后端分派;
- KV 长度每步增长,而图形状必须静态:按 2 的幂分桶(最后一个桶为
  max_seq_len),attention 读取 ``[:width]`` 前缀并用 mask 选出有效
  ``[:length]``。padding 只增加 <=2x 的 KV 读流量(带宽受限下可接受),
  不改变有效位置的数值;
- mask 中 padding 位为 ``-inf``:softmax 里 ``exp(-inf)=0`` 精确成立,
  ``x+0.0=x`` 亦精确成立,故带 mask 的 softmax 与无前缀截断版本在
  IEEE fp32 下按位等价(与规约顺序无关);padding 位的 QK^T 值是陈旧
  但有限的 KV 内容,不会传播 NaN;
- 缓冲(token/pos/len)地址固定,每步宿主侧 fill;KV 写位置与有效长度
  通过设备端索引缓冲在 replay 时按值读取,跨桶/跨回合无需重新捕获。

状态契约:
- 会话的 KV 就是传入的 ``TorchStaticKV``;图会话只读写其页缓冲、只推进
  ``length``。eager prefill(音频 chunk)与 graph decode 可交替进行;
- 每个桶的第一个 token 以 eager 执行(真实结果,同时为该宽度的 bmm
  预热 cuBLAS),第二个 token 执行捕获(捕获步本身真实执行),其后 replay;
- 满容量显式失败,不静默截断。

质量门禁:在 ``scripts/p1_graph_decode_check.py --dtype fp32
--kv-backend static`` 输出 parity PASS 之前,不得接入默认路径。
"""

from __future__ import annotations

import dataclasses
import time

import torch

from channellm.engine.blocks import TorchStaticKV, rotate_half

DEFAULT_BUCKETS: tuple[int, ...] = (
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    40960,
)


def select_bucket(length: int, buckets: tuple[int, ...]) -> int:
    """返回容纳 ``length`` 个有效 token 的最小桶宽。"""
    if length <= 0:
        raise ValueError("length must be positive")
    for width in buckets:
        if width >= length:
            return width
    raise MemoryError(f"static graph decode 宽度不足: {length} > {buckets[-1]}")


@dataclasses.dataclass
class _BucketEntry:
    width: int
    primed: bool = False
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None
    hidden: torch.Tensor | None = None


class StaticGraphDecodeSession:
    """把 fp32 decode 步捕获为按宽度分桶的 CUDA graph。"""

    def __init__(
        self,
        thinker,
        kv: TorchStaticKV,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    ) -> None:
        if not isinstance(kv, TorchStaticKV):
            raise TypeError("StaticGraphDecodeSession 只支持 TorchStaticKV")
        cfg = thinker.config
        if cfg.num_q_heads % cfg.num_kv_heads != 0:
            raise ValueError("GQA 分组要求 num_q_heads 整除 num_kv_heads")
        buckets = tuple(sorted(set(int(b) for b in buckets)))
        if not buckets or buckets[-1] > kv.max_seq_len:
            raise ValueError(
                f"buckets 必须在 (0, {kv.max_seq_len}] 内,得到 {buckets}"
            )
        self.thinker = thinker
        self.kv = kv
        self.buckets = buckets
        self.device = thinker.embed_tokens.weight.device
        self.dtype = thinker.embed_tokens.weight.dtype
        self.group = cfg.num_q_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5

        self.token_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.pos_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.len_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self._arange = torch.arange(kv.max_seq_len, device=self.device)
        self._entries: dict[int, _BucketEntry] = {}
        self._logits_out: torch.Tensor | None = None
        self._hidden_out: torch.Tensor | None = None
        self.capture_count = 0
        self.capture_ms = 0.0

    # ------------------------------------------------------------------
    # 单 token 前向(全设备算子,可捕获;宽度 ``width`` 决定静态形状)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_step(self, width: int) -> None:
        thinker = self.thinker
        cfg = thinker.config
        kv = self.kv

        hidden = thinker.embed_tokens(self.token_buf)
        cos, sin = thinker.rotary(self.pos_buf, dtype=self.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # mask 每步一次,36 层共用。padding 位 -inf,有效位 0。
        valid = self._arange[:width] < self.len_buf
        mask = torch.where(valid, 0.0, float("-inf"))

        for li, layer in enumerate(thinker.layers):
            h = layer.input_layernorm(hidden)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(h).view(1, cfg.num_q_heads, cfg.head_dim))
            k = attn.k_norm(attn.k_proj(h).view(1, cfg.num_kv_heads, cfg.head_dim))
            v = attn.v_proj(h).view(1, cfg.num_kv_heads, cfg.head_dim)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            # 写位置由设备端 pos_buf 决定,replay 安全。
            kv.k_pages[li].index_copy_(0, self.pos_buf, k)
            kv.v_pages[li].index_copy_(0, self.pos_buf, v)

            # 显式 GQA attention:q [HKV, G, D] @ k^T [HKV, D, W] -> [HKV, G, W]
            k_slice = kv.k_pages[li, :width].permute(1, 0, 2)
            v_slice = kv.v_pages[li, :width].permute(1, 0, 2)
            scores = torch.matmul(
                q.squeeze(0).view(cfg.num_kv_heads, self.group, cfg.head_dim),
                k_slice.transpose(1, 2),
            )
            scores = scores * self.scale + mask.view(1, 1, width)
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v_slice)
            out = out.reshape(1, cfg.num_q_heads * cfg.head_dim)

            hidden = hidden + attn.o_proj(out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        self._hidden_out = hidden
        self._logits_out = thinker.lm_head(thinker.norm(hidden))

    # ------------------------------------------------------------------
    # 宿主侧驱动
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, token_id: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """replay/eager/capture 一步 decode。

        返回 ``(next_token_id, logits, last_hidden)``;logits/hidden 是会话
        持有的缓冲视图,下一次 step 会覆写其内容,调用方需要保留就立刻克隆。
        """
        length = self.kv.length
        if length >= self.kv.max_seq_len:
            raise MemoryError(
                f"static KV 容量已满: {length} >= {self.kv.max_seq_len}"
            )
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
            # 桶内第一个 token:eager 执行(真实结果 + 预热该宽度的 bmm)。
            self._decode_step(width)
            entry.primed = True
            entry.logits = self._logits_out
            entry.hidden = self._hidden_out
        else:
            # 桶内第二个 token:捕获。本机实测(Blackwell SM120 + torch 2.13
            # cuBLAS):捕获期执行的 GEMM 结果与 eager 分歧,而 replay 逐位
            # 一致——捕获期执行的副作用不可信。因此捕获后立即 replay 一次,
            # 用正确结果覆写捕获期写进 KV 位置 L 的垃圾 k/v 与 logits 缓冲。
            graph = torch.cuda.CUDAGraph()
            t0 = time.monotonic()
            with torch.cuda.graph(graph):
                self._decode_step(width)
            entry.graph = graph
            entry.logits = self._logits_out
            entry.hidden = self._hidden_out
            entry.graph.replay()
            torch.cuda.synchronize()
            self.capture_count += 1
            self.capture_ms += (time.monotonic() - t0) * 1000

        self.kv.length = length + 1
        next_id = int(entry.logits[0].argmax().item())
        assert entry.logits is not None and entry.hidden is not None
        return next_id, entry.logits[0], entry.hidden[0]
