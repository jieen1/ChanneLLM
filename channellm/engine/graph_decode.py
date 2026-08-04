"""GraphDecodeSession —— CUDA graph 捕获的 decode 步(P1 性能面)。

把一个 decode token 的完整前向(embed -> 36 层 -> lm_head)捕获成一张
图,每 token 只有 fill 静态缓冲 + replay 两件事,消除 36 层 x 数百个
kernel 的 launch 开销与 Python 循环开销。

状态契约:
- 会话的 KV 仍在 SparkinferPagedKV/PagedKVPool 里,图会话只读页池、
  只推进 seq.length —— speak 段结束后回到 eager forward_embeds
  (下一个音频 chunk)时状态完全一致;step() 通过 _expected_length 检测
  eager prefill 的外部推进并重新同步静态页表;
- hidden_buf 给出末层隐层(Talker hidden_text_merge 条件化所需),
  与 StaticGraphDecodeSession 的 step 返回签名一致;
- page_table/slot 缓冲每步 fill_ 更新;跨页边界(每 page_size 个
  token)由宿主侧更新 pt_buf 内容;
- sparkinfer 侧用 use_cuda_graph=True 的独立 plan + disable_split_kv,
  metadata(cache_seqlens/page_table)在 replay 时按缓冲内容重建。

用法:
    g = GraphDecodeSession(thinker, kv)
    g.capture(first_token_id)   # 捕获并执行第一步
    next_id = g.step(token_id)  # 之后每 token 一次 replay
"""

from __future__ import annotations

import torch

from channellm.engine.blocks import rotate_half


class GraphDecodeSession:
    def __init__(self, thinker, kv, max_width: int = 64) -> None:
        from sparkinfer.attention import paged

        self.thinker = thinker
        self.kv = kv
        self.device = thinker.embed_tokens.weight.device
        self.dtype = thinker.embed_tokens.weight.dtype
        cfg = thinker.config
        pool = kv.pool

        self.token_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.pos_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.pt_buf = torch.zeros(1, max_width, dtype=torch.int32, device=self.device)
        self.cs_buf = torch.empty(1, dtype=torch.int32, device=self.device)
        self.cu_buf = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        self.slot_phys_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.slot_off_buf = torch.empty(1, dtype=torch.long, device=self.device)
        self.max_width = max_width

        self._paged = paged
        self._plan = paged.plan(
            paged.Caps(
                device=self.device,
                mode="decode",
                dtype=self.dtype,
                kv_dtype=self.dtype,
                num_q_heads=cfg.num_q_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim_qk=cfg.head_dim,
                head_dim_vo=cfg.head_dim,
                page_size=pool.page_size,
                max_total_q=1,
                max_batch=1,
                max_page_table_width=max_width,
                max_work_items=1024,
                max_partial_rows=0,
                num_cache_pages=pool.num_pages,
                use_cuda_graph=True,
                copy_runtime_metadata=True,
            )
        )
        # graph 模式契约:捕获前必须准备固定地址的 replay metadata
        self._plan.prepare_decode_graph_replay_state(
            batch=1,
            max_page_table_width=max_width,
            total_q_capacity=1,
            max_cache_page_count=pool.num_pages,
            window_left=-1,
        )
        spec = self._plan.scratch_specs()[0]
        self._scratch = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)

        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits_buf: torch.Tensor | None = None
        self.hidden_buf: torch.Tensor | None = None
        self._expected_length: int | None = None

    def _sync_page_table(self) -> None:
        """把当前 seq 的页表同步进静态缓冲(跨页/会话开始时调用)。"""
        pages = self.kv.seq.pages
        if len(pages) > self.max_width:
            raise RuntimeError(f"page_table 宽度不足: {len(pages)} > {self.max_width}")
        self.pt_buf.fill_(-1)
        if pages:
            self.pt_buf[0, : len(pages)] = torch.tensor(
                pages, dtype=torch.int32, device=self.device
            )

    def _fill_step_buffers(self, token_id: int) -> None:
        """为下一个 decode token 填静态缓冲(宿主侧,每步一次)。"""
        kv = self.kv
        pool = kv.pool
        length = kv.seq.length
        slot = pool.slot_for(kv.seq, 1)  # 需要时顺带扩页(宿主侧)
        self.token_buf.fill_(token_id)
        self.pos_buf.fill_(length)
        self.cs_buf.fill_(length + 1)
        self.slot_phys_buf.copy_(slot[0])
        self.slot_off_buf.copy_(slot[1])

    @torch.no_grad()
    def _decode_step(self) -> None:
        """单 token 完整前向,写入 self.logits_buf。全设备算子,可捕获。"""
        thinker = self.thinker
        cfg = thinker.config
        pool = self.kv.pool

        hidden = thinker.embed_tokens(self.token_buf)
        cos, sin = thinker.rotary(self.pos_buf, dtype=self.dtype)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        for li, layer in enumerate(thinker.layers):
            h = layer.input_layernorm(hidden)
            q = layer.self_attn.q_norm(
                layer.self_attn.q_proj(h).view(1, cfg.num_q_heads, cfg.head_dim)
            )
            k = layer.self_attn.k_norm(
                layer.self_attn.k_proj(h).view(1, cfg.num_kv_heads, cfg.head_dim)
            )
            v = layer.self_attn.v_proj(h).view(1, cfg.num_kv_heads, cfg.head_dim)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            pool.k_pages[li][self.slot_phys_buf, self.slot_off_buf] = k
            pool.v_pages[li][self.slot_phys_buf, self.slot_off_buf] = v

            attn_out = torch.empty(
                1, cfg.num_q_heads, cfg.head_dim, dtype=self.dtype, device=self.device
            )
            binding = self._paged.bind(
                self._plan,
                scratch=self._scratch,
                q=q,
                k_cache=pool.k_pages[li],
                v_cache=pool.v_pages[li],
                output=attn_out,
                page_table=self.pt_buf,
                cache_seqlens=self.cs_buf,
                cu_seqlens_q=self.cu_buf,
                active_total_q=1,
                disable_split_kv=True,
            )
            self._paged.run(binding=binding)

            hidden = hidden + layer.self_attn.o_proj(
                attn_out.view(1, cfg.num_q_heads * cfg.head_dim)
            )
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        self.hidden_buf = hidden
        self.logits_buf = thinker.lm_head(thinker.norm(hidden))

    @torch.no_grad()
    def capture(self, dummy_token_id: int = 0) -> None:
        """在会话开始处捕获 decode 图(必须在真实 prefill 之前调用)。

        warmup 两步 + 捕获一步都用 dummy token 在空池上执行:写入的
        3 个槽位是垃圾,但会被真实 prefill 的前几个 token 覆盖,
        会话状态零污染。捕获后 seq.length 复位为 0。
        """
        if self.graph is not None:
            raise RuntimeError("已捕获,不能重复 capture")
        if self.kv.seq.length != 0:
            raise RuntimeError("capture 必须在空 KV 上执行(prefill 之前)")
        self._sync_page_table()
        # warmup:与捕获完全相同的代码路径,保证 kernel 全部编译完
        for _ in range(2):
            self._fill_step_buffers(dummy_token_id)
            self._decode_step()
            self.kv.seq.advance(1)
            torch.cuda.synchronize()
        # 捕获(捕获本身会真实执行这一步)
        self._fill_step_buffers(dummy_token_id)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._decode_step()
        torch.cuda.synchronize()
        # 复位会话状态:释放 warmup/capture 占用的 dummy 页并同步静态页表,
        # 否则真实 prefill 的页表与捕获期残留不一致,第二个 token 起分歧。
        self.kv.pool.allocator.free(self.kv.seq.pages)
        self.kv.seq.pages = []
        self.kv.seq.length = 0
        self._sync_page_table()
        self._expected_length = None

    @torch.no_grad()
    def step(self, token_id: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """replay 一步 decode,返回 ``(next_token_id, logits, last_hidden)``。

        logits/hidden 是会话复用缓冲的视图,下一次 step 会覆写。duplex 环在
        graph 步之间会插入 eager prefill(音频 chunk),``_expected_length``
        检测到外部推进即重新同步静态页表,保证 replay 读到真实页表。
        """
        if self.graph is None:
            raise RuntimeError("先调用 capture()")
        seq = self.kv.seq
        external = self._expected_length is None or seq.length != self._expected_length
        pages_before = len(seq.pages)
        self._fill_step_buffers(token_id)
        if external or len(seq.pages) != pages_before:
            self._sync_page_table()  # 外部 prefill 或跨页边界:更新静态页表
        self.graph.replay()
        seq.advance(1)
        self._expected_length = seq.length
        assert self.logits_buf is not None and self.hidden_buf is not None
        return (
            int(self.logits_buf[0].argmax().item()),
            self.logits_buf[0],
            self.hidden_buf[0],
        )
