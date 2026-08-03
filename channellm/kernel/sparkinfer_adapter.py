"""L0 内核面 —— sparkinfer(SM120/SM121 CuTe DSL)适配层。

sparkinfer 是本项目唯一内核依赖(jieen1/sparkinfer fork,可深度优化):
JIT 编译、无构建步骤、attention.paged 支持 CUDA-graph replay。

MiniCPM-o 4.5 Thinker/Talker 骨干事实(设计文档 §1):
36 层、hidden 4096、32 heads / 8 KV heads(GQA)、head_dim 128、
rope_theta 1e6、上下文 40960、纯 full attention(无 hybrid/MTP)。

op 映射(P1 实施):
- prefill(变长批)        -> sparkinfer.attention.varlen
- decode/extend(paged KV) -> sparkinfer.attention.paged(FP8 KV 可选,graph-replayable)
- QKV/O + MLP 投影        -> bf16 起步;量化阶段用 gemm.block_fp8_linear / mxfp8_linear
- RMSNorm                -> 自带实现;sparkinfer.norm.mhc 是 RMSNorm+hyper-connection
                            融合核,MiniCPM-o 无 hyper-connection,不适用(勿误用)
- audio encoder(Whisper)/ TTS(llama)+ vocoder 的适配在 P1 逐模型评估

探测函数 probe() 在无 GPU/未安装时优雅降级,供 preflight 使用。
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

# P1 需要覆盖的最小 op 集
REQUIRED_OPS = (
    "attention.paged",
    "attention.varlen",
)
OPTIONAL_OPS = (
    "gemm.block_fp8_linear",
    "gemm.mxfp8_linear",
    "gemm.blockscaled",
    "quantization.nvfp4",
    "quantization.mxfp8",
)


@dataclasses.dataclass
class KernelProbe:
    available: bool
    version: str = ""
    ops: tuple[str, ...] = ()
    missing_required: tuple[str, ...] = ()
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.available and not self.missing_required


def probe() -> KernelProbe:
    """探测 sparkinfer 可用性与 op 覆盖。"""
    try:
        sparkinfer = importlib.import_module("sparkinfer")
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return KernelProbe(available=False, error=str(exc))

    ops: tuple[str, ...] = ()
    list_ops = getattr(sparkinfer, "list_ops", None)
    if callable(list_ops):
        try:
            ops = tuple(sorted(_op_qualname(op) for op in list_ops()))
        except Exception as exc:  # noqa: BLE001
            return KernelProbe(available=False, error=f"list_ops failed: {exc}")

    missing = tuple(op for op in REQUIRED_OPS if op not in ops) if ops else REQUIRED_OPS
    return KernelProbe(
        available=True,
        version=getattr(sparkinfer, "__version__", ""),
        ops=ops,
        missing_required=missing,
    )


def get_op(name: str) -> Any | None:
    """按 'group.op' 名取 sparkinfer op;不存在返回 None。"""
    try:
        sparkinfer = importlib.import_module("sparkinfer")
    except Exception:  # noqa: BLE001
        return None
    obj: Any = sparkinfer
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _op_qualname(op: Any) -> str:
    """list_ops() 返回 OpMeta 数据类;归一成 'group.name' 字符串。"""
    if isinstance(op, str):
        return op
    group = getattr(op, "group", "")
    name = getattr(op, "name", str(op))
    return f"{group}.{name}" if group else name
