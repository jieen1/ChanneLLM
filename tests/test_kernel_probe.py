import dataclasses

from channellm.kernel.sparkinfer_adapter import _op_qualname, probe


class FakeOpMeta:
    def __init__(self, group: str, name: str) -> None:
        self.group = group
        self.name = name


def test_op_qualname_normalizes():
    assert _op_qualname(FakeOpMeta("attention", "paged")) == "attention.paged"
    assert _op_qualname("attention.varlen") == "attention.varlen"


def test_probe_never_raises():
    result = probe()
    # 无 sparkinfer 时优雅降级;有时必须列出 op
    assert dataclasses.is_dataclass(result)
    if result.available:
        assert len(result.ops) > 0
