#!/usr/bin/env python
"""ChanneLLM 环境 preflight —— SM120 契约检查(R6:独立干净 venv,逐个解)。

用法:
    python scripts/preflight.py           # 基础检查(torch 缺失只 WARN)
    python scripts/preflight.py --gpu     # GPU 全量检查(torch 必须可用)
"""

from __future__ import annotations

import argparse
import importlib
import sys

REQUIRE_PY = (3, 10)
EXPECT_CAPABILITY = (12, 0)  # SM120 Blackwell


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = False

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append((name, status, detail))
        if status == "FAIL":
            self.failed = True

    def print(self) -> None:
        width = max(len(r[0]) for r in self.rows)
        for name, status, detail in self.rows:
            print(f"{name.ljust(width)}  [{status:4}]  {detail}")


def check_python(report: Report) -> None:
    ok = sys.version_info >= REQUIRE_PY
    report.add(
        "python",
        "PASS" if ok else "FAIL",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


def _try_import(name: str):
    try:
        return importlib.import_module(name), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def check_torch(report: Report, required: bool) -> object | None:
    torch, err = _try_import("torch")
    if torch is None:
        report.add("torch", "FAIL" if required else "WARN", err or "not installed")
        return None
    detail = f"{torch.__version__} cuda={torch.version.cuda}"
    if not torch.cuda.is_available():
        report.add("torch", "FAIL" if required else "WARN", detail + " (cuda unavailable)")
        return torch
    capability = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    ok = capability == EXPECT_CAPABILITY
    report.add(
        "torch.cuda",
        "PASS" if ok else ("FAIL" if required else "WARN"),
        f"{name} sm{capability[0]}{capability[1]} {mem_gb:.0f}GB",
    )
    return torch


def check_package(
    report: Report, name: str, required: bool, version_attr: str = "__version__"
) -> None:
    module, err = _try_import(name)
    if module is None:
        report.add(name, "FAIL" if required else "WARN", err or "not installed")
        return
    report.add(name, "PASS", str(getattr(module, version_attr, "")))


def check_sparkinfer(report: Report, required: bool) -> None:
    from channellm.kernel.sparkinfer_adapter import probe

    result = probe()
    if not result.available:
        report.add("sparkinfer", "FAIL" if required else "WARN", result.error)
        return
    status = "PASS" if result.ready else ("FAIL" if required else "WARN")
    detail = f"v{result.version} ops={len(result.ops)}"
    if result.missing_required:
        detail += f" missing={','.join(result.missing_required)}"
    report.add("sparkinfer", status, detail)


def check_weights(report: Report) -> None:
    from channellm.models.minicpmo import find_weights

    path = find_weights()
    if path is None:
        report.add("weights(MiniCPM-o-4_5)", "WARN", "not found in HF cache")
    else:
        report.add("weights(MiniCPM-o-4_5)", "PASS", str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="GPU 全量检查(torch 必须可用)")
    args = parser.parse_args()

    report = Report()
    check_python(report)
    check_torch(report, required=args.gpu)
    check_package(report, "transformers", required=args.gpu)
    check_package(report, "numpy", required=True)
    check_sparkinfer(report, required=args.gpu)
    check_weights(report)
    report.print()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
