#!/usr/bin/env bash
# ChanneLLM 环境搭建 —— 独立干净 venv(R6),版本契约见 pyproject.toml。
#
#   ./scripts/setup_env.sh            # 基础(dev)+ CUDA extra + sparkinfer fork
#   ./scripts/setup_env.sh --base     # 只装基础(CPU 可测试面)
set -euo pipefail
cd "$(dirname "$0")/.."

SPARKINFER_PATH="${SPARKINFER_PATH:-$HOME/project/sparkinfer}"
PY=".venv/bin/python"

# 国内源默认开启(阿里云);PYPI_MIRROR="" 可回退官方源。
# 注意:HF 权重下载另算 —— Xet 仓库禁用 hf-mirror(见 docs/environment.md)。
PYPI_MIRROR="${PYPI_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
if [ -n "$PYPI_MIRROR" ]; then
    export UV_DEFAULT_INDEX="$PYPI_MIRROR"
    echo "==> using PyPI mirror: $PYPI_MIRROR"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

if [ ! -x "$PY" ]; then
    uv venv .venv --python 3.12
fi

echo "==> base + dev"
uv pip install --python "$PY" -e ".[dev]"

if [ "${1:-}" != "--base" ]; then
    echo "==> cuda + omni extra (torch==2.13.0 钉版 + 官方模型路径依赖)"
    uv pip install --python "$PY" -e ".[cuda,omni]"

    echo "==> sparkinfer fork (editable,依赖同样走镜像): $SPARKINFER_PATH"
    if [ -d "$SPARKINFER_PATH" ]; then
        uv pip install --python "$PY" -e "$SPARKINFER_PATH"
    else
        echo "WARN: $SPARKINFER_PATH 不存在,跳过 sparkinfer(preflight 会报 WARN)" >&2
    fi
fi

echo "==> preflight"
"$PY" scripts/preflight.py || true
echo "done."
