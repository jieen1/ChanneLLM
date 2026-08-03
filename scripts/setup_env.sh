#!/usr/bin/env bash
# ChanneLLM 环境搭建 —— 独立干净 venv(R6),版本契约见 pyproject.toml。
#
#   ./scripts/setup_env.sh            # 基础(dev)+ CUDA extra + sparkinfer fork
#   ./scripts/setup_env.sh --base     # 只装基础(CPU 可测试面)
set -euo pipefail
cd "$(dirname "$0")/.."

SPARKINFER_PATH="${SPARKINFER_PATH:-$HOME/project/sparkinfer}"
PY=".venv/bin/python"

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
    echo "==> cuda extra (torch==2.13.0 钉版)"
    uv pip install --python "$PY" -e ".[cuda]"

    echo "==> sparkinfer fork (editable): $SPARKINFER_PATH"
    if [ -d "$SPARKINFER_PATH" ]; then
        uv pip install --python "$PY" -e "$SPARKINFER_PATH"
    else
        echo "WARN: $SPARKINFER_PATH 不存在,跳过 sparkinfer(preflight 会报 WARN)" >&2
    fi
fi

echo "==> preflight"
"$PY" scripts/preflight.py || true
echo "done."
