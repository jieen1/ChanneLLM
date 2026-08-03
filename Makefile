# ChanneLLM —— 开发与验证任务
PYTHON ?= .venv/bin/python
PKGS = channellm tests scripts

.DEFAULT_GOAL := help

help: ## 概览
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## 基础 + dev(CPU 可测试面)
	./scripts/setup_env.sh --base

install-cuda: ## 全量(torch 钉版 + sparkinfer fork)
	./scripts/setup_env.sh

lint: ## ruff 门禁
	$(PYTHON) -m ruff check .

format: ## 自动修复 + 格式化
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format $(PKGS)

test: ## 单元测试
	$(PYTHON) -m pytest -q

preflight: ## 基础环境检查
	$(PYTHON) scripts/preflight.py

preflight-gpu: ## GPU 全量检查
	$(PYTHON) scripts/preflight.py --gpu
