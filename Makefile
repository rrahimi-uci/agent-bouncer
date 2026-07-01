.DEFAULT_GOAL := help
.PHONY: help setup install data train-sft train-grpo eval bench baselines serve test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a venv and install dev + eval extras
	python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev,eval]'

install:  ## Editable install (core only)
	pip install -e .

data:  ## Download + unify the training datasets (WildGuardMix / BeaverTails / Aegis)
	python scripts/download_data.py

train-sft:  ## Supervised fine-tune (config=configs/model/sft_qwen3_0_6b.yaml)
	agent-bouncer train sft --config $(or $(config),configs/model/sft_qwen3_0_6b.yaml)

train-grpo:  ## GRPO reasoning guard (config=configs/model/grpo_qwen3_1_7b.yaml)
	agent-bouncer train grpo --config $(or $(config),configs/model/grpo_qwen3_1_7b.yaml)

eval:  ## Run the eval harness on the smoke set (uses the reference guard)
	agent-bouncer eval tests/data/smoke.jsonl --run-name keyword-baseline

bench:  ## Run the standard benchmarks (GuardBench / PINT / XSTest)
	python scripts/run_eval.sh

baselines:  ## Score incumbent guards (Llama Guard / ShieldGemma) on our harness
	python -m agent_bouncer.eval.baselines

serve:  ## Start the FastAPI screening server
	uvicorn agent_bouncer.serve.api:app --reload

test:  ## Run tests
	pytest

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Auto-format with ruff
	ruff format src tests && ruff check --fix src tests

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ dist build *.egg-info
