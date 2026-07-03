.DEFAULT_GOAL := help
.PHONY: help setup install data data-demo demo report train-sft train-grpo train-dpo build-dataset train-model test-model eval bench benchmarks benchmarks-full curves baselines incumbents serve studio test lint format clean clean-runs

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a venv and install dev + eval/benchmark extras
	python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev,eval]'

install:  ## Editable install (core only)
	pip install -e .

data:  ## Download + unify the training datasets (WildGuardMix / BeaverTails / Aegis)
	python scripts/data/download_data.py

train-sft:  ## Supervised fine-tune (config=configs/model/sft_qwen3_0_6b.yaml)
	agent-bouncer train sft --config $(or $(config),configs/model/sft_qwen3_0_6b.yaml)

train-grpo:  ## GRPO reasoning guard (config=configs/model/grpo_qwen3_1_7b.yaml)
	agent-bouncer train grpo --config $(or $(config),configs/model/grpo_qwen3_1_7b.yaml)

train-dpo:  ## DPO over-refusal tuning (config=configs/model/dpo_qwen3.yaml)
	agent-bouncer train dpo --config $(or $(config),configs/model/dpo_qwen3.yaml)

data-demo:  ## Build the balanced BeaverTails demo dataset (ungated)
	python scripts/data/prepare_beavertails_demo.py

demo:  ## Train Regime-A encoder and check it beats the baseline (end-to-end)
	python scripts/train/demo_train_eval.py

report:  ## Render results table + model card from outputs/demo_results.json
	python scripts/report/make_report.py

eval:  ## Run the eval harness on the smoke set (uses the reference guard)
	agent-bouncer eval tests/data/smoke.jsonl --run-name keyword-baseline

bench: benchmarks  ## Alias for `benchmarks`

benchmarks:  ## Download + run the standard benchmark suite (uses .env keys); writes outputs/BENCHMARKS.md
	python scripts/eval/run_benchmarks.py $(if $(per_class),--per-class $(per_class),)

benchmarks-full:  ## Download the full-size ungated benchmark datasets to data/benchmarks/full
	python scripts/data/download_full_benchmarks.py $(if $(benchmarks),--benchmarks $(benchmarks),)

baselines:  ## Score incumbent guards (Llama Guard / ShieldGemma) on our harness
	python -m agent_bouncer.evaluation.baselines

incumbents:  ## Compare vs OpenAI + gated incumbents on the test subset (uses .env keys)
	python scripts/eval/run_incumbents.py $(if $(limit),--limit $(limit),)

curves:  ## Compute ROC / PR / AUC curves for local guards -> outputs/curves.json
	python scripts/report/compute_curves.py

build-dataset:  ## Build a training set (strategy=balanced name=my-set sources="beavertails xstest")
	python scripts/data/build_dataset.py --strategy $(or $(strategy),balanced) --name $(or $(name),my-set) --sources $(sources)

train-model:  ## Train a registered model (model=smollm2-1.7b technique=sft [max_steps=40])
	python scripts/train/run_training.py --model $(or $(model),distilbert) --technique $(or $(technique),sft) \
	  $(if $(max_steps),--max-steps $(max_steps),) $(if $(epochs),--epochs $(epochs),)

test-model:  ## Test a trained version against benchmarks (exp=<experiment-id> [device=mps])
	python scripts/eval/run_testing.py --exp $(exp) $(if $(device),--device $(device),) $(if $(per_class),--per-class $(per_class),)

serve: studio  ## Alias for `studio`

studio:  ## Launch the Benchmark Studio dashboard + /screen API (http://127.0.0.1:8000)
	uvicorn agent_bouncer.serving.api:app --host 127.0.0.1 --port 8000

test:  ## Run tests
	pytest

lint:  ## Lint with ruff
	ruff check src tests

format:  ## Auto-format with ruff
	ruff format src tests && ruff check --fix src tests

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ dist build *.egg-info

clean-runs:  ## Remove ALL trained models + Studio results/experiments (keeps datasets + benchmark caches)
	rm -rf outputs/models outputs/model_store outputs/experiments outputs/predictions
	rm -rf outputs/demo-encoder outputs/demo-decoder-sft outputs/demo-grpo outputs/grpo-qwen3-0.6b \
	       outputs/decoder-sft-Qwen3-1.7B outputs/smoke-grpo outputs/smoke-sft
	rm -f  outputs/benchmark_results.json outputs/curves.json outputs/ensemble_results.json \
	       outputs/demo_results.json outputs/BENCHMARKS.md outputs/MODEL_CARD.md outputs/RESULTS.md
	@echo "✓ cleaned trained models + experiments + Studio diagrams"
	@echo "  kept: data/train_sets (datasets), data/benchmarks (caches), outputs/logs"
