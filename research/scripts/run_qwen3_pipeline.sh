#!/usr/bin/env bash
# Full Qwen3-4B guard pipeline (train -> eval), resumable/cached. Reuses model-independent baselines.
set -e
export MODEL_ID="Qwen/Qwen3-4B"
export ADAPTER="outputs/qwen3-4b-guard/adapter"
export OUT="outputs/qwen3-4b-guard"
export TAG="qwen3-4b"
export TOKENIZERS_PARALLELISM=false
echo "[pipeline] $(date) — training Qwen3-4B guard (skips if adapter exists) ..."
.venv/bin/python -u scripts/train_guard.py
echo "[pipeline] $(date) — eval (corrected + novel + base-vs-tuned) ..."
.venv/bin/python -u scripts/guard_eval_pipeline.py
echo "[pipeline] $(date) — DONE_QWEN3_PIPELINE"
