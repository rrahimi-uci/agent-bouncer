#!/bin/bash
# Cross-family guard pipeline for extra bases: train LoRA guard + full eval (in-house + novel + base-vs-tuned).
# Resumable (train_guard skips existing adapters; guard_eval_pipeline is TAG-cached). Run from repo root.
set -u
# repo-relative paths (this script lives in experiments/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GUARD_PY:-$ROOT/.venv/bin/python}"
cd "$ROOT" || exit 1
MODE="${1:-full}"   # 'smoke' = GUARD_SMOKE train only (validate); 'full' = train+eval

# MODEL_ID | TAG
MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct|qwen2.5-1.5b"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|deepseek-r1-1.5b"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct|smollm2-1.7b"
)
for entry in "${MODELS[@]}"; do
  MID="${entry%%|*}"; TAG="${entry##*|}"; OUT="notebooks/outputs/${TAG}-guard"
  echo "================ [$TAG] $MID ($MODE) ================"
  if [ "$MODE" = "smoke" ]; then
    GUARD_SMOKE=1 MODEL_ID="$MID" OUT="notebooks/outputs/_smoke_${TAG}" "$PY" -u legacy/experiments/train_guard.py \
      && echo "[$TAG] SMOKE_OK" || echo "[$TAG] SMOKE_FAIL"
    continue
  fi
  echo "--- train ---"
  MODEL_ID="$MID" OUT="$OUT" "$PY" -u legacy/experiments/train_guard.py || { echo "[$TAG] TRAIN_FAIL"; continue; }
  echo "--- eval (in-house + novel + base-vs-tuned) ---"
  MODEL_ID="$MID" ADAPTER="$OUT/adapter" TAG="$TAG" "$PY" -u legacy/experiments/guard_eval_pipeline.py || { echo "[$TAG] EVAL_FAIL"; continue; }
  echo "[$TAG] DONE"
done
echo "ALL_EXTRA_MODELS_${MODE}_DONE"
