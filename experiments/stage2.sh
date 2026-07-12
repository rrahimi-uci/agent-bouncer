#!/bin/bash
# Stage 2 (full remediation): the 5 non-primary bases x {SFT,DPO,GRPO} x 3 seeds,
# clean source-family-disjoint split (OR_BENCH_CAP=0), tie-aware eval on the frozen rows.
# Qwen3-8B is intended for an 80GB card (pass ONLY_8B=1 there); the 4 smaller bases run
# on a 40GB card (default, skips qwen3-8b). Resumable: train skips existing adapters and
# we skip a (base,obj,seed) whose summary already exists.
cd ~/guard
FR=notebooks/outputs/frozen_eval_rows.json
ND=notebooks/outputs/nb-smollm3-guard
declare -A MID=(
  [qwen2.5-1.5b]="Qwen/Qwen2.5-1.5B-Instruct"
  [smollm2-1.7b]="HuggingFaceTB/SmolLM2-1.7B-Instruct"
  [deepseek-r1-1.5b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  [qwen3-4b]="Qwen/Qwen3-4B"
  [qwen3-8b]="Qwen/Qwen3-8B"
)
if [ "${ONLY_8B:-0}" = "1" ]; then BASES=(qwen3-8b); else BASES=(qwen2.5-1.5b smollm2-1.7b deepseek-r1-1.5b qwen3-4b); fi
for base in "${BASES[@]}"; do
  for obj in sft dpo grpo; do
    for seed in 42 43 44; do
      TAG=${base}-clean-${obj}-s${seed}; OUT=outputs/$TAG
      if [ -f "$ND/summary_${TAG}.json" ]; then echo "===== [$(date +%H:%M)] SKIP (summary exists) $TAG ====="; continue; fi
      echo "===== [$(date +%H:%M)] TRAIN $TAG (${MID[$base]}) ====="
      if [ "$obj" = "sft" ]; then
        MODEL_ID="${MID[$base]}" OUT=$OUT OR_BENCH_CAP=0 GUARD_SEED=$seed python3 -u experiments/train_guard.py 2>&1 | tail -3
      else
        MODEL_ID="${MID[$base]}" OUT=$OUT TECHNIQUE=$obj OR_BENCH_CAP=0 GUARD_SEED=$seed python3 -u experiments/train_guard_pref.py 2>&1 | tail -3
      fi
      echo "===== [$(date +%H:%M)] EVAL $TAG ====="
      MODEL_ID="${MID[$base]}" ADAPTER=$OUT/adapter TAG=$TAG FROZEN_ROWS=$FR python3 -u experiments/guard_eval_pipeline.py 2>&1 | tail -4
      rm -rf "$OUT"/checkpoint-* 2>/dev/null   # drop intermediate checkpoints; keep adapter + summary + scores
      echo "===== [$(date +%H:%M)] SUMMARY_DONE $TAG ====="
    done
  done
done
echo "STAGE2_ALL_DONE (ONLY_8B=${ONLY_8B:-0})"
