#!/bin/bash
# Reproducible fleet launch for the starting-type adaptation study (proposal papers/unified-report).
# One VM per checkpoint; GPU-sized by params (>=3B -> A100-80GB a2-ultragpu-1g; <=2B -> A100-40GB
# a2-highgpu-1g). Each VM runs experiments/deploy/sta_startup.sh: preflight -> train (11 cells:
# 1 unmodified + 5 sft + 5 kl_sft@primary_beta) -> eval (per-checkpoint parquet) -> upload -> stop.
#
# Prereqs: gcloud auth; a HF_TOKEN file (gated gemma/llama/ai2 licenses already accepted) at
# $HF_TOKEN_FILE; the bundle uploaded to $BUCKET/bundle.tar.gz (see build step below).
#
#   export BUCKET=gs://jazztest-bucket/sta HF_TOKEN_FILE=/tmp/hftok
#   # build+upload bundle (from repo root):
#   tar czf /tmp/sta_bundle.tar.gz experiments/*.py configs/*.yaml guard_research \
#       artifacts/paper_a_sft_v2/manifests artifacts/paper_a_sft_v2/LOCK.json \
#       requirements-starting-type-adaptation.txt
#   # NOTE: guard_research MUST be bundled -- paper_a_common.content_sha256 uses
#   # guard_research.provenance (NFC-normalizing); without it the hash silently falls back and
#   # mismatches on unicode scoring rows (toxicchat), failing the eval integrity guard.
#   gsutil cp /tmp/sta_bundle.tar.gz $BUCKET/bundle.tar.gz
#   bash experiments/deploy/sta_launch.sh            # launches all 10
#   bash experiments/deploy/sta_launch.sh KEY1 KEY2  # launches a subset
# Portable to macOS bash 3.2 (no associative arrays).
set -uo pipefail
PROJECT=${PROJECT:-jazzx-gcp-poc-1}
BUCKET=${BUCKET:-gs://jazztest-bucket/sta}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/tmp/hftok}
IMAGE=${IMAGE:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260713}
IMAGE_PROJECT=${IMAGE_PROJECT:-deeplearning-platform-release}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_KEYS="qwen25_15b smollm2_17b qwen3guard_gen_06b llama_guard_3_1b shieldgemma_2b granite_guardian_31_2b smollm3_3b qwen3_4b qwen3guard_gen_4b wildguard_7b"

# echo "machine accel zone" for a checkpoint key (>=3B -> 80GB; small -> 40GB)
plan_for() {
  case "$1" in
    qwen25_15b|smollm2_17b|qwen3guard_gen_06b|llama_guard_3_1b) echo "a2-highgpu-1g nvidia-tesla-a100 us-central1-f" ;;
    shieldgemma_2b|granite_guardian_31_2b)                      echo "a2-highgpu-1g nvidia-tesla-a100 us-central1-c" ;;
    smollm3_3b|qwen3_4b|qwen3guard_gen_4b)                      echo "a2-ultragpu-1g nvidia-a100-80gb us-central1-a" ;;
    wildguard_7b)                                               echo "a2-ultragpu-1g nvidia-a100-80gb us-central1-b" ;;
    *)                                                          echo "" ;;
  esac
}

KEYS="$*"; [ -z "$KEYS" ] && KEYS="$ALL_KEYS"
for MK in $KEYS; do
  read -r MACHINE ACCEL ZONE <<< "$(plan_for "$MK")"
  if [ -z "$MACHINE" ]; then echo "[SKIP] unknown key: $MK"; continue; fi
  VM="sta-$(echo "$MK" | tr '_' '-')"
  # REEVAL=1 -> re-score from GCS adapters (no retrain); the VM name gets a -re suffix so it does not
  # collide with a still-existing training VM of the same key.
  META="model-key=$MK,bucket=$BUCKET"
  if [ "${REEVAL:-0}" = "1" ]; then META="$META,reeval=1"; VM="$VM-re"; fi
  echo "=== launching $VM ($MK) $MACHINE/$ACCEL @ $ZONE reeval=${REEVAL:-0} ==="
  gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE" --accelerator="type=$ACCEL,count=1" \
    --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
    --image="$IMAGE" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=200GB --scopes=cloud-platform \
    --metadata="$META" \
    --metadata-from-file="startup-script=$HERE/sta_startup.sh,hf-token=$HF_TOKEN_FILE" \
    && echo "  [OK] $VM created" \
    || echo "  [WARN] $VM launch failed (capacity?) -- retry a different zone"
done
