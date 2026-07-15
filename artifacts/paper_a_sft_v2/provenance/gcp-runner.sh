#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${RUN_BUCKET:?RUN_BUCKET is required}"
: "${RUN_ID:?RUN_ID is required}"
ROOT="$HOME/guard-ranking-fragility"
PY="$ROOT/.venv/bin/python"
RUN_HOME="$HOME/paper-a-run"
LOG="$RUN_HOME/pipeline.log"
ARTIFACT_ROOT="$ROOT/artifacts/paper_a_sft_v2"
mkdir -p "$RUN_HOME"
exec > >(tee -a "$LOG") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
# Exact-revision dataset and model caches were verified before launch. Keep the
# scientific run offline so a CDN failure or moving remote cannot affect it.
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage() {
  printf '%s\n' "$1" | timeout 120 gcloud storage cp - "$RUN_BUCKET/status/CURRENT_STAGE"
  printf '[stage] %s %s\n' "$(date -u +%FT%TZ)" "$1"
}

finish() {
  rc=$?
  trap - EXIT
  set +e
  timeout 120 gcloud storage cp "$LOG" "$RUN_BUCKET/logs/pipeline.log"
  if [ "$rc" -eq 0 ]; then
    printf 'success\n' | timeout 120 gcloud storage cp - "$RUN_BUCKET/status/SUCCESS"
  else
    printf 'failed rc=%s\n' "$rc" | timeout 120 gcloud storage cp - "$RUN_BUCKET/status/FAILED"
  fi
  exit "$rc"
}
trap finish EXIT

cd "$ROOT"
test -x "$PY"
test -f .env
set -a
# shellcheck disable=SC1091
source .env
set +a
test -n "${HF_TOKEN:-}"
test "$(git status --porcelain -- configs experiments guard_research pyproject.toml requirements.txt | wc -l | tr -d ' ')" = 0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [ -f artifacts/paper_a_sft_v2/LOCK.json ]; then
  stage verify_existing_lock
  make verify-lock PY="$PY"
else
  stage manifests
  # Dataset bytes are seeded from the locally verified exact-revision cache
  # because Hugging Face's CDN signed URLs failed on this GCP egress path.
  # Offline mode makes a missing cache entry fail instead of drifting online.
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 make manifests PY="$PY"

  stage audit
  make audit PY="$PY"

  stage tests_prelock
  "$PY" -m pytest -o addopts='' -q -ra

  stage analysis_selftest
  make selftest PY="$PY"

  stage prelock_clean_source
  PYTHONPATH=. "$PY" - <<'PY'
from experiments import paper_a_common as C
state = C.execution_git_provenance()
print(state)
assert state["execution_clean"], state
PY

  stage final_lock
  make lock PY="$PY"
fi

stage verify_lock
make verify-lock PY="$PY"

stage gpu_smoke
"$PY" experiments/run_paper_a_sft.py smoke \
  --lock artifacts/paper_a_sft_v2/LOCK.json --all-models --steps 5

stage train_20_cells
make train PY="$PY"

stage validate_20_cells
make validate-runs PY="$PY"

stage evaluate_24_bundles
make eval PY="$PY"
score_before=$(sha256sum artifacts/paper_a_sft_v2/scores/scores.parquet | awk '{print $1}')

stage strict_cache_replay
STRICT_CACHE_LOG="$RUN_HOME/strict-cache-replay.log"
"$PY" experiments/eval_paper_a_sft.py \
  --lock artifacts/paper_a_sft_v2/LOCK.json --strict-cache \
  | tee "$STRICT_CACHE_LOG"
test "$(grep -c '^  \[cache\] reuse ' "$STRICT_CACHE_LOG")" -eq 24
test "$(grep -c '^  \[scored\] ' "$STRICT_CACHE_LOG")" -eq 0
score_after=$(sha256sum artifacts/paper_a_sft_v2/scores/scores.parquet | awk '{print $1}')
test "$score_before" = "$score_after"

stage canonical_analysis
make analyze PY="$PY"

stage repeat_analysis
rm -rf /tmp/paper-a-v2-analysis-repeat
"$PY" experiments/analyze_paper_a_sft.py \
  --lock artifacts/paper_a_sft_v2/LOCK.json \
  --scores artifacts/paper_a_sft_v2/scores/scores.parquet \
  --out /tmp/paper-a-v2-analysis-repeat --nonfinal
diff -qr artifacts/paper_a_sft_v2/analysis /tmp/paper-a-v2-analysis-repeat

stage final_verification
make verify-lock PY="$PY"
"$PY" -m pytest -o addopts='' -q -ra
make selftest PY="$PY"

stage package_artifacts
test -z "$(find artifacts/paper_a_sft_v2 -type l -print -quit)"
test -z "$(find artifacts/paper_a_sft_v2 -type f \
  \( -name '*.tmp' -o -name '*.tmp.*' \) -print -quit)"
find artifacts/paper_a_sft_v2 -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > artifacts/paper_a_sft_v2/SHA256SUMS
sha256sum -c artifacts/paper_a_sft_v2/SHA256SUMS
tar --zstd -cf "$RUN_HOME/${RUN_ID}-artifacts.tar.zst" artifacts/paper_a_sft_v2
(cd "$RUN_HOME" && sha256sum "${RUN_ID}-artifacts.tar.zst" \
  > "${RUN_ID}-artifacts.tar.zst.sha256")

stage upload_artifacts
timeout 7200 gcloud storage cp "$RUN_HOME/${RUN_ID}-artifacts.tar.zst" "$RUN_BUCKET/results/"
timeout 120 gcloud storage cp "$RUN_HOME/${RUN_ID}-artifacts.tar.zst.sha256" "$RUN_BUCKET/results/"
timeout 120 gcloud storage cp "$LOG" "$RUN_BUCKET/logs/pipeline.log"
timeout 120 gcloud storage cp artifacts/paper_a_sft_v2/LOCK.json "$RUN_BUCKET/results/LOCK.json"
timeout 120 gcloud storage cp artifacts/paper_a_sft_v2/SHA256SUMS "$RUN_BUCKET/results/SHA256SUMS"

stage complete
