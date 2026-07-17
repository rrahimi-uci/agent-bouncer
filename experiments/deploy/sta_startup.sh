#!/bin/bash
# Starting-type adaptation study: one checkpoint per VM. preflight -> train (11 cells) -> eval
# (per-checkpoint parquet) -> upload -> self-stop (billing-safe). Mirrors the proven klsft runner.
exec > /var/log/sta.log 2>&1
set -x
hdr="Metadata-Flavor: Google"
MK=$(curl -s -H "$hdr" http://metadata/computeMetadata/v1/instance/attributes/model-key)
BUCKET=$(curl -s -H "$hdr" http://metadata/computeMetadata/v1/instance/attributes/bucket)
{ set +x; } 2>/dev/null   # keep the HF token out of the (GCS-uploaded) log
HFTOKEN=$(curl -s -H "$hdr" http://metadata/computeMetadata/v1/instance/attributes/hf-token)
set -x
PY=/usr/bin/python3
echo "=== STA MK=$MK ==="
$PY -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0))" || true
cd /root
gsutil cp "$BUCKET/bundle.tar.gz" /root/bundle.tar.gz
rm -rf /root/repo && mkdir -p /root/repo && tar xzf /root/bundle.tar.gz -C /root/repo
cd /root/repo
mkdir -p /root/staout/runs /root/staout/scores
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{ set +x; } 2>/dev/null; export HF_TOKEN="$HFTOKEN"; set -x   # export the token without echoing it
export HF_HUB_OFFLINE=0
$PY -m pip uninstall -y torchaudio torchvision 2>&1 | tail -1
$PY -m pip install -q --no-input transformers==5.12.1 peft==0.19.1 "jinja2>=3.1.0" "pyarrow>=14" \
    pandas scikit-learn scipy accelerate safetensors sentencepiece protobuf tiktoken PyYAML 2>&1 | tail -3
$PY -c "import transformers,peft,jinja2,sentencepiece,tiktoken; from peft import PeftModel; print('IMPORTS OK')"

# Phase-0 structural eligibility preflight (records report; non-fatal to the run)
$PY experiments/preflight_starting_type_adaptation.py --key "$MK" --device cuda --dtype float32 \
    --skip-training --out /root/staout/preflight_${MK}.json || echo "preflight rc=$?"

# Train all 11 cells (1 unmodified + 5 sft + 5 kl_sft primary beta) -- FINAL. In REEVAL mode
# (metadata reeval=1) skip training and pull the already-trained adapters from GCS (recover a run
# whose eval was broken without paying to retrain).
REEVAL=$(curl -s -H "$hdr" http://metadata/computeMetadata/v1/instance/attributes/reeval 2>/dev/null)
if [ "$REEVAL" = "1" ]; then
    echo "=== REEVAL mode: pulling trained adapters from GCS ==="
    mkdir -p /root/staout/runs/${MK}
    gsutil -m rsync -r "$BUCKET/results/adapters_${MK}" /root/staout/runs/${MK}
    TRC=$?
else
    $PY experiments/run_starting_type_adaptation.py --checkpoints "$MK" --final --device cuda \
        --out-root /root/staout/runs
    TRC=$?
fi
# Score the whole tree against the frozen Paper A scoring manifests -> per-checkpoint parquet -- FINAL.
# --out-root MUST match training (else all trained cells are "missing_adapter" and only the unmodified
# base is scored). batch 8 avoids the big-vocab (gemma 256k) logits OOM on the 40GB A100.
$PY experiments/run_eval_starting_type_adaptation.py --checkpoints "$MK" --final --device cuda \
    --out-root /root/staout/runs \
    --manifests-dir artifacts/paper_a_sft_v2/manifests --scores-dir /root/staout/scores --batch-size 8
ERC=$?
echo "=== $MK train_rc=$TRC eval_rc=$ERC reeval=$REEVAL ==="
gsutil cp /root/staout/scores/sta_scores_${MK}.parquet "$BUCKET/results/" 2>/dev/null
gsutil cp /root/staout/scores/sta_scores_${MK}.metadata.json "$BUCKET/results/" 2>/dev/null
gsutil cp /root/staout/preflight_${MK}.json "$BUCKET/results/" 2>/dev/null
[ "$REEVAL" = "1" ] || gsutil -m cp -r /root/staout/runs/${MK} "$BUCKET/results/adapters_${MK}" 2>/dev/null
gsutil cp /var/log/sta.log "$BUCKET/logs/sta_${MK}.log"
[ "$TRC" = "0" ] && [ "$ERC" = "0" ] && echo "train=$TRC eval=$ERC reeval=$REEVAL" | gsutil cp - "$BUCKET/results/DONE_${MK}"
echo "=== $MK done train=$TRC eval=$ERC -- self-stopping ==="
sudo shutdown -h +2
