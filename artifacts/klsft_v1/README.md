# KL-SFT experiment data (Act I anti-forgetting control)

Text-free scored data for the KL-regularized SFT control on the four general Paper A checkpoints
(SFT vs KL-SFT at beta {0, 0.5, 1.0} x 5 seeds). Feeds the Act I `sec:actI-klsft` subsection.

## What is committed here (durable reference)
- `scores/klsft_scores_<model_key>.parquet` — per-row scored margins (RAW logit margin
  `z_unsafe - z_safe`), keyed by `model_key`, `seed`, `kl_beta`, split, gold, `content_sha256`
  (NO raw prompt text — content is hashed, matching the Paper A text-free release policy).

## What is NOT in the repo (too large / regenerable) — in GCS `gs://jazztest-bucket/klsft/`
- `results/adapters_<model_key>/` — trained LoRA adapters.
- `logs/klsft_<model_key>.log` — run logs.
- `results/DONE_<model_key>` — completion markers.

## Regenerate
`experiments/run_klsft_sweep.py` (train+score one checkpoint's seed x beta grid) then
`experiments/analyze_klsft.py --klsft-dir artifacts/klsft_v1/scores --emit-dir papers/unified-report/generated`
(fills `tab_klsft_gen.tex` + `klsft_macros.tex`). See the `klsft-experiment` project memory for the
GCP run details.
