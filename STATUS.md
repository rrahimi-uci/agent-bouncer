# Overnight autonomous run — DONE

**Date:** 2026-07-17. **State:** ✅ complete (adaptation study run, analyzed, written up, reviewed, merged; all GCP torn down).

## What was delivered
- **KL-SFT control (Act I)** — 4/4 checkpoints; KL-SFT (β=0.5) recovers transfer SFT gives up (+0.061 vs SFT) at a small represented cost (−0.035). Data: `artifacts/klsft_v1/scores/`.
- **Confirmatory starting-type adaptation study** — 10 checkpoints (4 general + 6 released purpose-built guards), 6 families, 5 seeds, 2×3 grid, 363,880 scored rows. Data: `artifacts/starting_type_adaptation_v1/{scores,preflight,analysis}/`; adapters in GCS (`adapters_manifest.json`).
  - **RQ1 SUPPORTED** — ordinary SFT specializes released guards too: H_gain +0.174 (LCB +0.129), H_conc +0.239 (LCB +0.189).
  - **RQ2 NOT SUPPORTED** — KL-SFT preserves transfer (H_preserve LCB +0.035) but its represented cost (H_cost LCB −0.060) fails the −0.02 non-inferiority margin: a tradeoff, not a free win.
- **Paper** (`papers/unified-report/`): new §adaptation + `tab:adaptation`, a styled "what we learned → guideline" summary table (`tab:guidelines`, 7 rows), Act I KL-SFT numbers filled. PDF rebuilt clean.
- **Reviewer critique**: `report.md` (repo root) — 17/20 findings confirmed by adversarial verification; substantive fixes applied (real seed-count error 15/15→15/20; removed the KL-SFT "free" overclaim to match RQ2; abstract directional caveat; confirmatory-study consistency across contributions/limitations/roadmap; family-term disambiguation; Llama-Guard null-cell handling).

## Key engineering lessons (captured in memory)
- Eval **out-root mismatch** in the VM startup scored only the unmodified cell; recovered via a re-eval mode (adapters were safe in GCS) — no retrain. gemma 256k-vocab logits OOM at batch 32 → batch 8. `guard_research` must be bundled (NFC-normalizing content hash) or unicode rows mismatch.

## Housekeeping
- All 23 GCP VMs (klsft-*, sta-*, sta-*-re) deleted.
- Merged to `main` (see git log).
