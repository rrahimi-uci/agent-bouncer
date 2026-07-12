# agent-bouncer — paper

ACM-formatted (`acmart`, sigconf) measurement study of a laptop-trained SmolLM3-3B safety guard.

## Files
- `benchmark_chooses_the_winner.tex` — the paper (ACM `acmart`, `sigconf` + `nonacm`).
- `benchmark_chooses_the_winner.pdf` — compiled output.
- `refs.bib` — 32 fact-checked references (verified arXiv IDs).
- `figures/` — vector PDF figures (drawn by `experiments/make_figures.py` from **inline, hand-entered** values --- not auto-loaded from the result JSON, so keep in sync with the TeX tables by hand; Okabe-Ito colorblind-safe palette):
  - `fig1_inhouse_auprc.pdf` — in-house pooled AUPRC (guard vs open guards).
  - `fig2_novel_auprc.pdf` — novel held-out AUPRC (base > tuned > Llama-Guard).
  - `fig3_operating_point_flip.pdf` — native-F1 vs AUPRC vs matched-FPR ranking flip.
  - `fig4_base_vs_tuned.pdf` — per-benchmark base→tuned F1 delta.
  - `fig5_pareto.pdf` — F1 vs single-request latency Pareto.
- `DRAFT.md` — the prose source the LaTeX was converted from (kept for editing).
- `metrics_survey.md` — survey of related-work evaluation metrics (background for the eval protocol).
- `tables/operating_point_fairness.md` — the operating-point-fairness table.

## Build
```bash
make            # compile benchmark_chooses_the_winner.tex -> benchmark_chooses_the_winner.pdf (uses tectonic; auto-fetches acmart)
make figures    # regenerate figures/ (values are inline in make_figures.py; keep in sync with the tables by hand)
make clean
```
Or directly: `tectonic benchmark_chooses_the_winner.tex` (needs network on first run to fetch acmart).

## Provenance
All numbers trace to `notebooks/outputs/nb-smollm3-guard/*.json`. Producing scripts:
- `experiments/eval_corrected.py` → `summary_corrected.json`, `preds_corrected.json`
  (in-dist-only calibration T=2.10/τ=0.59, matched-FPR@0.10, AUPRC, batch=1 latency).
- `experiments/eval_novel_gaps.py` + `experiments/verify_novel.py` → `_cache_{guard,base,llama}_exp.json`,
  `summary_novel_full.json` (novel-set AUPRC + Optimal-F1; `verify_novel.py` reconstructs the
  gold/order deterministically and re-grounds base 0.886 / tuned 0.781 / llama 0.701).
- `experiments/score_base_inhouse.py` → `base_smollm3_inhouse.json` (base continuous in-house scores).
- `experiments/recompute_base_vs_tuned.py` → `base_vs_tuned_clean.json` (base-vs-tuned at CLEAN
  per-model in-dist calibration on the identical 2,018 rows: base 0.713 / tuned 0.794,
  Δ+0.081 [0.062,0.100]; base in-house AUPRC 0.696).
- `experiments/guard_eval_pipeline.py`, `experiments/eval_expanded_heldout.py` — parameterized / earlier eval.

Figures are drawn by `experiments/make_figures.py`, which holds the plotted values **inline** (hand-transcribed from the numbers above); it does not read the JSON, so it must be kept in sync by hand.

Note: the `notebooks/outputs/nb-smollm3-guard/*.json` artifacts are gitignored (derived); regenerate them with the producing scripts listed above.

## Status / TODO
- Cross-family (Qwen3): in-house results salvaged; **novel-held-out Qwen3 runs pending** (Gemma-2 / large-model
  scoring stalls on the thermally-throttled laptop — deferred to a CUDA GPU).
- Deferred (needs GPU): Qwen3-4B (size-matched), ShieldGemma-on-novel, adversarial-robustness eval,
  broad-data head-to-head.
- Known cosmetic LaTeX warnings: a few overfull `\hbox` lines and Libertine font-substitution notes.
