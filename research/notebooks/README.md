# Notebooks

## `smollm3_guard_reproduction.ipynb` — self-contained SmolLM3-3B guard pipeline

Trains **SmolLM3-3B** into an LLM/agent safety guard and evaluates it against the mini judges, end to end,
in one notebook. It implements [`../docs/smollm3-guard-plan.md`](../docs/smollm3-guard-plan.md).

> **This notebook is a companion demo, not the paper's reproduction path.** The paper's open-guard /
> AUPRC / matched-FPR results are produced by `../scripts/` (see `../README.md`). This notebook is a
> self-contained illustrative pipeline; its GPT baselines are a mini-judge parity check and use a
> demo-grade **fail-closed** default on API errors (the paper's *abstain* policy lives in the eval scripts).

**This folder is standalone.** The benchmark data is bundled in
[`data/benchmarks/full/`](data/benchmarks/full/) — copy or zip the `notebooks/` folder and the eval sets
travel with it, no download needed. The notebook builds class-balanced matched-n subsets from it (seed 42)
and also uses it for the offline training fallback. If the bundle is missing the notebook falls back to
Hugging Face. The only network requirement is the one-time model download.

- **SMOKE mode** (auto when no CUDA): a tiny proxy model + tiny data + a few steps, so the whole pipeline
  runs in minutes on CPU/MPS to prove it works.
- **FULL mode** (a CUDA GPU): trains the real SmolLM3-3B; batch size, sequence length and precision
  auto-scale to the GPU's VRAM (a 16 GB T4 runs without OOM).
- Set `OPENAI_API_KEY` (and optionally `HF_TOKEN`) to include the GPT-4o-mini / GPT-5-mini baselines.

Outputs (leaderboard, per-axis + per-benchmark P/R/F1, parity CIs, and the figures) are written to
`outputs/nb-smollm3-guard/`.

Regenerate the notebook: `python ../scripts/build_smollm3_notebook.py`.
Refresh the bundled data: `python ../scripts/bundle_notebook_data.py`.
