# `legacy/` — broad-study & Paper B research code

This directory holds the **earlier, broad measurement study** and the exploratory
code for a **planned follow-up ("Paper B")**. It is quarantined here, apart from the
clean [Paper A](../README.md) pipeline, so the public release stays focused and
reproducible while nothing is lost.

> **Not part of the Paper A reproduction.** Nothing under `legacy/` is needed to
> reproduce the results in the paper. The Paper A pipeline is entirely
> [`guard_research/`](../guard_research) + [`experiments/`](../experiments).

## Why it is kept

The broad study (*"The Benchmark Chooses the Winner: A Fair Evaluation of Small-LLM
Safety Guards"*) covered far more than the focused specialization result that became
Paper A: an objective comparison (SFT vs. DPO vs. GRPO), a mortgage-lending
compliance case study, a base⊕tuned ensemble mitigation, guardrail baselines
(Llama-Guard, WildGuard, …), GPT parity/latency, and a name-fairness probe. That
work is the seed for a second paper and is preserved here for reference and reuse.

## ⚠️ Metric caveat

Several scripts here **predate the centralized, tie-aware metrics** now in
[`guard_research/metrics.py`](../guard_research/metrics.py) and compute
average-precision with the earlier order-dependent implementation. They are retained
**as they were during the broad study**, deliberately *not* back-ported, so the
historical numbers remain reproducible. **For any new work, import the canonical
metrics from `guard_research`** rather than copying these.

## How to run

All scripts are **repo-root-relative** — run them from the repository root, not from
inside `legacy/`:

```bash
# examples (most training/eval scripts need a GPU + Hugging Face access)
MODEL_ID=HuggingFaceTB/SmolLM3-3B python legacy/experiments/train_guard.py
HPO_METHOD=dpo python legacy/experiments/hpo_guard.py
python legacy/experiments/make_figures.py
```

Benchmark data and frozen evaluation rows live in [`../notebooks/`](../notebooks)
(`notebooks/data/`, `notebooks/outputs/`); the scripts reference those paths
relative to the repo root.

## What is here

| Group | Scripts |
|---|---|
| **Training / objective study** | `train_guard.py` (SFT), `train_guard_pref.py` (DPO/GRPO/KTO), `hpo_guard.py` (Optuna HPO), `stage2.sh`, `run_extra_models.sh`, `run_qwen3_pipeline.sh` |
| **Core evaluation pipeline** | `guard_eval_pipeline.py`, `eval_corrected.py`, `eval_novel_gaps.py`, `eval_expanded_heldout.py`, `aggregate_clean_sweep.py`, `recompute_base_vs_tuned.py`, `diag_base_id_ood.py`, `score_base_inhouse.py`, `emit_inhouse_auprc_poolings.py`, `reground_gpt_inhouse.py`, `verify_novel.py` |
| **Baselines / large guards** | `eval_guardrail_baselines.py` (WildGuard, …), `eval_llamaguard_logprob.py` (Llama-Guard), `eval_large_guard.py`, `eval_base_ablation.py` |
| **Mortgage compliance case study** | `train_mortgage.py`, `eval_mortgage.py`, `eval_mortgage_tuned.py`, `eval_mortgage_hard.py`, `build_mortgage_split.py`, `expguard_eval.py` |
| **Ensemble mitigation** | `ensemble_deployable.py`, `ensemble_probe.py` |
| **Fairness** | `name_fairness_probe.py` |
| **Figures** | `make_figures.py` |
| **Hard-benchmark builders** | `build_hard_jsonl.mjs`, `wf_build_hard_benchmark.mjs`, `wf_build_hard_benchmark_v2.mjs`, `wf_harden_mortgage.mjs` |
| **Notebook builders** | `build_paper_reproduction_notebook.py`, `build_smollm3_notebook.py`, `bundle_notebook_data.py` |

The broad study's reproduction notebooks (`paper_reproduction.ipynb`,
`smollm3_guard_reproduction.ipynb`) remain alongside their bundled data in
[`../notebooks/`](../notebooks).
