# scripts/ — producing scripts for the paper

**Run every script from `research/`** (the parent of this folder), so their `notebooks/…` and `paper/…`
paths resolve — e.g. `python scripts/eval_mortgage_hard.py`. The `.py` scripts need the environment in
`../requirements.txt`; the `.mjs` files are Claude Code **Workflow** scripts (LLM-orchestration, run via the
Claude Code harness, not plain `node`) — the benchmark they generate is committed, so re-running them is
optional.

## Paper: in-house, novel/OOD, and base-vs-tuned
| script | purpose |
|---|---|
| `eval_corrected.py` | Corrected in-house eval on the 2,018 rows (fixes calibration/latency/threshold/GPT bugs). |
| `eval_expanded_heldout.py` | Guard vs open guards on the 4 novel held-out sets (batched, resumable). |
| `eval_novel_gaps.py` | Fill novel-held-out gaps, serialized on one MPS device (cached, resumable). |
| `verify_novel.py` | Re-ground the novel numbers (base 0.886 / tuned 0.781 / llama 0.701). |
| `score_base_inhouse.py` | Score the zero-shot base (no adapter) on in-house dev+test. |
| `recompute_base_vs_tuned.py` | Base-vs-tuned at clean per-model in-dist calibration (Δ+0.081). |
| `eval_base_ablation.py` | Base-vs-tuned ablation on identical rows. |
| `diag_base_id_ood.py` | Diagnostic: why base novel AUPRC ≫ base in-house AUPRC. |
| `guard_eval_pipeline.py` | Parameterized eval pipeline for any decoder guard. |
| `eval_large_guard.py` | Eval-only re-run on a large balanced test set. |
| `train_guard.py` | Parameterized guard trainer (SmolLM3 recipe for any decoder base). |
| `make_figures.py` | Regenerate `paper/figures/*.pdf` (numbers inline — self-contained). |

## Mortgage domain case study (§ hardened-benchmark)
| script | purpose |
|---|---|
| `build_mortgage_split.py` | Family-safe train/dev/test split of the mortgage benchmark. |
| `train_mortgage.py` | Fine-tune SmolLM3-3B into a mortgage guard (TECHNIQUE=sft/dpo/ipo/kto). |
| `eval_mortgage.py` | Guard protocol on the (saturated) mortgage red-team benchmark. |
| `eval_mortgage_tuned.py` | Mortgage-tuned adapters on the family-disjoint test split. |
| `eval_mortgage_hard.py` | Hardened-benchmark metrics: Recall@FPR, worst-family, guard-fairness gap, minimal-pair, wrapper-flip + gpt ceiling. |

## Hardened-benchmark construction (Workflow scripts, `.mjs`)
| script | purpose |
|---|---|
| `wf_harden_mortgage.mjs` | Design workshop: multi-lens design → adversarial verify → synthesize the hardening spec. |
| `wf_build_hard_benchmark.mjs` | First-pass minimal-pair generation + blind 3-juror gate. |
| `wf_build_hard_benchmark_v2.mjs` | Scaled generation (16 families) + chunked jury → `notebooks/data/benchmarks/hard_admitted.json`. |
| `build_hard_jsonl.mjs` | (plain node) split the admitted items family-safe + add wrapper variants → `guard_benchmark_hard.jsonl`. |

## Notebook build / data bundle
| script | purpose |
|---|---|
| `build_smollm3_notebook.py` | Regenerate `notebooks/smollm3_guard_reproduction.ipynb`. |
| `bundle_notebook_data.py` | Bundle benchmark subsets into `notebooks/data/` so the notebook is standalone. |
| `run_qwen3_pipeline.sh` | Cross-family Qwen3-4B train→eval pipeline (future work; resumable). |
