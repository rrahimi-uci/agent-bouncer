# `experiments/` — the Paper A pipeline

Six scripts produce every number in the paper, in order. They read a single config
([`configs/paper_a_sft.yaml`](../configs/paper_a_sft.yaml)) and write an auditable
artifact chain under [`artifacts/paper_a_sft/`](../artifacts/paper_a_sft). Run each
**from the repo root**.

> The broad-study / Paper B scripts that used to live here now sit under
> [`legacy/`](../legacy). Nothing in this folder depends on them.

## Pipeline

| # | Script | What it does |
|---|---|---|
| 1 | `prepare_paper_a_manifests.py` | Build the immutable, **decontaminated** 1,200-row training manifest (400/source × 200/label over ToxicChat + Prompt-Injections + Jailbreak-Classification) plus the held-out evaluation sets, from Hugging Face sources pinned by revision and `notebooks/outputs/frozen_eval_rows.json`. Emits content/family hashes and a manifest with full provenance. |
| 2 | `audit_paper_a_splits.py` | Recompute the decontamination facts independently of the builder and **hard-assert** them: zero exact/conflicting train↔eval overlap, label balance, family disjointness. Writes `audit.json` / `audit.md`. |
| 3 | `lock_paper_a_sft.py` | Freeze the study: write `LOCK.json` binding the config, manifest hashes, and audit result. Created **after** manifests + tests + smoke pass and **before** the final training run. |
| 4 | `run_paper_a_sft.py` | Train the fixed panel — **4 checkpoints × 5 seeds = 20** completion-only LoRA-SFT adapters (Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B). Subcommands: `train` / `smoke` / `validate-runs`. GPU required. |
| 5 | `eval_paper_a_sft.py` | Score the 4 untuned bases (once each) + all 20 adapters on every benchmark → `scores/scores.parquet` (**row-keyed hashes + logits only, no raw text**). A completeness gate requires all 24 model-runs before scoring. |
| 6 | `analyze_paper_a_sft.py` | From the keyed score table, compute macro-AP with the **canonical tie-aware metric** ([`guard_research`](../guard_research)), the family+seed hierarchical bootstrap, the represented-vs-transfer decomposition, and the intersection-union **claim gates** → tables, figures, `results.json`, `claim_checks.json`. |

### Support libraries (imported by the pipeline, not run directly)

| Module | Role |
|---|---|
| `paper_a_common.py` | Shared config loading, artifact I/O, and metric/threshold helpers that wrap `guard_research`. |
| `paper_a_manifest_lib.py` | Manifest construction + provenance (NFKC normalization, content/family SHA-256, MinHash). |

## Run it

The whole pipeline is wired into the top-level [`Makefile`](../Makefile):

```bash
make manifests   # 1. prepare
make audit       # 2. audit  (hard assertions)
make lock        # 3. lock
make train       # 4. run_paper_a_sft train   (GPU)
make eval        # 5. eval_paper_a_sft         (GPU)
make analyze     # 6. analyze  -> tables + figures
```

Or call a script directly, e.g. a CPU-only re-analysis from the committed scores:

```bash
python experiments/analyze_paper_a_sft.py \
    --scores artifacts/paper_a_sft/scores/scores.parquet \
    --out artifacts/paper_a_sft/analysis
```

Steps 4–5 need a GPU and Hugging Face access (set `HF_TOKEN`; see
[`.env.example`](../.env.example)). Steps 1–3 and 6 run on CPU. Because
`scores.parquet` is committed, **step 6 reproduces every table and figure without a
GPU** — see the repository [README](../README.md#reproduce).
