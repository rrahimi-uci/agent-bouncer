# `experiments/` — the Paper A pipeline

Six scripts produce every number in the paper, in order. They read a single config
([`configs/paper_a_sft.yaml`](../configs/paper_a_sft.yaml)) and write an auditable
clean artifact chain under `artifacts/paper_a_sft_v2/`; the committed legacy
evidence remains read-only under [`artifacts/paper_a_sft/`](../artifacts/paper_a_sft). Run each
**from the repo root**.

> The broad-study / Paper B scripts that used to live here now sit under
> [`legacy/`](../legacy). Nothing in this folder depends on them.

## Pipeline

| # | Script | What it does |
|---|---|---|
| 1 | `prepare_paper_a_manifests.py` | Build the 1,200-row training manifest and evaluation sets from pinned Hugging Face revisions using deterministic hash ranking. Preserve upstream pair/family IDs, route any family seen on a reported test/stress surface out of calibration, and emit a recursively text-free public index. The old seed-7 cache is available only through `--allow-legacy-frozen-cohorts`. |
| 2 | `audit_paper_a_splits.py` | Independently recompute hashes, counts, roles, selection provenance, licenses, exact/conflicting overlap, upstream/MinHash family links, near duplicates, and train/eval plus calibration/all-reported-test family disjointness. Any failed invariant exits nonzero. |
| 3 | `lock_paper_a_sft.py` | Write a self-hashed v2 contract that binds the clean source state, raw/canonical config hashes, every manifest, audit, prompt/tokenizer probes, model/runtime settings, expected score schema, and analysis mode. It never silently upgrades the historical v1 lock. |
| 4 | `run_paper_a_sft.py` | Train **4 checkpoints × 5 seeds = 20** completion-only LoRA-SFT adapters. Final mode rejects recipe overrides, preserves the classifier instruction under truncation, and rehashes completed adapters; development overrides require a separate nonfinal namespace. |
| 5 | `eval_paper_a_sft.py` | Revalidate all adapters, score 4 bases + 20 adapters, and write a complete row-keyed score matrix plus sibling metadata and a score-file SHA-256. Raw prompt text is never included. Debug/incomplete modes require explicit nonfinal output. |
| 6 | `analyze_paper_a_sft.py` | Verify the lock, score hash/metadata, exact v1-or-v2 schema, complete bundle×sample matrix, and row identity before computing observed macro-AP points, descriptive hierarchical-bootstrap intervals, sensitivity analyses, complete RQ4/stress tables, narrative macros, and figures. Precision mode emits no formal rejection or Holm decision. |

### Support libraries (imported by the pipeline, not run directly)

| Module | Role |
|---|---|
| `paper_a_common.py` | Shared config loading, artifact I/O, and metric/threshold helpers that wrap `guard_research`. |
| `paper_a_manifest_lib.py` | Manifest construction + provenance (NFKC normalization, content/family SHA-256, pinned NumPy MinHash, upstream-family extraction, recursive public redaction). |

## Run it

The whole pipeline is wired into the top-level [`Makefile`](../Makefile):

```bash
make manifests   # 1. prepare
make audit       # 2. audit  (hard assertions)
make lock        # 3. write artifacts/paper_a_sft_v2/LOCK.json
make train       # 4. run_paper_a_sft train   (GPU)
make eval        # 5. eval_paper_a_sft         (GPU)
make analyze     # 6. strict v2 analysis -> v2 tables + figures
```

The committed scores belong to the historical v1 lock. Reproduce them only via
the explicit compatibility path:

```bash
python3 experiments/analyze_paper_a_sft.py \
    --allow-legacy-lock \
    --lock artifacts/paper_a_sft/LOCK.json \
    --scores artifacts/paper_a_sft/scores/scores.parquet \
    --out artifacts/paper_a_sft/analysis
```

Steps 4–5 need a GPU. Gated source access can be configured by copying
[`.env.example`](../.env.example) to `.env`; the manifest builder reads that file
directly. Steps 1–3 and 6 run on CPU. Because the historical score bundle is
committed, `make repro-legacy` reproduces its tables and figure without a GPU;
strict `make analyze` requires a clean v2 lock and matching score artifact.
