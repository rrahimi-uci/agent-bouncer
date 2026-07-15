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
| 6 | `analyze_paper_a_sft.py` | Verify the lock, score hash/metadata, exact v1-or-v2 schema, complete bundle×sample matrix, and row identity before computing observed macro-AP points, descriptive hierarchical-bootstrap intervals, sensitivity analyses, complete RQ4/stress tables, narrative macros, and figures. `--release-cache` is an explicit strict-final-v2 path for a text-free score release; it never enables legacy mode. Precision mode emits no formal rejection or Holm decision. |

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

`make analyze` is valid for the same-source pipeline or an immutable full archive
whose execution-source bytes match the lock. Post-run hardened checkouts use the
public `make repro` cache path; full-archive verification additionally requires
the separately preserved exact source snapshot described in
[`docs/reproducibility.md`](../docs/reproducibility.md).

### Fresh-clone v2 release-cache path

When the release contains `artifacts/paper_a_sft_v2/LOCK.json`, its self-hashed
`RELEASE.json`, the separately tracked
`configs/paper_a_sft_v2_release_anchor.json`, the complete `public_manifests/`
directory, and `scores/scores.parquet` plus sibling `scores/metadata.json`, the
primary CPU-only reproduction is:

```bash
make install
make repro
```

`make repro` invokes `analyze_paper_a_sft.py --release-cache`, writes the v2
analysis, and byte-compares every manuscript-consumed table/figure/macro with
the checked-in copy without overwriting it first. Maintainer synchronization is
the separate `make paper-sync` action. Reproduction fails if the lock is not
strict and final or if public evidence,
score metadata, the score matrix, row identities, or runtime requirements do not
match. The resulting `analysis_metadata.json` explicitly records that raw manifest
files, run metadata, adapter bytes, and the original execution-source bytes were
not locally reverified. The last of those is checked against the separately
archived immutable source bundle, not inferred from the current checkout. See
[`docs/reproducibility.md`](../docs/reproducibility.md) for the contract and
limitations. If the final cache is absent, the target is expected to fail.

Use `make release-package` to build the distributable overlay. The packager
copies from an explicit allowlist and rejects raw manifests, run/adapter trees,
base-score caches, audit inputs, smoke outputs, and symlinks; the full internal
execution archive is never treated as the public release cache.

### Downstream composition prototype

The composition analyzer uses the same Paper A score evidence through three
explicit, output-disjoint targets:

```bash
make composition         # primary strict-v2 release cache -> v2 analysis/composition
make composition-full    # full strict-v2 evidence       -> v2 analysis/composition-full
make composition-legacy  # explicit archived-v1 input    -> v1 analysis/composition
```

The release target uses text-free public identity rows and the release analysis
runtime contract; only raw manifests, run metadata, and adapter bytes may be
locally absent. `composition.json` records the fixed prototype repetitions, RNG
seed, target FPR, and primary combiner; `composition_metadata.json` separately
records input verification, the complete current analysis-source aggregate,
runtime, and output hashes so full/release scientific files remain byte-identical.
These downstream constants are not a separate Paper B lock; analytical overrides
require `--nonfinal` and an output outside canonical artifact roots.

### Archived v1 compatibility

The committed historical v1 scores are not a fallback for v2. Reproduce them only
through the explicit compatibility target:

```bash
make repro-legacy
```

Steps 4–5 need a GPU. Gated source access can be configured by copying
[`.env.example`](../.env.example) to `.env`; the manifest builder reads that file
directly. Steps 1–3 and 6 run on CPU. Because the historical score bundle is
committed, `make repro-legacy` reproduces its tables and figure without a GPU;
strict `make analyze` requires the full clean-v2 artifact chain, while the default
`make repro` requires the smaller final v2 release cache. Neither silently reads
the v1 namespace.
