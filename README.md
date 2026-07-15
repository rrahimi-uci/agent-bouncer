# The Benchmark Chooses the Winner

Reproduction code, data manifests, and an auditable evidence chain for the paper

> **The Benchmark Chooses the Winner: Measuring Fine-Tuning Specialization Across Safety-Guard Benchmarks**
> Reza Rahimi (JazzX AI)

This repository compares each prompt-safety guard with its own untuned base on a
fixed panel of four instruction checkpoints, five SFT seeds each. It separates
changes on sources represented during training from transfer to held-out datasets.
The publication result is a **clean-v2 execution analyzed retrospectively**. It
binds corrected provenance, family links, instruction-preserving truncation,
adapters, scores, and analysis under the strict v2 contract. That clean execution
does not make previously inspected benchmark cohorts prospective; confirmatory
evidence still requires a separately locked study on a genuinely uninspected cohort.

## Headline result

Across the panel (Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B; seeds 42–46),
LoRA-SFT on the locked 1,200-row clean-v2 training manifest:

| Effect | Observed Δ macro-AP (base → SFT) | Descriptive 95% two-sided bootstrap interval |
|---|---:|---:|
| **Represented** sources | **+0.323** | **[+0.265, +0.369]** |
| **Transfer** to held-out datasets | **−0.059** | **[−0.084, −0.032]** |

SFT lifts every checkpoint to about 0.98 macro-AP on represented sources. Transfer
is heterogeneous by checkpoint—SmolLM2 +0.040, Qwen2.5 −0.039, SmolLM3 −0.087,
Qwen3-4B −0.150—while every leave-one-checkpoint-out and
leave-one-transfer-benchmark-out aggregate remains negative. Benchmark-macro
transfer FPR rises from **8.1% to 15.5%** (pooled-negative: **4.3% to 17.0%**),
and HarmBench recall falls from **78.0% to 60.0%**. Fifteen of the twenty
seed-level points improve represented-source AP while losing transfer AP.
These are clean-v2, retrospective, estimation-only results—not formal claim gates
or a universal "fine-tuning hurts" conclusion.

The v2 numbers regenerate from the released score cache through the strict
[`make repro`](#reproduce-a-released-v2-cache-fresh-clone-no-gpu) path. The older
v1 numbers remain available only through the explicit compatibility target.

## Repository layout

```text
guard_research/      canonical, auditable library: tie-aware metrics, provenance
                     (hashing + MinHash), the frozen prompt, calibration thresholds
experiments/         the six-step Paper A pipeline (prepare → audit → lock →
                     train → eval → analyze); see experiments/README.md
configs/             paper_a_sft.yaml — the single study config
tests/               unit tests for the canonical metrics, thresholds, manifests
artifacts/paper_a_sft/   the evidence chain: LOCK.json, audit/, analysis/,
                     scores/scores.parquet (row-keyed hashes + logits, no raw text),
                     runmeta/; this namespace is immutable legacy evidence
artifacts/paper_a_sft_v2/   primary clean-v2 artifact root for manifests, lock,
                     runs, scores, and analysis
paper-a/             the manuscript (tectonic) + generated tables/figures
paper-a/paper-html/  the focused HTML edition (self-contained, offline math);
                     explorer/ is an unlinked archive of the earlier broad study
paper-a-simplified/  plain-language edition of the paper for readers with basic
                     stats + fine-tuning knowledge (same numbers) + a glossary
docs/                design/planning notes
legacy/              the earlier broad study + planned Paper B code (quarantined,
                     still runnable; not part of this reproduction)
```

## Install

Python 3.12 (see `.python-version`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install-all                  # pinned training/scoring + analysis + dev
# or, for pinned CPU analysis + tests only:
make install
```

Both Make targets constrain the editable package with `requirements.txt`; do not
use an unconstrained extras install for a final v2 run.
The exact executed clean-v2 GPU environment and the older partially recorded v1
environment are intentionally separated in
[docs/reproducibility-environments.md](docs/reproducibility-environments.md).

## Reproduce a released v2 cache (fresh clone, no GPU)

The primary no-GPU path uses the strict final v2 `LOCK.json`, its text-free
`public_manifests/`, `scores/scores.parquet` with sibling `scores/metadata.json`,
the self-hashed `RELEASE.json`, and the separately tracked
`configs/paper_a_sft_v2_release_anchor.json`. The last two bind the mutable score
metadata and current verifier/analyzer sources. From a fresh clone:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install
make repro       # verify evidence, analyze, and compare checked-in paper inputs
make paper       # re-verify v2 generated inputs, then compile (needs tectonic)
```

`make repro` fails closed when any required release file, binding, score row, or
runtime requirement is missing or mismatched. Its explicit `--release-cache` mode
verifies the text-free public evidence and the score/metadata bindings without
pretending that a fresh clone contains licensed raw prompts, training run metadata,
or adapter bytes. It records those local omissions and the current analysis-source
hashes in `analysis/analysis_metadata.json`; byte verification of the original GPU
execution sources remains the job of the separately archived immutable source
snapshot in [`artifacts/paper_a_sft_v2/provenance/`](artifacts/paper_a_sft_v2/provenance/).
See [docs/reproducibility.md](docs/reproducibility.md) for the exact contract.
If any required score-cache component is absent or changed, this command stops.

Maintainers stage a distributable overlay with `make release-package`. That
target uses an explicit allowlist and rejects raw manifests, adapters, run
directories, base-score caches, audit inputs, smoke outputs, and symlinks. The
full internal execution archive is verified separately and is not the public
score-only package.

## Reproduce the archived v1 scores (no GPU)

The historical score bundle regenerates its tables and figures on CPU (about
10–15 minutes for 10,000 bootstrap replicates on the reviewed laptop), but it is
rejected by the strict v2 contract unless legacy mode is explicit:

```bash
make repro-legacy  # explicit compatibility mode → archival v1 analysis only
make test          # unit and release-contract tests
make selftest       # synthetic end-to-end check of the analysis
```

This v1 path is archival only and never updates the v2 manuscript. `make repro`,
`make paper-verify`, and `make paper` all point to v2 and never silently fall
back to these files. `make paper-sync` is a separate, explicit maintainer action;
ordinary reproduction verifies before changing any paper input. `make help` lists
every target.

## Generate v2 from scratch (GPU)

The full pipeline runs in order; steps 1–3 and 6 are CPU, steps 4–5 need a GPU and
Hugging Face access (`HF_TOKEN`; copy [.env.example](.env.example) → `.env`):

```bash
make manifests   # 1. build pinned, hash-ranked manifests
make audit       # 2. independently recompute and hard-assert split integrity
make lock        # 3. write artifacts/paper_a_sft_v2/LOCK.json
make train       # 4. train the 4 × 5 LoRA-SFT panel
make eval        # 5. score bases + adapters → scores.parquet
make analyze     # 6. strict validated analysis → tables/figures
```

`make analyze` is the same-source in-pipeline path. Full-archive verification
must run from the immutable execution-source snapshot/bundle whose bytes match
the lock; the later hardened checkout cannot honestly impersonate that snapshot.
See [docs/reproducibility.md](docs/reproducibility.md).

See [experiments/README.md](experiments/README.md) for what each step does.

## Auditable evidence chain

The study is designed to be checkable without rerunning the GPU work:

- **V2 locks** are self-hashed and bind the config, every manifest, the audit,
  prompt rendering, source state, and expected score identity. The historical
  `LOCK.json` remains immutable and is accepted only with `--allow-legacy-lock`.
- **`audit/`** independently recomputes and fail-closes on exact/conflicting
  overlap, label balance, upstream family disjointness, selection provenance,
  pinned revisions, and deterministic near-duplicate dispositions.
- **`scores/scores.parquet`** stores per-row **content hashes and model logits
  only — never raw prompt text** — so the third-party benchmark text is not
  redistributed, yet every metric is recomputable.
- **`analysis/`** holds `results.json`, descriptive `claim_checks.json`, per-seed
  and per-benchmark outputs, and figures. Analysis validates the complete score
  matrix and sibling metadata before computing anything.

All metrics come from [guard_research/metrics.py](guard_research/metrics.py)
(sklearn-backed, tie-aware, permutation-invariant) — there is no ad-hoc
average-precision loop anywhere in the pipeline.

## Data & provenance

The corrected builder writes to the separate `artifacts/paper_a_sft_v2/` namespace,
pulls every source at a pinned revision, and uses deterministic hash-ranked cohorts.
The old `data/frozen_eval_rows.json` seed-7 cohorts are available only through an
explicit legacy option. Raw third-party rows are gitignored. The released v2
public-manifest tree under
[`artifacts/paper_a_sft_v2/public_manifests/`](artifacts/paper_a_sft_v2/public_manifests/)
is recursively text-free and retains identifiers, revisions, hashes, licenses,
selection provenance, and family links. The historical v1 public index remains
archival evidence only. Provenance uses one pinned NumPy MinHash implementation
so the installed environment cannot change family assignments.

## Earlier broad study & Paper B

The broad measurement study this paper was distilled from — protocol-dependent
ranking, the SFT/DPO/GRPO objective comparison, a mortgage-compliance case study,
guardrail baselines, an ensemble mitigation, and a fairness probe — is preserved
under [legacy/](legacy) as the basis for a planned follow-up. It is **not** needed to
reproduce this paper. Note the metric caveat in [legacy/README.md](legacy/README.md).

## Citation & license

If you use this work, please cite it via [CITATION.cff](CITATION.cff).
Licensed under [Apache 2.0](LICENSE).
