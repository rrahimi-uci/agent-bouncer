# The Benchmark Chooses the Winner

Reproduction code, data manifests, and an auditable evidence chain for the paper

> **The Benchmark Chooses the Winner: Measuring Fine-Tuning Specialization Across Safety-Guard Benchmarks**
> Reza Rahimi (JazzX AI)

This repository compares each prompt-safety guard with its own untuned base on a
fixed panel of four instruction checkpoints, five SFT seeds each. It separates
changes on sources represented during training from transfer to held-out datasets.
The committed scores are a **legacy, retrospective run**; corrected provenance,
family-link, truncation, and lock code is included, but those fixes require a new
GPU run before the result can be treated as clean-run evidence.

## Headline result

Across the panel (Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B; seeds 42–46),
LoRA-SFT on the legacy 1,200-row training subset:

| Effect | Observed Δ macro-AP (base → SFT) | Descriptive 95% two-sided bootstrap interval |
|---|---:|---:|
| **Represented** sources | **+0.333** | **[+0.272, +0.379]** |
| **Transfer** to held-out datasets | **−0.050** | **[−0.076, −0.025]** |

SFT lifts every checkpoint to about 0.98 macro-AP on represented sources. Transfer
is heterogeneous by checkpoint—SmolLM2 +0.05, Qwen2.5 −0.03, Qwen3-4B −0.10,
SmolLM3 −0.12—while every leave-one-checkpoint-out and
leave-one-transfer-benchmark-out aggregate remains negative. Benchmark-macro
transfer FPR rises from **8.3% to 13.7%** (pooled-negative: **4.4% to 14.6%**),
and HarmBench recall falls from **78.4% to 57.5%**.
These are estimation-only legacy results, not formal claim gates or a universal
"fine-tuning hurts" conclusion.

The legacy numbers regenerate from the committed score table only through the
explicit compatibility path, [`make repro-legacy`](#reproduce-the-legacy-scores-no-gpu).

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
artifacts/paper_a_sft_v2/   output root for clean manifests, lock, runs, scores,
                     and analysis (created by the corrected pipeline)
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

Python 3.11+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # current library + training/scoring + dev
# or, for a fully pinned current environment:
pip install -r requirements.txt
```

CPU-only analysis needs the core package; tests need `pip install -e ".[dev]"`.
The pinned current environment and the partially recorded historical GPU
environment are intentionally separated in
[docs/reproducibility-environments.md](docs/reproducibility-environments.md).

## Reproduce the legacy scores (no GPU)

The historical score bundle regenerates its tables and figures on CPU (about
10–15 minutes for 10,000 bootstrap replicates on the reviewed laptop), but it is
rejected by the strict v2 contract unless legacy mode is explicit:

```bash
make repro-legacy  # explicit compatibility mode → tables + figures, then paper sync
make test          # unit and release-contract tests
make selftest   # synthetic end-to-end check of the analysis
make paper      # build the PDF (needs tectonic)
```

`make help` lists every target.

## Reproduce from scratch (GPU)

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

See [experiments/README.md](experiments/README.md) for what each step does.

## Auditable evidence chain

The study is designed to be checkable without rerunning the GPU work:

- **New v2 locks** are self-hashed and bind the config, every manifest, the audit,
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
explicit legacy option. Raw third-party rows are gitignored. The tracked
[public manifest index](artifacts/paper_a_sft/public_manifests/manifest.json) is a
recursively text-free snapshot of the historical cohorts and explicitly marks
itself incompatible with a clean rerun; a v2 build writes its own public index.
Both retain identifiers, revisions, hashes, licenses, selection provenance, and
family links. Provenance uses one pinned NumPy MinHash implementation so the
installed environment cannot change family assignments.

## Earlier broad study & Paper B

The broad measurement study this paper was distilled from — protocol-dependent
ranking, the SFT/DPO/GRPO objective comparison, a mortgage-compliance case study,
guardrail baselines, an ensemble mitigation, and a fairness probe — is preserved
under [legacy/](legacy) as the basis for a planned follow-up. It is **not** needed to
reproduce this paper. Note the metric caveat in [legacy/README.md](legacy/README.md).

## Citation & license

If you use this work, please cite it via [CITATION.cff](CITATION.cff).
Licensed under [Apache 2.0](LICENSE).
