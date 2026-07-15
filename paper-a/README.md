# Paper — *The Benchmark Chooses the Winner*

Single-column 11-point article (`\documentclass[11pt]{article}`) for the focused Paper A:
**measuring fine-tuning specialization across safety-guard benchmarks** (4 checkpoints
× 5 seeds). Clean-v2 analysis is the only publication workflow. Any retrospective
v1 values retained in the repository are explicitly archival; publication inputs
must be regenerated from a final v2 score cache through the strict path below.

## Files

- `benchmark_chooses_the_winner.tex` — the paper (authoritative source).
- `benchmark_chooses_the_winner.pdf` — compiled output.
- `paper-a-review.md` — deep code/data/statistical critique, repair record, and
  residual scientific blockers.
- `refs.bib` — bibliography.
- `figures/specialization_plane.pdf` — the generated empirical figure:
  represented-source gain vs. held-out transfer (the specialization plane).
- `figures/study_design.svg` — accessible HTML counterpart of the manuscript's
  inline TikZ study-design figure.
- `tab_primary_gen.tex`, `tab_sensitivity_gen.tex`, `tab_seed_values_gen.tex` —
  generated tabulars consumed by the manuscript.
- `results_macros_gen.tex` — generated aggregate/RQ4/stress narrative values.

The authoritative generated inputs for a publication build live under
`artifacts/paper_a_sft_v2/analysis/` and are copied into this folder only after a
strict v2 analysis. A released score-only cache can be reproduced from the
repository root without raw prompts, adapters, or GPU access:

```bash
make repro            # verify v2 cache, analyze, compare checked-in inputs
make paper            # compare v2 inputs again, then compile
```

Full-archive verification is instead run from the immutable execution-source
snapshot/bundle whose exact bytes match the lock, using the verified in-pipeline
`make analyze` workflow documented in `docs/reproducibility.md`. Both workflows
write the canonical outputs (`tables/table3_primary.tex`,
`tables/table4_per_benchmark.tex`,
`tables/table5_seed_values.tex`, `tables/results_macros.tex`,
`figures/specialization_plane.pdf`) under the v2 analysis root; the copies here are
what the `.tex` consumes. `make repro` compares these copies without overwriting
them; maintainers update them only with an explicit `make paper-sync`, then rerun
verification. If the final cache is absent or inconsistent, these targets fail
rather than using v1 values.

The historical v1 outputs remain available only through the explicit archival
target `make repro-legacy`, which analyzes
`artifacts/paper_a_sft/scores/scores.parquet` without updating publication paper
files. It must not be presented as v2.

## Build
```bash
make paper      # from repository root: verify v2 inputs and compile
make -C paper-a clean
```
Running `make` inside this directory compiles the copies already present but does
not prove where they came from. Use the repository-root `make paper` for a checked
build. Tectonic may need network access on its first run to fetch LaTeX packages
and fonts.

## Provenance
V2 values must trace from the final lock, self-hashed `RELEASE.json`, tracked
release anchor, and bound score metadata through the
canonical metric module (`guard_research/metrics.py`) and generated analysis
metadata. See the repository [`../README.md`](../README.md) “Auditable evidence
chain” and [`../docs/reproducibility.md`](../docs/reproducibility.md). V1 analysis
requires `--allow-legacy-lock`; strict v2 analysis never implies it. The earlier
broad study's code lives under [`../legacy/`](../legacy) and is not part of this
focused paper.
