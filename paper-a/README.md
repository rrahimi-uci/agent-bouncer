# Paper — *The Benchmark Chooses the Winner*

ACM-formatted (`acmart`, `sigconf` + `nonacm`) manuscript for the focused Paper A:
**measuring fine-tuning specialization across safety-guard benchmarks** (4 checkpoints
× 5 seeds). The reported values are a retrospective legacy run; the manuscript
states which corrected pipeline changes require a new GPU run.

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

Tables and figures are generated from the committed legacy scores and copied into
this folder by an explicit compatibility reproduction:

```bash
make repro-legacy    # from the repository root: analyze, sync, verify
```

From the repository root, `artifacts/paper_a_sft/analysis/` holds the canonical outputs
(`tables/table3_primary.tex`, `tables/table4_per_benchmark.tex`,
`tables/table5_seed_values.tex`, `tables/results_macros.tex`,
`figures/specialization_plane.pdf`); the copies
here are what the `.tex` consumes.

## Build
```bash
make            # compile benchmark_chooses_the_winner.tex -> .pdf (tectonic; fetches acmart)
make clean      # remove build artifacts
```
After `make repro-legacy` has synced generated inputs, you can also run
`tectonic benchmark_chooses_the_winner.tex` directly (network may be needed on
the first run to fetch `acmart`).

## Provenance
Legacy values trace to `artifacts/paper_a_sft/scores/scores.parquet` via the canonical
metric module (`guard_research/metrics.py`) — see the repository
[`../README.md`](../README.md) "Auditable evidence chain". Strict analysis requires
a v2 lock; legacy reproduction requires `--allow-legacy-lock`. The earlier broad study's
manuscript, prose draft, and its five figures are preserved in git history (they are
not part of this focused paper); the broad-study code lives under [`../legacy/`](../legacy).
