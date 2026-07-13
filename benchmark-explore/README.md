# Benchmark Explorer

A self-contained, single-page web app for exploring the safety-guard evaluation
suites used in *"The Benchmark Chooses the Winner: A Fair Evaluation of Small-LLM
Safety Guards."*

## Open it

Just double-click **`index.html`** (or drag it into any browser). No server, no
build step, no dependencies — everything is inlined into the one file.

## What's inside

- **13 benchmarks** spanning 4 evaluation pools (in-distribution, in-house
  held-out, novel/OOD, domain use case) and 3 safety axes (guardrail, red-team,
  over-refusal), plus one landscape-only survey set.
- For each set: the source repo/citation, what it tests, prompt-pool
  composition (unsafe vs. safe row counts), the role it plays in the paper, and
  a plain-language caveat.
- Threshold-free **tie-aware AUPRC** (sklearn `average_precision_score`) is used
  throughout, matching the paper's corrected metric.

## Interactions

- **Pool** / **Axis** filter chips narrow the card grid.
- Click any card to open a detail drawer with sources, composition, per-system
  results, and caveats.

The page is intentionally dependency-free so it stays openable and archivable
alongside the paper.
