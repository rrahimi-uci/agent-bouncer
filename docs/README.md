# Design & reference notes

What remains here is the **live reference material** the code and papers depend on, plus
**forward-looking** design for the deferred Paper C. The historical/applied planning documents
(the Paper A refactor and specialization plans, the Paper B compose/joint-stack/feasibility
plans, the SmolLM3 guard plan, the 2026-07-12 review, the metrics survey, and the hardened-benchmark
results) have been removed now that they are applied; they remain in git history. The authoritative
results live in [`../papers/`](../papers) and [`../artifacts/paper_a_sft_v2/`](../artifacts/paper_a_sft_v2).

## Reproducibility contracts (live)

- [reproducibility.md](reproducibility.md) — the Paper A reproducibility contract (what the
  fail-closed lock/audit/analysis chain guarantees). Linked from the root README and
  `experiments/README.md`.
- [reproducibility-environments.md](reproducibility-environments.md) — the pinned
  training/scoring vs. analysis environments behind `requirements.txt`.

## Mortgage benchmark spec (live)

- [mortgage-benchmark-build-spec.md](mortgage-benchmark-build-spec.md) — the *what*: the row
  schema and dual-label design that [`../mortgage-benchmark/magen/`](../mortgage-benchmark/magen)
  mirrors (request-screening shape, general-safety + mortgage-policy labels, quadrant/fairness/
  prevalence coverage, decontamination, canonical-tooling evaluation).
- [mortgage-benchmark-build-runbook.md](mortgage-benchmark-build-runbook.md) — the *how*:
  the phase-by-phase build procedure (scaffold → freeze rubric → author families → generate →
  decontaminate → split/seal → validate → evaluate → package).

## Paper C — objective axis (future work, not applied)

- [paper-c-objective-axis-reward-and-design.md](paper-c-objective-axis-reward-and-design.md) —
  grounded design for the objective axis (SFT · DPO · KTO · GRPO): what reward/preference signal
  works (verifiable label, no learned RM), which objectives to keep vs. drop, and a literature
  review grounding feasibility and novelty.
- [paper-b-topic-proposal.md](paper-b-topic-proposal.md) — the deferred topic proposal (objective ×
  independent-competence specialization) that motivates that Paper C.
