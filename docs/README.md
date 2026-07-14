# Design & planning notes

Working notes behind the study. They are historical design documents, not the paper;
the authoritative results live in [`../paper-a/`](../paper-a) and
[`../artifacts/paper_a_sft/`](../artifacts/paper_a_sft).

## Paper A — *The Benchmark Chooses the Winner*

- [paper-a-minimal-refactor-plan.md](paper-a-minimal-refactor-plan.md) — the plan
  that contracted the broad study into the focused specialization result, and
  specified the auditable pipeline and repository structure this repo implements.
- [paper-a-specialization-frontier-plan.md](paper-a-specialization-frontier-plan.md) —
  the design of the represented-vs-transfer specialization measurement.
- [paper-a-improvement-and-extension-recommendations.md](paper-a-improvement-and-extension-recommendations.md) —
  reviewer recommendations grounded in the current results: run the clean v2 rerun,
  add an uninspected benchmark, and develop the base-competence law (who FT helps vs hurts).

## Paper B / broad study (future work)

Planning notes for the earlier broad study and the planned follow-up. The
corresponding code is quarantined under [`../legacy/`](../legacy).

- [paper-b-joint-compliance-stack-plan.md](paper-b-joint-compliance-stack-plan.md) —
  a joint general + domain-specific compliance guard stack.
- [paper-b-development-checkpoint-plan.md](paper-b-development-checkpoint-plan.md) —
  development checkpoints for the follow-up (the governing `v2-adversarial-review` plan).
- [paper-b-feasibility-investigation.md](paper-b-feasibility-investigation.md) —
  independent assessment of whether the (mortgage joint-stack) Paper B plan is feasible
  and can reach a solution; recommends a de-scoped measurement-paper landing.
- [paper-b-compose-dont-tune-plan.md](paper-b-compose-dont-tune-plan.md) —
  **recommended near-term Paper B**: "Compose, Don't Tune." A composed guardrail (calibrated
  average of the untuned base + the SFT adapter) recovers held-out transfer that fine-tuning
  erodes, while keeping in-domain gains — grounded in a prototype on the committed scores and a
  4-lens adversarial review. Zero new training; gated on Paper A's clean rerun.
- [paper-b-topic-proposal.md](paper-b-topic-proposal.md) —
  the objective × base-competence study (SFT vs DPO vs GRPO). **Deferred to a contingent Paper C**
  (needs a GPU retrain; reproduces published SFT-vs-RL results) — see the compose-don't-tune plan
  for why it moves behind the composition paper.
- [smollm3-guard-plan.md](smollm3-guard-plan.md) — plan behind the SmolLM3 guard
  reproduction notebook (the broad-study notebooks were removed from the repo;
  their builders remain under [`../legacy/experiments/`](../legacy/experiments)).
- [mortgage-benchmark-build-spec.md](mortgage-benchmark-build-spec.md) — target build
  spec (v0.2) for the mortgage guard benchmark (the *what*): request-screening shape, dual
  labels (general-safety + mortgage-policy), quadrant/fairness/prevalence coverage,
  decontamination, evaluation via the canonical tooling. Supersedes the v0.1.0 draft under `data/`.
- [mortgage-benchmark-build-runbook.md](mortgage-benchmark-build-runbook.md) — the *how*:
  phase-by-phase build procedure (scaffold → freeze rubric → author label-bearing template
  families → generate → decontaminate → split/seal → validate → evaluate → package), with
  scripts, a worked example, and the one-command pipeline.
- [mortgage-benchmark-hard-results.md](mortgage-benchmark-hard-results.md) — results
  from the hardened mortgage-compliance benchmark.

## Reviews & background

- [review-2026-07-12.md](review-2026-07-12.md) — a technical re-review of the earlier
  broad-study manuscript; the recommendation to narrow to the focused specialization
  result is what this repo now implements.
- [metrics-survey.md](metrics-survey.md) — survey of related-work evaluation metrics
  (background for the evaluation protocol).
