# Design & planning notes

Working notes behind the study. They are historical design documents, not the paper;
the authoritative results live in [`../paper-a/`](../paper-a) and
[`../artifacts/paper_a_sft_v2/`](../artifacts/paper_a_sft_v2). The v1
[`../artifacts/paper_a_sft/`](../artifacts/paper_a_sft) tree is archival.

## Paper A — *The Benchmark Chooses the Winner*

- [paper-a-minimal-refactor-plan.md](paper-a-minimal-refactor-plan.md) — the plan
  that contracted the broad study into the focused specialization result, and
  specified the auditable pipeline and repository structure this repo implements.
- [paper-a-specialization-frontier-plan.md](paper-a-specialization-frontier-plan.md) —
  the design of the represented-vs-transfer specialization measurement.
- [paper-a-improvement-and-extension-recommendations.md](paper-a-improvement-and-extension-recommendations.md) —
  reviewer recommendations grounded in the current results: add an uninspected benchmark and
  test the base-competence hypothesis on an independent development/outcome split.

## Paper B / broad study (future work)

Planning notes for the earlier broad study and the planned follow-up. The Compose, Don't Tune
plan is the authoritative near-term Paper B plan; the mortgage joint-stack/checkpoint plans are
historical alternatives. Their corresponding code is quarantined under [`../legacy/`](../legacy).

- [paper-b-joint-compliance-stack-plan.md](paper-b-joint-compliance-stack-plan.md) —
  archived historical joint general + domain-specific compliance guard-stack plan.
- [paper-b-development-checkpoint-plan.md](paper-b-development-checkpoint-plan.md) —
  archived historical checkpoints for the superseded mortgage direction; these do not govern
  composition work.
- [paper-b-feasibility-investigation.md](paper-b-feasibility-investigation.md) —
  historical assessment explaining why the mortgage joint-stack direction was dropped.
- [paper-b-compose-dont-tune-plan.md](paper-b-compose-dont-tune-plan.md) —
  **recommended near-term Paper B**: "Compose, Don't Tune." A composed guardrail (calibrated
  average of the untuned base + the SFT adapter) trades a small represented-source loss for
  transfer recovery relative to SFT in the completed clean-v2 retrospective estimate. It is not
  prospective or confirmatory; Paper B still requires its own lock, a prospective cohort,
  WiSE-FT GPU rescoring, a same-inference-cost SFT+SFT control, and matched-compute KL/replay
  baselines.
- [paper-b-topic-proposal.md](paper-b-topic-proposal.md) —
  the objective × base-competence study (SFT vs DPO vs GRPO). **Deferred to a contingent Paper C**
  (needs a GPU retrain; reproduces published SFT-vs-RL results) — see the compose-don't-tune plan
  for why it moves behind the composition paper.
- [paper-c-objective-axis-reward-and-design.md](paper-c-objective-axis-reward-and-design.md) —
  **grounded design layer for the objective axis (Paper C)**: answers *what reward/preference
  signal works* (verifiable label, no learned RM), *which objectives to select and drop*
  (SFT · DPO · KTO · GRPO core, +ORPO; drop SimPO/PPO/BCO), and a five-facet literature review
  that grounds feasibility and novelty. Ties the axis to the composition remedy via reference-KL
  anchoring; pre-registers GRPO's likely single-token null. Companion to the topic proposal.
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
