# Paper B plan — "Compose, Don't Tune"

**Status: authoritative near-term Paper B plan; clean-v2 retrospective estimation completed,
but no prospective or separately locked Paper B study has run.** The older mortgage joint-stack
plans are historical alternatives, not governing plans.

Plan for the next paper, extending the "Benchmark Chooses the Winner" line of work with
the **techniques + guardrail-design** ideas (DPO/GRPO objectives, a fintech/mortgage
high-compliance domain, and an ensemble/fine-tune guardrail). Written 2026-07-14, after an
empirical prototype on Paper A's clean-v2 scores **and** a 4-lens adversarial review. For review.

---

## 0. Verdict (the honest answer to "does it make sense / make it better?")

**Yes — but only if collapsed to one axis.** Your three ideas map to three axes:

1. **Objectives** — does the specialization trade-off depend on SFT vs DPO vs GRPO?
2. **Domain** — does it hold in a high-compliance fintech/mortgage domain?
3. **Guardrail design** — can composing the untuned base with the tuned adapter (ensemble)
   keep in-domain gains *and* recover generalization, instead of "more tuning"?

Bundling all three into one paper **recreates the exact "too many stories" defect** that got
the earlier broad manuscript rejected and that the Paper A refactor was built to escape. Two
of the three are also the *weak* kind of axis right now: the objective arms have **no
re-analyzable per-row scores** (legacy, single-seed) and need a GPU retrain, and the
*naturalistic* mortgage cohort was already found **infeasible** for a solo researcher.

**Axis 3 (composition) is the candidate paper.** Its output-space prototype needs no new
training beyond Paper A's clean SFT rerun, and it tests a candidate remedy rather than adding
another measurement axis. It extends the arc **measure and characterize (Paper A) → test a
candidate remedy (Paper B)**.
Objectives become a contingent **Paper C**; the domain becomes an optional exploratory
appendix. This supersedes [`paper-b-topic-proposal.md`](paper-b-topic-proposal.md) (the
objective-axis proposal) as the recommended *near-term* Paper B.

### One caveat I must correct up front (found by the review)

My first read of the prototype said "the ensemble **beats both** base and SFT on transfer."
That is **overstated**. The clean-v2 retrospective paired bootstrap estimates a
calibrated-average advantage over **SFT for all 4 models**, but an advantage over the **base for
only 2 of 4**; the descriptive interval overlaps zero for SmolLM3 and is entirely below zero for
Qwen3-4B (0.914 vs 0.944). In this four-model panel, the positive ensemble-minus-base estimates
occur for the two weaker base-transfer scores. With only four checkpoints, that association is a
hypothesis, not a law or validated decision rule. The defensible prototype summary is:

> **The composition trades a small represented-source loss for a larger transfer recovery relative
> to SFT** on this fixed panel, pulling transfer back toward the untuned base's level. The observed
> variation by base competence is exploratory. A prospectively fixed and supported
> represented-source noninferiority margin could justify a bounded-retention claim, but not literal
> Pareto improvement: Pareto requires being no worse on every objective and better on at least one.

Not "dominates both." That honesty is the paper's spine, not a footnote.

---

## 1. The evidence that makes this worth doing

Computed by the current reproducible analyzer
[`experiments/analyze_composition.py`](../experiments/analyze_composition.py) on
[`artifacts/paper_a_sft_v2/scores/scores.parquet`](../artifacts/paper_a_sft_v2/scores/scores.parquet)
(score SHA-256 `b941ddba...a1c3`, Paper A lock SHA-256 `cabc8dee...c25`; canonical tie-aware
macro-AP; **mean of per-seed AP**, matching Paper A; panel mean over the 4 checkpoints). The
analyzer reports `clean_v2_retrospective_estimation`: transfer rows were not used to train SFT or
fit calibration/combiner parameters, but the transfer benchmarks and earlier scores were inspected
during method development. This is dataset-held-out retrospective evidence, not a leak-free
prospective test. Combiner = mean of the two guards' calibrated unsafe-probabilities. Exact
machine-readable results are in
[`composition.json`](../artifacts/paper_a_sft_v2/analysis/composition/composition.json).

| Guard | represented macro-AP | transfer macro-AP | min(rep, tr) |
|---|---:|---:|---:|
| base (untuned) | 0.658 | 0.866 | 0.658 |
| SFT (tuned) | **0.982** | 0.807 | 0.807 |
| **base+SFT, calibrated avg (fixed primary)** | 0.962 | 0.883 | 0.883 |
| base+SFT, logit avg (ablation) | 0.943 | **0.891** | **0.891** |
| base+SFT, calibrated max (ablation) | 0.961 | 0.826 | 0.826 |

The fixed-primary calibrated average changes represented macro-AP from 0.982 to 0.962 and
transfer macro-AP from 0.807 to 0.883 relative to SFT. The paired-bootstrap means and percentile
intervals are **−0.019 [−0.031, −0.010]** on represented and **+0.075 [+0.058, +0.093]** on
transfer. Relative to base they are **+0.297 [+0.250, +0.346]** on represented and
**+0.017 [+0.005, +0.030]** on transfer. Logit averaging has the largest descriptive transfer and
minimum score in the table, but it is an ablation; selecting it from these exposed transfer
results would violate the transfer-blind primary rule.

**Per-model (transfer), the honest picture** — recovery is heterogeneous, and the difference from
base is positive only for the two weaker base-transfer scores:

| Base (transfer strength) | base | SFT | avg-ensemble | ensemble − base [95% CI] |
|---|---:|---:|---:|---|
| SmolLM2 (weak, 0.790) | 0.790 | 0.830 | 0.857 | **+0.066 [+0.038, +0.095]** (positive interval) |
| Qwen2.5 (0.819) | 0.819 | 0.780 | 0.855 | **+0.036 [+0.008, +0.066]** (positive interval) |
| SmolLM3 (0.910) | 0.910 | 0.823 | 0.907 | −0.003 [−0.012, +0.005] (interval spans zero) |
| Qwen3-4B (strong, 0.944) | 0.944 | 0.794 | 0.914 | −0.029 [−0.043, −0.018] (negative interval) |

**Verified descriptive intervals (current analyzer, hierarchical bootstrap):** ensemble − SFT
on transfer is positive for all 4 models (panel +0.075 [+0.058, +0.093]); ensemble − base has
intervals above zero for two bases, spanning zero for SmolLM3, and below zero for Qwen3-4B.
These are precision summaries, not formal significance decisions. A transfer-blind convex-weight
sweep selected on `calibration` chooses `w_sft≈0.95`. Two shuffle diagnostics show that destroying
the SFT ranking removes the observed gain, while within-class row shuffling does not; this narrows
possible explanations but does not establish a mechanism. At the 5%-FPR deployment diagnostic,
realized transfer macro-FPR exceeds the calibration target for both SFT (0.155) and the ensemble
(0.114) — a finding to report, not hide.

The apparent base-competence pattern is especially fragile: base transfer AP appears both in the
putative predictor and, through `ensemble − base`, in the same-row outcome. That mathematical
coupling can manufacture a negative association. The table motivates a prospective hypothesis;
it does not estimate a competence rule. A valid test must measure base competence on a locked
development cohort and predict composition recovery on a disjoint prospective cohort.

**Why it makes the work better:** Paper A ends on a problem ("fine-tuning specializes"). This
turns the line of work into problem → **candidate remedy**. The primary output-space operation
needs no shared weights or retraining, but it doubles inference passes and has not yet been
validated on prospective traffic. Its apparent relation to base competence is a hypothesis.

---

## 2. The paper

**Title (working):** *Compose, Don't Tune: Recovering Safety-Guard Generalization by Keeping
the Untuned Base in the Decision.*

**Through-line thesis to test:** *A weight-free calibrated average of the untuned base and one
SFT adapter may exchange a small amount of represented-source AP for transfer recovery relative
to SFT. Independently measured pre-transfer base competence may predict the size of that recovery.*

**Setup (cite Paper A, do not re-litigate):** the fixed-panel specialization estimate is the
diagnosis; base-competence heterogeneity remains exploratory.

**Research questions**
- **RQ1 (retention):** Does calibrated base+SFT composition keep the represented-source gains?
  (Clean-v2 retrospective point estimate: 0.982 → 0.962; paired-bootstrap mean difference
  −0.019 [−0.031, −0.010].)
- **RQ2 (recovery):** What is the estimated transfer change *relative to SFT* per model and per
  benchmark, and how close does composition come to the untuned base?
- **RQ3 (heterogeneity hypothesis):** Does base competence measured on a locked development cohort
  predict composition recovery on a disjoint prospective cohort? Four checkpoints and same-row
  retrospective contrasts can motivate this hypothesis, not validate a decision rule.

**Contributions**
1. A reproducibly verified **clean-v2 retrospective estimate** of the
   represented-retention/transfer-recovery trade-off across a fixed four-checkpoint panel, with
   prior benchmark exposure disclosed. This is not yet a separately locked Paper B result.
2. An exploratory **when-to-compose hypothesis** based on independently measured base competence,
   explicitly requiring a larger model panel and a disjoint prospective outcome before it becomes
   a decision rule.
3. A clean-v2 **operator ablation** (avg-of-calibrated-probs vs max vs convex-weight/PIT vs
   logit-average), with a transfer-blind selection rule, plus planned submission-grade WiSE-FT,
   same-inference-cost SFT+SFT, and matched-compute tuning baselines.

---

## 3. Novelty positioning (so it survives "known result")

Base + fine-tuned combination under distribution shift has close prior art — **WiSE-FT**
(Wortsman 2022), **model soups**, and **calibrated ensembling** (Kumar 2022). If framed as "a
new ensemble," it draws a limited-novelty rejection. Position the contribution as:

- **Output-space, not weight-space:** a calibrated *score* average — architecture-agnostic,
  needs no shared weights (WiSE-FT/soups need the same architecture and interpolable weights).
- **Base-competence-conditioned recovery hypothesis:** test whether development-cohort base
  competence predicts recovery on a disjoint prospective cohort; do not regress a same-row
  `ensemble − base` contrast on base transfer AP or call four panel points a law or mechanism.
- **Observed heterogeneous recovery, not a rule:** relative to SFT, calibrated averaging improves
  transfer for every checkpoint in this panel. Relative to base, its intervals are positive for
  Qwen2.5 and SmolLM2, span zero for SmolLM3, and are negative for Qwen3-4B. A larger model panel
  with independently measured competence and disjoint prospective outcomes must test whether base
  competence explains that pattern.

**Submission-grade baselines** (or reviewers dismiss it as "output-space WiSE-FT"): run actual
WiSE-FT weight interpolation, plain logit-average, a same-inference-cost SFT+SFT ensemble of two
independently seeded adapters, and a matched-compute KL-regularized or replay-tuning comparison.
Pre-specify how SFT seed pairs are aggregated so ten pairs are not treated as independent model
replicates. WiSE-FT needs GPU rescoring; KL/replay and the two-adapter control need retained or new
weights. Until those exist, do not claim the gain comes specifically from **keeping the base** or
that "compose beats more tuning." Report latency, throughput, peak memory, and both training and
inference compute for every baseline. Use a fixed averaging rule, not a learned router.

---

## 4. Analysis protocol (the discipline that got Paper A to PASS)

- **Metrics:** benchmark-macro AP (primary, threshold-free) **and** recall at a common
  calibration-FPR target
  (threshold from held-out/calibration negatives) as a deployment diagnostic — and **disclose the
  OOD calibration failure** rather than reporting only the flattering threshold-free number.
- **Combiners:** the clean-v2 retrospective analyzer covers calibrated-prob average (primary),
  raw-prob average and PIT/rank average (calibration-robustness rows), max, convex-weight
  (grid-selected on `calibration`), and logit-avg. The prospective protocol must add WiSE-FT and
  the same-inference-cost SFT+SFT control.
  The calibrated-probability average is the fixed primary operator; **any tunable combiner
  parameter is selected without touching `id_test` or `transfer_test`** — otherwise it is an
  implicit peek at a reported test axis.
- **Uncertainty:** the same **family × seed hierarchical bootstrap** as Paper A, reported as
  **per-model and per-benchmark paired CIs**, plus leave-one-benchmark-out. Keep the four-checkpoint
  panel mean and interval as a conditional fixed-panel descriptive summary, not a substitute for
  checkpoint-level replication. State plainly that **`base` is single-pass (seed −1)**: family-row
  resampling affects both sides of every paired contrast, while seed resampling estimates only
  adapter-side variation and does not estimate base-training-seed uncertainty.
- **Equal-inference-compute control:** compare base+SFT with a fixed SFT+SFT calibrated ensemble
  built from independently seeded adapters. Aggregate at the checkpoint level and account for
  shared adapters across seed pairs; this distinguishes a base-retention effect from generic
  two-pass ensembling.
- **Null diagnostics (both implemented in the prototype):** report a signal-destroying shuffle
  and a within-class row shuffle under the same mean-of-per-seed estimand. Treat them as
  diagnostic ablations, not proof of a causal mechanism.
- **Leak checks:** confirm `probability_calibrated` was fit only on the `calibration` split,
  never on transfer rows (the averaging step makes any calibration leak matter).

The current analyzer now produces the clean-v2 retrospective result in §1:
[`experiments/analyze_composition.py`](../experiments/analyze_composition.py); Paper B still needs
its own prospective protocol and analysis lock before any claim-bearing run.
Legacy ensemble scripts may supply implementation ideas, but are not claim-bearing evidence.

---

## 5. Provenance gating (non-negotiable)

The numbers in §1 have now been regenerated from Paper A's clean-v2, row-keyed score artifact.
The release-cache run strictly reverified the score/metadata hashes, complete 24-bundle matrix,
and public-manifest identities against Paper A's v2 lock. This repairs the prototype's execution
provenance; it does **not** make the analysis prospective because the transfer benchmarks and
earlier scores informed method development. It also does not turn Paper A's lock into a Paper B
lock: the composition analyzer and its fixed constants are post-lock Paper B development work.
Therefore:

- **Completed for retrospective estimation:** Paper A's clean SFT rerun emitted the base+SFT
  per-row scores, and the output-space composition was regenerated without extra training.
- **Keep the v2 result labeled retrospective:** the exact status is
  `clean_v2_retrospective_estimation`, not prospective or confirmatory. The score-only release
  cache verifies its bound inputs. The 4.43 GB full cloud archive was downloaded and checksum-
  verified during this audit, but it is not committed or distributed by this repository; the
  public cache therefore cannot independently reverify run metadata or adapter bytes. What is
  tracked is the compact execution-source snapshot plus the archive digest and internal-checksum
  verification record. Publish the full archive at a durable immutable location before claiming
  third-party re-verification of those omitted artifacts.
- **Create a separate Paper B analysis lock before a prospective claim-bearing run.** It must bind
  the full imported source/dependency set, analysis configuration and RNGs, exact runtime/package
  versions, operator hierarchy, noninferiority margin (if one is used), and prospective cohort.
  The tracked Paper A release anchor binds the retrospective analyzer source for reproducibility,
  but it is not a prospectively frozen Paper B analysis lock.

---

## 6. Feasibility & minimal-viable scope

**Axis 3's CPU-only clean-v2 retrospective re-analysis is complete.** It fits calibration on
`calibration` only, keeps the calibrated average fixed as primary, and reports per-seed paired
deltas with hierarchical-bootstrap intervals, leave-one-benchmark-out checks, logit/rank/max
ablations, shuffle diagnostics, and calibration-target operating-point diagnostics. The result is promising
enough to continue, but it is not enough for the final novelty or generalization claim.

**Remaining effort/compute:** a submission-grade study needs GPU rescoring for WiSE-FT and new
training for a matched-compute KL/replay baseline. Prospective generalization also needs a
genuinely uninspected cohort or dataset selected before the Paper B lock.

---

## 7. Deferred / de-scoped axes (honest reasons)

- **Axis 1 — objectives (SFT/DPO/GRPO) → contingent Paper C, gated on regaining GPU.**
  No re-analyzable per-row DPO/GRPO scores survive (legacy is single-seed, legacy-metric); GRPO is
  online RL and would dominate wall-clock on an M4 Max with no GPU; and "SFT specializes, RL-family
  generalizes" is close to published results (Chu 2025, Kirk 2024), so re-establishing it is
  high-cost/low-novelty. **Use the surviving directional SFT>DPO>GRPO table as motivation only**,
  and state plainly no auditable per-row artifacts exist. *If GPU returns:* a minimal 3-seed
  DPO+GRPO rerun on the same 4 bases, scored on the frozen rows, then fed back through the Axis-3
  combiner to test "does the less-specialized GRPO leave less for composition to recover?" — which
  is the one genuinely novel way to make the objective axis pay its way.
- **Axis 2 — mortgage/fintech domain → optional exploratory appendix only.** The naturalistic
  dual-labeled cohort is infeasible (see [`paper-b-feasibility-investigation.md`](paper-b-feasibility-investigation.md)),
  and the existing synthetic bench is **saturated (base ~0.995 AUPRC)**, so it cannot show either
  specialization or a composition benefit. Pursue *only* if a hardened synthetic challenge set
  (via `legacy/experiments/wf_harden_mortgage.mjs` / the [build spec](mortgage-benchmark-build-spec.md))
  shows real base headroom in a quick zero-shot probe; otherwise drop it. Never headline a
  saturated, single-labeled, un-reviewed synthetic set as a compliance result.

---

## 8. Top risks → de-risks

1. **Overstated headline ("beats both" / "Pareto").** False per-model; a noninferiority margin can
   support bounded retention, not literal Pareto improvement. → Report the measured
   retention/recovery trade-off, per-model/per-benchmark intervals, and the Qwen3-4B loss.
2. **Retrospective reuse is mistaken for prospective evidence.** The clean-v2 rerun repairs
   execution provenance, not transfer-cohort exposure. → Keep §1 estimation-only; freeze a
   separate Paper B protocol and lock before evaluating a genuinely uninspected cohort.
3. **Novelty collision with WiSE-FT / soups.** → Run WiSE-FT + logit-avg + SFT+SFT + matched-compute
   KL/replay baselines; position as output-space + base-competence-conditioned; include the
   shuffle-null and systems-cost metrics. Fixed rule, not a learned router.
4. **Ranking gains are mistaken for deployable calibration.** The 5%-FPR target transfers poorly
   (macro-FPR 0.155 for SFT and 0.114 for the ensemble). → Pre-specify the operating-point
   analysis and report calibration shift alongside macro-AP.
5. **Generic ensembling is mistaken for a base-retention effect.** Two passes alone may help. →
   Add a same-inference-cost, independently seeded SFT+SFT ensemble and account for shared seed
   pairs in uncertainty.
6. **Mathematical coupling is mistaken for a competence law.** Same-row base AP is embedded in
   `ensemble − base`. → Measure competence on a locked development cohort and predict recovery
   only on a disjoint prospective cohort.

---

## 9. What to do next (concrete)

1. **Completed — Paper A clean v2:** the 24-bundle, 79,392-row base+SFT score artifact is bound to
   Paper A lock `cabc8dee...c25` and score hash `b941ddba...a1c3`.
2. **Completed — retrospective composition analyzer:** calibrated/raw/PIT/max/convex/logit
   operators, calibration-only tuning, family×seed bootstrap with per-model and per-benchmark
   intervals, LOO checks, shuffle diagnostics, and calibration-target operating-point outputs now
   produce §1.
3. **Next — freeze a separate Paper B protocol and lock:** pre-specify the primary operator,
   estimands, multiplicity policy, any represented-source noninferiority margin, model panel,
   development-only competence measure, prospective cohort, code/dependency hashes, and the
   WiSE-FT/SFT+SFT/KL/replay baseline grid.
4. **Then run the missing submission-grade evidence:** GPU-rescore WiSE-FT, train matched-compute
   KL/replay baselines, form the pre-specified SFT+SFT control, and evaluate all locked operators
   once on the genuinely uninspected cohort with latency, throughput, memory, and compute reported.
5. **Only after that** consider the Axis-2 appendix (if a hardened synthetic set shows headroom)
   and, if justified, the Axis-1 Paper C.

---

## 10. A plain-language edition is a first-class deliverable

This paper must ship with a **plain-language, paper-format edition** for the same audience as
[`../paper-a-simplified/`](../paper-a-simplified) — a reader with **basic statistics** and **one
LoRA fine-tune** of experience — in the same annotated style (teach-as-you-go "Background"
boxes, worked mini-examples, honest takeaway boxes, full tables + figure). Crystal-clear for that
reader is a requirement, not a nice-to-have.

- **Already done (clean-v2 preview):** the annotated Paper A edition has been extended with a
  plain-language section, *"A fix: compose, don't tune"*
  ([`../paper-a-simplified/the-benchmark-chooses-the-winner-annotated.pdf`](../paper-a-simplified/the-benchmark-chooses-the-winner-annotated.pdf),
  §5), teaching what an ensemble is, the clean-v2 base/SFT/composed tables, and the honest
  *small represented loss / transfer recovery toward base / exploratory heterogeneity* framing
  (not "beats both" or "Pareto"). It is explicitly marked a **preliminary clean-v2 retrospective
  Paper B analysis**, not a Paper A result or prospective confirmation.
- **Required for the eventual Paper B submission:** promote that preview into a standalone
  plain-language Paper B edition after the separately locked study runs. Keep it in lockstep with
  the formal Paper B using generated tables/figures rather than hand-copied values.
- **Non-negotiables for the lay edition:** define every new term inline (ensemble, calibration
  transfer, recall at a common calibration-FPR target); never headline "beats both"; always show the per-model
  heterogeneity table; state the two-passes cost and the ranking-not-calibration caveat in plain
  words.

---

*Bottom line: the clean-v2 retrospective composition estimate is promising enough to justify a
focused Paper B, but it supports a retention/recovery estimate, not dominance, Pareto optimality,
a universal remedy, a base-competence law, or a mechanism. Lock Paper B separately, add WiSE-FT
GPU rescoring, a same-inference-cost SFT+SFT control, and matched-compute KL/replay baselines;
evaluate a prospective cohort, and keep objectives/domain out of the headline.*
