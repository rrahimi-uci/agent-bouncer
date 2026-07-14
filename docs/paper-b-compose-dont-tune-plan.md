# Paper B plan — "Compose, Don't Tune"

Plan for the next paper, extending the "Benchmark Chooses the Winner" line of work with
the **techniques + guardrail-design** ideas (DPO/GRPO objectives, a fintech/mortgage
high-compliance domain, and an ensemble/fine-tune guardrail). Written 2026-07-14, after an
empirical prototype on the committed scores **and** a 4-lens adversarial review. For review.

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

**Axis 3 (composition) is the real paper.** It is the cheapest (zero new training beyond the
clean SFT rerun Paper A already needs), the most novel-feeling (a *fix*, not another
measurement), and it closes the arc **measure (Paper A) → explain (Paper A) → fix (Paper B)**.
Objectives become a contingent **Paper C**; the domain becomes an optional exploratory
appendix. This supersedes [`paper-b-topic-proposal.md`](paper-b-topic-proposal.md) (the
objective-axis proposal) as the recommended *near-term* Paper B.

### One caveat I must correct up front (found by the review)

My first read of the prototype said "the ensemble **beats both** base and SFT on transfer."
That is **overstated**. Independent re-run with a paired bootstrap shows the calibrated-average
ensemble beats **SFT for all 4 models**, but beats the **base for only 2 of 4** — it *ties* on
SmolLM3 and is **significantly worse than the base on Qwen3-4B** (0.930 vs 0.945). The
panel-level edge over base is carried by the weak-base models — Paper A's base-competence
pattern resurfacing *inside* the ensemble. The defensible claim is:

> **The composition Pareto-improves over the SFT guard** — it keeps almost all represented-source
> gains **and** recovers dataset-held-out transfer (significantly, every model, every transfer
> benchmark), pulling it *back toward the untuned base's level*. **How much it recovers is
> predicted by base competence.** It recovers generalization by *composing*, not by tuning harder.

Not "dominates both." That honesty is the paper's spine, not a footnote.

---

## 1. The evidence that makes this worth doing

Prototype on the committed `artifacts/paper_a_sft/scores/scores.parquet` (tie-aware macro-AP,
panel mean over the 4 checkpoints; **transfer is held out from both base and SFT, so it is
leak-free**). Combiner = mean of the two models' calibrated unsafe-probabilities.

| Guard | represented macro-AP | transfer macro-AP | min(rep, tr) |
|---|---:|---:|---:|
| base (untuned) | 0.651 | 0.867 | 0.651 |
| SFT (tuned) | **0.989** | 0.842 | 0.842 |
| **base+SFT, calibrated avg** | 0.973 | **0.897** | **0.897** |
| base+SFT, max | 0.968 | 0.859 | 0.859 |

The calibrated-average composition keeps represented at 0.973 (−0.016 vs SFT) and lifts
transfer to 0.897 — **above SFT (0.842) and, at the panel level, above base (0.867)**. Best
worst-case (min) score by a wide margin.

**Per-model (transfer), the honest picture** — composition helps most exactly where SFT hurt
most, and least where the base was already strong:

| Base (transfer strength) | base | SFT | avg-ensemble | ensemble vs base |
|---|---:|---:|---:|---|
| SmolLM2 (weak, 0.787) | 0.787 | 0.855 | 0.869 | **+0.082 (win)** |
| Qwen2.5 (0.822) | 0.822 | 0.821 | 0.874 | **+0.052 (win)** |
| SmolLM3 (0.914) | 0.914 | 0.820 | 0.913 | ≈ tie |
| Qwen3-4B (strong, 0.945) | 0.945 | 0.872 | 0.930 | −0.015 (loss vs base) |

**Adversarially reproduced by the review (to be re-confirmed with the locked pipeline):**
ensemble > SFT is bootstrap-significant for all 4 models; a within-source label-shuffle **null
control collapses to 0.712** (so the gain is real composition, not variance reduction); a
convex-weight sweep peaks near `w_sft≈0.25` at ~0.901; and at a deployed 5%-FPR threshold, OOD
calibration fails for *both* SFT (realized FPR 0.21–0.38) and the ensemble (0.15–0.30) — a
finding to report, not hide.

**Why it makes the work better:** Paper A ends on a problem ("fine-tuning specializes"). This
turns the line of work into problem → **remedy**, and the remedy is cheap, deployable
(output-space, no shared weights, no retrain), and *mechanistically tied to Paper A's
base-competence law*.

---

## 2. The paper

**Title (working):** *Compose, Don't Tune: Recovering Safety-Guard Generalization by Keeping
the Untuned Base in the Decision.*

**Through-line thesis:** *Fine-tuning specialization is a composition problem, not a tuning
problem. A weight-free, calibrated average of the untuned base and the SFT adapter retains
represented-source gains while recovering dataset-held-out transfer that SFT specialized away —
and how much it recovers is governed by base competence. You recover generalization by
composing, not by tuning harder.*

**Setup (cite Paper A, do not re-litigate):** the specialization trade-off and the
base-competence law are the given diagnosis.

**Research questions**
- **RQ1 (retention):** Does calibrated base+SFT composition keep the represented-source gains?
  (prototype: 0.989 → 0.973.)
- **RQ2 (recovery):** Does it recover transfer *relative to SFT* — significantly, per model and
  per benchmark — and *toward* the untuned base's level?
- **RQ3 (when/why):** The base-competence map — weak bases gain, strong bases only tie or lose.
  This heterogeneity is the honest, novel core; foreground it, don't bury it.

**Contributions**
1. Composition **recovers transfer** across the fixed 4-checkpoint panel, leak-free, from
   released row-keyed scores — Pareto-improving over the SFT guard.
2. A **when-to-compose decision rule** grounded in base competence (compose to protect a strong
   base's OOD behavior; a weak base benefits everywhere).
3. An **operator ablation** (avg-of-calibrated-probs vs max vs convex-weight/PIT vs
   WiSE-FT weight-interpolation vs logit-average), with a transfer-blind selection rule.

---

## 3. Novelty positioning (so it survives "known result")

Base + fine-tuned combination under distribution shift has close prior art — **WiSE-FT**
(Wortsman 2022), **model soups**, and **calibrated ensembling** (Kumar 2022). If framed as "a
new ensemble," it draws a limited-novelty rejection. Position the contribution as:

- **Output-space, not weight-space:** a calibrated *score* average — architecture-agnostic,
  needs no shared weights (WiSE-FT/soups need the same architecture and interpolable weights).
- **Base-competence-conditioned recovery:** *when and why* composing beats tuning is predicted by
  the base's OOD competence (ties the fix to Paper A's law). This is the map, not just a knob.
- **Recovers precisely where tuning hurts most:** strong bases, adversarial-jailbreak OOD
  (WildJailbreak/JailbreakBench), and the HarmBench recall collapse (78→57%).

**Mandatory baselines** (or reviewers dismiss it as "output-space WiSE-FT"): run actual WiSE-FT
weight interpolation, plain logit-average, and a **matched-compute KL-regularized / replay-tuning**
comparison — so "compose beats more tuning" is *measured*, not asserted. Use a **fixed** averaging
rule, **not a learned router** (avoids the leakage/bypass surface).

---

## 4. Analysis protocol (the discipline that got Paper A to PASS)

- **Metrics:** benchmark-macro AP (primary, threshold-free) **and** recall at a matched FPR
  (threshold from held-out/calibration negatives) as the deployable metric — and **disclose the
  OOD calibration failure** rather than reporting only the flattering threshold-free number.
- **Combiners:** calibrated-prob average (primary), raw-prob average and PIT/rank average
  (calibration-robustness rows), max, convex-weight (learned with nested CV), WiSE-FT, logit-avg.
  **The combiner is selected by a rule that never touches `transfer_test`** (fit on `calibration`
  only) — otherwise it's an implicit peek at the test axis.
- **Uncertainty:** the same **family × seed hierarchical bootstrap** as Paper A, reported as
  **per-model and per-benchmark paired CIs** (not a 4-point panel mean), plus
  leave-one-benchmark-out. State plainly that **`base` is single-pass (seed −1)**, so
  base-vs-ensemble CIs rest on adapter-side seed variance only.
- **Null control:** within-source label shuffle (prototype collapsed to 0.712) as positive
  evidence the effect is composition, not averaging-variance-reduction.
- **Leak checks:** confirm `probability_calibrated` was fit only on the `calibration` split,
  never on transfer rows (the averaging step makes any calibration leak matter).

Most of this reuses [`legacy/experiments/ensemble_deployable.py`](../legacy/experiments/ensemble_deployable.py)
(leak-free PIT averaging + CIs) and `ensemble_probe.py` (convex-weight sweep, oracle bound).

---

## 5. Provenance gating (non-negotiable)

The prototype numbers ride on the **same legacy, estimation-only scores** as un-rerun Paper A,
so they are **no more confirmatory than Paper A itself**. Therefore:

- **Sequence Paper B immediately behind Paper A's clean SFT rerun.** That rerun emits the
  base+SFT **per-row scores** the ensemble consumes — so Paper B needs **zero extra training**.
- **Regenerate every ensemble number from the locked, row-keyed, hashed v2 artifacts.** Treat all
  numbers in §1 as *preliminary* until then.

---

## 6. Feasibility & minimal-viable scope

**Do Axis 3 first, as pure re-analysis of the locked parquet — zero new training.** Fit
calibration on `calibration` only; pick the combiner by a transfer-blind rule; report per-seed
paired deltas with hierarchical-bootstrap CIs + LOO across the 4 models; add WiSE-FT + logit-avg
+ shuffle-null baselines. **Days of work**, reusing existing code, gated only on Paper A's clean
rerun landing. This alone is a complete, defensible Paper B.

**Effort/compute:** CPU-only analysis (the ensemble is post-hoc over existing scores). The only
GPU dependency is Paper A's own already-scoped clean SFT rerun. No annotation. No new datasets.

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

1. **Overstated headline ("beats both").** False per-model. → Reframe to "Pareto-improves over
   SFT; recovers transfer toward base; gated by base competence"; report per-model/per-benchmark
   CIs and disclose the Qwen3-4B loss.
2. **Numbers ride on un-rerun Paper A scores.** → Sequence behind Paper A's clean rerun;
   regenerate from locked artifacts; confirm no calibration leak.
3. **Novelty collision with WiSE-FT / soups.** → Run WiSE-FT + logit-avg + matched-compute
   KL/replay baselines; position as output-space + base-competence-conditioned; include the
   shuffle-null. Fixed rule, not a learned router.

---

## 9. What to do first (concrete)

1. **Land Paper A's clean SFT rerun** (its own top recommendation) → locked base+SFT per-row scores.
2. **Build the composition analyzer** (extend `analyze_paper_a_sft.py` or adapt
   `ensemble_deployable.py`): calibrated-avg / raw-avg / PIT / max / convex / WiSE-FT / logit-avg,
   transfer-blind combiner selection, family×seed bootstrap with per-model + per-benchmark CIs +
   LOO, shuffle-null, matched-FPR recall.
3. **Regenerate §1's numbers** from the locked artifacts; write RQ1–RQ3 with the honest per-model
   heterogeneity foregrounded.
4. **Only then** consider the Axis-2 appendix (if a hardened synthetic set shows headroom) and,
   if GPU returns, the Axis-1 Paper C.

---

*Bottom line: the composition idea is genuinely strong and empirically supported — it makes the
work better by turning "fine-tuning specializes" into "so compose instead of tuning harder,"
tied to the base-competence law. Ship it as one focused, locked-artifact paper. Keep objectives
and the domain out of the headline (Paper C / appendix) so this doesn't relapse into the sprawl
the Paper A refactor fixed.*
