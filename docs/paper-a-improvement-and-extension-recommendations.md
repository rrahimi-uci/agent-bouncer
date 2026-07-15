# Paper A — Improvement & Extension Recommendations

Reviewer recommendations for *"The Benchmark Chooses the Winner"*
([`../paper-a/benchmark_chooses_the_winner.tex`](../paper-a/benchmark_chooses_the_winner.tex)),
grounded in the clean-v2 retrospective results under
[`../artifacts/paper_a_sft_v2/analysis/`](../artifacts/paper_a_sft_v2/analysis).
Written 2026-07-13; execution status and numbers updated 2026-07-14.

This is a strong, honestly-hedged measurement paper. The recommendations below are
ordered by leverage: what most increases the paper's credibility and contribution per
unit of effort. Nothing here reopens the broad-study sprawl the refactor deliberately
removed — the scope discipline is a feature; keep it.

---

## 1. What the current numbers actually say

Read straight off `results.json` / `sensitivity.json`:

- **Represented gain is largely a ceiling effect.** SFT drives *every* checkpoint to
  ~0.98 represented macro-AP regardless of where its base started (0.452–0.886). So the
  headline "+0.323" represented gain is mostly headroom: `Δ_rep ≈ 0.98 − base_rep`. True
  but nearly mechanical — it should be framed as *saturation*, not *improvement*.
- **The interesting effect is on transfer, with checkpoint heterogeneity.**
  SFT compresses transfer AP toward a common ~0.79–0.84 band: the range across
  checkpoints shrinks from **0.158 (base) to 0.052 (SFT), ~3×**. Concretely SmolLM2
  changes by +0.040, Qwen2.5's interval crosses zero, and SmolLM3/Qwen3-4B change by
  −0.087/−0.150. The aggregate −0.059 hides this. Because `Δ = SFT − base`, a same-row
  regression of change on base AP is mathematically coupled; four purposively chosen
  checkpoints cannot establish a base-competence law.
- **Transfer loss is concentrated in jailbreak-style OOD.** Per-benchmark Δ: WildJailbreak
  −0.079, JailbreakBench −0.078, WildGuardTest −0.067, XSTest −0.012 (≈flat). The
  over-refusal contrast set is barely touched; the loss is on adversarial jailbreaks.
- **A concrete safety cost is buried in the stress diagnostics.** HarmBench recall falls
  **78.0% → 60.0% (−18.0pp)** after SFT — the guard newly misses substantially more harmful
  prompts on a held-out harm set. This is arguably the most policy-relevant single number
  in the paper and currently gets one line.
- **Deployment FPR inflates.** Transfer pooled-negative FPR rises **4.3% → 17.0%**
  (macro 8.1% → 15.5%) at calibration-targeted thresholds.
- **Status: clean-v2 execution, retrospective estimation.** `analysis_mode =
  precision_focused`; the lock, manifests, 20 adapters, 24 score bundles, and analysis
  now satisfy the v2 contract. Prior cohort exposure still blocks a prospective or
  confirmatory interpretation.

---

## 2. Improvements — strengthen the claim you already have

### P0 — Execute the clean v2 rerun — completed 2026-07-14

The full `manifests → audit → lock → train → eval → analyze` pipeline has now run under
`artifacts/paper_a_sft_v2/`. It:

- repairs the documented family-link defects (36 JailbreakBench + 58 XSTest unjoined
  pairs; 2 represented families crossing calibration/ID);
- fixes the truncation bug (26 eval prompts + ~16–17 train rows/checkpoint lost the
  system instruction under left-truncation);
- replaces 605 of the 1,220 archived WildGuard/WildJailbreak rows under the locked
  hash-ranked construction, while retaining and disclosing 615 previously inspected rows;
- produces clean, reproducible execution evidence while correctly retaining the
  `precision_focused` retrospective analysis mode.

**Executed compute:** the five-run medians were 8.7, 7.4, 11.3, and 12.6 minutes for
Qwen2.5, SmolLM2, SmolLM3, and Qwen3 respectively on one A100 40GB. This closes the
execution-provenance blocker, not the prospective-cohort blocker.

### P1 — Add at least one genuinely uninspected transfer benchmark

The biggest internal-validity threat is stated in Limitations: the transfer sets were
inspected during development (HPO scored the cache), and even the rebuild retains 615
previously-inspected rows. A confirmatory transfer claim needs a benchmark that was
*never* looked at. Two low-cost options:

- **Run the ExpGuard replication the appendix already promises** (finance/health/law,
  expert-labeled, differently sourced). This simultaneously fixes the "uninspected" gap
  and tests external recurrence of the pattern.
- Pull one fresh public guard benchmark at a pinned revision and score it *once*, blind.

### P1 — Report a low-prevalence / calibration-target operating-point view

Limitations concede balanced pools overstate production precision. Add, at realistic
unsafe prevalence (e.g. 1–5%): recall at a fixed pooled-FPR budget, and precision/AP under
re-weighting. This converts the operating-point section from "diagnostic" to a genuinely
useful deployment statement and makes the FPR-inflation finding land harder.

### P2 — Confirmatory design (only after a new prospective cohort is locked)

Do not convert the current clean-v2 retrospective run into confirmatory evidence by adding
post-hoc tests. First lock a justified null, estimand, power/precision plan, seed count, and
genuinely uninspected cohort. A null-calibrated test may then be primary; mixed-effects or
hierarchical models are optional only if their population and sampling assumptions match
the design. Keep the fixed-panel paired estimate when inference remains conditional on the
named checkpoints and benchmarks.

---

## 3. Extensions — new contributions

### E1 (flagship hypothesis) — Does independently measured base performance predict the effect?

The current data (§1) show **SFT compressing guards toward a narrower transfer band** while
saturating everyone on represented sources. The four same-row contrasts motivate a hypothesis,
but their apparent relationship with base AP is mathematically coupled. Test the hypothesis
without building that coupling into the design:

- Pre-specify base competence on an independent locked development cohort, then model its
  interaction with treatment against `Δ_transfer` on a disjoint prospective cohort. Do not
  regress `Δ_transfer` on `base_transfer_AP` from the same rows.
- **Extend the panel to ~10–15 checkpoints** across sizes (0.5B–8B) and families
  (add e.g. Llama-3.2, Gemma-2, Phi, Qwen3 sizes) and report a family-aware interaction
  estimate with a slope and interval; a larger panel alone does not cure same-row coupling.
- Treat any **decision rule**—for example, fine-tune weak bases but retain or compose a
  strong base for transfer screening—as a hypothesis to validate, not a rule established
  by the current four-model panel.

This is a high-value extension if the independently measured relationship survives: it reuses
the evaluation machinery, but needs a new locked development/outcome split as well as more
checkpoints. Only then could the contribution move from "heterogeneity appears in four named
models" to "specialization is predictable from a pre-transfer competence measure."

### E2 — Training dose–response frontier

Vary training volume (e.g. 100 / 300 / 1,200 / 4,000 rows) and number of represented
sources (1/2/3), holding everything else fixed. Trace the represented-vs-transfer frontier
as a function of data. Tests whether the trade-off is a small-data artifact or widens with
scale. Cheap (retraining only), adds one strong frontier figure.

### E3 — Mechanism: is the transfer loss recoverable or structural?

Macro-AP is calibration-invariant, so the transfer loss is genuine *ranking* degradation,
not miscalibration — state and verify this (re-fit temperature on transfer; AP should not
move — a clean sanity check). Then probe the cause: does SFT increase reliance on
lexical/surface cues shared with the training sources? Test with simple ablations
(e.g. token-overlap-stratified transfer AP) or representation similarity (CKA between base
and SFT hidden states on represented vs transfer prompts). Even a partial answer to *why*
elevates the paper above pure measurement.

### E4 — HarmBench recall-decline deep-dive

The −18.0pp HarmBench recall drop deserves its own short analysis: which behavior
categories are newly missed post-SFT, and does the miss rate scale with distance from the
training sources? This is a safety-relevant, self-contained result (needs a per-category
rescore of HarmBench, available once the rerun exists).

### E5 — A mitigation *measurement* (frame carefully)

A small replay ablation — mix k% held-out-family general rows into training and measure
whether transfer loss shrinks while represented gain holds — makes the practical takeaway
constructive. **Frame it as measuring the trade-off's tunability, not as a new method:**
KL-preservation and calibrated ensembling are established prior art (see the related-work
boundary), so claim measurement, not novelty.

---

## 4. Scope discipline — what NOT to fold back in

The focused refactor removed the objective comparison (DPO/GRPO), ensembling, GPT parity,
fairness, and the mortgage case study for good reason. Keep them out of Paper A:

- **SFT-vs-DPO-vs-GRPO** is a *separate* paper (the objective axis), not a Paper A
  extension — the manuscript explicitly scopes it out. Do not re-merge.
- **Mortgage and fairness** remain separate follow-on lines; output-space composition is
  governed by the retrospective Paper B prototype plan
  ([`paper-b-compose-dont-tune-plan.md`](paper-b-compose-dont-tune-plan.md)).
- Adding checkpoints (E1) and datasets (P1) is *deepening the same claim*; adding new
  axes (objectives, domains, architectures-as-contributions) is *sprawl*. Prefer depth.

---

## 5. Suggested sequencing

1. **P0 clean rerun — complete** → fixes execution and known preprocessing/provenance defects.
2. **P1 uninspected benchmark (ExpGuard) + low-prevalence view** → addresses the two biggest
   validity gaps. (days)
3. **E1 prospectively locked panel expansion** → tests the base-performance interaction
   hypothesis with adequate breadth. (1–2 weeks of mostly-unattended training)
4. **E3/E4 mechanism + HarmBench** → depth, if aiming for a stronger venue.
5. **E2/E5** → optional frontier/mitigation if space and time allow.

**Minimum next upgrade:** P1. **Strongest realistic paper:** completed P0 + a prospectively
locked P1/E1 design, with P2 only if its inferential assumptions are prespecified and met.
E3 is separately required before making a mechanism claim, and an actionable decision rule
must be validated rather than inferred from the current four-model panel.
