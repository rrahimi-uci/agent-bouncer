# Paper A — Improvement & Extension Recommendations

Reviewer recommendations for *"The Benchmark Chooses the Winner"*
([`../paper-a/benchmark_chooses_the_winner.tex`](../paper-a/benchmark_chooses_the_winner.tex)),
grounded in the current committed results under
[`../artifacts/paper_a_sft/analysis/`](../artifacts/paper_a_sft/analysis). Written 2026-07-13.

This is a strong, honestly-hedged measurement paper. The recommendations below are
ordered by leverage: what most increases the paper's credibility and contribution per
unit of effort. Nothing here reopens the broad-study sprawl the refactor deliberately
removed — the scope discipline is a feature; keep it.

---

## 1. What the current numbers actually say

Read straight off `results.json` / `sensitivity.json`:

- **Represented gain is largely a ceiling effect.** SFT drives *every* checkpoint to
  ~0.98 represented macro-AP regardless of where its base started (0.447–0.878). So the
  headline "+0.333" represented gain is mostly headroom: `Δ_rep ≈ 0.98 − base_rep`. True
  but nearly mechanical — it should be framed as *saturation*, not *improvement*.
- **The interesting effect is on transfer, and it is a base-competence interaction.**
  SFT compresses transfer AP toward a common ~0.79–0.84 band: the range across
  checkpoints shrinks from **0.158 (base) to 0.052 (SFT), ~3×**. Concretely it **helps
  the weakest base** (SmolLM2 +0.051) and **hurts the strongest** (SmolLM3 −0.120,
  Qwen3-4B −0.102). The aggregate −0.050 hides this.
- **Transfer loss is concentrated in jailbreak-style OOD.** Per-benchmark Δ: WildJailbreak
  −0.079, JailbreakBench −0.073, WildGuardTest −0.039, XSTest −0.010 (≈flat). The
  over-refusal contrast set is barely touched; the loss is on adversarial jailbreaks.
- **A concrete safety cost is buried in the stress diagnostics.** HarmBench recall falls
  **78.4% → 57.5% (−20.9pp)** after SFT — the guard newly *misses* one in five harmful
  prompts on a held-out harm set. This is arguably the most policy-relevant single number
  in the paper and currently gets one line.
- **Deployment FPR inflates.** Transfer pooled-negative FPR rises **4.4% → 14.6%**
  (macro 8.3% → 13.7%) at calibration-targeted thresholds.
- **Status: estimation-only on a legacy artifact.** `analysis_mode = precision_focused`,
  `score_artifact.legacy = true`. The paper itself names a clean rerun as the gate to
  confirmatory status.

---

## 2. Improvements — strengthen the claim you already have

### P0 — Execute the clean v2 rerun (the paper's own blocker)

The manuscript repeatedly defers to "a clean rerun is required." The v2 pipeline already
exists in the `Makefile` (`manifests → audit → lock → train → eval → analyze` into
`artifacts/paper_a_sft_v2/`). Running it:

- repairs the documented family-link defects (36 JailbreakBench + 58 XSTest unjoined
  pairs; 2 represented families crossing calibration/ID);
- fixes the truncation bug (26 eval prompts + ~16–17 train rows/checkpoint lost the
  system instruction under left-truncation);
- replaces the partially-inspected seed-7 WildGuard/WildJailbreak cohorts;
- unlocks `powered_confirmatory` analysis (intersection–union test, Holm, bootstrap
  *p*-value) instead of descriptive-only wording.

**Compute reality:** legacy runs were 6–10 min each on A100; the full 4×5 panel is ~5–10
A100-hours, or feasible (slowly) on the M4 Max (per project notes SmolLM3-3B converges by
~step 40; the smaller three are faster). This is the single highest-leverage action and is
cheap. **Do this first;** most items below compose with it.

### P1 — Add at least one genuinely uninspected transfer benchmark

The biggest internal-validity threat is stated in Limitations: the transfer sets were
inspected during development (HPO scored the cache), and even the rebuild retains 615
previously-inspected rows. A confirmatory transfer claim needs a benchmark that was
*never* looked at. Two low-cost options:

- **Run the ExpGuard replication the appendix already promises** (finance/health/law,
  expert-labeled, differently sourced). This simultaneously fixes the "uninspected" gap
  and tests external recurrence of the pattern.
- Pull one fresh public guard benchmark at a pinned revision and score it *once*, blind.

### P1 — Report a low-prevalence / matched-FPR deployment view

Limitations concede balanced pools overstate production precision. Add, at realistic
unsafe prevalence (e.g. 1–5%): recall at a fixed pooled-FPR budget, and precision/AP under
re-weighting. This converts the operating-point section from "diagnostic" to a genuinely
useful deployment statement and makes the FPR-inflation finding land harder.

### P2 — Confirmatory statistics (after P0)

With the clean lock: the intersection–union claim gate with Holm control, a bootstrap
*p*-value, and a **mixed-effects model** (checkpoint fixed panel, seed nested, benchmark
random) to produce a principled aggregate with proper uncertainty — replacing the
panel-mean + hierarchical bootstrap as the *primary* (keep the bootstrap as sensitivity).

---

## 3. Extensions — new contributions

### E1 (flagship) — The base-competence law: *who* fine-tuning helps vs hurts

The strongest latent result in the current data (§1) is that **SFT compresses guards
toward a common specialized regime**: it lifts weak bases and degrades strong ones on
transfer, and saturates everyone on represented sources. Turn this from a buried
heterogeneity note into the paper's second headline:

- Formalize: regress `Δ_transfer` on `base_transfer_AP` (n=4 today shows a clean negative
  relationship; the weakest base is the *only* transfer gainer).
- **Extend the panel to ~10–15 checkpoints** across sizes (0.5B–8B) and families
  (add e.g. Llama-3.2, Gemma-2, Phi, Qwen3 sizes) to power the relationship into a
  quantitative claim with a real slope and CI.
- Deliver an **actionable decision rule**: *fine-tune weak bases; keep strong bases
  zero-shot for OOD screening.* That is a practitioner takeaway no current guard paper
  states, and it is directly supported by your data.

This is the highest-value extension: it reuses all existing machinery, needs only more
checkpoints (cheap), and upgrades the contribution from "specialization happens on 4
models" to "specialization is predictable from base competence."

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

### E4 — HarmBench recall-collapse deep-dive

The −20.9pp HarmBench recall drop deserves its own short analysis: which behavior
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
- **Mortgage / fairness / ensembling** belong to the Paper B line (see
  [`paper-b-feasibility-investigation.md`](paper-b-feasibility-investigation.md)).
- Adding checkpoints (E1) and datasets (P1) is *deepening the same claim*; adding new
  axes (objectives, domains, architectures-as-contributions) is *sprawl*. Prefer depth.

---

## 5. Suggested sequencing

1. **P0 clean rerun** → unlocks confirmatory status and fixes the known defects. (days)
2. **P1 uninspected benchmark (ExpGuard) + low-prevalence view** → closes the two biggest
   validity gaps. (days)
3. **E1 panel expansion + base-competence law** → the flagship upgrade; also supplies the
   confirmatory power P2 wants. (1–2 weeks of mostly-unattended training)
4. **E3/E4 mechanism + HarmBench** → depth, if aiming for a stronger venue.
5. **E2/E5** → optional frontier/mitigation if space and time allow.

**Minimum credible upgrade:** P0 + P1. **Strongest realistic paper:** P0 + P1 + E1 (+ P2),
which converts an honest but panel-limited estimation study into a confirmatory,
mechanism-bearing result with a practitioner decision rule — without leaving the scope the
refactor established.
