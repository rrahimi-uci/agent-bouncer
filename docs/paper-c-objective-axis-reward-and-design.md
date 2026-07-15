# Paper C — Objective Axis: Reward Design, Objective Selection, and Grounding

> **Status: grounded design layer for the objective axis (Paper C).** This is the *how*
> and *why-it-is-defensible* companion to
> [`paper-b-topic-proposal.md`](paper-b-topic-proposal.md), which already fixes the research
> questions, the surviving-asset inventory, and the run-plan. This doc adds the three things
> that proposal deferred and that the reader asked for: (1) **what reward / preference signal
> actually works** for a single-token binary guard, (2) **which objectives to select and
> which to drop**, and (3) a **literature review** that grounds the plan's **feasibility and
> novelty**. It reuses Paper A's clean v2 lock/manifest so SFT/DPO/…/GRPO are directly
> comparable, and it ties the objective axis to the composition remedy of
> [`paper-b-compose-dont-tune-plan.md`](paper-b-compose-dont-tune-plan.md).

Written 2026-07-14. Grounded in a five-facet literature review (see §9 for sources and their
verification status). **Citations are tiered**: some arXiv IDs were fetched/verified this
session, some are recalled and should be reconfirmed before they appear in the manuscript,
and a few very recent (2026) preprints are flagged as *claim-only, do not cite the ID*.

---

## 0. TL;DR — the decisions

1. **No learned reward model. Anywhere.** Our labels are verifiable ground truth. Every
   canonical RL-with-verifiable-rewards recipe (DeepSeek-R1, Tulu 3) deliberately avoids a
   neural reward model because it only adds reward-hacking surface. The "reward model" for a
   safety guard **is the gold label**.
2. **Objectives to run:** **SFT · DPO · KTO · GRPO** as the core four, **+ ORPO** strongly
   recommended as a fifth. **Drop** SimPO, PPO, BCO, and treat IPO/RLOO as optional
   robustness cross-checks. Rationale is mechanistic (§4), not taste.
3. **The organizing mechanism is reference-KL anchoring** ("RL's Razor"): how far an
   objective moves the tuned model from its base *predicts* both its transfer loss and how
   much the base+tuned composition remedy can recover. This makes the objective axis and the
   composition paper one story, and it is the genuinely novel question (§5, §7).
4. **GRPO will probably underperform on a single token — say so up front.** The
   "RL generalizes, SFT memorizes" result is established only for *multi-token generative*
   tasks. A one-token, two-outcome verdict removes the exploration/reasoning degrees of
   freedom that make RL win, and makes GRPO's group-relative advantage degenerate. We
   **pre-register** GRPO's likely null; a bounded negative result is a contribution, not a
   failure (§5, §8).
5. **Feasibility is high and mostly done:** TRL has stable trainers for all core objectives;
   1-token rollouts make GRPO cheap; the v2 lock/manifest/analyzer are reused; the only real
   new code is a frozen v2 preference/reward recipe and an `objective` dimension in the
   scores + composition analyzer (§6).

---

## 1. Scope and how this fits the program

The topic proposal frames the axis as *"which fine-tuning objective specializes least,"* with
the surviving legacy directional table (SFT > DPO > GRPO on how much they move transfer). That
directional pattern is real signal but **not publishable** — single-seed, pre-decontamination,
old pooled metric, no per-row scores. This doc does not re-argue that; it answers the design
questions needed to turn it into a clean study:

- **§3 Reward design** — the crux the proposal left open ("DPO/GRPO need a preference/reward
  construction over the same rows — freeze it as a v2 recipe").
- **§4 Objective selection** — upgrades the proposal's fixed {SFT, DPO, GRPO} to a
  mechanistically-justified menu, adding KTO (the most natural fit for our labels) and ORPO
  (the reference-free endpoint that probes the composition interaction).
- **§5 Mechanism + hypotheses** — connects the axis to the composition remedy via KL-anchoring.
- **§6–8 Feasibility, novelty, matrix, risks** — grounded in the literature review.

**One campaign, two/three papers.** Per the proposal's synergy argument, the SFT slice of this
matrix *is* Paper A's own recommended clean rerun; the composed-guard evaluation (§5, H2) is an
eval-time operation over the same scores, so it also feeds the composition paper. This axis does
not require a separate training campaign beyond adding the DPO/KTO/GRPO/ORPO conditions.

---

## 2. What the literature actually says (grounded review)

Five facets were searched and cross-checked. The honest headline: **no located paper studies
SFT vs DPO vs GRPO on a binary, single-token safety guard measured as represented-vs-transfer,
let alone its interaction with an output-space composition remedy.** The neighbors each miss
≥2 of our axes. That gap is the opportunity; the caveats below keep us from overclaiming.

### 2.1 "RL generalizes, SFT memorizes" — true, but only for generative reasoning
- **Chu et al. 2025** (*SFT Memorizes, RL Generalizes*, arXiv:2501.17161, ICML 2025): RL with an
  outcome reward generalizes OOD where SFT collapses — but on **multi-step generative reasoning**
  (an arithmetic card game, V-IRL navigation) with an **11B vision-language** backbone. OOD deltas
  are large (e.g. V-IRL text RL +11.0 vs SFT −79.5). Crucially, the benefit is credited to
  *outcome-reward exploration over multi-token reasoning/recognition traces*, and **SFT is still
  required first** to stabilize output format.
- **Kirk et al. 2024** (arXiv:2310.06452, ICLR 2024): RLHF generalizes to new inputs better than
  SFT, and the advantage **grows with distribution shift** — but on **summarization and
  instruction-following**, and bundled with a *diversity* collapse. Their own analysis notes RL
  "fails to explore beyond the SFT model's solution set," so the gain is modest, not a regime change.
- **Neither paper attributes generalization to a KL term.** In RLHF, the reference-KL is more
  naturally tied to the diversity/mode-collapse side.

> **Implication for us.** Both results live in the multi-token generative regime. Our verdict is
> **one token, two outcomes**, so the exploration/reasoning degrees of freedom that make RL win
> are *absent*. Predict the objective moves our specialization/transfer trade-off **less** than
> these papers suggest, and pre-register that GRPO's transfer edge over SFT will be small. This
> is the correct, defensible reading — not "RL will help us because it helped them."

### 2.2 Reward design — verifiable label, no reward model
- **GRPO** (DeepSeekMath, arXiv:2402.03300): critic-free PPO variant; per prompt, sample a group
  of *G* completions, advantage = reward normalized by group mean/std, plus a KL-to-reference term
  (β≈0.04 in the paper).
- **RLVR** (DeepSeek-R1, arXiv:2501.12948; Tulu 3, arXiv:2411.15124): for verifiable tasks the
  standard reward is a **deterministic 0/1 correctness check**; both papers **explicitly avoid a
  neural reward model** to prevent reward hacking. Minimal reward = fewer degrees of freedom to hack.
- **Dr. GRPO** (*Understanding R1-Zero-Like Training*, arXiv:2503.20783) shows GRPO's
  std-normalization introduces a difficulty bias; removing it recovers an unbiased REINFORCE-style
  objective and is essentially RLOO up to a constant.

> **Implication for us.** Use the label as the reward. GRPO here reduces to **group-baselined
> REINFORCE on a two-action contextual bandit** with a KL leash — see §3 for the single-token
> degeneracy this creates and the graded-reward fix. Prefer Dr.GRPO/RLOO-style (unnormalized)
> advantages given only two reward values.

### 2.3 The objective menu splits three ways
- **Pointwise binary → KTO** (arXiv:2402.01306) and its sibling **BCO** (arXiv:2404.04656): trains
  on a per-example desirable/undesirable label — *exactly our data* — with built-in class-imbalance
  weights. Still reference/β-anchored (β=0.1 default in TRL; reference defaults to the initial policy).
- **Pairwise → DPO** (arXiv:2305.18290), **IPO** (arXiv:2310.12036), **ORPO** (arXiv:2403.07691),
  **SimPO** (arXiv:2405.14734): need a (chosen, rejected) pair, which we synthesize losslessly from
  the label. DPO/IPO are reference-anchored; **ORPO and SimPO are reference-free**. IPO's bounded
  loss matters because our synthetic pairs are near-deterministic (DPO's overfitting regime).
  **SimPO's whole method is length-normalization, which is moot at length 1** → it reduces to a plain
  one-token margin loss and loses its distinguishing property.
- **Reward → GRPO / RLOO / PPO** (RLOO: *Back to Basics*, arXiv:2402.14740): turn the label into a
  reward. Two-outcome single-token action space makes group/leave-one-out advantages degenerate.
- **TRL maturity** (docs snapshot ~v1.8.0, reconfirm against the repo's pinned version): stable/
  first-class = SFT, DPO (incl. `loss_type='ipo'`), KTO, GRPO (vLLM), RLOO (vLLM); experimental =
  PPO, ORPO, CPO (hosts SimPO via `loss_type='simpo'`), BCO.

### 2.4 Prior art on RL/preference-tuned guards
- **The field standard is SFT.** Llama Guard (arXiv:2312.06674), WildGuard (arXiv:2406.18495),
  ShieldGemma (arXiv:2407.21772) are all supervised instruction fine-tuning of a generative LLM that
  emits a safe/unsafe verdict. Our SFT baseline **is** the field recipe; our single verdict token is a
  thin variant.
- **DuoGuard** (arXiv:2502.05163) uses RL, but as a two-player minimax for **synthetic-data
  generation**; the guard itself stays a classifier. "RL guardrail" exists, in a *different* sense.
- **GSPR** (arXiv:2509.24418) is the closest: GRPO on a safeguard after an SFT cold-start, measuring
  OOD across benchmarks — but it is **reasoning/CoT-based, multi-class (up to 167 labels), 7–8B, GRPO-only**
  (no SFT-vs-DPO-vs-GRPO head-to-head), and has **no composition remedy**.
- **"Objective Matters"** (arXiv:2601.12639, *reconfirm*) compares six objectives on **general**
  LLM safety and finds constrained objectives (ORPO, KL-reg) reduce adversarial vulnerability at
  scale — validates our "objective matters" framing but not for a guard/classifier.

### 2.5 KL-anchoring ↔ forgetting ↔ composition (the mechanism)
- **RL's Razor** (arXiv:2509.04259): online RL forgets prior/OOD capabilities *less* than SFT for
  the same task gain, and the amount of forgetting is **predicted by the forward KL between tuned and
  base on the new task**. Important caveat the same work reports: the driver is **on-policy sampling
  keeping updates near the base, not the explicit KL penalty term** (non-regularized GRPO forgot about
  as little as KL-regularized GRPO).
- **Forgetting in SFT→preference pipelines** (arXiv:2410.15483): a DPO/RLHF KL anchor **does not**
  prevent forgetting when applied as a second stage after SFT (data-distribution shift dominates);
  the fix is joint SFT+preference optimization.
- **WiSE-FT** (arXiv:2109.01903) and **model soups** (arXiv:2203.05482): interpolating base and
  fine-tuned models recovers OOD robustness while keeping ID accuracy — the canonical "combine
  base+tuned to recover OOD" result, but for **CLIP/vision weight-space**.
- **Output-space composition** (overadaptation-ensemble theory, arXiv:2506.01901; GLA,
  arXiv:2310.08106): averaging base and tuned *outputs* recovers information lost to overadaptation;
  helps more as fine-tuning difficulty/movement grows. Caveat: theory assumes near-base tuning and
  measures upstream retention, not held-out domain shift.

> **Implication for us.** This is the spine of the paper: **objective → how far it moves from base
> → transfer loss → how much composition recovers.** Our composition remedy (averaging base and
> tuned calibrated probabilities) is an *output-space* WiSE-FT/soup analog that matches our
> one-forward-pass scoring, and doing it **as a function of the training objective** is unexplored.

---

## 3. The reward-model question (answer + per-objective construction)

**Answer: there is no reward model to design.** A safety guard has verifiable ground-truth labels,
so the reward *is* the label. Introducing a learned RM would (a) be unnecessary and (b) add the exact
reward-hacking surface DeepSeek-R1/Tulu-3 avoid. Evaluation stays identical across every arm — the
score is always `z_unsafe − z_safe` from one forward pass — so the arms are compared on the *same*
metric. (Action item: verify each trainer's verdict/reward extraction still reads the two verdict-token
logits and not a sampled string.)

| Objective | Signal built from the binary label | Reference model? | Key knob |
|---|---|---|---|
| **SFT** | cross-entropy on the correct verdict token (`" safe"`/`" unsafe"`) | no | lr, steps |
| **DPO** | pair: `chosen`=correct token, `rejected`=wrong token | yes (frozen base) | β (KL leash) |
| **IPO** | same pair; bounded squared-error loss (safer for near-deterministic pairs) | yes | β, `loss_type='ipo'` |
| **KTO** | pointwise `{prompt, completion=correct token, label=True}` (optionally add wrong token `label=False`) | yes (initial policy) | β; `desirable/undesirable_weight` |
| **ORPO** | same pair; odds-ratio penalty on the SFT/NLL loss | **no** (single-stage) | λ (OR weight) |
| **GRPO** | reward = correctness of the sampled verdict vs label | yes (KL, or reference-free à la DAPO) | β, group size *G*, reward shape |

### 3.1 The one real reward-design problem: GRPO on a single token
A one-token, two-outcome action space breaks GRPO's group-relative advantage. Once the guard is
confident, all *G* sampled verdicts agree → group std → 0 → advantage vanishes → **gradient
starvation** (that prompt contributes no gradient). Mitigations, in priority order:

1. **Graded reward instead of 0/1 — the single most important GRPO choice.** Reward the *margin*:
   use the calibrated `P(unsafe)` (from `z_unsafe − z_safe`) as a continuous reward, or a
   hinge/margin on the logit difference toward the correct side. This restores within-group spread
   that raw 0/1 destroys, and it uses the score the guard is actually evaluated on.
2. **Dr.GRPO / RLOO-style advantages** — drop group-std normalization; with two reward values,
   std-normalization is pure difficulty bias.
3. **Dynamic sampling / zero-variance-group filtering** (DAPO-style) so agreed-verdict prompts
   don't add noise; raise rollout temperature for diversity. *G* > 1 is cheap (1-token rollouts).
4. **Guard against majority-class collapse** — the only live reward-hacking mode once length hacking
   is impossible (our output is one token). If the 1,200-row manifest is class-skewed, use a
   class/severity-weighted correctness reward. (Check the manifest balance; the v2 audit already
   asserts label balance, so skew should be small — confirm the number.)

> Even with these fixes, expect GRPO to gain little here: with no reasoning trace to diversify (unlike
> GSPR), the graded reward mostly recovers the *same* signal SFT/KTO already use. That is the
> pre-registered prediction, not a bug.

---

## 4. Objective selection — what to run, what to drop, why

**Run these five.** Each earns its place by anchoring a distinct point on the KL/reference axis:

| Arm | Class | Reference/KL anchor | Role on the axis |
|---|---|---|---|
| **SFT** | supervised | none | field-standard baseline; the "moves-most, β-free" endpoint |
| **KTO** | pointwise binary | yes (β leash) | most *natural* fit for our labels; anchored |
| **DPO** | pairwise | yes (β leash) | canonical reference-anchored objective; label→pair lossless |
| **ORPO** | pairwise | **none** (reference-free) | the "specializes hardest, recovers most via composition" endpoint; also lighter |
| **GRPO** | reward (RLVR) | yes / optional | the headline "RL generalizes?" test; pre-registered likely-null |

**Optional cross-checks (only if compute allows):**
- **IPO** — DPO with a bounded loss; include *iff* DPO shows the near-deterministic-pair overfitting
  the theory predicts. Robustness check, not a primary arm.
- **RLOO** — GRPO cross-check (Dr.GRPO ≈ RLOO); include only to show GRPO's result isn't an artifact
  of std-normalization.

**Drop, with reasons:**
- **SimPO** — its length-normalization *is* the method and is **moot at length 1**; reduces to a
  plain one-token margin loss (uninteresting, near-indistinguishable from a margin DPO).
- **PPO** — experimental in TRL and strictly dominated by GRPO/RLOO for this task.
- **BCO** — a KTO sibling; redundant with KTO for a first study.

This upgrades the topic proposal's fixed {SFT, DPO, GRPO} in two deliberate ways: **add KTO** (the
objective whose input format matches our labels one-to-one, and whose asymmetric weights address any
class imbalance) and **add ORPO** (the only reference-free arm — scientifically load-bearing because it
removes the β leash and thus anchors the opposite end of the mechanism in §5).

---

## 5. Mechanism and the two hypotheses

**Mechanism (grounded in §2.5): reference-KL / on-policy anchoring.** How far an objective moves the
tuned model from its base is the mediator variable. In our binary setting, measure "distance moved" as
the forward KL — or simply the shift in the calibrated `P(unsafe)` distribution — between tuned and
base **on the represented training rows**.

**H1 — objective orders the specialization/transfer trade-off by distance moved.**
Weakly/unanchored objectives move most → biggest represented gain, biggest transfer loss;
strongly-anchored objectives move least → smaller gain, smaller transfer loss. Predicted movement
order: **ORPO (reference-free) ≳ SFT > KTO ≈ DPO/IPO (β-anchored) > GRPO (KL-close / low-signal).**
*This is already directionally visible in the legacy table* (topic proposal §2): SFT buys the most
in-distribution gain and risks transfer; GRPO barely moves transfer (novel ≈ base); DPO sits between
— i.e. **SFT > DPO > GRPO** in movement, exactly what the KL-anchor mechanism predicts. The clean
rerun's job is to confirm this under the corrected metric with seed variance.

**H2 — objective × composition interaction (the genuinely new question).**
Output-space composition (averaging base and tuned calibrated probabilities — the same operation the
compose-don't-tune analyzer already computes) recovers transfer **in proportion to how far the
objective moved from base.** So composition should help **most for SFT/ORPO** (large move, much OOD
lost, much to recover) and **least for DPO/KTO/GRPO** (little lost, little to recover — and averaging
may even dilute their modest represented gain). Concrete prediction: **a monotone relation between
(base↔tuned KL) and (composition's transfer recovery)**, traceable by sweeping β as the mediator.

**Honest caveat to state in the paper.** RL's Razor attributes low forgetting to *on-policy sampling*,
not the KL penalty term; and arXiv:2410.15483 shows a static DPO KL anchor fails under stage-to-stage
data shift. So DPO's static reference anchor and on-policy GRPO may behave differently *even at matched
KL*. Report KL as the *predictor*, not the *cause*, and separate the anchored-static (DPO/KTO) from the
on-policy (GRPO) arms when interpreting.

---

## 6. Feasibility

**Trainers.** All core arms are first-class in TRL (SFT, DPO+IPO, KTO, GRPO with vLLM); ORPO is
usable but experimental. RLOO is available for the GRPO cross-check. Reconfirm the "experimental"
tags against the TRL version pinned in `requirements.txt` before freezing the recipe.

**Memory on a single L4 (24 GB).** LoRA r=32 on 1.5–8B fits comfortably. The reference-anchored arms
(DPO/IPO/KTO/GRPO-with-KL) hold a second frozen model — use `precompute_ref_log_probs` or a frozen
LoRA-base reference to stay in budget, especially on Qwen3-8B / DeepSeek-R1-1.5B. ORPO needs no second
model (lighter). *(MEMORY: 3B on MPS converges by ~step 40, cap ~300; the L4 is comparable in steps and
far faster.)*

**RL cost.** Verdict rollouts are **one token**, so GRPO's per-step cost is trivial; the cost is
steps-to-convergence, not rollout length. The dominant feasibility *risk* is GRPO gradient starvation
(many prompts contribute zero gradient once the guard is confident) — mitigated by the graded reward
of §3.1. There is **no length-based reward hacking** (one token). β-anchored arms risk *undertraining*
if β is too high (no movement) — so the β sweep doubles as a training-signal sanity check, and a
learning-curve check separates a genuine "transfer-preserving" objective from an undertrained one.

**Reuse of the existing machinery (what's already done vs. new).**
- *Reuse as-is:* the v2 lock/manifest/audit, the identity-keyed scorer emitting per-row logits, the
  canonical tie-aware macro-AP + represented/transfer split + paired hierarchical bootstrap
  (`analyze_paper_a_sft.py`), and the composition analyzer (`analyze_composition.py`).
- *New code (small, well-scoped):*
  1. **A frozen v2 preference/reward recipe** — a builder that turns each of the 1,200 locked rows
     into (a) the SFT target token, (b) the DPO/ORPO/IPO `(chosen, rejected)` verdict-token pair, and
     (c) the KTO `{completion, label}` record, plus the GRPO graded-reward function over the logit
     margin. Legacy `train_guard_pref.py` (DPO/IPO/KTO) shows the shape; freeze it, seed it, and bind
     it into the lock so the preference data is reproducible.
  2. **An `objective` dimension in the scores + analyzer.** The scorer already keys rows by
     `(model_key, condition, seed, split, source)`; today `condition ∈ {base, sft}`. Extend it to
     `{base, sft, dpo, kto, grpo, orpo}` and generalize `analyze_composition.build()` /
     `combiner_score()` so composition can pair the base with *any* tuned condition (not just `sft`).
     This is the single change that lets H2 be computed per objective with zero new training.
  3. **A KL/movement report** — per (base, objective, seed), the base↔tuned shift in calibrated
     `P(unsafe)` on represented rows, so H1/H2 can be plotted against the mediator.

---

## 7. Novelty — defensible, and how to frame it

**Genuinely new (no surveyed paper combines these):**
1. A **controlled head-to-head SFT vs DPO vs KTO vs GRPO (+ORPO)** ablation where the objective is
   applied to the **guard's own verdict** — not to data generation (DuoGuard) and not GRPO-only (GSPR).
2. On a **binary single-token** guard measured as **represented-vs-transfer macro-AP** with a
   decontaminated, locked protocol and seed variance.
3. The **objective × output-space-composition interaction** (H2) — *unexplored* in the surveyed
   literature.

**Cite these so reviewers cannot call it derivative:**
- Chu 2025 / Kirk 2024 — "RL generalizes" holds for *multi-token generative* tasks; we test whether it
  survives when the reasoning/exploration degrees of freedom are removed. **Frame GRPO's likely null
  as a contribution**: it *bounds where the RL-generalizes claim applies*.
- WiSE-FT / model soups / overadaptation-ensemble — "combine base+tuned to recover OOD" for
  vision/weight-space; we do it in **output space for a binary-token guard, as a function of the
  objective**.
- GSPR / DuoGuard / "Objective Matters" — nearest neighbors, each missing ≥2 of our axes; cite as
  related work, not competitors.

**Framing to avoid the "known result" trap:** lead with the **composition × objective interaction**
and the **single-token-collapses-RL** mechanism as the thesis; treat the SFT/DPO/GRPO comparison as the
*vehicle*, not the headline. "RL for guards" already exists — *this specific measurement* does not.

---

## 8. Minimal experiment matrix, risks, and pre-registration

**Matrix (minimal set that answers H1 + H2 + mechanism):**
- **Objectives:** SFT, DPO, KTO, GRPO core + ORPO endpoint = **5 arms**.
- **Composition:** each arm evaluated ×2 — tuned alone vs base+tuned output-space average. *(Eval-time,
  nearly free; no extra training.)*
- **Seeds:** **5** per arm (matches Paper A's 4×5 protocol → comparable variance bars).
- **Checkpoints:** early + converged (2) for GRPO and the β-anchored arms to catch under/over-training;
  a single converged checkpoint suffices for SFT.
- **Bases:** ≥2 spanning the size range (e.g. **Qwen2.5-1.5B** and **Qwen3-4B/8B**) to test the prior
  SFT finding that specialization cost is worst for the strongest bases; expand to the 6-base legacy
  panel for the stronger "base-competence × objective" version (topic proposal §5).
- **Mediator sweep (recommended, cheap):** β ∈ {low, default, high} on DPO (or KTO) to trace the H2
  monotone relation without multiplying full arms.

Roughly **5 objectives × 5 seeds × ~2 checkpoints × 2 bases ≈ 50 training runs**, × 2 composition
evals — feasible on the L4 given 1-token rollouts.

**Risks → mitigations (superset of the topic proposal's):**
1. *GRPO degenerates to SFT-on-labels* → graded logit-margin reward + Dr.GRPO advantages + dynamic
   sampling; **pre-register the null** so the outcome publishes either way.
2. *β confounds the objective comparison* → sweep β; **report KL-moved per run** and compare objectives
   at matched movement where possible.
3. *Majority-class collapse under GRPO* → class/severity-weighted correctness reward if the manifest is
   skewed.
4. *Reference-model OOM on 8B* → `precompute_ref_log_probs` / frozen LoRA-base reference.
5. *Undertraining masquerading as "transfer-preserving"* → learning-curve check + HPO artifacts.
6. *Effect shrinks under corrected metric/decontamination* (as Paper A cautions for its own legacy
   numbers) → the study is a **paired, same-manifest** comparison, so a null is itself a clean result.

**Pre-registered predictions (write these before running):**
- H1 movement order **ORPO ≳ SFT > KTO ≈ DPO > GRPO**, with transfer loss tracking movement.
- GRPO's transfer edge over SFT is **small or absent** on a single-token verdict.
- H2: composition's transfer recovery is **monotone in base↔tuned KL** — largest for SFT/ORPO,
  smallest for the anchored arms.

---

## 9. References (tiered by verification status)

**Verified by fetch this session (safe to cite as-is):**
- KTO — Ethayarajh et al., *Model Alignment as Prospect Theoretic Optimization*, arXiv:2402.01306.
- DuoGuard — arXiv:2502.05163 · GSPR — arXiv:2509.24418 · RL's Razor — arXiv:2509.04259 ·
  "Objective Matters" — arXiv:2601.12639 · OPT-350M SFT/DPO study — arXiv:2509.09055 ·
  overadaptation-ensemble — arXiv:2506.01901.
- TRL trainer docs (huggingface.co/docs/trl) — trainer availability/maturity.

**Well-known, IDs recalled with high confidence (reconfirm the ID string before final citation):**
- DPO — Rafailov et al., arXiv:2305.18290 · GRPO/DeepSeekMath — Shao et al., arXiv:2402.03300 ·
  IPO/Azar — arXiv:2310.12036 · ORPO — Hong et al., arXiv:2403.07691 · SimPO — Meng et al.,
  arXiv:2405.14734 · RLOO/*Back to Basics* — Ahmadian et al., arXiv:2402.14740 · BCO — arXiv:2404.04656.
- Chu et al., *SFT Memorizes, RL Generalizes* — arXiv:2501.17161 · Kirk et al. — arXiv:2310.06452 ·
  DeepSeek-R1 — arXiv:2501.12948 · Tulu 3 — arXiv:2411.15124 · Dr. GRPO/*Understanding R1-Zero-Like
  Training* — arXiv:2503.20783.
- WiSE-FT — Wortsman et al., arXiv:2109.01903 · Model soups — arXiv:2203.05482 · GLA — arXiv:2310.08106 ·
  Forgetting in SFT→preference — arXiv:2410.15483.
- Llama Guard — arXiv:2312.06674 · WildGuard — arXiv:2406.18495 · ShieldGemma — arXiv:2407.21772.

**Claim-only — recent (2026) preprints surfaced in search but NOT verified; do NOT cite the ID
without re-checking** (used only as directional support, superseded by the reasoning in §3.1):
"gradient starvation under binary-reward GRPO," "GRPO/Dr.GRPO/DAPO as operations on the group std,"
DAPO dynamic sampling, DynaGuard, YuFeng-XGuard, "Learning When to Act or Refuse," "Can GRPO Transcend
Its Pretraining Origin?" The single-token GRPO-degeneracy argument in this doc rests on GRPO's own
advantage definition and Dr. GRPO's verified std-normalization critique, **not** on these preprints.

---

*Cross-references:* [`paper-b-topic-proposal.md`](paper-b-topic-proposal.md) (RQs, asset inventory,
run-plan) · [`paper-b-compose-dont-tune-plan.md`](paper-b-compose-dont-tune-plan.md) (the composition
remedy this axis interacts with) · [`paper-a-improvement-and-extension-recommendations.md`](paper-a-improvement-and-extension-recommendations.md)
(the clean rerun whose SFT slice this campaign shares).
