# Paper C — Deferred Topic Proposal (objective × independent competence specialization)

> **Repositioned 2026-07-14 → this is now Paper C (deferred).** A later prototype + adversarial
> review found the *composition* idea (ensemble the untuned base with the tuned adapter) is a
> cheaper, more novel, GPU-free near-term paper. The objective axis below needs a GPU retrain
> (no per-row DPO/GRPO scores survive) and reproduces published SFT-vs-RL results, so it moves
> behind the composition paper. See [Paper B, *Compose, Don't Tune*](../papers/base-adapter-composition/).

A concrete, feasibility-grounded proposal written 2026-07-13 after inventorying what survives
on disk, and corrected after the 2026-07-14 review. Legacy directional summaries survive, but no
re-analyzable DPO/GRPO result does. The topic remains a natural contingent sequel to Paper A; it
no longer governs Paper B.

---

## 1. Recommended contingent topic

> **Which fine-tuning objective specializes least? A controlled study of SFT vs DPO vs
> GRPO for small prompt-safety guards, and whether the specialization trade-off is
> predictable from an independently measured base-competence covariate.**

**Core question.** Paper A fixed the objective (LoRA-SFT) and varied the base, and found
*in-source specialization*: fine-tuning saturates represented-source AP while producing
heterogeneous transfer changes across four checkpoints. Paper C fixes the specialization lens and varies
the **training objective**:

- **RQ1.** Does the objective change the represented-vs-transfer trade-off — does a
  preference/RL objective (DPO, GRPO) preserve dataset-held-out transfer better than SFT
  at a given represented-source gain?
- **RQ2.** Does base competence measured on an independent locked development cohort predict
  transfer change on a disjoint prospective cohort, and does the objective modulate that
  interaction? Same-row base AP and `SFT − base` must not be used as predictor/outcome.
- **RQ3 (deployment).** How do the objectives differ on the realized FPR / HarmBench-recall
  costs that Paper A surfaced?

This is a coherent, single-axis extension — not a return to the broad-study sprawl.

---

## 2. Preliminary directional evidence (legacy, single-seed — motivating only)

From the surviving legacy summaries (held-out "novel" guard AUPRC vs the untuned base;
in-distribution AUPRC in parentheses). **These use the old pooled metric and benchmark
taxonomy, are single-seed, and predate decontamination — directional signal only, not a
result.**

| Base | base novel AUPRC | SFT (ind / nov) | DPO (ind / nov) | GRPO (ind / nov) |
|---|---|---|---|---|
| DeepSeek-R1-1.5B | 0.600 | 0.822 / 0.772 | 0.584 / 0.585 | 0.496 / 0.601 |
| Qwen2.5-1.5B | 0.797 | 0.869 / 0.792 | 0.691 / 0.815 | 0.649 / 0.801 |
| SmolLM2-1.7B | 0.774 | 0.845 / 0.797 | 0.643 / 0.850 | 0.560 / 0.778 |
| SmolLM3-3B | 0.885 | 0.881 / 0.812 | 0.804 / 0.880 | 0.639 / 0.886 |
| Qwen3-4B | 0.742 | 0.879 / 0.757 | 0.681 / 0.756 | 0.650 / 0.737 |
| Qwen3-8B | 0.765 | 0.882 / 0.812 | — | 0.693 / 0.769 |

The pattern is strikingly consistent and motivates the whole paper: **SFT buys the most
in-distribution gain but risks transfer; GRPO barely moves transfer (novel ≈ base) at the
cost of little in-distribution specialization; DPO sits between.** In other words the
objective appears to be a *dial on specialization intensity* (SFT > DPO > GRPO). If a
clean rerun confirms this, the contribution is clear and useful: *pick the objective by
how much OOD behavior you are willing to disturb.*

---

## 3. What survives on disk (asset inventory)

**Reusable now:**

- **A near-complete 6×(base+SFT+DPO+GRPO) result matrix** under
  `notebooks/outputs/nb-smollm3-guard/summary_*.json`: DeepSeek-R1-1.5B, Qwen2.5-1.5B,
  SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B, Qwen3-8B. Missing only Qwen3-8B-DPO and a
  standalone SmolLM3-3B base file (recoverable). Each summary carries in-dist +
  novel/held-out AUPRC (+CI), per-benchmark P/R/F1, base-vs-tuned decomposition, legacy
  fixed-FPR points, HarmBench recall, and open-guard baselines.
- **HPO results per objective** (`outputs/hpo/hpo_{sft,dpo,grpo}_smollm3-3b.json` + `best`):
  the objective hyperparameters were already searched — the recipe is not guesswork.
- **Training + eval code** in `legacy/experiments/` (`train_guard.py`, `train_guard_pref.py`
  for DPO/IPO/KTO, `hpo_guard.py`, `guard_eval_pipeline.py`, `ensemble_*`, guardrail
  baselines) — the exact pipeline that produced the above.
- **Open-guard baselines** (LlamaGuard-3-1B/8B, WildGuard-7B, Qwen3Guard-4B, ShieldGemma,
  PromptGuard2) result summaries, for context.
- **Paper A's clean v2 pipeline + canonical metrics** (`guard_research/`, `experiments/`)
  — the machinery a publishable version must run under.

**Gone / not reusable as-is (be honest about this):**

- **No per-row score caches for the SFT/DPO/GRPO arms.** Only ExpGuard and mortgage/hard
  caches survive. Consequently the objective AUPRCs **cannot be recomputed** under the
  canonical tie-aware macro-AP or under Paper A's represented/transfer partition from
  surviving data — the legacy summary numbers are all that remain.
- **Single-seed, legacy pipeline.** The summaries have no seed dimension; the "clean
  multi-seed rerun" that replicated the findings is **not** on disk (it lived on the
  now-deleted VMs). Paper A's rigor is 5 seeds.
- **Legacy benchmark taxonomy differs.** Legacy "in-house" mixes benchmarks Paper A splits
  (JailbreakBench/XSTest are Paper A *transfer*; BeaverTails is *excluded*); "novel" =
  {WildGuardTest, WildJailbreak, OR-Bench-hard}. Not comparable line-for-line to Paper A.
- **Adapters deleted** (weights removed in the repo cleanup; retrainable from the code +
  the pinned recipe/HPO).

**Net:** the experiment is designed, HPO'd, proven runnable, and shows a clear directional
result — but it is **not publishable from the surviving files**. A publishable Paper C
needs a clean rerun; the surviving data's job is to de-risk it and scope it precisely.

---

## 4. What a publishable Paper C requires (the plan)

Run the objective matrix through **Paper A's v2 pipeline**, reusing its locks, canonical
metrics, decontaminated manifest, and analysis:

1. **Retrain** the matrix: `{4–6 bases} × {SFT, DPO, GRPO} × {5 seeds}` LoRA adapters on
   the same 1,200-row decontaminated manifest (DPO/GRPO need a preference/reward
   construction over the same rows — the legacy `train_guard_pref.py` shows how; freeze it
   as a v2 recipe). Reuse the surviving HPO for the objective hyperparameters.
2. **Score** base + all adapters on the *locked* Paper A benchmark set for retrospective
   development/comparability and once on a genuinely uninspected prospective cohort after the
   Paper C protocol is locked, emitting per-row logits with the identity-keyed scorer.
3. **Analyze** with `analyze_paper_a_sft.py`'s canonical macro-AP + represented/transfer
   split + hierarchical bootstrap, adding the objective as a factor. Measure competence on the
   locked development cohort and estimate its treatment interaction only on the disjoint
   prospective outcome; never fit the same-row `base AP` versus `adapter − base` regression.

**Synergy with Paper A.** Paper A's clean five-seed SFT slice is complete and supplies a verified
recipe plus retrospective baseline. Paper C still requires new DPO/GRPO training and a separately
locked prospective cohort; reusing Paper A's exposed transfer rows cannot make the objective study
prospective.

---

## 5. Scope, framing, and two size options

- **Minimal Paper C (objective axis):** 4 Paper-A bases × {SFT, DPO, GRPO} × 5 seeds, same
  benchmarks. Clean, tight, directly answers RQ1/RQ3.
- **Stronger Paper C (objective × base-competence):** add DeepSeek-R1-1.5B, Qwen3-8B (both
  already in the legacy panel) to get 6 bases spanning 1.5–8B and two-plus lineages, and test
  the independent-covariate base-competence hypothesis across objectives (RQ2). A practitioner
  decision surface is allowed only if that prospectively locked interaction replicates.

**Keep out of scope** (Paper A's discipline): ensembling, GPT parity, fairness, the
mortgage/domain case study, and a guardrail leaderboard. Objective is the single new axis.

---

## 6. Compute, effort, risk

- **Compute:** ~`bases × 3 objectives × 5 seeds` LoRA runs (minimal: 60; stronger: 90),
  1.5–8B. Legacy SFT runs were 6–10 min each on A100; DPO/GRPO are somewhat heavier, and
  8B is slower. Order ~15–40 A100-hours total — feasible on cheap cloud; slow but possible
  on the M4 Max for the ≤4B tier (8B is the throughput bottleneck). **No human annotation.**
- **Effort:** dominated by freezing the DPO/GRPO preference construction as a v2 recipe and
  wiring the objective factor into the analyzer; the rest reuses Paper A code.
- **Primary risk:** the effect could shrink under the corrected metric/decontamination
  (as Paper A cautions for its own legacy numbers). Mitigation: the study is a *paired,
  same-manifest* comparison, so even a null ("objective does not change the trade-off") is
  an interpretable result rather than an automatic positive claim.
- **Second risk:** GRPO's low in-distribution gain may mean it is simply undertrained
  rather than "transfer-preserving." The HPO artifacts and a learning-curve check guard
  against reporting an undertraining artifact as an objective effect.

---

## 7. Why this remains a contingent Paper C

| Candidate | Feasible solo, no annotation? | Data on hand | Verdict |
|---|---|---|---|
| **Objective × independent competence covariate (this)** | ⚠️ needs GPU rerun + prospective cohort | legacy matrix + HPO + code are motivation only | **Deferred Paper C** |
| Base-competence interaction (standalone) | ⚠️ needs independent predictor/outcome data | same-row Paper A pattern is coupled | Merge only with corrected RQ2 |
| Financial-policy request screening | ⚠️ needs external data + G-labeling | single-labeled external benchmarks | Fallback; harder (see feasibility doc) |
| Decontaminated guardrail landscape | ✅ | baselines survive | Weak novelty (prior review: "methodology, not contribution") |

**Bottom line:** this is a viable objective-axis Paper C, not the governing near-term Paper B.
The legacy experiments motivate it but do not supply claim-bearing evidence. It needs disciplined
GPU retraining, an independently measured competence covariate, and a separately locked prospective
cohort; the completed Paper A rerun cannot substitute for those steps.
