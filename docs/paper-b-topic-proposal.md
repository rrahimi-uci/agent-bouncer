# Paper B — Topic Proposal (objective × base-competence specialization)

A concrete, feasibility-grounded proposal for the next paper, written 2026-07-13 after
inventorying what survives on disk. It supersedes the mortgage/joint-stack direction for
practical purposes: that line is blocked on an uncollectable dual-labeled naturalistic
cohort (see [`paper-b-feasibility-investigation.md`](paper-b-feasibility-investigation.md)).
This topic is chosen because **its core experiments were already run and their results
survive**, it needs **no human annotation**, and it is the natural sequel to Paper A.

---

## 1. Recommended topic

> **Which fine-tuning objective specializes least? A controlled study of SFT vs DPO vs
> GRPO for small prompt-safety guards, and whether the specialization trade-off is
> predictable from base competence.**

**Core question.** Paper A fixed the objective (LoRA-SFT) and varied the base, and found
*in-source specialization*: fine-tuning saturates represented-source AP while degrading
transfer, worst for the strongest bases. Paper B fixes the specialization lens and varies
the **training objective**:

- **RQ1.** Does the objective change the represented-vs-transfer trade-off — does a
  preference/RL objective (DPO, GRPO) preserve dataset-held-out transfer better than SFT
  at a given represented-source gain?
- **RQ2.** Does the **base-competence law** (fine-tuning helps weak bases, hurts strong
  ones on transfer) hold across objectives, and does the objective modulate its slope?
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
  novel/held-out AUPRC (+CI), per-benchmark P/R/F1, base-vs-tuned decomposition, matched-FPR
  points, HarmBench recall, and open-guard baselines.
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
result — but it is **not publishable from the surviving files**. A publishable Paper B
needs a clean rerun; the surviving data's job is to de-risk it and scope it precisely.

---

## 4. What a publishable Paper B requires (the plan)

Run the objective matrix through **Paper A's v2 pipeline**, reusing its locks, canonical
metrics, decontaminated manifest, and analysis:

1. **Retrain** the matrix: `{4–6 bases} × {SFT, DPO, GRPO} × {5 seeds}` LoRA adapters on
   the same 1,200-row decontaminated manifest (DPO/GRPO need a preference/reward
   construction over the same rows — the legacy `train_guard_pref.py` shows how; freeze it
   as a v2 recipe). Reuse the surviving HPO for the objective hyperparameters.
2. **Score** base + all adapters on the *locked* Paper A benchmark set (represented,
   transfer, stress) with the correct prompt and the identity-keyed scorer, emitting
   per-row logits (so canonical macro-AP is recomputable — the thing the legacy run lost).
3. **Analyze** with `analyze_paper_a_sft.py`'s canonical macro-AP + represented/transfer
   split + hierarchical bootstrap, adding the objective as a factor and fitting the
   base-competence regression across the expanded panel.

**Synergy with Paper A.** Paper A's own top recommendation is to run its clean SFT rerun.
That rerun **is the SFT slice of this matrix.** Doing them together — one v2 campaign that
trains SFT/DPO/GRPO across 4–6 bases × 5 seeds — yields Paper A (the SFT specialization
result) *and* Paper B (the objective × base-competence result) from one pipeline execution.
This is the efficient program.

---

## 5. Scope, framing, and two size options

- **Minimal Paper B (objective axis):** 4 Paper-A bases × {SFT, DPO, GRPO} × 5 seeds, same
  benchmarks. Clean, tight, directly answers RQ1/RQ3.
- **Stronger Paper B (objective × base-competence):** add DeepSeek-R1-1.5B, Qwen3-8B (both
  already in the legacy panel) to get 6 bases spanning 1.5–8B and two-plus lineages, and
  formalize the base-competence law across objectives (RQ2). Delivers a practitioner
  decision surface: *which objective for which base, given a transfer budget.*

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
  a clean, publishable result — the honest-negative property Paper A already relies on.
- **Second risk:** GRPO's low in-distribution gain may mean it is simply undertrained
  rather than "transfer-preserving." The HPO artifacts and a learning-curve check guard
  against reporting an undertraining artifact as an objective effect.

---

## 7. Why this over the alternatives

| Candidate | Feasible solo, no annotation? | Data on hand | Verdict |
|---|---|---|---|
| **Objective × base-competence (this)** | ✅ needs rerun compute only | near-complete legacy matrix + HPO + code | **Recommended** |
| Base-competence scaling law (standalone) | ✅ | subsumed here as RQ2 | Merge into this |
| Financial-policy request screening | ⚠️ needs external data + G-labeling | single-labeled external benchmarks | Fallback; harder (see feasibility doc) |
| Decontaminated guardrail landscape | ✅ | baselines survive | Weak novelty (prior review: "methodology, not contribution") |

**Bottom line:** the objective × base-competence specialization paper is the strongest,
most feasible Paper B on hand. Its experiments are designed, HPO-tuned, and show a clear
directional result; its only real cost is a disciplined clean rerun that **shares the
pipeline execution with Paper A's own recommended rerun** — no annotation, no naturalistic
cohort, no domain-SME dependency.
