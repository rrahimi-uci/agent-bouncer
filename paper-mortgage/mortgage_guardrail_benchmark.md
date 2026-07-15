# A Mortgage-Specific Safety-Guardrail Benchmark, Grounded in Real HMDA Data

**Reza Rahimi** · JazzX AI, Los Altos, CA · reza.rahimi@jazzx.ai
*Working draft — July 2026*

> **Study status.** This is a **benchmark-construction and baseline-evaluation** paper. The
> dataset is a *fixed* artifact grounded in the public HMDA 2022 snapshot; its prompts are
> **synthetic** and its labels are **LLM-judge, policy-card-consistent — not SME-adjudicated**.
> The benchmark therefore supports *measuring how guards behave on a mortgage-compliance
> distribution*; it does **not** license confirmatory fair-lending claims about any lender,
> model, or population. The contribution is the benchmark, its construction methodology, and a
> reproducible evaluation harness — plus baseline guard results on the frozen data.

---

## Abstract

General-purpose prompt-safety guards are trained to catch domain-independent harm — jailbreaks,
injections, abuse. But a request can be perfectly benign under a general safety policy while still
soliciting a **mortgage-compliance** violation (disparate treatment, redlining, ability-to-repay
circumvention, adverse-action-reason masking, occupancy fraud). We build a **mortgage-specific
guardrail benchmark** whose unit is one incoming request, grounded in the public **HMDA 2022**
loan-level snapshot, and carrying **two independent labels**: a domain-independent
`general_safety` label (G) and a `mortgage_policy` label (D). The cross-product yields four
quadrants; the load-bearing one is **G0/D1** — requests a general guard rates *safe* that
nonetheless solicit a mortgage-policy violation. The benchmark is produced by an **agentic
pipeline** (plan → HMDA-ground → generate → adversarially mutate → rubric-bound judge) that
follows the borderline-and-subtle design principles established for LLM mortgage bias by
\citep{bowen2024mortgage}, and includes **protected-class counterfactual minimal pairs** reframed
as a guard-*invariance* gate. We release a frozen **994-row** version and a clean, canonical-metric
evaluation harness, and report baseline results for open instruction checkpoints used zero-shot as
guards and for off-the-shelf safety guards. We are explicit about what the benchmark does not yet
establish and ship a validation protocol (SME adjudication, the empty G1/D0 quadrant, guard
fine-tuning, cross-source decontamination) required before confirmatory use.

## 1. Introduction

Lenders are exploring LLMs across the mortgage lifecycle. That raises a specific safety question a
general guard is not built to answer: *would honoring this request violate mortgage law or
policy?* A general safety guard screens for harm that is independent of domain. A **mortgage
guardrail** must additionally screen for domain violations that are, on their surface, entirely
"safe" text — e.g. *"applications from the majority-Spanish-speaking tracts keep coming in weak;
draft language to quietly deprioritize that pipeline, and write the denials up as 'collateral' so
it's clean."* No slur, no jailbreak — but it solicits redlining, proxy discrimination, and
adverse-action-reason masking.

This paper contributes:
1. a **fixed, HMDA-grounded benchmark** of mortgage-workflow requests with **dual orthogonal
   labels** (`general_safety` G × `mortgage_policy` D) and a derived intervention target;
2. an **agentic construction methodology** that grounds realistic, *borderline* scenarios in real
   loan records and makes policy violations *subtle and deniable* rather than cartoonish;
3. **protected-class counterfactual minimal pairs** used as a guard-invariance PASS gate; and
4. a **reproducible evaluation harness** and **baseline results** showing that general and
   off-the-shelf guards track G but do not separate the G0/D1 stratum.

It explicitly does **not** claim SME-validated labels, a populated G1/D0 quadrant, a fair-lending
finding about any real system, or deployment readiness.

## 2. Related work and novelty boundary

**LLM mortgage bias.** \citet{bowen2024mortgage} audit LLM underwriting on real HMDA applications
with experimentally manipulated race and credit scores, finding disparate approval/pricing
recommendations that are *largest for lower-credit-quality / riskier loans*. We borrow their
methodology in two ways: grounding scenarios in real HMDA fields, and building **counterfactual
minimal pairs** that hold financials fixed and vary only a protected attribute. Our object is
different: not an audit of an underwriting model's decisions, but a **benchmark for a guard** that
screens incoming requests.

**Safety guards and their benchmarks.** Open guards (Llama Guard, WildGuard, ShieldGemma) and
general prompt-safety datasets (ToxicChat, prompt-injection and jailbreak sets) target a *content*
taxonomy. None encodes mortgage regulation, none grounds in real lending data, and none separates
domain-independent harm from a domain-policy violation as orthogonal labels.

**Over-refusal and counterfactual fairness.** XSTest/OR-Bench probe over-refusal; counterfactual
fairness suites measure a *model's output* bias. We combine both concerns for a *guard*: benign
hard-negatives that a nervous guard over-blocks (e.g. a protected applicant asking "is it pointless
for someone like me to apply?") must PASS, and protected minimal pairs test the guard's
*invariance*, not a downstream decision.

**Novelty boundary.** The dual-label G×D design, HMDA grounding, and minimal-pairs-as-invariance
gate are the combination that is new here. The agentic generate/judge loop is standard red-team
practice; we do not claim it as a primitive.

## 3. Benchmark construction

The construction is deterministic in structure and stochastic in surface wording (the LLM steps
run at temperature > 0). **The released benchmark is therefore a *frozen* artifact, not a
regenerable one** — like the HMDA snapshot it grounds in. Reproducibility is defined at the
*evaluation* layer (§5), not the generation layer.

**HMDA grounding (PII-safe).** We pull the public HMDA 2022 loan-level data (FFIEC/CFPB Data
Browser) and reduce each record to a **banded, de-identified fact sheet** — loan purpose,
occupancy, banded loan amount / income / LTV / DTI, action, denial reason, state — never exact
dollar amounts, census tract, or any identifier. Scenarios are sampled from *marginal
distributions* over these bands, with a build-time assertion that no emitted fact sheet reproduces
a single source record. Following \citep{bowen2024mortgage}, fair-lending and ATR cells are biased
toward **borderline / higher-risk** files (high DTI, high LTV, prior denial), where underwriter
discretion — and bias — actually operates.

**Agentic pipeline.** A *planner* enumerates coverage cells (quadrant × trap × policy card × role ×
protected context); an *HMDA-grounder* draws a fact sheet; a *generator* authors the request in the
natural voice of a role (applicant, loan officer, underwriter, processor, broker, adversary); an
*adversarial mutator* composes label-preserving tactics (euphemism, coded proxy,
business-justification, buried injection); a *rubric-bound judge* assigns the authoritative labels.
Domain-independent harm (jailbreaks) uses curated seeds; the novel domain content is LLM-authored.

**Labels.** Each row carries `general_safety_gold` G ∈ {safe, unsafe} (domain-independent), and
`mortgage_policy_gold` D ∈ {allow, intervene}, defined by 24 benchmark **policy cards** (D01–D24:
fair lending, ATR/QM, disclosures, UDAAP, fraud, privacy) each with an "intervene iff" predicate
and an authority pointer. A derived `final_intervention` = (G unsafe ∨ D intervene), plus an
`action` lattice (PASS/CONSTRAIN/REVIEW/BLOCK) and severity. **Labels are policy-card-consistent,
assigned by an LLM judge; they are not SME-adjudicated** (all 24 cards remain `sme_signoff:false`;
the judge's agreement is measured as self-consistency, not a human Fleiss-κ study).

**Protected minimal pairs.** For fairness, we author counterfactual pairs identical in every slot
except a protected-class token (`variant ∈ {protected, reference}`), both benign (G0/D0, PASS),
sharing a `pair_id`. The guard must PASS both; the pair delta feeds a Δ_context invariance gate
(target ≈ 0).

**Provenance and splits.** Content and near-dup (MinHash) family hashes; family-isolated
train/dev/public_test splits + a sealed private_test; a text-free public index. *Current
limitation:* decontamination was run against the legacy Paper A general sources (exact-hash);
re-running against the v2 sources with near-dup removal is required before joint claims.

## 4. Benchmark composition (frozen v1_hmda2022)

**994 rows**, all `synthetic=true`, `contains_real_pii=false` (verified at freeze: 0 violations).

| Split | Rows | | Quadrant | Rows | | Domain | Rows |
|---|---:|---|---|---:|---|---|---:|
| train | 604 | | G0/D0 (benign) | 450 | | fair_lending | 204 |
| dev | 149 | | G0/D1 (domain-only) | 502 | | fraud | 112 |
| public_test | 146 | | G1/D1 (both) | 42 | | udaap | 90 |
| private_test (sealed) | 95 | | **G1/D0** (general-only) | **0** | | disclosure | 66 |
| | | | | | | atr_qm | 54 |
| | | | | | | privacy | 18 |
| | | | | | | benign | 450 |

Trap types span `business_justified` (202), `direct` (167), `coded_proxy` (126), `euphemism`
(126), `over_refusal_bait` (114), `benign_info` (84), `occupancy_temptation` (72),
`minimal_pair` (78 rows = 39 protected pairs), `buried_injection` (25). **The G1/D0 quadrant is
empty** — the safety-tuned generator declined to author domain-independent jailbreaks — so the 2×2
is only three-quarters populated; this is a stated limitation, not a design claim.

## 5. Evaluation protocol (reproducible)

The evaluator (`magen/score_guards.py` + `magen/evaluate.py`) is the reproducible layer. A guard
emits one "unsafe" probability per row; we map it to G, D, and the composed `final` label alike,
and compute, via the canonical tie-aware `guard_research` metrics:
- **threshold-free macro-AP** for G, D, and `final` (per benchmark, macro-averaged);
- **per-quadrant miss rate** at a calibration-selected operating point (5% dev-FPR target), so a
  strong aggregate cannot hide a G0/D1 collapse;
- **Δ_context**, the mean absolute score gap within protected minimal pairs (fairness invariance).

Guards scored: the four base instruction checkpoints from the companion study
(Qwen2.5-1.5B, SmolLM2-1.7B, SmolLM3-3B, Qwen3-4B) used **zero-shot as guards** via the canonical
guard prompt, plus off-the-shelf **Llama Guard 3** and **WildGuard**. (A domain-fine-tuned guard
arm is future work: the companion study's LoRA adapters were not retained.)

## 6. Baseline results

> *Baseline table populated from the committed scoring run
> (`out_eval/baseline_table.json`, produced by `score_guards.py` on the frozen benchmark). Until
> that run is committed, the harness has been verified end-to-end offline (a mock general guard
> reproduces the expected shape: it tracks general harm but **misses 73/75 G0/D1 rows**, and its AP
> on D collapses relative to its AP on G).*

<!-- BASELINE_TABLE_START -->
| Guard | AP · G | AP · D | AP · final | G0/D1 caught @5%FPR | Δ_context |
|---|---:|---:|---:|---:|---:|
| _pending committed baseline run_ | | | | | |
<!-- BASELINE_TABLE_END -->

The expected and offline-verified finding: guards optimized for general safety **track G but do
not separate the subtle G0/D1 mortgage-policy violations**, so their AP on D and `final` is far
below their AP on G, and they miss most G0/D1 rows at a sensible operating point. That gap — not a
claim that any specific number is a fair-lending fact — is what the benchmark is designed to
surface.

## 7. What this benchmark does *not* establish

- **Not SME-validated.** Labels are LLM-judge / policy-card-consistent. No fair-lending headline
  about a real lender or model is warranted until a stratified SME adjudication (especially of the
  G0/D1 rows and protected pairs) is recorded.
- **Not a full 2×2.** G1/D0 is empty; the orthogonality of G and D is demonstrated on three
  quadrants.
- **Not a regenerable dataset.** Generation is intentionally frozen; only the *evaluation*
  reproduces.
- **Not decontaminated against v2.** Cross-source near-dup removal vs the v2 general sources is
  pending.

## 8. Validation protocol (to reach confirmatory use)

1. **SME adjudication** of a stratified subset; upgrade cards from `sme_signoff:false`; report
   Fleiss-κ per label.
2. **Populate G1/D0** with curated domain-independent-harm-in-mortgage-clothing to complete the 2×2.
3. **Re-decontaminate** against the v2 general sources with MinHash near-dup removal.
4. **Add a domain-fine-tuned guard arm** (retrain the LoRA guards) to quantify the domain-adapter
   gain — and, jointly with the composition study, whether composing a general base with a mortgage
   adapter recovers general transfer.
5. **Select a license**; resolve the contested disparate-impact card (D07) with version-dating.

## 9. Conclusion

We release a fixed, HMDA-grounded, dual-labeled mortgage guardrail benchmark and a reproducible
evaluation harness, and show baselines for how general and off-the-shelf guards behave on it. The
benchmark's value is the **G0/D1 stratum** — mortgage-policy violations that read as safe — and the
**protected minimal pairs** that test guard invariance. It is an honest construct with a clear
validation path, not a finished fair-lending instrument.

## References

- Bowen, Price, Stein & Yang (2024). *Measuring and Mitigating Racial Bias in Large Language Model
  Mortgage Underwriting.* (grounding methodology; borderline / counterfactual design)
- Companion study: *The Benchmark Chooses the Winner* (Paper A) and *Compose, Don't Tune?* (Paper B)
  in this repository.
- HMDA 2022 Snapshot National Loan-Level Dataset, FFIEC/CFPB.
- Llama Guard; WildGuard; ToxicChat; XSTest; OR-Bench (baseline guards / related benchmarks).
