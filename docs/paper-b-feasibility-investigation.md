# ARCHIVED/HISTORICAL — Mortgage Paper B Feasibility Investigation

> **Historical role:** This investigation explains why the mortgage joint-stack axis was
> dropped. The authoritative near-term Paper B direction is now
> [`paper-b-compose-dont-tune-plan.md`](paper-b-compose-dont-tune-plan.md).

Investigator analysis of whether the former mortgage Paper B plan was feasible and could
end in a "solution." Written 2026-07-13 against the then-governing plan
[`paper-b-development-checkpoint-plan.md`](paper-b-development-checkpoint-plan.md)
(`v2-adversarial-review`, baseline commit `df604d9`) and its archived rationale
[`paper-b-joint-compliance-stack-plan.md`](paper-b-joint-compliance-stack-plan.md).

This is a historical judgement document, not a current plan amendment. Its checkpoint
contract no longer governs composition work.

---

## TL;DR

- **The idea is sound and the core experiment is trivial to run.** Paper B's
  primary contrast (E3 vs E4) is a clean, falsifiable, well-posed threshold-selection
  question with a closed-form basis. The optimizer, scorer, and decontamination
  machinery are mostly already built (Paper A + `legacy/`).
- **The binding constraint is not compute or code — it is data and construct
  validity.** The claim-bearing path needs a *powered, naturalistic, fully
  dual-labeled* mortgage-request cohort that **does not exist anywhere** and is
  expensive-to-impossible for a solo researcher to collect, plus a mortgage-policy
  target that currently **fails construct validity**.
- **A publishable paper is reachable; the *advertised* positive solution is a real
  research bet.** The plan has honest fallbacks (precision-measurement / controlled
  challenge-set) that still yield a paper. But "we end up with a demonstrably better
  joint stack" is conditional on (a) executing a heavy protocol and (b) the effect
  being real on untouched test data — and the available evidence suggests a **null
  result is quite possible**.

**Verdict:** Feasible as a *measurement / challenge-set* paper. The full
*confirmatory positive-claim* version is, for a solo researcher with no live
compute, **unlikely to complete as written**, and even if executed has a
substantial probability of a null outcome. Recommendation: pursue the de-scoped
landing in [§7](#7-recommended-feasible-path) deliberately, not as a failure mode.

> **Update ([§9](#9-external-fintech-benchmarks-addendum), 2026-07-13):**
> External fintech-safety benchmarks (FinRED, FinSafetyBench, FairHome) offer *citable,
> expert/real-world-grounded* financial-compliance labels that help the P4a construct
> problem and can seed the domain (`D`) dimension — **most usably FinRED/FinSafetyBench**
> (request-oriented). **FairHome is weaker than it first looked** (a deep-dive found it is
> mostly fair-housing *steering* with lending a minority slice, labels *responses* not
> requests, and is *request-gated*). None solves the crux: all are single-labeled (not
> dual) and none is *naturalistic request traffic*. Net: the de-scoped measurement paper
> gets modestly cheaper and better-grounded (likely reframed to "financial-policy
> screening"); the confirmatory naturalistic claim is unchanged.

---

## Conclusion — current-data feasibility (2026-07-13)

Stated bluntly, after reviewing every dataset within reach (the repo's own mortgage
sets plus the public fintech-safety benchmarks in [§9](#9-external-fintech-benchmarks-addendum)):

- **Paper B's flagship claim is infeasible on available datasets.** The confirmatory
  result — E4 beats E3 on *untouched, naturalistic, fully dual-labeled* mortgage requests —
  requires a cohort with two properties **no downloadable dataset has, and that no
  combination of them can synthesize**: (1) *dual* labels (`G` **and** `D` on the same
  rows — every available set is single-labeled, and single-labeled sets cannot be
  concatenated into joint gold), and (2) *naturalistic* request traffic (everything
  available is authored, adversarial, curated-from-cases, or response-labeled). The
  fintech benchmarks change only the *cost* of annotation, not this fact.
- **The blocker is data that must be *created*, not *found*.** Reaching the flagship
  claim is a resourcing decision — expert dual-annotation, authentic-traffic sourcing,
  SME construct signoff, and multi-party governance (months; not solo) — not a technical
  gap a dataset release will close.
- **Paper B is not dead — the *claim* has to change.** Exactly three paths remain:
  1. **Change the data** — fund/partner for the annotation + naturalistic cohort, then
     run the confirmatory protocol. Makes the impossible part possible.
  2. **Change the claim** — the de-scoped *measurement / challenge-set* paper of
     [§7](#7-recommended-feasible-path): a modest dual-labeled challenge set, the E3/E4
     overlap mechanism reported *descriptively* (`precision_focused_measurement`), fairness
     lens retained. Reachable solo in weeks–months; a real contribution, just not
     "we built a better joint stack."
  3. **Shelve Paper B** and let Paper A stand.

**Bottom line: the confirmatory-naturalistic Paper B is a no-go on current data; the
honest measurement paper (§7) is the only version a solo researcher can actually ship.**

---

## 1. What Paper B actually is

Two request screens over the same input `x`:

- `G` — a **general-safety** guard (harmful content / jailbreak / injection).
- `D` — a **mortgage-policy** guard (fair-lending / compliance violations).

The system intervenes when either fires: `b_OR = b_G ∨ b_D`. On benign-under-both
(`G0/D0`) requests, the observed false-intervention rate obeys the finite-sample
inclusion–exclusion identity

```
r_OR = r_G + r_D − r_GD          (r_GD = both fire on the same benign row)
```

The whole paper turns on how you *budget* benign risk when picking the two
thresholds:

| Arm | Constraint | Meaning |
|---|---|---|
| **E3** | `r_G + r_D ≤ α` | marginal-sum: ignores overlap (conservative, wastes budget) |
| **E4** | `r_OR ≤ α` i.e. `r_G + r_D − r_GD ≤ α` | measured-union: credits the overlap actually observed |

The primary question: does crediting measured overlap (E4) let you keep more
benign traffic un-blocked than the naive sum (E3), **on a locked, untouched test
set**, without breaching preregistered missed-intervention caps `β_q` in each
positive quadrant?

This is a legitimately nice idea: it is exact (no independence assumption), it
isolates a single accounting rule, and it is preregisterable.

### The one subtlety that governs everything

`r_OR ≤ r_G + r_D` always, so **E4's feasible set is a superset of E3's, and E4 is
mathematically guaranteed to be no worse *on development data*** (plan §2.1,
line 184). That guarantee is worthless as evidence. The *only* real finding is
whether the development advantage **transfers to the locked test set** without
violating absolute test constraints. Everything hard about Paper B exists to make
that one transfer comparison trustworthy.

---

## 2. Two feasibility questions, kept separate

People conflate these; they have very different answers.

1. **Can the protocol be executed?** (engineering + data + governance)
2. **Will it yield the positive "solution"?** (a scientifically real E4 > E3)

| | Optimistic | Realistic (solo, no live compute) |
|---|---|---|
| Run the E3/E4 mechanics on *some* data | ✅ already demonstrated (smoke test) | ✅ |
| Execute the **confirmatory** protocol as written | possible with a team + budget | ❌ unlikely (see §5) |
| Reach **a** publishable paper | ✅ | ✅ via fallback (§7) |
| Reach the **positive** joint-stack claim | plausible | ⚠️ genuine bet; null is likely (§6) |

---

## 3. What already exists vs. what is missing

### Assets that materially reduce the work (already in-repo)

- **Canonical, tie-aware metrics** (`guard_research/metrics.py`) and
  **FPR-budget threshold selection** with a Clopper–Pearson upper bound
  (`guard_research/thresholds.py`) — reusable for per-guard operating points and
  the risk accounting.
- **Provenance / de-leaking** (`guard_research/provenance.py`: NFKC normalization,
  content/family SHA-256, MinHash near-dup families) and the **family-safe splitter**
  pattern (`legacy/experiments/build_mortgage_split.py`). This is exactly the
  decontamination rigor Paper B's `split_lineage_graph` needs, and it is Paper A's
  proven strength.
- **A full general-safety guard pipeline** (Paper A: prepare→audit→lock→train→
  eval→analyze) with calibrated scores in `artifacts/paper_a_sft/scores/scores.parquet`.
- **Score-fusion primitives** (`legacy/experiments/ensemble_probe.py`,
  `ensemble_deployable.py`): convex weighting, rank-average, PIT normalization — a
  starting point for the selector.
- **A mortgage guard trainer + eval protocol** (`legacy/experiments/train_mortgage.py`,
  `eval_mortgage.py`, `eval_mortgage_hard.py`) and the mortgage system prompt.
- **A fair-lending fairness probe** (`legacy/experiments/name_fairness_probe.py`,
  ECOA/Reg-B framing) — a distinctive, defensible angle mostly ready to reuse.
- **The 334-row hardened mortgage benchmark** (relocated to
  `paper-a/paper-html/explorer/sources/guard_benchmark_hard.jsonl`), with rationales
  and 30 minimal-pair groups.

### Gaps that must be built or collected from scratch

- **A dual-labeled cohort does not exist.** No row anywhere carries *both* a
  general-safety label and a mortgage-policy label. This is the central missing asset.
- **A naturalistic mortgage cohort does not exist.** All mortgage data is authored
  red-team / template-augmented synthetic. The plan forbids study-authored or
  existing-benchmark rows for the *primary* claim (dev plan lines 1261, 1597).
- **The constrained selector itself** — no E3/E4 budget-constrained grid search, no
  overlap term `r_GD`, no per-quadrant `β_q` caps. Only unconstrained AUPRC-max
  weighting exists.
- **Simultaneous two-guard scoring** on one cohort with two gold columns and the
  *correct* prompts (`B/G` under the general prompt, `D` under the mortgage prompt).
  Existing caches score everything under the mortgage prompt and grade against a
  single policy — unusable for the joint claim.

### Current-state deltas since the plan's baseline (`df604d9`)

The recent repo cleanup changed the starting assets the plan enumerates in §3.1:

- **Both SmolLM3-3B LoRA adapters (general + legacy mortgage) were deleted locally.**
  Their `notebooks/outputs/nb-smollm3-guard/**/adapter/` dirs now hold only configs.
- **The 1,563-row mortgage source and 1,000-row red-team source were deleted** with
  `notebooks/`; only positional `P(unsafe)` score caches survive (raw text/gold
  unrecoverable at row level).
- **Survived (relocated):** the 334-row hard set and `data/frozen_eval_rows.json`.

This looks worse than it is: P4a (below) was already going to declare the legacy
mortgage target construct-invalid and likely force a retrain, so the deleted adapter
and corpus were on the chopping block regardless. The loss mainly removes the
"quick prototype" convenience, not the claim-bearing path. But it does mean **there
is no runnable component today** — the general guard is retrainable from committed
HF sources; the mortgage guard retrain is blocked until a *valid* target and
training corpus are (re)constructed.

---

## 4. Binding constraint #1 — construct validity (P4a)

**This is the most likely place the whole paper dies, and it is upstream of
everything.**

The legacy mortgage adapter was trained on a `flag` target. The plan's own audit of
the join distribution shows `flag` collapses ~11 heterogeneous response behaviors
into one label — 552/995 rows are `refuse_and_educate`, others are
`acknowledge_uncertainty`, `enforce_verification`, etc. The plan states plainly this
is **"a broad 'non-default handling' target, not a hard-block target"** and gives a
fair-lending example where the current label is *actively wrong* (a protected
applicant asking an information-seeking question is flagged, when the criterion
required reassurance/education).

P4a must return one of `VALIDATED_TARGET`, `RETRAIN_REQUIRED`, or `NO_GO`. Two
uncomfortable facts:

- A `VALIDATED_TARGET` on the existing target is implausible given the evidence
  above; the honest outcome is `RETRAIN_REQUIRED` at best.
- A defensible mortgage-policy *request-screening* construct needs a policy registry
  grounded in real regulation (ECOA/Reg-B, TILA/RESPA, UDAAP…) and **mortgage-SME
  adjudication**. The author is an ML researcher, not a mortgage-compliance lawyer;
  without credible domain expertise, the construct itself is contestable, and a
  reviewer can reject the paper on construct validity alone.

If the mortgage-policy screen cannot be given a valid, expert-approved binary
intervention target that depends only on the *request* (not the response or a credit
decision), Paper B has no `D` and no paper.

---

## 5. Binding constraint #2 — the powered naturalistic dual-labeled cohort

The primary claim lives or dies on data the repo does not have and cannot cheaply make.

**Volume (dev plan P7, §13):**

- False-intervention precision: for a ~1% rate at ±0.5%, `n ≈ 1,522` benign
  (`G0/D0`) units — and the plan is explicit that 1,522 is an **IID Wald
  lower-bound illustration, not the target**. The real number inflates for every
  clustering level and may count **only expert-validated** `G0/D0` rows.
- Separately, a *paired noninferiority* comparison on **severe** missed-intervention
  in **each** of three sparse positive quadrants (`G1/D0`, `G0/D1`, `G1/D1`). Rare
  severe events in naturalistic traffic are exactly where effective sample size
  collapses.
- The archived rationale estimates "≈1,500 or more independent, expert-validated
  dual-benign units" for the low-FPR precision target alone, and says the powered
  total "cannot be estimated honestly" until the pilot runs.

**Labeling (dev plan P3/P6):** every row needs *two independent general-policy
judgments and two independent mortgage-policy judgments*, blind to model identity,
scores, the other policy's label, split, and the intended result; then joint-panel
adjudication. Pilot gates require Gwet AC1 ≥ 0.70 overall and ≥ 0.65 per major
category. The archived labor table: ~32h for a 240-row pilot alone; powered cohorts
"substantially more"; policy registry 10–20h. Realistically **hundreds of hours of
expert annotation**.

**"Naturalistic" is the killer adjective.** Authored quartets and existing
benchmarks are explicitly disallowed for the primary claim. Naturalistic mortgage
*requests* must pre-exist and be independent of the study (deidentified logs,
authentic public inquiries). A solo academic has no obvious pipeline to thousands of
authentic, consented, deidentified mortgage-request texts — and this domain carries
privacy/fair-lending sensitivity that raises the acquisition and IRB bar further.

If this cohort is unobtainable, the plan self-demotes to a controlled challenge-set
paper (dev plan lines 32, 1841) — i.e. the headline comparative claim is dropped.

---

## 6. Binding constraint #3 — governance vs. a solo researcher, and the null-result risk

### The independence apparatus is sized for a lab, not a person

The protocol demands **nine distinct signing roles with separate KMS keys**
(research owner, general-policy owner, statistician, mortgage SME, data owner,
custodian, auditor, reproducer, release owner), a **custodian↔selector firewall**
(the test set is secretly ordered/sealed and scored exactly once by someone with no
model access), an **auditor-supervised fit process**, and an **independent
reproducer** on a fresh clone. "One generic key cannot prove independence."

A single person cannot *genuinely* hold independent custodian, auditor, and
reproducer roles; self-signing all nine keys defeats the very property the locks
exist to demonstrate. This is not fatal to a measurement paper, but it **guts the
credibility of the confirmatory, one-shot-test framing** — the strongest version of
the paper is precisely the version a solo author cannot legitimately certify.

### Even flawless execution may return a null

Three independent signals point to a plausible E4 ≈ E3 outcome:

1. The archived smoke test on the 334-row set: composition **did not** beat the
   single best component — "the optimizer selects the mortgage adapter alone."
2. `corr(G, D) = 0.39` and `corr(B, D) = 0.53`: the guards are correlated. The
   benefit of E4 comes from a **large, stable benign overlap** `r_GD`; correlated
   guards that co-fire on the same benign rows can make `r_GD` either helpfully large
   *or* the two screens redundant (mortgage dominates, `G` adds little).
3. E4 is guaranteed to win on development **by construction**, which means any
   apparent advantage that fails to survive the locked test is exactly the outcome
   the protocol is built to expose. The honest prior on a small, transfer-fragile
   effect is not favorable.

A null is **publishable and useful** ("measured evidence against unnecessary
layering"), and the plan rightly retains it. But it is not the "solution" of a
better joint stack. So: conditional on flawless execution, P(positive confirmatory
result) is maybe coin-flip-ish; unconditional (including the chance the cohort/
governance never comes together) it is low.

---

## 7. Recommended feasible path

Treat the plan's fallback as the **primary target**, chosen on purpose:

> **A controlled, fully-decontaminated *measurement* study of guard-composition error
> overlap in mortgage-request screening, with a fair-lending fairness lens.**

Concretely, a version a solo researcher can actually finish:

1. **Fix the construct first, honestly (P3/P4a-lite).** Write a small, regulation-grounded
   mortgage request-screening policy with explicit `PASS/CONSTRAIN/REVIEW/BLOCK`
   criteria. If you cannot get a mortgage SME to sign it, **say so** and frame the
   domain label as a documented proxy — do not overclaim a validated compliance screen.
2. **Build a *challenge-set* dual-labeled cohort you can afford** (hundreds, not
   thousands of rows), reusing the 334-row hard set + new authored quadrant quartets,
   fully family-decontaminated with the existing MinHash machinery. Label both
   policies; report agreement statistics honestly.
3. **Implement the E3/E4/E_sep selector** on top of `guard_research/thresholds.py`
   and the inclusion–exclusion identity. Add the temperature-invariance unit test the
   plan specifies (a cheap, high-value correctness check).
4. **Report as `precision_focused_measurement`:** effect estimates and intervals for
   `r_GD`, `Δ_intervene`, and per-quadrant missed-intervention — *descriptively*.
   State up front you are measuring the mechanism, not certifying a superiority claim.
   This removes the one-shot-custodian and multi-role-independence requirements that a
   solo author cannot satisfy.
5. **Keep the fairness probe** (`name_fairness_probe.py`) as a genuine contribution:
   protected-context counterfactual sensitivity of a composed mortgage screen is
   novel and mostly built.
6. **Only if a positive, powered claim is later wanted:** partner with a lab/SME for
   the naturalistic cohort and the independent custodian/auditor roles, and upgrade to
   the confirmatory protocol. Do not attempt the one-shot confirmatory claim solo.

This lands a defensible paper in weeks-to-a-couple-months of solo effort, reuses the
repo's real strengths (decontamination, calibrated metrics, fairness), and keeps the
door open to the confirmatory upgrade — without betting the whole project on data and
governance that a solo researcher cannot control.

---

## 8. Bottom line

- **Is it feasible?** Yes — *as a measurement / challenge-set study*. The science is
  clean and most of the tooling exists.
- **Will we end up with the solution?** If "the solution" means a rigorously
  certified, naturalistic, confirmatory demonstration that a measured-union joint
  stack beats the marginal-sum baseline: **not on the current solo, no-live-compute
  footing, and not reliably even with heroic effort** — construct validity, an
  unobtainable powered naturalistic dual-labeled cohort, a lab-grade governance
  apparatus, and a real chance of a null all stand in the way.
- **The intellectually honest move** is to build the de-scoped measurement paper in
  §7 deliberately, report whatever the overlap mechanism actually does (positive or
  null), and reserve the confirmatory positive claim for a future, resourced,
  multi-party effort.

Cost is dominated by **expert annotation and domain expertise, not GPUs**. That is
the number to plan around.

### Where to start (from the plan's own §24, trimmed to the feasible path)

1. Stand up the Paper B package skeleton + Python 3.12 toolchain (P1). *(Note the
   plan's flagged defect: the active `.venv` is on 3.14.4; `.python-version` says 3.12.)*
2. Freeze a small, honest policy/action registry (P3) — get SME signoff or label the
   proxy as a proxy.
3. Run the P4a construct audit on whatever mortgage target you adopt; expect
   `RETRAIN_REQUIRED`.
4. Implement + unit-test the E3/E4 selector and the correct dual-prompt scorer before
   collecting any labels.

---

## 9. External fintech benchmarks (addendum)

*Added 2026-07-13 after a review of public fintech-safety benchmarks (prompted by a
pointer to `datumo/FinRED`). These change the feasibility arithmetic of the de-scoped
path in §7; they do not change the crux.*

### What's out there

| Benchmark | What it is | Size | Row = | Label(s) | Naturalistic? | License |
|---|---|---|---|---|---|---|
| [FairHome](https://arxiv.org/abs/2409.05990) | Fair-**housing** (mostly *steering*) + some fair-lending | ~75,000 | **(query, LLM-response) pair**, labeled at response-sentence level | **binary compliance-risk** + protected-category tags (22.4% carry ≥1) | Zillow/search queries + expert + synthetic aug | CC-BY-NC-**SA**-4.0; **dataset request-gated (email Zillow)** |
| [FinRED](https://huggingface.co/datasets/datumo/FinRED) | Financial-safety **red-team** prompts (Korea FSI, 12 experts) | 5,805 | adversarial request | 26-way risk taxonomy | ❌ adversarial/authored | CC-BY-NC-4.0 |
| [FinSafetyBench](https://arxiv.org/abs/2605.00706) | Financial-compliance request-refusal benchmark | (n/s) | user request | 14 crime/ethics subcats | ~ real-world-**case-grounded** (curated, not raw traffic) | (n/s) |
| [CNFinBench](https://www.arxiv.org/pdf/2512.09506) | Chinese finance safety/compliance | — | — | multi-dimension | ❌ | — |
| [TRIDENT](https://arxiv.org/pdf/2507.21134) | LLM safety across finance/medicine/law | — | request | domain safety | — | — |

### FairHome, on closer inspection (deep-dive, 2026-07-13)

A closer read of the paper + the [Zillow guardrail repo](https://github.com/zillow/fair-housing-guardrail)
substantially tempers FairHome's usefulness for Paper B:

- **Lending/mortgage is a minority, and no published percentage exists.** The paper
  gives *no* domain breakdown. But the dataset's *primary policy target is illegal
  "steering" in real estate* — a fair-**housing**/brokerage concept (directing clients
  to neighborhoods by protected class), not mortgage/credit underwriting. Fair *lending*
  (ECOA) is named in the intro but is secondary. Provenance skews the same way: Zillow
  plugin + real-estate search queries, i.e. mostly property search/browse/rent/buy.
  **Estimated mortgage/lending share: a minority — plausibly single-digit to low-tens
  percent, not the majority.** The exact figure is unknowable from public materials;
  getting it requires the gated data + filtering for lending/credit/mortgage content.
- **It is a *response* screen, not a *request* screen.** Rows are (query, LLM-response)
  pairs labeled at the **response-sentence** level. Paper B's `D` screens the *request*.
  So FairHome does not drop in as the `D` dimension without relabeling.
- **The labeled dataset is request-gated** (email Zillow + use-case review); only the
  guardrail *code* is open. Add an access dependency and possible use restrictions on
  top of CC-BY-NC-**SA** copyleft.

**Net:** FairHome is a fair-housing *steering* benchmark with a modest, response-labeled,
gated lending subset — **not** the clean ~75k expert-labeled *mortgage-request* backbone
the first draft of this addendum implied. Treat it as, at most, a citable source of a few
thousand lending-adjacent rows (after access + relabeling), or as prior-art to cite —
not as the `D` cohort.

### What they genuinely help

- **The `D` (financial/mortgage-policy) construct — constraint #1 / P4a.** FinRED (FSI
  expert taxonomy) and FinSafetyBench (real-world-case-grounded compliance) provide
  *externally authored, citable* financial-policy labels; grounding `D` in published
  expert work is more defensible than the legacy `flag` target. FairHome contributes
  here only for its *fair-lending steering* slice (see deep-dive above), and as prior art.
- **The fairness gate almost for free.** FairHome's protected-category tagging (22.4% of
  rows) is a template for the protected-context counterfactual gate (`Δ_context`) and the
  existing `name_fairness_probe.py`, even if its rows aren't reused directly.
- **Annotation burden roughly halved.** The single biggest cost was dual-annotation.
  If a row arrives with an expert-provided *domain* label, only the **general-safety**
  label `G` must be added (annotated, or scored by the Paper A guard with light human
  verification). Dual-labeling becomes single-dimension top-up on expert-labeled data.
- **A bigger, better challenge set.** §7 step 2 can now use these instead of thin
  authored quartets — hundreds-to-thousands of expert rows, easily past the ~1,522
  precision floor for the benign (`G0/D0`) rate.

### What they do NOT solve (the crux is unchanged)

- **None is dual-labeled.** Every one carries only its own policy label. The
  general-safety label `G` on the *same rows* still has to be produced. This is the
  irreducible task, only made cheaper, not eliminated.
- **None is "naturalistic" in the plan's strict sense.** FinRED is adversarial;
  FinSafetyBench is real-world-*case-grounded* but curated; FairHome's provenance is
  unstated. The plan forbids authored/benchmark rows for the **primary** naturalistic
  claim (dev plan lines 1261, 1597), so these strengthen the **challenge-set /
  `precision_focused_measurement`** landing — not the confirmatory naturalistic claim.
- **Domain drift: "financial" ≠ "mortgage," and even FairHome ≠ "lending."**
  FinRED/FinSafetyBench are broad fintech (fraud, authentication, product misinfo);
  FairHome is housing but *mostly steering*, with lending a minority slice (deep-dive
  above). None is a mortgage-request corpus. Using them either **reframes** Paper B to
  "financial-" or "housing-policy screening" (broader, loses fair-lending sharpness) or
  requires filtering to a small mortgage/lending subset.
- **Licensing.** All are non-commercial (CC-BY-NC / NC-SA), consistent with the repo's
  existing reconstruct-only, text-free-artifact handling. FairHome's **ShareAlike** is a
  genuine copyleft complication for any mixed released cohort — keep sources separate and
  redistribute only text-free artifacts, as Paper A already does.

### Due-diligence before relying on them

1. ~~Verify FairHome's row type~~ **Resolved:** FairHome labels (query, response) pairs at
   the response-sentence level (a response screen), and its dataset is request-gated —
   so it does not drop into the request-screen `D` without access + relabeling.
2. **Request FinSafetyBench/FinRED data and confirm the mortgage/lending subset size** —
   how many rows survive filtering to mortgage/credit/housing-finance is the number that
   decides whether an external set can seed `D` at all.
3. **Decontaminate against Paper A general sources** with the existing MinHash/family
   machinery before mixing (some financial-safety rows may overlap general jailbreak sets).
4. **Check license compatibility** for a combined release; default to text-free artifacts
   (note FairHome's ShareAlike copyleft).

### Revised recommendation

Use these as **citable prior art and a construct scaffold**, not as a drop-in `D` cohort.
The most usable domain backbone is **FinRED / FinSafetyBench** (request-oriented,
expert/real-world-grounded financial-compliance labels), filtered to the mortgage/lending
subset if a mortgage framing is kept — accepting that this likely **reframes Paper B to
"financial-policy screening."** FairHome is best cited as prior art and mined for its
protected-category/steering structure, not adopted as `D` (it is response-labeled, gated,
and mostly non-lending). In all cases you still **annotate the general-safety label `G`**
on the chosen rows, and report the E3/E4 overlap mechanism descriptively. This is still
the measurement paper, not the confirmatory naturalistic claim, which remains gated on
authentic request traffic (§5) and a multi-party governance
apparatus (§6).
