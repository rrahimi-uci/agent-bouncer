# Presentation and implementation proposal for the unified report

## Status update (2026-07-17): this proposal was written against a superseded build

> **Read this before acting on any page-anchored recommendation below.** A deep re-review
> (`presentation-style-proposal-review.md`, 50 findings) reconciled this proposal against the current
> repo. The plan's architecture is still right; its *premise* is stale.
>
> **The report now contains a confirmatory study.** `sections/act-adaptation.tex` (`sec:adaptation`,
> `tab:adaptation`) is a fully-run, **preregistered confirmatory** study over 10 released checkpoints
> spanning 6 model families (incl. six purpose-built guards: ShieldGemma, Qwen3Guard ×2, Llama-Guard,
> Granite, WildGuard). Verdicts were locked in a non-HARKing claim registry before any score existed
> (`artifacts/starting_type_adaptation_v1/analysis/claim_checks.json`): **RQ1 SUPPORTED** — ordinary SFT
> specializes even already-purpose-built vendor guards (H_gain LCB **+0.129**, H_conc LCB +0.189);
> **RQ2 NOT SUPPORTED** — KL-SFT preserves transfer (H_preserve LCB +0.035) but its represented cost
> (H_cost LCB **−0.060**) fails the −0.02 non-inferiority margin, so it is a genuine tradeoff, not free.
> Every reference below to this work as "future research" or "results do not yet exist" (esp. lines in
> §1.6 and §6.5) is now **false and must be deleted**. The only genuinely-future item is a fresh,
> genuinely-*uninspected* cohort — `sec:adaptation` still scores on the inspected Paper A manifest, so
> its bounds are preregistered-but-panel-conditional, **not** sealed-cohort.
>
> **Consequence — the report is FOUR bodies of evidence, not three:** three retrospective acts **plus one
> confirmatory adaptation study.** The rewrite must give that confirmatory study a first-class slot in the
> narrative spine, page budget, evidence-badge taxonomy, executive spread, and source-migration map
> (see the revised §3.1/§3.2 and new §1.7 below). A confirmatory *not-supported* result (RQ2) is a
> finding, not a gap.
>
> **Several P0 corrections are already applied** (commits + a `report.md` critique pass this proposal
> predates) — do not re-litigate them: 15/15 → **15/20** (`unified_report.tex:314`); the KL-SFT
> "free/keeps-specialization" overclaim removed and cross-linked to the RQ2 failure (`tab:guidelines`
> Row 3); abstract mortgage "winner flips" → "top-ranked guard differs (directional, small split)";
> base-only/zero-shot domain labels. The §2.1 **preregistration P0 is inverted** — the preregistration is
> real; do **not** remove it, *propagate* it.
>
> **The synthesis object is now `tab:guidelines`** — a styled 7-row "what we learned → evidence →
> guideline" table (`unified_report.tex`, Row 7 = the confirmatory result), not "Table 16." Use it as the
> *seed* for the executive cards, not a from-scratch build.
>
> **One high-severity correctness item is still OPEN and is now the top Phase-0 task:**
> `unified_report.tex:130` still reads *"All evidence here is retrospective and estimation-only,"* which
> contradicts `sec:adaptation`. Scope that clause to Acts I–III and carve out the confirmatory study
> (keep the panel-conditional / no-fair-lending clause). This is the single highest-value remaining claim
> fix.

## Review record

<!-- REVISED (v2): flag the reviewed build as superseded; every page anchor / SHA / "Table 16" below is stale. -->
> **This review build is superseded — see the Status update above.** All page anchors, the SHA-256,
> and every table number in this record predate `sec:adaptation` and the styled `tab:guidelines`. Do not
> act on any page-anchored recommendation without a re-review against the current PDF. In particular, the
> flow diagnosis below must now account for the new confirmatory section (currently mis-placed — its
> opener references "Acts I–III" but it is `\input` right after Act I), and "Table 16" no longer
> identifies the synthesis object (that is now `tab:guidelines`, which already carries the 15/20 and KL
> fixes and a confirmatory row).

Reviewed artifact: [unified-report/unified_report.pdf](unified-report/unified_report.pdf)

Review date: July 17, 2026

Reviewed build:

- 57 pages, US Letter;
- SHA-256: 8e8741624a7bab7d15e2a539b07a541f33b15e2efe7b58e02ac3b851c117c4f8;
- abstract across PDF pages 1–2;
- contents across pages 3–4;
- the first empirical act starts on page 19, its primary-result subsection on page 20, and its first numeric table on page 21;
- consolidated synthesis begins on page 45;
- practitioner guidance appears on pages 46–47;
- conclusion and references begin together on page 53;
- the PDF contains 11 figures, 19 numbered tables, and 29 blue/green callouts;
- the PDF is not tagged for accessibility; and
- several plot fonts are embedded as Type 3 fonts, which weakens screen and print quality.

This proposal is based on a page-by-page visual review, text extraction, a source-to-PDF audit, and a check of the generated-artifact path. It addresses narrative flow, data presentation, explanation of the approach, conclusion design, practical guidance, accessibility, and reproducibility.

The intended readers are:

1. executives who need to make product, risk, staffing, and release decisions;
2. software engineers who need an implementable evaluation and deployment workflow; and
3. R&D researchers who need estimands, uncertainty, evidence boundaries, and reproducibility.

The rewrite must not change locked measurements to make the story cleaner. It must instead make every claim visibly proportional to the evidence.

---

## Executive verdict

<!-- REVISED (v2): the report is now FIVE bodies, not four — three retrospective acts PLUS one preregistered confirmatory study. -->
The report contains a valuable and practical result, but the current presentation hides it inside a 57-page audit-style narrative. It currently behaves like five bodies combined:

- an introductory tutorial on guards and statistics;
- a detailed experimental protocol;
- three retrospective empirical studies (Acts I–III);
- one preregistered confirmatory adaptation study (`sec:adaptation`, RQ1 supported / RQ2 not supported); and
- a reproducibility and limitations ledger.

That structure mainly serves a specialist who reads from beginning to end. It does not serve an executive who has five minutes, an engineer who needs to build a release gate, or an R&D reader who wants to distinguish established findings from exploratory controls.

The report should be rebuilt around one decision story:

> A benchmark winner is not automatically a deployment winner. Measure how adaptation changes the same checkpoint, test threshold behavior separately from ranking, evaluate domain requirements directly, and ship only a candidate that passes a frozen multi-gate contract.

The recommended result is:

- a 35–40-page main paper, with 38 pages as the working layout target;
- a two-page executive spread near the front;
- a 20–30-minute engineering route through the core evidence and decision workflow;
- appendices that retain the complete research and audit trail;
- a set of principal composite visuals (nine aspirational; build the 3–4 load-bearing ones first — see §4.4) instead of a long sequence of small tables and repeated callouts; and <!-- REVISED (v2): "nine" is aspirational, not a gate. -->
<!-- REVISED (v2): the confirmatory study already exists; the open item is a fresh, uninspected cohort. -->
- a conclusion that states the decision rule, the evidence boundary, and the next *prospective* (fresh, genuinely-uninspected-cohort) study — the preregistered confirmatory adaptation study is already in the report, not future work.

The paper should progress in this order:

> Decision problem → measurement design → observed failure → conditional repairs → domain boundary → deployment workflow → evidence limits.

The current “Act I / Act II / Act III” framing should be removed from section titles. It is memorable for the authors but forces readers to remember an internal sequence instead of the decision each section answers.

### Current-PDF flow and layout diagnosis

| PDF pages | Current reader experience | Rewrite decision |
|---|---|---|
| 1–2 | The abstract is roughly 700 words, mixes findings, methods, implications, and commercialization, then leaves most of page 2 empty. Several strongest claims exceed the evidence. | Replace with a 120–180-word abstract on page 1 and move decisions to the executive spread. |
| 3–4 | A subsection-level contents list consumes two pages while page 4 remains underfilled. It provides navigation but no answer. | Replace with a two-page evidence/action and reader-route spread; retain PDF bookmarks and a compact contents strip. |
| 5–18 | Fourteen pages of definitions, methods, dataset inventory, and related work precede the empirical answer. The page-7 study map uses a full page with small labels; the page-14 dataset table is nearly float-only. | Retain only five concepts in Section 2; move detailed methods, inventory, and literature to appendices. |
| 19–28 | The first result section begins low on page 19. The primary table is dense, a 20-seed table interrupts the result, an n=4 attractor analysis is visually prominent, and KL point estimates look coequal with supported evidence. | Lead with one paired-delta composite; move raw seeds and attractor analysis; visually subordinate KL as exploratory. |
| 29–34 | The repair section starts with method and fragments its main result across several tables. The 11.4% FPR miss is separated from the headline AP recovery. The domain section then begins at the bottom of page 34. | Combine composition evidence and threshold failure into one result module; start the domain section on a fresh page. |
| 35–44 | Mortgage construction precedes the decision result; the pipeline appears after it is discussed and consumes a page. Mortgage visuals are small and separated. ExpGuard uses a truncated y-axis and “tie” language, then leaves substantial white space. | Put the Mortgage construct and result together; move construction detail; use three aligned ExpGuard domain panels on a common scale. |
| 45–47 | The best synthesis object appears only after the evidence, but it contains the 15/15 bug, an exploratory KL recommendation, “fairness gate” language, and unsupported self-host economics. | Rebuild this logic as the corrected executive spread and compact engineering workflow near the front and in Section 6. |
| 48–53 | Five pages of limitations follow the recommendation. The four-sentence conclusion shares page 53 with reproducibility and references. | Put result-specific boundaries beside results, consolidate the ledger in Section 7, and give the conclusion its own page. |

<!-- REVISED (v2): the p.45 row describes the SUPERSEDED build — 15/15 and the KL overclaim are fixed and the synthesis object is now tab:guidelines. -->
> **Note (v2):** the "45–47" row above describes the superseded build. The synthesis object is now
> `tab:guidelines` (a styled 7-row "what we learned → evidence → guideline" table), not "Table 16"; its
> 15/15 → 15/20 and KL-SFT overclaim issues are already fixed, and Row 7 carries the confirmatory result.
> A new confirmatory section (`sec:adaptation`) has also been added and is currently mis-placed (input
> after Act I, opener references "Acts I–III"); the flow diagnosis must be re-run against the current PDF.

### Scope-safe title

The current title and subtitle can imply that SFT, KL-SFT, and composition were tested in high-compliance domains. The Mortgage and ExpGuard arms only evaluate the four untuned base checkpoints zero-shot.

Recommended title:

> **The Benchmark Chooses the Winner: How Tuning Changes Guards and Evaluation Changes Their Ranking**
>
> *Paired SFT and repair evidence, with base-only zero-shot evaluations in regulated domains*

<!-- REVISED (v2): the subtitle must also credit the confirmatory released-guard adaptation study (it adapts released guards, not base-only). -->
> **Subtitle addition (v2):** append a clause crediting the confirmatory arm, e.g. *"…and a preregistered
> adaptation study across ten released guards."* The confirmatory arm **does** adapt released/purpose-built
> guards (not base-only), so the "base-only, zero-shot" caveat applies to the *domain* arms, not to the
> whole paper — do not let the scope-safe framing erase the strongest contribution.

Repeat “base-only, zero-shot” in the abstract, study map, domain-section opening, figure captions, synthesis, and conclusion (scoping it to the domain arms).

### What the paper can safely claim

- On this fixed retrospective panel, ordinary SFT increases represented-source ranking performance.
- Transfer behavior is heterogeneous: 15 of 20 seed-level runs specialize, while five show uniform gain.
<!-- REVISED (v2): add the confirmatory result to the safe-claims list. -->
- On a fixed 10-checkpoint / 6-family panel, a preregistered confirmatory study shows the specialization tradeoff extends to already-purpose-built released guards (RQ1 supported, H_gain LCB +0.129), and that KL-SFT preserves transfer but is *not* a free improvement (RQ2 not supported: H_preserve LCB +0.035, but H_cost LCB −0.060 fails the −0.02 non-inferiority margin). These bounds are panel-conditional and still scored on the inspected Paper A manifest.
- On this panel, fixed output-space base+SFT composition recovers transfer relative to SFT and numerically has the highest observed minimum of represented and transfer AP among the main base, SFT, and fixed calibrated-composition candidates.
- That composition result is not the same as deployment readiness; the reported operating-point result still misses the illustrated 5% FPR target.
- The inspected KL-SFT cohort is an exploratory recipe control, not a validated default.
- Mortgage, Finance, Health Care, and Law results are base-only, zero-shot domain evaluations.
- The Mortgage protected-pair result is a small-sample directional invariance diagnostic, not a fairness or legal-compliance certification.
- A production decision requires ranking, operating-point, domain, reliability, and service gates; the report recommends this lifecycle but did not validate it end to end.

---

## 1. Design the paper for three readers

### 1.1 Reader contracts

| Reader | Time budget | Questions the paper must answer | Required exit state |
|---|---:|---|---|
| Executive | 5 minutes | What changed? Why does it matter? What decision is required? What remains uncertain? | Can approve an evaluation plan, assign an accountable owner, and understand why “pick the top benchmark model” is unsafe. |
| Software engineer | 20–30 minutes for the decision path; Appendix D for implementation | What do I compute? Which data is used when? What gates block release? What happens when no candidate passes? | Can implement the control-flow scaffold and knows where the complete schema, values, uncertainty, sample-size, and operations requirements live. |
| R&D researcher | Full paper plus appendices | What is the estimand? What was frozen? How was uncertainty computed? Which results are confirmatory, retrospective, exploratory, external, or pending? <!-- REVISED (v2): add "confirmatory" to the reader-contract question set. --> | Can audit each claim, reproduce each number, and design the next sealed study. |

The current PDF predominantly supports the third reader. The redesign should add the first two without weakening the audit trail.

### 1.2 One claim, three layers

Every major result should be presented in three layers.

| Layer | Reader | Form |
|---|---|---|
| Decision | Executive | One sentence: evidence → risk → action. |
| Implementation | Engineer | One figure or compact table, the operating consequence, and a release-gate implication. |
| Audit | Researcher | Estimand, paired comparison, interval, frozen artifact, sensitivity, and limitation. |

Example:

- Decision: “SFT can improve the benchmark it sees while weakening transfer; do not replace the base guard on represented AP alone.”
- Implementation: show each checkpoint’s represented and transfer movement, then require both absolute and relative gates.
- Audit: define the paired delta, fixed panel, seed aggregation, bootstrap unit, and pending overlap audit.

This pattern prevents two common failures: executives receiving statistical detail without a decision, and researchers receiving a simplified claim without its boundary.

### 1.3 Visible reader routes

Place a “How to read this report” strip on the executive spread:

- Executive core route: pages 1–3 only. Result-section decision boxes and the linked Conclusion are optional drill-down.
- Engineer decision route: executive core plus Sections 2–6.
- Engineer implementation route: decision route plus Appendix D.
- R&D route: full main text, Appendices A–E, artifact ledger, and reproduction command.

Use small route icons only if they remain accessible in grayscale and have text labels. Do not create three separate versions of the paper.

The executive core should be explicitly timed:

- 0:00–0:45 — page 1: title, abstract, and evidence boundary;
- 0:45–2:30 — page 2: six Evidence → Risk → Action cards;
- 2:30–3:40 — page 3 top: study map and plain-language metric legend;
- 3:40–4:40 — page 3 middle: five-branch candidate selector; and
- 4:40–5:00 — page 3 bottom: owners, no-ship rule, and optional reader routes.

### 1.4 The executive route

The executive spread must answer five questions without requiring knowledge of AP, bootstrap intervals, LoRA, or logits:

1. What business decision is at risk?
2. What did the study observe?
3. Which remedies have stronger versus weaker evidence?
4. Which regulated-domain results were actually measured?
5. What must be true before release?

Recommended executive actions:

- require each proposed guard to be compared with its own base;
- prohibit selection from one aggregate leaderboard score;
- fund a separate target-calibration set and one-time sealed acceptance set;
- keep the base when it passes every required gate;
- consider base+SFT composition only after measuring an SFT transfer regression, recalibrating the composition, and passing the two-pass service gate;
- establish a no-ship branch whenever no candidate passes all required gates;
<!-- REVISED (v2): KL-SFT now has a confirmatory RQ2 verdict — it is a tradeoff dial, not "research until confirmed." -->
- treat KL-SFT as a transfer-vs-represented tradeoff dial with a *confirmed* represented-source cost (RQ2 not supported: it clears transfer retention but fails the −0.02 non-inferiority margin), gated like any other candidate — not a free upgrade and not merely "future research";
<!-- REVISED (v2): RQ1 is the most decision-relevant executive takeaway — surface it as an action. -->
- do not assume a purpose-built / vendor guard is exempt: fine-tuning a released guard on your data specializes it too (RQ1 confirmed), so compare it with its own base as well;
- require domain SME review before calling a mortgage diagnostic a compliance gate; and
- require a serving-mode load and cost study before making self-host versus API claims.

If a required gate, evidence dependency, or fresh acceptance cohort is missing, the candidate is not acceptance-passed. Shadow use or human review may continue under a separate control, but release does not.

<!-- REVISED (v2): executives approve budget from cost/time, not risk alone — add an order-of-magnitude cost annotation. -->
> **Cost/time to run (order of magnitude, for staffing/budget approval):** building a one-time sealed
> acceptance set plus running the multi-gate battery is roughly a small number of engineer-weeks plus modest
> GPU-hours; anchor rough figures to the measured serving/latency basis (`tab:latency`) and the confirmatory
> study's own compute footprint. Attach a one-line "what this costs to run" note to each executive action so
> an executive can approve headcount and wall-clock, not just accept the risk framing.

### 1.5 The engineer route

The engineer should leave with:

- a five-role data split;
- a frozen candidate registry;
- candidate-specific calibration and thresholds;
- explicit required and not-applicable gates;
- a one-time blind acceptance procedure;
- a deterministic selector;
- failure actions for timeout, malformed output, model unavailability, policy mismatch, and drift;
- shadow, canary, rollback, and monitoring requirements; and
- a clear no-feasible-threshold outcome.

### 1.6 The R&D route

The R&D path should preserve:

- the paired same-checkpoint design;
- represented-source and dataset-held-out estimands;
- row/family and seed uncertainty;
- per-checkpoint results rather than panel means alone;
- the exact fixed panel and revisions;
- full sensitivity and raw tables;
- evidence provenance by study arm;
- overlap and contamination checks;
- artifact hashes and reproduction status;
- the completed preregistered confirmatory adaptation study on released/purpose-built guards (RQ1 supported, RQ2 not supported); and <!-- REVISED (v2): purpose-built adaptation is a completed confirmatory arm, not a prospective protocol. -->
- a prospective protocol for a fresh, genuinely-uninspected-cohort re-run and tuned domain evaluation.

<!-- REVISED (v2): the purpose-built/adaptation study is DONE, not future. -->
The purpose-built adaptation study is **complete** (`sec:adaptation`) and is the report's one
confirmatory piece; the R&D route must present it as completed confirmatory evidence (RQ1 supported, RQ2
not supported), not a roadmap item. The only genuinely-future work is a fresh, genuinely-*uninspected*
cohort (the current study still scores on the inspected Paper A manifest).

### 1.7 One confirmatory study among retrospective acts — what each reader must take from it

This report is deliberately mixed-tier: **Acts I–III are retrospective** (the panel was inspected during
method development — estimation-only, not confirmed), and **`sec:adaptation` is the one preregistered
confirmatory study** (estimands, decision rules, and the −0.02 non-inferiority margin locked before any
score existed). The rewrite must make this distinction unmissable for all three readers, without letting
the confirmatory badge bleed onto the retrospective acts or overclaiming the confirmatory study as
sealed-cohort.

- **Executive:** lead with the confirmed finding — *"CONFIRMED: fine-tuning any released/vendor guard on
  your data specializes it (you gain represented ranking, lose transfer); KL-SFT buys the transfer back
  but at a confirmed represented cost — a tradeoff dial, not a free upgrade."* The retrospective acts are
  the *why*; the confirmatory study is the *how-sure*. A confirmatory **not-supported** result (RQ2) is a
  finding, not a gap.
- **Software engineer:** RQ1 makes `starting_checkpoint_type` a decision axis — "compare every tune to its
  own base" applies to purpose-built guards too. RQ2 gives an implementable rule: KL-SFT is a quantified
  transfer-vs-represented dial, gated like any candidate, not shelved as "future." Validate the native
  verdict interface byte-for-byte and confirm the output head is LoRA-movable before trusting a near-zero
  delta (the Llama-Guard-3-1B pruned-head null cell).
- **R&D researcher:** the confirmatory study is the template for the next study — one-sided 97.5% LCBs, a
  declared non-inferiority margin, Bonferroni across RQ families, an explicit between-model-family-variance
  exclusion, and a leave-one-family-out sensitivity. Its bounds are **panel-conditional** and it still
  scores on the inspected manifest, so the residual open work is a genuinely-uninspected cohort.

---

## 2. Correct the claims before restyling

<!-- REVISED (v2): point to the Status update + companion review; several P0s below are already DONE and one is INVERTED. -->
> **Read the Status update at the top of this document and the companion review
> (`presentation-style-proposal-review.md`, 50 findings) before using this section.** Several P0 rows
> below are already applied (marked DONE), the preregistration row is INVERTED (the preregistration is
> real — propagate it, do not remove it), and the top remaining task is the abstract carve-out at
> `unified_report.tex:130`. Cross-walk each row against `report.md` (17 findings, most applied) rather than
> re-litigating settled work.

Presentation changes will amplify whatever the paper says. The following correctness changes therefore precede layout work.

### 2.1 Publication-blocking corrections

<!-- REVISED (v2): added a Status column; promoted the abstract carve-out to the top OPEN task; marked DONE rows; INVERTED the preregistration row; updated the reproduction count. -->

| Priority | Status | Current location | Problem | Required language or analysis |
|---|---|---|---|---|
| P0 | **OPEN — top task** | Abstract, `unified_report.tex:130` | The abstract still reads "All evidence here is retrospective and estimation-only," which contradicts the preregistered confirmatory study (`sec:adaptation`, `\input` into the report and advertised at :119–120). The limitations carve-out was applied but the abstract sentence was not (`report.md` rec #1 — flagged as the single most damaging class of issue). | Scope the "estimation-only" clause to Acts I–III and carve out the preregistered adaptation study; keep the panel-conditional / no-fair-lending clause. This is the highest-value remaining claim fix. |
| P0 | **DONE** (value fixed; residual infra ask only) | `unified_report.tex:314` (now `tab:guidelines` Row 2) | Fixed: the ratio now renders 15/20 via `\SpecializationSeedCount/\TotalSeedCount`. | Residual (still absent): add a source lint for the duplicated-macro wiring and a compiled-PDF regression test asserting the displayed 15/20; a generator-only value assertion cannot catch this wiring error. |
| P0 | **INVERTED — do NOT remove** | `tab:guidelines` caption / abstract / intro / ledger | The preregistration is real and load-bearing (`sec:adaptation`'s locked, non-HARKing claim registry). Acts I–III being retrospective is now a correct, intentional contrast, not a contradiction. | Do **not** remove the preregistration statement — *propagate* the confirmatory status through abstract/intro/ledger/conclusion (see the top OPEN row). Retain only the still-valid sub-point: define LCB/UCB on first use (now done in `sec:adaptation`) and keep two-sided-CI vs one-sided-LCB terminology consistent. |
| P0 | **OPEN** | `unified_report.tex:110, :255, :270` | Overlapping marginal CIs still call two guards "statistically tied" in three places, with no paired-difference interval. | Compute a paired difference interval. Without it, say "the ordering is unresolved." |
| P0 | **DONE** (overclaim removed) + reframed | `tab:guidelines` Row 3 | Fixed: Row 3 no longer calls KL-SFT "free/keeps specialization" and cross-refs the RQ2 non-inferiority failure. KL-SFT is exploratory on general checkpoints (Act I, n=4) but now has a **confirmatory** verdict on released guards (RQ2 not supported). | Present the general-checkpoint KL result as exploratory and the released-guard KL result as confirmatory-not-supported (a finding, not a gap). Do not recommend a default beta. |
| P0 | **DONE** | Title, abstract, study map, synthesis | Fixed: abstract/body label Mortgage and ExpGuard base-only, zero-shot. | Keep the base-only, zero-shot scoping on the domain arms; do not infer domain remedy efficacy from the general-safety panel. |
| P0 | **OPEN** | PDF p.30 versus p.33 | Composition is called the "safest single choice," but that is AP-based while the operating point misses the illustrated 5% FPR target (11.4%). | Say it numerically has the highest observed minimum of represented and transfer AP among the main base, SFT, and fixed calibrated-composition candidates on this panel. Reserve "safe" and "deployable" for candidates that pass every frozen gate. |
| P0 | **DONE** | Act I title and abstract | Fixed: body uses 15/20 and "transfer is heterogeneous." | Keep "represented-source ranking improves; transfer is heterogeneous and usually falls on this panel." |
| P0 | **OPEN** (count now stale) | reproducibility claim and [reproduce.py](unified-report/reproduce.py) | `generated/` now holds **17** consumed inputs (incl. `adaptation_macros.tex`, `tab_adaptation_gen.tex`, `klsft_macros.tex`); the newest adaptation/KL stages are not registered/regenerated. | Enumerate the consumed set programmatically; add deterministic adaptation, KL, and mortgage-composition stages; fail release checks on missing or drifted outputs. |
| P0 | **OPEN** | Figure check path | Check mode overwrites figures rather than producing a temporary candidate and comparing it. | Render to a temporary directory, hash-compare, report drift, and replace only in explicit write mode. |

### 2.2 Required evidence qualifications

| Priority | Current claim | Correct treatment |
|---|---|---|
| P1 | Prevalence reweighting is exact. | It is exact scenario analysis only under prior-probability shift with fixed class-conditional score behavior. It is not a forecast under covariate or concept drift. |
| P1 | Composition works because of AP diversity or error correlation. | This is a mechanism hypothesis. Measure disagreement and error correlation before presenting it as an explanation. |
| P1 | Delta-context is a fairness gate. | The public result has only three complete protected pairs. Call it a directional invariance diagnostic; do not make a legal or fairness certification. |
| P1 | Self-hosting is faster and cheaper than an API. | The current artifact measures batched per-row latency on one A100 at batch 16. It does not measure batch 1, queueing, concurrency, availability, API latency, or cost break-even. Replace the conclusion with a serving-study checklist. |
| P1 | The result is attributable to the fine-tune, while elsewhere no causal claim is made. | Say the paired contrast describes applying this fixed intervention to these fixed runs; it does not establish a mechanism or population-wide causal law. |
| P1 | Transfer is cleanly dataset-held-out. | Keep “formal overlap audit pending” beside the main result until n-gram and near-duplicate checks against the current v2 manifest are complete. |
| P1 | ExpGuard ranks the top two models. | Without paired per-domain difference intervals, report descriptive points and say ordering is unresolved. |
| P1 | Mortgage demonstrates a production compliance guard. | It is an LLM-judge diagnostic benchmark. It has not been adjudicated by mortgage SMEs or validated as a legal control. |

### 2.3 Reproduction contract to fix

<!-- REVISED (v2): the count is stale — generated/ now holds 17 consumed inputs incl. the adaptation + KL macros; enumerate programmatically rather than hard-coding a number. -->
The report now consumes **17** generated TeX inputs (do not hard-code this — enumerate the current consumed set programmatically), including the four newest — `adaptation_macros.tex`, `tab_adaptation_gen.tex`, `klsft_macros.tex`, and `tab_klsft_gen.tex` — which the reproduction entry point does **not** yet register or regenerate. The registry must add:

- a deterministic **adaptation** stage (owner e.g. `experiments/emit_adaptation_tex.py`) emitting `adaptation_macros.tex` + `tab_adaptation_gen.tex`, with both included in the byte-identity assertion;
- a deterministic **KL-SFT** stage emitting `klsft_macros.tex` + `tab_klsft_gen.tex`; and
- the **mortgage-composition** table generated from the frozen public index and manifest (currently an orphan).

Feasibility upside: [reproduce.py](unified-report/reproduce.py) needs **no GPU and no network** (it works from committed scores), so these stages are achievable on CPU despite all compute VMs being torn down — Phase 4 can run as an independent parallel workstream. The `expguard`, `sftsft`, and `latency` PENDING branches and the check-mode write path (render to temp, hash-compare, write only in explicit `--build`) still need fixing.

The release check should:

1. enumerate all generated TeX inputs consumed by [unified_report.tex](unified-report/unified_report.tex) and included section files;
2. assert exact equality with the registered output set;
3. regenerate into a temporary tree;
4. byte- or hash-compare tables and figures;
5. fail on drift, missing artifacts, PENDING, PINNED-ENV REQUIRED, or skipped stages;
6. print the artifact source and hash for every principal claim; and
7. compile twice, then verify page-count and unresolved-reference gates.

This requires new output-directory controls in [unified-report/figures/make_figures.py](unified-report/figures/make_figures.py) and the Mortgage generators; check mode must never write into the committed tree. Propagate and check every subprocess return code before accepting an existing output. Generate mortgage_composition_table.tex from mortgage-benchmark/benchmark/v1_hmda2022/public_index.json and verify that source against its MANIFEST.json and CHECKSUMS.txt before emission.

Generator-owned captions must be corrected in their generators, not only in generated TeX. Relevant owners include:

- [experiments/analyze_klsft.py](../experiments/analyze_klsft.py);
- [unified-report/reproduce.py](unified-report/reproduce.py);
- [unified-report/figures/make_figures.py](unified-report/figures/make_figures.py); and
- [base-adapter-composition/code/build_pilot_artifacts.py](base-adapter-composition/code/build_pilot_artifacts.py).

---

## 3. Recommended narrative architecture

### 3.1 Narrative spine

The paper should answer one question at a time.

<!-- REVISED (v2): 8 questions; Q5 is the new preregistered confirmatory question. -->

1. Why is selecting the leaderboard winner a deployment risk?
2. How does the paired experiment measure the within-checkpoint effect of the fixed SFT recipe?
3. What changes on represented sources vs. held-out datasets? (retrospective, Act I)
4. Which repairs recover transfer, and how strong is each? (composition; KL-SFT as an Act I control)
5. **Does the specialization tradeoff — and KL-SFT's transfer recovery — also hold on already-purpose-built released guards? [Preregistered confirmatory: RQ1 supported, RQ2 not supported]**
6. Does a general-safety score answer regulated-domain compliance questions?
7. How should a team select, gate, deploy, and monitor a guard?
8. What remains unproven, pending, or future (a fresh, genuinely-uninspected-cohort confirmation)?

Each section should end with a three-line box:

- Evidence: what was measured.
- Decision: what the reader can do.
- Boundary: what the result does not establish.

### 3.2 Page budget

Count the main paper from the title page through the conclusion. References and appendices are outside this 35–40-page target unless a venue explicitly counts them.

<!-- REVISED (v2): re-derived to include the confirmatory adaptation study (RQ1 folded into §3, RQ2/KL verdict into §4, plus a dedicated §4b confirmatory subsection with its own badge). -->

| Part | Designed | Reserve | Pages | Purpose |
|---|---:|---:|---:|---|
| Title + abstract | 1 | 0 | p.1 | Scope-safe claim; **four** findings incl. the confirmatory result |
| Executive spread + reader map | 2 | 0 | pp.2–3 | Five-minute route; confirmed-vs-directional split leads |
| 1. Leaderboard winner ≠ deployment decision | 2 | +1 | pp.4–6 | Decision problem, scope |
| 2. How to read the experiment | 3 | 0 | pp.7–9 | Minimum stats + SLM concepts |
| 3. SFT improves represented ranking; transfer heterogeneous (+ RQ1 confirmatory module) | 6 | +1 | pp.10–16 | Primary evidence; RQ1 released-guard confirmation |
| 4. Two conditional repair paths (+ RQ2 KL-SFT confirmatory verdict) | 5 | 0 | pp.17–21 | Composition primary; RQ2 is the *authoritative* KL-SFT verdict; n=4 Act I control demoted to corroboration |
| 4b. Confirmatory adaptation study (badge: *preregistered, panel-conditional*) | 2 | 0 | pp.22–23 | RQ1+RQ2 design, `tab:adaptation`, null-cell disclosure, panel-conditional limits |
| 5. Base-only, zero-shot domain evaluations | 7 | +1 | pp.24–31 | Mortgage + three ExpGuard panels |
| 6. Engineering decision guide | 4 | 0 | pp.32–35 | Gates, selector, lifecycle |
| 7. Evidence boundaries + reproducibility | 3 | 0 | pp.36–38 | Ledger, dependencies, reproduction |
| 8. Conclusion | 1 | 0 | p.39 | Workflow, not winner |
| **Total** | **36** | **4** | **≈39–40 pages** | Confirmatory content added |
| **Hard stop** |  |  | **42 pages** | Contingency only; if 40 is a hard venue limit, keep 4b as a labeled subsection inside §3–§4 and push the full 10×3 `tab:adaptation` grid to Appendix C |

Add to the **"do not cut"** list (§3.3): *the confirmatory RQ1/RQ2 verdict and `tab:adaptation`* — a confirmatory result outranks retrospective raw tables in the cut order.

If references must fit inside a 40-page venue limit, first measure the rebuilt bibliography. Then define the main-body cap as 40 minus the measured reference pages. Do not silently squeeze figures or fonts.

### 3.3 Overage cut order

If the rebuilt PDF exceeds 40 pages, cut in this order:

1. duplicate prose already stated in a result box;
2. extended related-work discussion;
3. the attractor figure and full mechanism speculation;
4. raw per-seed and per-benchmark tables;
5. full KL mechanics and beta sensitivities;
6. composition ablations and secondary controls;
7. the mortgage construction pipeline and policy-card inventory;
8. expanded engineering YAML and monitoring examples.

Do not cut:

- the two-page executive spread;
<!-- REVISED (v2): a confirmatory verdict outranks retrospective raw tables in the cut order. -->
- the confirmatory RQ1/RQ2 verdict and `tab:adaptation`;
- paired per-checkpoint deltas and uncertainty;
- the composition operating-point failure;
- the Mortgage result and its label caveat;
- separate Finance, Health Care, and Law panels;
- the overlap-audit status;
- the no-ship branch;
- the evidence ledger; or
- the reproduction command.

### 3.4 Front matter

#### Title and subtitle

Use the scope-safe title above.

#### Abstract

<!-- REVISED (v2): add a seventh move for the preregistered confirmatory result. -->
The abstract should use seven short moves:

1. deployment problem;
2. paired design;
3. represented and transfer result;
4. composition and exploratory KL result;
5. base-only domain result;
6. **preregistered confirmatory adaptation result (released guards; RQ1 supported, RQ2 not supported)**;
7. engineering implication and evidence boundary.

It should fit on page 1. Remove tutorial definitions, detailed latency claims, and future-roadmap material.

#### Evidence, Risk, Action: The Five-Minute Version

Use a two-page spread rather than a conventional contents page.

Page 2:

- a one-line micro-legend above the cards: AP = ranking, not cutoff; FPR = legitimate requests falsely blocked; represented = sources represented in tuning; transfer = datasets excluded from tuning;
- a CONFIRMED (preregistered) lead card for RQ1, then six Evidence → Risk → Action cards in a 2×3 grid (or drop the weakest card to keep a strict 2×3); <!-- REVISED (v2): lead with the confirmed RQ1 card above the 2×3 grid. -->
- one compact evidence-status tag per card and one spread-level boundary strip linked to the detailed sections;
- no more than 45 words per card;
- one decision per card; and
- no full confidence intervals.

Page 3:

<!-- REVISED (v2): split "sealed" into two concepts — the adaptation study is preregistered-analysis, NOT sealed-cohort. -->
- top 35%: compact study map, two distinct definitions — *"preregistered analysis = estimands/decision rules/margin locked before scoring"* (what `sec:adaptation` **is**) and *"sealed cohort = data held back and opened once on uninspected rows"* (the still-future step) — and evidence badges;
- middle 45%: five-branch candidate selector;
- bottom 20%: evidence status, accountable owners, no-ship rule, and optional reader routes.

A short linked contents strip can remain at the bottom or in the PDF bookmarks. The current two-page detailed contents should not occupy prime narrative space.

<!-- REVISED (v2): seed the cards from tab:guidelines' 7 rows (do NOT "replace Table 16"); lead with the CONFIRMED RQ1 card. -->
Recommended card copy — **seed these from the seven rows of `tab:guidelines`** (the existing styled synthesis object) rather than building from scratch, and lead with the confirmed finding. Hold the page-2 grid to six cards plus a lead banner (or drop the weakest card to keep a 2×3 grid):

| Card | Evidence | Risk | Action |
|---|---|---|---|
| **CONFIRMED — released guards (RQ1, preregistered)** | On a fixed 10-checkpoint / 6-family panel, fine-tuning an already-purpose-built released guard still specializes it (H_gain LCB +0.129): you gain represented ranking, lose transfer. | Assuming a vendor/purpose-built guard is exempt hides the same tradeoff. | Compare every tune to its own base — released guards included; do not treat a purpose-built guard as exempt. |
| Benchmark choice | A model can rank differently across represented, transfer, and domain datasets. | One aggregate winner can hide a deployment failure. | Select against the declared deployment matrix, not one leaderboard. |
| Ordinary SFT | Represented ranking +32.3 AP points; transfer −5.9 on average; 15/20 runs moved represented-up/transfer-down. | A tuning gain can conceal transfer loss. | Compare every tune with its own base on identical rows. |
| Operating point | Pooled FPR 4.3%→17.0%; hard-attack recall 78%→60%. | Better ranking does not preserve the production cutoff. | Recalibrate every candidate and require threshold gates. |
| Composition repair | Versus SFT: transfer +7.6 AP points, represented −1.9; 4/4 recover, 2/4 beat base; transfer macro FPR 11.4% at a 5% target. | A repair can still fail release criteria. | Recalibrate and gate it; do not infer dominance over base. |
| Domain evidence | Mortgage and ExpGuard show domain-dependent orderings, but all reported domain candidates are untuned bases evaluated zero-shot. | General-safety evidence can be mistaken for compliance evidence. | Require separate target-domain validation and domain ownership. |
<!-- REVISED (v2): KL-SFT now has a confirmatory RQ2 verdict — it is confirmed NOT a free upgrade. -->
| KL-SFT (confirmed not free) | Preregistered RQ2: KL-SFT preserves transfer (LCB +0.035) but its represented cost (LCB −0.060) fails the −0.02 non-inferiority margin — confirmed *not* a free improvement. The overlap audit, domain tuning, SME adjudication, and serving study remain incomplete. | A confirmed tradeoff — or a missing dependency — can be promoted by presentation alone. | Treat KL-SFT as a tradeoff dial, not a free upgrade; use explicit evidence badges; missing required evidence means no ship. |

Put the AP, FPR, represented, and transfer micro-legend above the page-2 cards, before the terms are used.
<!-- REVISED (v2): do not label the adaptation study "sealed"; it scores on the inspected manifest. -->
Put both definitions beside the page-3 evidence badges — *"preregistered analysis"* (the adaptation study locks its estimands/decision rules/−0.02 margin before scoring but still scores on the inspected Paper A manifest) versus *"sealed cohort = held back and opened once"* (reserved for the roadmap) — so the confirmatory badge is not read as a sealed-cohort claim.

The rounded point estimates belong on the cards because they make the material decision visible. Full intervals belong in the linked result visuals:

- ordinary SFT represented change +0.3234 [0.2647, 0.3690] and average transfer change −0.0589 [−0.0837, −0.0321]; and
- composition versus SFT transfer change +0.076 [0.058, 0.093] and represented change −0.019 [−0.031, −0.010].

<!-- REVISED (v2): the Act I gain and the confirmatory gain are different numbers on different panels — never merge them. -->
> **Do not conflate the two represented-gain numbers.** The Act I represented gain (+0.3234 → +32.3 AP
> points) is on the **4-checkpoint** retrospective panel; the confirmatory represented gain (H_gain +0.174,
> LCB +0.129) is on the **10-checkpoint / 6-family** released-guard panel. They are different estimands on
> different panels and must never be averaged, merged, or presented as one number. The Act I number leads the
> retrospective SFT card; the +0.174 number leads the confirmatory RQ1 card.

The page-3 selector must show five outcomes:

1. keep the base when it passes every required gate;
2. consider SFT only for a stated represented-source need;
3. treat KL-SFT as a tradeoff dial with a confirmed represented-source cost (RQ2 not supported), gated like any candidate — not a future sealed candidate; <!-- REVISED (v2): KL-SFT has a confirmed cost, not a future/sealed status. -->
4. use composition only to repair a measured SFT regression when recalibration and the two-pass service gate pass; and
5. do not release when a required gate or evidence dependency is missing.

### 3.5 Recommended sections and subsections

<!-- REVISED (v2): give the confirmatory study an explicit home — RQ1 into §3, RQ2 into §4, plus a §4b subsection; fix its mis-placement in migration. -->
> **Where the confirmatory study lives (consistent with the revised §3.1 spine and §3.2 budget).** Fold the
> RQ1 result (SFT specializes released guards too) into Section 3 as a confirmatory module, make the RQ2
> verdict (KL-SFT not free) the *authoritative* KL-SFT result in Section 4 (demoting the n=4 Act I control to
> corroboration), and gather the design + `tab:adaptation` into a short **§4b "Confirmatory adaptation study"**
> subsection carrying the *preregistered, panel-conditional* badge. Migration note: `sections/act-adaptation.tex`
> is currently `\input` right after Act I even though its opener references "Acts I–III" (`report.md` flow-2) —
> fix its placement during the source migration, do not merely move the file.

#### 1. Why a Leaderboard Winner Is Not a Deployment Decision

1.1 One score hides three deployment questions
1.2 A general-safety failure and a domain-policy failure
1.3 What this study measures—and what it does not
1.4 Contribution relative to the closest work

Section purpose:

- open with one general-safety example and one polite-but-noncompliant mortgage example;
- show why ranking, threshold behavior, and domain fit are different questions;
- disclose that the model panel is fixed and retrospective;
- give only the closest related-work contrast.

Move the full literature review to Appendix A.

#### 2. How to Read the Experiment

2.1 From two verdict logits to one guard score
2.2 Compare each tuned checkpoint with its own base
2.3 Represented sources versus dataset-held-out transfer
2.4 Ranking, calibration, and deployment are different tests
2.5 Fixed panel, data roles, and uncertainty

Section purpose:

- teach only the concepts required to interpret the results;
- retain the current AP worked example;
- use one diagram for base → SFT → candidate scoring;
- explain paired deltas before bootstrap detail;
- introduce evidence badges.

Full equations, LoRA recipe, benchmark inventory, prompt rendering, and bootstrap implementation move to Appendix B.

#### 3. On This Fixed Panel, Ordinary SFT Improves Represented-Source Ranking; Transfer Is Heterogeneous

3.1 The represented-source movement is consistent
3.2 Fifteen of twenty seed runs specialize
3.3 Ranking gains do not guarantee an acceptable threshold
3.4 Prevalence changes the operating picture
3.5 Interpretation boundary: fixed panel and pending overlap audit
3.6 Confirmatory module (RQ1): ordinary SFT specializes released guards too [preregistered, panel-conditional] <!-- REVISED (v2): RQ1 confirmatory module folded into Section 3. -->

Section purpose:

- show the primary paired movement immediately;
- show per-checkpoint and seed heterogeneity;
- translate ranking movement into missed positives and false positives at a declared threshold;
- state the prior-shift assumption beside the prevalence plot;
- label the overlap audit as pending.

The attractor analysis should be one cautious sentence in the main text. Full analysis and raw seed tables move to Appendix C.

#### 4. Two Conditional Repair Paths: Recover After Tuning or Regularize During Tuning

4.1 The repair decision
4.2 Base+SFT composition recovers transfer on this panel
4.3 Recovery in AP does not repair the operating point automatically
4.4 KL-SFT verdict: confirmatory RQ2 (preserves transfer, fails non-inferiority — not free); n=4 Act I control corroborates <!-- REVISED (v2): RQ2 is the authoritative KL-SFT verdict, not "promising but exploratory." -->
4.5 What evidence would promote either method
4b. Confirmatory adaptation study (RQ1+RQ2 design, `tab:adaptation`, null-cell disclosure, panel-conditional limits) [preregistered, panel-conditional]

Section purpose:

- lead with composition because it has the more complete fixed-panel comparison;
- compare composition with SFT and the original base;
- display the 11.4% FPR versus 5% target;
<!-- REVISED (v2): RQ2 confirmatory verdict supersedes the exploratory inset. -->
- present the confirmatory RQ2 verdict as the authoritative KL-SFT result (preserves transfer but fails the −0.02 non-inferiority margin — not free), and keep the n=4 Act I KL result as a visually subordinate corroborating inset, clearly distinct from the confirmatory verdict;
- state promotion gates and fallbacks.

Avoid “compose, do not tune” as a universal command. The evidence supports a conditional candidate, not a general law.

#### 5. Base-Only, Zero-Shot Domain Evaluations: Mortgage, Finance, Health Care, and Law

5.1 Scope: domain evaluation without adaptation
5.2 General harm and domain-policy violations are different constructs
5.3 Mortgage: the dual-label quadrant
5.4 Mortgage results and directional invariance
5.5 Finance: external expert-annotated prompts
5.6 Health Care: external expert-annotated prompts
5.7 Law: external expert-annotated prompts
5.8 What these domain results establish—and what they do not

Section purpose:

- make all four domains visible in the main text;
- keep Mortgage and ExpGuard separate by construct and evidence tier;
- show Finance, Health Care, and Law as three panels, not one aggregate;
- state that no tuned or composed domain candidates were tested;
- prevent a general-safety ranking from being read as compliance readiness.

#### 6. Engineering Decision Guide: Gate Candidates, Not Leaderboards

6.1 Freeze the deployment contract
6.2 Give each dataset one role
6.3 Calibrate and freeze every candidate
6.4 Open the blind acceptance set once
6.5 Apply absolute, relative, domain, and service gates
6.6 Select deterministically—or do not ship
6.7 Shadow, canary, monitor, and roll back

Section purpose:

- turn the findings into a small executable workflow;
- include one illustrative mini-contract;
- include one worked candidate decision;
- show the no-feasible-threshold branch;
- label the workflow “recommended; not validated end to end by this study.”

The full contract schema, sample-size planning, multiplicity policy, load-test plan, and monitoring templates move to Appendix D.

#### 7. Evidence Boundaries and Reproducibility

7.1 Evidence matrix by study arm
7.2 Open dependencies and claim fallbacks
7.3 Exact artifact and reproduction contract
7.4 Next studies: a fresh, genuinely-uninspected cohort <!-- REVISED (v2): the confirmatory analysis is done; the next study is a fresh cohort. -->

Section purpose:

- consolidate limitations instead of repeating them everywhere;
- keep only result-specific caveats beside results;
<!-- REVISED (v2): the evidence matrix must carry the confirmatory arm as a distinct, tagged row. -->
- give the confirmatory adaptation arm a distinct row in the by-arm evidence matrix (RQ1 supported, RQ2 not supported; tagged "preregistered confirmatory, panel-conditional"; note the general-vs-purpose descriptive/blocked limit and the null Llama-Guard cell);
- make pending work visible and actionable (scope the residual gap to a fresh, uninspected cohort, not "any confirmation");
- link every main claim to an artifact owner.

#### 8. Conclusion: A Workflow, Not a Winner

The conclusion should occupy its own page and contain no new evidence.

#### Appendices

A. Related Work
B. Methods, Frozen Objects, and Full Statistical Protocol
C. Complete Results, Sensitivities, and Mechanism Diagnostics
D. Engineering Playbook and Templates
E. Evidence Ledger, Limitations, and Research Roadmap

---

## 4. Make the data presentation decision-oriented

### 4.1 One result module everywhere

Every principal result should use the same visual grammar:

1. Question
2. Comparison
3. Estimate
4. Uncertainty
5. Evidence status
6. Operational consequence
7. Boundary

Example:

> Question: Does composition recover held-out transfer relative to SFT?
> Comparison: Same checkpoint and same rows, composition minus SFT.
> Estimate: show checkpoint deltas and panel summary.
> Uncertainty: paired interval.
> Status: retrospective fixed-panel result.
> Consequence: keep composition as a candidate, then recalibrate it.
> Boundary: the illustrated threshold still misses the target FPR.

Readers should not need to infer the practical meaning from a caption or several pages of prose.

### 4.2 Evidence badges

Use a small, text-bearing badge beside each headline:

| Badge | Meaning |
|---|---|
| Preregistered confirmatory (panel-conditional) | Estimands, decision rules, and the non-inferiority margin were locked before scoring (non-HARKing claim registry); the strongest tier here. Caveat: bounds are conditional on the fixed model panel and still scored on the inspected manifest — confirmatory, but **not** sealed-cohort. Used only by `sec:adaptation`. |
| Retrospective fixed-panel result | Result from the declared retrospective panel with locked, reproducible artifacts; not a prospective confirmation. |
| Exploratory inspected cohort | Useful estimate, but not a promotion result. |
| Base-only, zero-shot | No SFT, KL-SFT, or composition was evaluated in this arm. |
| Directional, small n | Descriptive evidence without a deployment-grade gate. |
| Recommended workflow | Engineering guidance inferred from the findings, not tested end to end. |
| Pending dependency | Claim remains qualified until the named audit or artifact exists. |

Do not use “validated,” “safe,” “fair,” or “production-ready” as badge labels.

### 4.3 Four-domain facts that must remain separate

| Domain | Source and construct | Reported rows | Positive rows | Current base AP range | Allowed interpretation |
|---|---|---:|---:|---:|---|
| Mortgage | HMDA-grounded, dual general-safety and mortgage-policy labels assigned by an LLM judge | Public test n=146 | Domain positive n=81; general positive n=6; G0/D1 n=75 | Domain AP 0.672–0.851 | Diagnostic performance on this frozen public test; not legal certification. |
| Finance | ExpGuard, external expert-annotated single-label prompts | n=964 | n=576 | 0.887–0.958 | Base-only descriptive domain ranking; use paired differences before winner/tie language. |
| Health Care | ExpGuard, external expert-annotated single-label prompts | n=771 | n=393 | 0.892–0.955 | Base-only descriptive domain ranking; not clinical safety validation. |
| Law | ExpGuard, external expert-annotated single-label prompts | n=540 | n=287 | 0.868–0.958 | Base-only descriptive domain ranking; not legal advice or compliance certification. |

Mortgage also has three complete protected pairs in the public test. Show their observed direction, sample count, and limitation together.

Never average or pool Mortgage and ExpGuard values. They use different constructs, label sources, and evidence tiers.

The committed ExpGuard artifact provides per-domain AP points but not per-domain confidence intervals. The main paper has two acceptable choices:

1. compute and register paired, per-domain uncertainty as a new artifact-derived analysis; or
2. show descriptive points, row count, positive count, and “ordering unresolved—no paired interval.”

Do not hide the three domains behind the aggregate ExpGuard result.

### 4.4 Principal composite visuals (nine aspirational; build the load-bearing ones first)

<!-- REVISED (v2): "nine composites" is aspirational, not a gate; build 3-4 load-bearing ones first and restyle existing act figures for the rest. -->
> **Treat "nine composites" as aspirational, not a Definition-of-Done gate.** Build the 3–4 load-bearing
> ones first — the executive evidence/action map (#1), the SFT paired-delta + seed quadrant (#3), the
> composition-with-FPR-miss result (#5), and the three-domain plot (#7) — and reuse/lightly restyle existing
> act figures for the rest. Several composites need per-domain paired CIs the committed artifacts do **not**
> provide (see §4.3); use the descriptive-dots "ordering unresolved" fallback rather than computing new
> per-domain CIs. The real cost here is figure engineering plus new bootstrap analyses (weeks), so gate on
> "every main claim has one legible visual (composite where built, restyled existing figure otherwise)," not
> on all nine.

| No. | Composite | Main message | Required fallback |
|---:|---|---|---|
| 1 | Executive evidence/action map | What was observed, why it matters, and what to decide | Text cards remain readable without color. |
| 2 | Study and data-role strip | Which models, transformations, datasets, and evidence tiers belong to each arm | Explicitly label retrospective, base-only, and external arms. |
| 3 | SFT paired-delta plus seed quadrant | Represented gains are consistent; transfer is heterogeneous | Keep 15/20 denominator and pending-overlap badge. |
| 4 | Operating KPI plus prevalence scenario | AP movement is not a threshold guarantee | State threshold source and prior-shift assumption. |
| 5 | Composition result with KL verdict | Composition is the supported repair candidate; KL-SFT has a confirmatory RQ2 verdict (preserves transfer, fails non-inferiority — not free), with the n=4 Act I result as a subordinate exploratory inset <!-- REVISED (v2): KL now carries a confirmatory verdict, not just an exploratory inset. --> | If confirmatory RQ2 detail must move, keep the "not free" verdict + margin in-text; reduce the Act I inset to points or Appendix C. |
| 6 | Mortgage quadrant plus base results and invariance inset | General-safety and mortgage-policy labels disagree on important rows | State LLM-judge source and n=3 protected pairs. |
| 7 | Three-panel Finance / Health Care / Law plot | Model ordering is domain-dependent | If paired intervals are absent, use descriptive dots and no winner language. |
| 8 | Frozen candidate-gating lifecycle | How an engineering team turns evidence into a release decision | Include no-ship and rollback branches. |
| 9 | Evidence matrix plus reproduction box | Which claims are confirmatory (preregistered, panel-conditional), locked/retrospective, exploratory, pending, or future <!-- REVISED (v2): add "confirmatory" as an evidence-matrix category. --> | Link every row to an artifact or protocol; give the confirmatory RQ1/RQ2 rows their own tier. |

Cap the main text at ten objects larger than one-third of a page. Use only three to five compact tables.

<!-- REVISED (v2): the caps collide with the confirmatory objects — reconcile main vs appendix explicitly. -->
> **Reconcile the object/table caps against the confirmatory content.** Once `tab:guidelines`,
> `tab:adaptation`, and a confirmatory RQ1/RQ2 composite are counted, the 3–5-table and ≤10-object caps are
> tight. Explicitly designate main-text objects (candidates: the executive map / `tab:guidelines`,
> `tab:adaptation`, the worked-candidate table, one gate table, and the confirmatory composite) versus
> Appendix D (data-role, failure, full gate family, contract schema), and move a retrospective composite
> (e.g. #4 KPI + prevalence detail) to the appendix so the object budget still closes.

### 4.5 Number language

Use the same grammar for every result:

> Starting value → ending value; paired change; interval; practical meaning.

Avoid:

- naked percentages with no denominator;
- “large,” “small,” or “significant” without a reference;
- rank-only statements;
- a model winner without a paired difference;
- pooled values that erase domains or checkpoints; and
- false precision beyond what the artifact supports.

Where a result is descriptive, say “numerically higher on this sample,” not “better.”

### 4.6 Uncertainty rules

- Use paired intervals for same-row candidate comparisons.
- Use model- or family-aware summaries only for the population actually represented.
- Do not infer equivalence from overlapping marginal intervals.
- Do not turn a two-sided descriptive interval into a one-sided promotion gate after seeing results.
- Put the interval next to the estimate, not in a separate appendix-only table.
- Show the sample unit and denominator in the caption.
- Distinguish row uncertainty, seed uncertainty, and model-panel uncertainty.
<!-- REVISED (v2): fold the adaptation study's design vocabulary in as the reference template for the next study. -->
- State the panel-conditional caveat explicitly: a bootstrap that holds model identities fixed carries **no between-model-family variance**, so a single dominant family can drive an equal-family mean — require a leave-one-family-out sensitivity (the null Llama-Guard cell moved H_gain 0.174 → 0.208 when dropped).

For a production gate, define the estimand, direction, bound, tolerance, confidence level, and failure action before opening the acceptance set.

<!-- REVISED (v2): use the executed confirmatory study as the reference design, not a reinvented subset. -->
> **Reference template — the adaptation study already executed this design.** For the next confirmatory
> study, template on `sec:adaptation`: one-sided 97.5% lower bounds, an explicit non-inferiority margin
> (−0.02), Bonferroni multiplicity across RQ families, the between-model-family-variance exclusion above, and
> a leave-one-family-out sensitivity — all declared before unblinding.

### 4.7 Table rules

- One decision per table.
- No more than seven visible columns in the main paper.
- Put candidate names in rows and decisions in columns when possible.
- Replace repeated decimals with point plus interval formatting.
- Use em dashes only for truly not-applicable cells; explain missing evidence explicitly.
- Put units in headers.
- Keep full precision in generated artifacts, not the reader-facing table.
- Never let a caption carry the only scope qualification.

### 4.8 Figures and captions

Each caption should answer:

1. What is compared?
2. What does movement or position mean?
3. What should the reader conclude?
4. What should the reader not conclude?

Target:

- at least 9 pt text inside figures;
- captions no longer than about 90 words or four lines where layout permits;
- direct labels instead of remote legends;
- consistent checkpoint colors and method shapes;
- grayscale-safe contrast;
- no red/green-only meaning; and
- no screenshot-style pipeline with unreadable labels.

The current full-page experiment and mortgage-pipeline diagrams should be redrawn as simpler vector diagrams or moved to appendices.

### 4.9 Callouts

Retain callouts, but give them stable semantics:

- Evidence: measured result.
- Decision: recommended action.
- Boundary: limitation or non-claim.
- Definition: one concept needed by a basic-statistics reader.

Use eight to ten editorial callouts in the main paper, each no more than about 60 words. Repeated callouts currently interrupt flow and weaken hierarchy.

### 4.10 Page composition and accessibility

The current PDF has several underfilled pages, isolated tables, and orphaned section headings. The rebuild should:

- remove forced H float placement from the main narrative;
- use top or bottom floats with declared barriers only at major section boundaries;
- keep a heading with at least two body lines;
- avoid a table alone in the top half of a mostly empty page;
- avoid introducing a new major section at the bottom of a page;
- use a baseline grid and consistent vertical rhythm;
- add PDF bookmarks and logical heading structure;
- tag figures, tables, headings, and reading order;
- provide alt text or an accessible text equivalent for every principal figure;
- set document language and metadata;
- verify link text is meaningful; and
- test keyboard navigation, screen-reader extraction, and grayscale printing.

<!-- REVISED (v2): demote blocking PDF/UA to best-effort; ship the HTML companion as the a11y deliverable; do NOT switch off Tectonic. -->
Run an accessibility feasibility spike before the layout rebuild: produce one page containing a tagged heading, paragraph, table, figure alternative, and link. The build is Tectonic (XeTeX-based), whose LaTeX tagging path is immature — a machine-passing PDF/UA validator on a 38-page multi-panel document is specialist, multi-week work. Therefore **full PDF/UA conformance is best-effort, not a blocking gate.** Ship the accessible **HTML companion** as the a11y deliverable. For the PDF, require only: bookmarks, document language, meaningful link text, grayscale-safe / non-Type-3 figures, and alt-text stubs. Attempt tagging as a time-boxed Phase-0 spike; **if the smoke page cannot pass, drop the PDF/UA target — do not switch the whole compiler off Tectonic.** An accessible HTML edition remains the primary a11y artifact.

For Matplotlib PDFs, set pdf.fonttype to 42 and make the build fail when pdffonts reports Type 3 plot text.

---

## 5. Explain the approach with the minimum necessary statistics

### 5.1 A five-term glossary

Place a compact glossary near Section 2:

| Term | Plain-language definition |
|---|---|
| Base guard | The checkpoint before this study’s adaptation recipe. |
| SFT guard | The same checkpoint after supervised fine-tuning on the declared rows. |
| Represented source | A benchmark source represented in the adaptation data. |
| Dataset-held-out transfer | A benchmark dataset whose rows were withheld from adaptation; this is not a claim of universal out-of-distribution generalization. |
| Average precision | A ranking summary over all thresholds; it does not supply a production cutoff. |

Define LoRA, logit, calibration, bootstrap, and KL only where they first matter.

### 5.2 Explain the paired design before the metric

Use a three-box diagram:

> Same checkpoint → unchanged base score
> Same checkpoint + fixed SFT recipe → SFT score
> Same evaluation rows → paired difference

Then state:

> This comparison describes the effect of applying this fixed recipe to these fixed checkpoint runs. It does not identify why the behavior changed or establish a law for all guards.

This is more intuitive than opening with the bootstrap or a model leaderboard.

### 5.3 Separate four questions

The paper should explicitly distinguish:

1. Ranking: does the candidate order positives above negatives?
2. Calibration: does a score have a stable meaning?
3. Operating point: can a threshold meet the required false-positive and recall constraints?
4. Deployment: does the complete service pass domain, reliability, latency, cost, and governance gates?

AP answers only the first question.

### 5.4 Explain the two repairs without implying equivalence

Composition:

- combines output scores from the base and SFT candidates;
- is evaluated across the fixed panel;
- recovers transfer relative to SFT on this panel;
- requires its own calibration and threshold;
- is supported as a candidate, not automatically as a deployable service.

KL-SFT:

- adds a penalty intended to keep the adapted model close to a frozen reference on selected positions;
- is currently based on an inspected cohort;
- anchors supervised verdict/completion positions, not the model’s entire behavior;
- needs achieved-KL diagnostics and a new sealed confirmation;
- should not be promoted as a default beta.

### 5.5 Keep evidence tiers explicit

<!-- REVISED (v2): add the preregistered confirmatory adaptation study as the seventh, strongest evidence flavor. -->
The report contains several distinct evidence flavors (strongest first):

- **preregistered confirmatory adaptation study (panel-conditional)** — locked estimands/decision rules/−0.02 margin before scoring (non-HARKing); the strongest tier here, ranked above the retrospective panel, but its bounds are conditional on the fixed 10-model panel and it still scores on the inspected Paper A manifest, so it is preregistered-analysis, not sealed-cohort;
- retrospective general-safety panel;
- fixed-panel composition pilot;
- inspected-cohort KL-SFT control;
- Mortgage LLM-judge dual-label benchmark;
- ExpGuard external expert-annotated domain benchmark;
- recommended production lifecycle.

These can support one decision narrative, but their numeric results must not be pooled as if they came from one experiment.

---

## 6. Practical guidelines by audience

### 6.1 Executive decision policy

The main paper should provide this compact policy:

1. Do not approve a guard from one aggregate leaderboard rank.
2. Require target-domain data and an explicit operating contract.
3. Compare every adapted candidate with its own base.
4. Retain the base when it passes every frozen requirement.
5. Fund a sealed acceptance set that is separate from training, development, and calibration.
6. Make “no candidate passes” a valid outcome.
7. Require an accountable domain owner for high-compliance use.
8. Require measured service evidence for the actual deployment mode.
9. Treat unconfirmed remedies as R&D, not production defaults.

Executive scorecard:

| Decision | Green only when | Owner |
|---|---|---|
| Continue adaptation | Represented gain is needed and transfer/domain floors remain achievable | Model/R&D lead |
| Promote a challenger | The base fails a frozen need that the challenger satisfies, and the challenger passes all statistical, domain, service, and governance gates | Release authority |
| Use domain claims | Label source, SME status, sample scope, and legal boundary are explicit | Domain risk owner |
| Choose self-host or API | Batch, concurrency, queueing, availability, privacy, and cost are measured for the intended load | Platform owner |
| Release | Rollback, monitoring, and incident actions are tested | Product and operations owners |

<!-- REVISED (v2): add a cost/time dimension so an executive can approve staffing/budget, not just accept risk. -->
> **Add a "what this costs to run" annotation to each scorecard row** (order-of-magnitude engineer-weeks
> and GPU-hours to build a sealed acceptance set and run the gate battery). Owner and risk alone do not let
> an executive approve headcount, wall-clock, or GPU budget; anchor rough figures to the measured serving
> basis (`tab:latency`) and the confirmatory study's compute footprint.

### 6.2 Engineer-facing lifecycle

Label this lifecycle:

> Recommended engineering workflow; not evaluated end to end by the present study.

#### Step 1: Freeze the deployment contract

Define:

- request and response screening scope;
- policy taxonomy and severity;
- target prevalence scenarios;
- maximum FPR;
- minimum recall and precision where meaningful;
- domain-specific floors;
- minimum sample sizes and the simultaneous-error or multiplicity policy across candidates and gates;
- service-level objectives;
- abstention and human-review behavior;
- failure defaults;
- protected-group or invariance requirements;
- model/version ownership; and
- rollback authority.

#### Step 2: Give every dataset one role

| Data role | Allowed use | Prohibited use |
|---|---|---|
| Training | Fit SFT or KL-SFT candidates | Threshold selection or final acceptance |
| Development | Tune recipe and debug implementation | Final acceptance claim |
| Target calibration | Fit candidate-specific calibrator and threshold | Candidate redesign after blind results |
| Blind acceptance | Evaluate the complete frozen candidate once | Repeated tuning or threshold adjustment |
| Monitoring | Detect production drift and incidents | Retroactively justify the original acceptance |

If the candidate, calibrator, threshold, gate, or selector changes after blind acceptance, use a new sealed cohort or a prospectively accumulated replacement set.

#### Step 3: Freeze the complete candidate registry

For each candidate record:

- checkpoint and revision;
- scorer and prompt contract;
- base, SFT, KL-SFT, or composition status;
<!-- REVISED (v2): RQ1 makes starting-checkpoint type a confirmed decision axis; record it + the native verdict-token contract. -->
- `starting_checkpoint_type: {general | released_guard}` (RQ1 confirmed this is a decision axis — the "compare every tune to its own base" rule applies to released/purpose-built guards too);
- native verdict-token contract (e.g. Yes/No vs Safe/Unsafe) validated byte-for-byte against the real tokenizer at the decision position;
- training rows and hashes;
- seed and recipe;
- calibration method;
- threshold-selection rule;
- runtime and quantization;
- failure behavior; and
- artifact owner.

Do not add a candidate after inspecting blind acceptance results.

<!-- REVISED (v2): the Llama-Guard-3-1B null cell — a near-zero delta can be a degenerate cell, not "no regression." -->
> **Pre-acceptance precheck (degenerate-cell guard).** Before trusting a near-zero adaptation delta,
> validate the candidate's native verdict interface byte-for-byte against the real tokenizer at the decision
> position **and** confirm the output head is actually movable by the chosen PEFT method. Llama-Guard-3-1B's
> 20-token-pruned, embedding-tied head is effectively unmovable by LoRA, so a flat delta there is a
> *degenerate cell — re-check trainability*, not evidence of "no regression."

#### Step 4: Calibrate every candidate separately

A base, SFT, KL-SFT, and composed score can have different scales. Fit and freeze each calibrator on target-calibration data. Then select a threshold by the predeclared rule.

If no threshold meets the required constraints, return:

> NO_FEASIBLE_THRESHOLD

The response is stop, escalate, collect better data, redesign the candidate, or change the product requirement through governance. It is not to relax the cutoff silently.

#### Step 5: Open blind acceptance once

Evaluate all frozen candidates on the same rows. Preserve paired row keys. Compute:

- absolute ranking metrics;
- threshold metrics;
- candidate-versus-base changes;
- candidate-versus-SFT recovery where applicable;
- domain and subgroup diagnostics;
- service metrics; and
- uncertainty according to the frozen plan.

#### Step 6: Apply all required gates

| Gate family | Example question | Failure action |
|---|---|---|
| Absolute ranking | Is domain AP above the declared floor? | Do not promote. |
| Operating point | Is FPR below its maximum while recall meets its floor? | NO_FEASIBLE_THRESHOLD or redesign. |
| Statistical adequacy | Do sample minima, paired uncertainty, and the prospectively frozen simultaneous-error or multiplicity rule pass? | Fail acceptance or evaluate on a separate confirmation set. |
| Relative retention | Did adaptation preserve the required transfer level versus base? | Keep base or test a repair candidate. |
| Recovery | Does composition or KL-SFT recover versus SFT without violating the base floor? | Reject the repair candidate. |
| Domain | Does the candidate pass each required domain separately? | Do not substitute an aggregate pass. |
| Invariance | Is the declared group or paired diagnostic adequately sampled and within bounds? | Route to domain review; do not call it fairness-certified. |
| Reliability | Are malformed, timeout, unavailable, and policy-mismatch behaviors correct? | Fail release. |
| Service | Do latency, throughput, availability, and cost pass under intended load? | Resize or change serving design. |
| Governance | Are the policy/version, privacy, audit, accountable-owner, and release approvals complete? | Fail release and route to the named approval owner. |

Every gate must be marked required or not_applicable with a frozen rationale. A required gate with missing evidence fails.

The pooled Clopper–Pearson diagnostic used in the current report is not a substitute for a family-aware production guarantee.

#### Step 7: Select deterministically—or do not ship

Use an incumbent-first selector that is consistent with the executive rule:

1. if the base passes every required gate, retain the base;
2. otherwise filter challengers to those that pass every required gate;
3. among feasible challengers, maximize the declared worst-regime metric;
4. break ties by lower measured service cost; and
5. break remaining ties by simpler operational topology.

If the organization needs a represented-source improvement that the base does not provide, encode that need as a frozen absolute gate before evaluation. Do not bypass the incumbent-first rule after seeing a challenger’s score.

The selector should not be changed after results are visible.

If the feasible set is empty, the outcome is no ship.

#### Step 8: Validate failure behavior

| Failure | Required declared action |
|---|---|
| Timeout | Fail closed, fail open, abstain, or human review; choose explicitly by risk class. |
| Malformed verdict | Retry once or route to fallback; never parse silently as safe. |
| Model unavailable | Invoke a versioned fallback or block the operation. |
| Policy/checkpoint mismatch | Reject the request and alert the owner. |
| Drift alarm | Increase review, freeze rollout, or roll back within a declared deadline. |

Each row needs an owner, alert threshold, and response deadline.

#### Step 9: Shadow, canary, monitor, and roll back

Before full release:

- shadow on representative traffic;
- verify policy routing and score logging;
- canary by risk tier;
- monitor prevalence, score distributions, FPR proxies, recall audits, abstentions, timeouts, and drift;
- preserve human escalation;
- version every decision artifact; and
- test rollback.

Use a rotating sealed holdout for material model, policy, prompt, calibrator, or threshold changes.

### 6.3 Illustrative mini-contract

Keep only a small example in the main paper:

~~~yaml
scope: prompt_screening
candidates: [base, sft, composition]
starting_checkpoint_type: general   # or released_guard (RQ1: a confirmed decision axis)   # REVISED (v2)
native_verdict_contract: {token_scheme: yes_no, validated_byte_for_byte: true}   # REVISED (v2)
data_roles: [train, development, target_calibration, blind_acceptance, monitoring]
required_gates:
  - domain_ap_floor
  - max_fpr_with_min_recall
  - statistical_adequacy
  - transfer_retention_vs_base
  - reliability
  - serving_slo
  - governance_release_approval
selector: incumbent_first_then_maximize_worst_required_domain_metric
empty_feasible_set: no_ship
blind_openings: 1
rollback_owner: named_release_authority
~~~

State that the values, uncertainty method, sample-size requirements, and failure actions must be supplied before this becomes executable. Put the complete schema in Appendix D.

### 6.4 One worked candidate decision

Use a compact fictional example, clearly labeled illustrative:

| Candidate | Ranking | Operating | Transfer retention | Domain, reliability, governance | Service | Outcome |
|---|---|---|---|---|---|---|
| Base | Pass | Pass | Not applicable | Pass | Pass | Retain |
| SFT | Pass | Pass | Fail | Pass | Pass | Reject |
| Composition | Pass | Fail | Pass | Pass | Pass | Reject |

Decision: keep the base. A higher AP candidate does not override a failed required gate.

This teaches the practical lesson more effectively than another leaderboard table.

### 6.5 R&D protocol for the next studies

The research appendix should convert open questions into sealed protocols.

#### Confirm KL-SFT on a fresh, uninspected cohort

<!-- REVISED (v2): the preregistered KL-SFT analysis (RQ2) is already done and returned "not supported"; only a fresh cohort remains. -->
> The preregistered confirmatory analysis of KL-SFT (RQ2 in `sec:adaptation`) is **already done** — it
> returned *not supported* (preserves transfer, fails the −0.02 non-inferiority margin). The remaining gap is
> **not "any confirmation"** but a fresh, genuinely-*uninspected* cohort (the current study still scores on
> the inspected Paper A manifest — `report.md` rec #11). Scope this protocol to that residual step:

- freeze one beta and one recipe before claim-bearing evaluation;
- declare represented non-inferiority (margin −0.02) and transfer superiority/retention estimands, one-sided 97.5% LCBs, and Bonferroni multiplicity, before unblinding (template on the executed study);
- record achieved train and held-out KL;
- use paired rows and seed-aware uncertainty;
- open a new **genuinely-uninspected** sealed cohort once; and
- report failures and incomplete cells.

#### Extend adaptation to regulated domains

- score base, SFT, KL-SFT, and composition on each domain separately;
- preserve Mortgage dual labels and ExpGuard single labels as separate constructs;
- do not choose a method from aggregate domain AP;
- add domain SME adjudication where a compliance claim is intended;
- define domain-specific operating thresholds; and
- keep legal and clinical certification outside the statistical claim.

#### Audit transfer overlap

- hash exact rows;
- run normalized n-gram, semantic near-duplicate, and source-family checks;
- quarantine collisions before analysis;
- recompute the primary result if material overlap is found; and
- retain an auditable collision ledger.

#### Test the proposed mechanisms

- measure base/SFT disagreement;
- measure error correlation by benchmark and class;
- separate arithmetic coupling from behavioral effects;
- preregister the mechanism analysis; and
- avoid causal mechanism claims from four checkpoints.

#### Evaluate purpose-built starting checkpoints

<!-- REVISED (v2): this study is DONE and confirmatory — not a "separate, prospective study." -->
The purpose-built adaptation study is **already run and confirmatory** (`sec:adaptation`): RQ1 supported
(ordinary SFT specializes released guards too, H_gain LCB +0.129), RQ2 not supported (KL-SFT preserves
transfer but fails the −0.02 non-inferiority margin). Present it as completed confirmatory evidence, not a
roadmap item. The only genuinely-future extension is a fresh, genuinely-*uninspected* cohort (the current
study still scores on the inspected Paper A manifest) plus tuned-domain evaluation. Keep within-checkpoint
adaptation effects separate from cross-family comparisons and vendor-native product performance.

---

## 7. Rebuild the conclusion

The current conclusion is one paragraph immediately followed by references. It should be a full page with five compact paragraphs.

### Paragraph 1: Primary fixed-panel result

State:

> Applying the fixed SFT recipe to the four fixed checkpoints from two model lineages reliably improved represented-source ranking on this retrospective panel, while transfer movement was heterogeneous and usually negative; 15 of 20 seed runs fell in the specialization quadrant.

Add the pending overlap-audit boundary in the same paragraph.

### Paragraph 2: Repair evidence

State:

<!-- REVISED (v2): KL-SFT now has a confirmatory RQ2 verdict on released guards — not wholly exploratory. -->
> Fixed base+SFT output composition recovered transfer relative to SFT and numerically had the highest observed minimum of represented and transfer AP among the main base, SFT, and fixed calibrated-composition candidates, but it still required recalibration and did not meet the illustrated 5% FPR target. On general checkpoints KL-SFT remains an exploratory inspected-cohort control; on released guards it has a preregistered confirmatory verdict (RQ2 not supported) — it preserves transfer but fails the −0.02 non-inferiority margin, so it is a confirmed tradeoff, not free.

Do not say “safest,” “solves forgetting,” or recommend a beta.

### Paragraph 3: Domain evidence

State:

> Base-model orderings also varied across Mortgage, Finance, Health Care, and Law. These were base-only zero-shot evaluations: Mortgage used LLM-judge dual labels, while ExpGuard used external expert-annotated single labels. Neither arm demonstrates tuned-domain efficacy or legal, clinical, or fairness certification.

### Paragraph 4: Practical workflow

State:

> Teams should freeze a deployment contract, calibrate each candidate, open a blind acceptance set once, and require all absolute and relative gates. Retain a passing base; otherwise apply the frozen selector to feasible challengers. When none passes, the correct outcome is no ship.

Label the workflow recommended, not end-to-end validated by the study.

### Paragraph 5: Next evidence

State:

<!-- REVISED (v2): credit the completed confirmatory study; scope the remaining gap to a fresh uninspected cohort. -->
> This report already includes a preregistered confirmatory adaptation study (RQ1 supported, RQ2 not supported on released guards). The next decisive evidence is therefore a fresh, genuinely-*uninspected*-cohort confirmation, a formal overlap audit, paired tuned-domain evaluation, and serving-mode load/cost measurement. The contribution is a measurement and decision workflow plus one confirmatory result, not a universal winning model.

End on:

> **Choose through a frozen, auditable workflow—not from a leaderboard.**

---

## 8. Source migration and implementation plan

### 8.1 Source-to-destination map

| Current source | New main-text use | Appendix destination |
|---|---|---|
| Root abstract and introduction in [unified_report.tex](unified-report/unified_report.tex) | Scope-safe abstract, executive spread, Section 1 | Repeated tutorial material removed. |
| [sections/background-setup.tex](unified-report/sections/background-setup.tex) | Minimum score, paired-design, represented/transfer, and ranking/threshold concepts in Section 2 | Full equations, recipes, benchmark inventory, bootstrap, prompts, and domain construction in Appendix B. |
| [sections/related-work.tex](unified-report/sections/related-work.tex) | One closest-work paragraph in Section 1 | Full review in Appendix A. |
| [sections/act1.tex](unified-report/sections/act1.tex) | Primary delta, seed quadrant, operating consequence, and prevalence scenario in Section 3; compact exploratory KL inset in Section 4 | Raw tables, per-benchmark decomposition, attractor analysis, sensitivities, and full KL detail in Appendix C. |
| [sections/act3.tex](unified-report/sections/act3.tex) | Composition primary result and threshold failure in Section 4 | Composition mechanics, weight-space comparison, controls, and ablations in Appendices B–C. |
<!-- REVISED (v2): the confirmatory section + its generated inputs were missing from the migration map. -->
| [sections/act-adaptation.tex](unified-report/sections/act-adaptation.tex) + `generated/adaptation_macros.tex` + `generated/tab_adaptation_gen.tex` | RQ1 confirmatory module in Section 3, RQ2 authoritative KL verdict in Section 4, and the §4b confirmatory subsection with `tab:adaptation` — **fix its current mis-placement** (it is `\input` after Act I but references "Acts I–III") | Full protocol, claim registry, null-cell detail, and the complete 10×3 `tab:adaptation` grid in Appendix C. |
| [sections/act4-mortgage.tex](unified-report/sections/act4-mortgage.tex) | Mortgage construct, base results, and directional invariance in Section 5 | Pipeline, HMDA transformations, policy cards, rubric, and full protocol in Appendix B. |
| Root ExpGuard synthesis | Three separate Finance, Health Care, and Law panels in Section 5 | Exact tables and additional comparisons in Appendix C. |
| Root synthesis and practitioner guide | Executive actions and compact Section 6 workflow | Full contract, predicates, service plan, monitoring, and economics template in Appendix D. |
| [sections/limitations-validation.tex](unified-report/sections/limitations-validation.tex) | Evidence matrix and open dependencies in Section 7 | Exhaustive limitations and roadmap in Appendix E. |
| Root reproducibility and conclusion | Section 7 reproduction box and the five-paragraph conclusion | Full artifact manifest in Appendix E. |

<!-- REVISED (v2): the migration must also read report.md and this review as inputs. -->
> **Inputs to the migration:** in addition to the source `.tex` files, read `report.md` (17 findings, most
> applied) and the companion `presentation-style-proposal-review.md` (50 findings) so the refactor starts
> from the current build and does not re-litigate applied fixes or orphan the confirmatory section.

### 8.2 Recommended source structure

Refactor toward:

~~~text
unified-report/
  unified_report.tex
  sections/
    executive-spread.tex
    decision-problem.tex
    how-to-read.tex
    sft-result.tex
    repair-paths.tex
    confirmatory-adaptation.tex   # REVISED (v2): home for sec:adaptation (RQ1/RQ2 + tab:adaptation)
    domain-evaluations.tex
    engineering-guide.tex
    evidence-reproducibility.tex
    conclusion.tex
  appendices/
    related-work.tex
    methods.tex
    full-results.tex
    engineering-playbook.tex
    evidence-ledger.tex
~~~

Keep generated numbers in generated files and presentation prose in authored section files. Do not put hand-edited result values into section prose.

### 8.3 Implementation phases

#### Phase 0: Correct evidence language

<!-- REVISED (v2): most P0s are done; the top task is the abstract carve-out; the preregistration item is inverted; PDF/UA smoke is best-effort. -->
- **Top task — carve out the abstract:** scope `unified_report.tex:130` ("All evidence here is retrospective and estimation-only") to Acts I–III and carve out the preregistered adaptation study (`report.md` rec #1, still OPEN);
- **Reconciliation pass:** cross-walk §2.1 against `report.md`, marking each item DONE / SUPERSEDED / STILL-OPEN, so applied work (15/20, KL Row-3 overclaim, mortgage directional caveat, base-only labels) is not re-litigated;
- **propagate** (do NOT remove) the preregistration/confirmatory status through abstract/intro/ledger/conclusion; keep Acts I–III labeled retrospective as the intended contrast;
- finish the genuinely-remaining P0 claim fixes: replace "statistically tied" with a paired-difference interval or "ordering unresolved" at the three sites (`unified_report.tex:110, :255, :270`); reserve "safe/deployable" for gate-passing candidates (composition "highest observed min" wording); replace any self-host-cheaper-than-API conclusion with the serving-study checklist; remove any remaining causal/domain-remedy overclaims;
- keep the general-checkpoint KL result exploratory while presenting the released-guard KL result as confirmatory-not-supported (a finding, not a gap);
- verify base-only zero-shot labels and the pending overlap-audit badge are present (already applied — verify only);
- add reproduction coverage assertions over the enumerated **17** consumed inputs (incl. the adaptation/KL stages) and fix the figure-check write path; and
- attempt the one-page tagged-PDF smoke as a **best-effort** spike (do not block on it; do not switch off Tectonic — see §4.10).

Exit gate: no known P0 claim contradiction remains in source or generated captions (the abstract carve-out landed), and §2.1 is reconciled against `report.md`.

#### Phase 1: Build the 38-page skeleton

- add the executive spread;
- create the new section files and headings;
- move content without rewriting measurements;
- add end-of-section evidence/decision/boundary boxes;
- place a clear page break before references and emit a references_start page label after floats; and
- assert that references_start minus one is between 35 and 40.

Exit gate: a text-only build follows the new order and the measured last main-text page is between 35 and 40.

#### Phase 2: Rebuild principal visuals

- implement the nine composites;
- use shared style constants;
- provide fallbacks for missing uncertainty;
- move raw displays to appendices;
- add accessible text equivalents.

Exit gate: every main claim has one legible visual and one artifact source.

#### Phase 3: Add the compact decision guide

- add the mini-contract;
- add gate families and the no-ship branch;
- add one worked candidate decision;
- add failure actions and lifecycle;
- put full templates in Appendix D.

Exit gate: an engineer can turn the guide into test cases without inferring missing control flow.

#### Phase 4: Harden reproduction and provenance

<!-- REVISED (v2): register the 17 consumed inputs and add the adaptation stage; all CPU-only, so this can run in parallel. -->
- register all **17** consumed generated inputs (enumerate programmatically, not a hard-coded count);
- implement deterministic **adaptation, KL, and mortgage-composition** stages (`adaptation_macros.tex` + `tab_adaptation_gen.tex`; `klsft_macros.tex` + `tab_klsft_gen.tex`), runnable CPU-only from committed scores (no GPU/network) so this whole phase can run in parallel with the restyle;
- add output-directory controls for every table and figure generator;
- generate the mortgage composition table from the frozen public index after manifest/checksum validation;
- compare figures through temporary outputs;
- propagate every subprocess failure instead of accepting a stale file;
- fail on pending, skipped, pinned-only, missing, or drifted stages;
- print a machine-readable artifact ledger.

Exit gate: a clean checkout can reproduce or explicitly fail; it cannot report success with unresolved stages.

#### Phase 5: Editorial, visual, and accessibility QA

- compile twice;
- run reference, citation, overflow, and float-placement checks;
- render all pages and inspect contact sheets;
- verify 9 pt minimum figure text;
<!-- REVISED (v2): PDF/UA and formal reader-tests are best-effort/recommended, not blocking; HTML companion is the a11y deliverable. -->
- tag reading order and objects (best-effort);
- run the PDF/UA conformance check as a **best-effort, non-blocking** step; ship the accessible HTML companion as the a11y deliverable; for the PDF require only bookmarks, language, meaningful links, grayscale-safe/non-Type-3 figures, and alt-text stubs;
- reject Type 3 plot fonts with pdffonts (blocking);
- run the three reader tasks as a **recommended self-run** check, not a blocking gate;
- perform an adversarial claim audit.

Exit gate: all acceptance criteria below pass.

---

## 9. Feasibility and stress test

### 9.1 Overall feasibility

<!-- REVISED (v2): correct the "just a reorder" framing — this is a multi-week source refactor + figure/repro/a11y work. -->
The rewrite keeps all evidence and changes its disclosure order, but it is **not** merely a reorder. It also entails a ~180 KB TeX source refactor into a new sections/appendices tree (re-wiring every `\label`/`\Cref`/macro cross-ref — high regression risk), new figure pipelines, a reproduction-harness rewrite, best-effort PDF tagging, and (recommended) reader tests. Budget it as multi-week specialist work and refactor the source **incrementally** — move one act at a time and compile-and-diff after each — rather than a big-bang tree move, to contain cross-ref regressions.

The design relies on four constraints:

- three to four load-bearing composites first (nine aspirational, not a gate — see §4.4);
- three to five compact main tables;
- eight to ten short callouts;
- full detail moved to appendices rather than compressed into unreadable text.

The working plan has 34 pages of designed content and four pages of layout reserve. The reserve is needed because real LaTeX floats, captions, section transitions, and accessibility-friendly text sizes consume space.

### 9.2 Evidence fallback matrix

| Missing or failed dependency | Main-paper fallback | Prohibited claim |
|---|---|---|
| No KL paired uncertainty or achieved-KL diagnostics | Amber point-estimate inset or Appendix C only | No default beta or production recommendation |
| No fresh sealed KL cohort (general checkpoints) | “Exploratory inspected cohort” (Act I, n=4) | No confirmatory promotion from the general-checkpoint control |
| Preregistered confirmatory result on released guards (RQ1/RQ2); bounds panel-conditional, scored on the inspected manifest | Label “Preregistered confirmatory (panel-conditional)”; report the SUPPORTED (RQ1) and NOT-SUPPORTED (RQ2) verdicts as decisions | No sealed-cohort or fresh-data claim; do not merge with Act I panel numbers |
| No per-domain ExpGuard paired differences | Descriptive dots, n, positives, and “ordering unresolved” | No winner or statistical tie |
| Only three Mortgage protected pairs | Directional observed result | No fairness gate or certification |
| Transfer overlap audit pending | Visible pending badge | No “clean transfer” claim |
| Material overlap found | Quarantine, rebuild, and recompute | No unchanged primary conclusion |
| No Mortgage SME adjudication | LLM-judge diagnostic | No legal compliance validation |
| No production serving study | Service checklist only | No self-host/API latency or cost superiority |
| No fresh blind acceptance cohort | Shadow or human-review status | No acceptance-passed release |
| Required gate has missing evidence | Candidate fails feasibility | No waiver by omission |

### 9.3 Adversarial stress tests

#### “The executive spread oversimplifies the science.”

Pass condition: every card has a compact evidence-status tag; one spread-level boundary strip states the non-claims and links each card to its detailed section.

#### “The paper now looks like a product playbook instead of research.”

Pass condition: the paired design, uncertainty, per-checkpoint evidence, fixed panel, artifact provenance, and full appendices remain auditable.

#### “The engineering guide appears empirically validated.”

Pass condition: every lifecycle visual says “recommended; not evaluated end to end by this study.”

#### “The four regulated domains look like one experiment.”

Pass condition: Mortgage and ExpGuard have separate evidence badges, label sources, panels, captions, and limitations. No pooled domain metric appears.

#### “Composition is presented as safe because it has the best AP.”

Pass condition: the 11.4% versus 5% FPR miss appears in the same result module.

#### “KL-SFT is still a hidden recommendation.”

Pass condition: it is visually subordinate, has no selected default beta, and names its sealed promotion requirements.

#### “A model is called tied or best without the required comparison.”

Pass condition: paired-difference evidence exists, or the text says “ordering unresolved.”

#### “A required gate can disappear.”

Pass condition: every gate is required or not_applicable with a frozen rationale; missing required evidence fails.

#### “The page limit forces removal of Finance, Health Care, or Law.”

Pass condition: the three-panel ExpGuard composite is nonnegotiable; detailed tables move first.

#### “The reproduction command passes despite incomplete work.”

Pass condition: unresolved, skipped, missing, drifted, and pinned-only stages return nonzero in release-check mode.

### 9.4 Reader-task tests

<!-- REVISED (v2): downgraded from a blocking gate to a recommended, non-blocking sanity check. -->
> **Recommended, non-blocking.** Self-run the executive 5-minute and engineer 25-minute routes against the
> checklists below; recruit an uninvolved outside reader per persona only if time permits. The hard gates are
> the automated page-budget/float/font checks in §9.5, not the reader tests.

Recruit at least one representative of each audience who was not involved in drafting (time-permitting).

Executive, after five minutes:

- identifies that there is no universal winner;
- distinguishes measured repairs from exploratory ones;
- names the required organizational decisions;
- does not infer regulated-domain tuning evidence.

Then test three short scenarios:

- base passes all required gates → keep the base;
- SFT improves represented ranking but fails transfer retention → reject it or evaluate a predeclared repair, not promote it; and
- no fresh acceptance cohort exists → allow only controlled shadowing or human review, not an acceptance-passed release.

Engineer, after 25 minutes:

- identifies all five data roles;
- explains why each candidate needs its own calibration;
- implements required versus not-applicable gates;
- returns no ship when no candidate passes;
- describes timeout, drift, and rollback behavior.

R&D reader:

- identifies the primary paired estimand;
- distinguishes row, seed, and fixed-panel uncertainty;
- locates the overlap-audit status;
- reproduces a headline number;
- separates Mortgage, ExpGuard, KL-SFT, and recommended-workflow evidence tiers.

Any systematic failure requires a content or navigation revision, not a footnote.

### 9.5 Page-budget test

Automate:

- clear all floats before references, emit references_start, and require references_start minus one to be between 35 and 40;
- no figure/table text below 9 pt;
- no main figure larger than one page;
- no more than ten large main objects;
- no unresolved overfull boxes;
- no major heading orphaned at a page bottom;
- no forced H floats in main sections; and
- references and appendices begin after a deliberate page break.

---

## 10. Definition of done

### Flow

- [ ] The abstract fits on page 1.
- [ ] An executive reaches the decision and evidence boundaries by page 3.
- [ ] The first primary result starts by page 11, not page 20.
- [ ] Composition follows the problem it repairs.
- [ ] KL-SFT is visibly exploratory.
- [ ] Mortgage, Finance, Health Care, and Law are all visible in the main paper.
- [ ] The engineering guide appears before limitations.
- [ ] The conclusion occupies a deliberate final main-text page.

### Correctness

<!-- REVISED (v2): mark applied items DONE; add the abstract carve-out, confirmatory-tier, preregistered-vs-sealed, and panel-conflation items. -->
- [x] Specialization is 15/20, not 15/15. (DONE — `unified_report.tex:314`)
- [ ] **The abstract's "estimation-only" clause is scoped to Acts I–III and carves out the preregistered adaptation study (`unified_report.tex:130` — STILL OPEN, top task).**
- [ ] The confirmatory RQ1/RQ2 results carry a "preregistered confirmatory (panel-conditional)" tier, distinct from retrospective and directional evidence.
- [ ] "Preregistered analysis" (estimands locked before scoring) and "sealed cohort" (uninspected data opened once) are kept distinct; the adaptation study is labeled the former, not the latter.
- [ ] The Act I (+0.32, 4-checkpoint) and confirmatory (+0.174, 10-checkpoint) represented gains are never merged.
- [ ] No marginal-CI overlap is called a statistical tie. (three sites still OPEN)
- [x] No "free/keeps-specialization" KL overclaim; released-guard KL is confirmatory-not-supported, general-checkpoint KL is exploratory. (DONE — `tab:guidelines` Row 3)
- [ ] Composition is not called safe or deployable from AP alone. (OPEN)
- [x] Transfer is described as heterogeneous. (DONE)
- [ ] Prior-shift assumptions are explicit.
- [ ] Delta-context is a directional small-n diagnostic.
- [ ] Self-host/API claims match measured serving evidence. (OPEN)
- [x] Domain arms are labeled base-only, zero-shot. (DONE)
- [ ] The pending overlap audit is visible beside the primary result.

### Practicality

- [ ] Executive actions have owners.
- [ ] The five data roles are distinct.
- [ ] Candidate-specific calibration is required.
- [ ] Blind acceptance opens once.
- [ ] Gates are executable predicates.
- [ ] Missing required evidence fails.
- [ ] The selector is deterministic.
- [ ] NO_FEASIBLE_THRESHOLD and no ship are explicit.
- [ ] Failure actions, shadow, canary, monitoring, and rollback are present.
- [ ] The workflow is labeled recommended, not end-to-end validated.

### Research integrity

- [ ] Each claim-bearing result module has an estimand, interval or explicit no-interval label, and evidence status; each executive card links to that module.
- [ ] Per-checkpoint results remain visible.
- [ ] Mortgage and ExpGuard values are never pooled.
- [ ] Directional and exploratory results are not promoted.
<!-- REVISED (v2): the purpose-built study is DONE and confirmatory, not future. -->
- [ ] The confirmatory adaptation study (RQ1/RQ2) is presented as completed confirmatory evidence, distinct from the retrospective acts; only a fresh, genuinely-uninspected-cohort re-run is future.
- [ ] Full methods and sensitivities remain in appendices.

### Reproducibility

<!-- REVISED (v2): 17 inputs, enumerated programmatically; add the adaptation stage. -->
- [ ] All **17** consumed generated TeX inputs are registered (enumerated programmatically, not a hard-coded count) or the consumed set is deliberately changed and asserted.
- [ ] Adaptation (`adaptation_macros.tex` + `tab_adaptation_gen.tex`), KL, and mortgage-composition outputs have deterministic owners.
- [ ] Figure check mode does not overwrite source artifacts.
- [ ] Release checks fail on unresolved states.
- [ ] Main claims link to hashes and sources.
- [ ] A clean reproduction either succeeds completely or fails visibly.

### Visual and accessibility quality

<!-- REVISED (v2): composites aspirational; PDF/UA and reader-tests demoted to best-effort/recommended. -->
- [ ] Every main claim has one legible visual (3–4 load-bearing composites built; existing act figures restyled otherwise) — "nine composites" is aspirational, not a gate.
- [ ] Main table count is three to five (main-text objects designated vs Appendix D — see §4.4).
- [ ] Main callout count is eight to ten.
- [ ] Figure text is at least 9 pt.
- [ ] pdffonts reports no Type 3 text in principal figures.
- [ ] Color is not the only signal.
- [ ] The accessible HTML companion ships as the a11y deliverable; the PDF has bookmarks, document language, meaningful link text, grayscale-safe/non-Type-3 figures, and alt-text stubs. Full PDF/UA conformance is best-effort, not a blocking gate.
- [ ] Every page through the measured main-text end passes rendered visual inspection.
- [ ] Executive, engineer, and R&D reader-task tests are run as a recommended self-run check (non-blocking).

---

## Recommended first editing pass

<!-- REVISED (v2): derive from tab:guidelines (do not "replace Table 16"); reconcile against report.md; add a lean critical-path note. -->
Do not begin with typography. The first pass should:

1. correct the remaining P0 claims (top: the abstract carve-out at `unified_report.tex:130`) and their generator-owned text, after reconciling §2.1 against `report.md`;
2. create the new section skeleton and executive spread;
3. move existing material into the new evidence order (including a first-class home for the confirmatory study);
4. derive the executive evidence/action map from the seven rows of `tab:guidelines` and add a confirmatory card — do not "replace Table 16" (that object no longer exists);
5. build the 3–4 load-bearing composites (SFT, composition/KL with the confirmatory verdict, Mortgage, three-domain); restyle existing figures for the rest;
6. add the compact candidate-gating workflow;
7. repair the reproduction registry (17 inputs; add the adaptation/KL stages; CPU-only, can run in parallel);
8. compile and enforce the page gate; and
9. run the three reader-task tests as a recommended self-run check before polishing prose.

**Lean critical path** (defer the expensive/risky tail): (1) finish the remaining P0 claim fixes; (2) re-scope to current reality — the confirmatory study is in-paper, not future; (3) build the 2-page executive spread; (4) incremental source refactor + 3–4 core composites; (5) run repro hardening in parallel (CPU-only). Defer/downgrade: the other composites, blocking PDF/UA (ship the HTML companion), formal user tests, and the strict 40-page hard stop.

This order reduces the risk of spending time polishing claims or layouts that the evidence audit later requires changing.

The target paper is not a shorter version of the current PDF. It is a layered decision document: executives see the decision, engineers see the control flow, and researchers see the complete evidence chain.
