> **Review status (this pass).** 17 findings confirmed by adversarial verification (of 20 raised; 3 rejected/downgraded on re-check). Substantive fixes APPLIED in commit alongside this file:
> 1. **numbers-1** (real error): guidelines table seed count `15/15` -> `15/20` (`\TotalSeedCount`=20).
> 2. **overclaim-1 / flow-5**: guidelines Row 3 no longer calls KL-SFT "free / keeps specialization" -- now states the represented-source cost and cross-refs the RQ2 non-inferiority failure (removes the Row 3 <-> Row 7 contradiction).
> 3. **overclaim-2/3**: abstract mortgage "winner flips" -> "top-ranked guard differs across benchmarks (directionally, on a small split)".
> 4. **consistency-2**: contributions now list the preregistered confirmatory adaptation study.
> 5. **stats-4**: disambiguated the overloaded word "family" (model vs evaluation vs research-question) in Sec.~adaptation.
> 6. **stats-1**: Llama-Guard-3-1B null/degenerate cell -- added the inclusion + drop-sensitivity statement (verdicts unchanged).
> 7. **flow-1 / consistency-1**: limitations "not confirmatory" bullet + validation roadmap now carve out the adaptation study as the one confirmatory piece (resolves the "no pre-registration" self-contradiction).
>
> DEFERRED (cosmetic/polish, non-blocking, listed below): figure-float placement (flow-6), evidence-ledger row (consistency-3/flow-3), conclusion-sentence mention (consistency-4), a held-SFT UCB macro (numbers-2), opener forward-ref (flow-2), and minor wording (stats-2/3, flow-4). None affect correctness or any verdict.

---

# Reviewer Report: *Guard Ranking Fragility* (unified report, Paper A edition)

## 1. Summary assessment

The paper is fundamentally **sound in its core evidence and unusually disciplined in its epistemics** — it separates evidence tiers, binds every number to generated macros from committed artifacts, and preregisters its one confirmatory study with locked estimands and a non-inferiority margin. The underlying data support the headline that fine-tuning specializes rather than transfers, and the confirmatory adaptation study is a genuine strength.

However, the write-up has three recurring problems that a careful reviewer will flag:

1. **The abstract, introduction, conclusion, and evidence ledger systematically erase the one confirmatory study.** The abstract asserts "All evidence here is *retrospective and estimation-only*," which directly contradicts `sec:adaptation` ("the one *confirmatory* piece") and the guidelines-table caption ("only the adaptation study is preregistered"). The paper's strongest contribution is invisible in its own framing spine (abstract, intro roadmap, contributions list, unified evidence ledger, conclusion). This is the single most damaging class of issue because it is an *internal contradiction on epistemic status* in a paper whose entire selling point is honest evidence tiering.

2. **The KL-SFT story is overclaimed and internally colliding.** Guidelines Row 3 sells KL-SFT as "free at inference" that "keeps specialization," while Row 7 and the confirmatory RQ2 verdict say it is "not a free upgrade" and *fails* the non-inferiority margin (H_cost LCB −0.060 < −0.02). The same method is described with opposite valence in two rows and against its own confirmatory verdict.

3. **Number and hedging slips.** Row 2 renders the specialization fraction as "15/15 seeds" (should be 15/20), overstating a headline result as 100%; the abstract drops the "small split / directional" qualifier the body attaches to every mortgage claim; and the confirmatory means silently retain a degenerate null cell (Llama-Guard-3-1B) without a drop-one sensitivity, despite the paper advertising leave-one-out checks elsewhere.

None of these overturn a *result* (RQ1 remains supported, RQ2 remains correctly not-supported), but together they make the paper read as less rigorous than its actual methodology, and in the KL-SFT case they risk misleading a skim reader.

## 2. Major-claims verification

| Claim | Supported by data? | Evidence / LCB | Verdict or needed qualification |
|---|---|---|---|
| **RQ1** — specialization tradeoff also hits *already purpose-built* released guards | **Yes** | H_gain LCB **+0.129 > 0**; H_conc LCB **+0.189 > 0** (`claim_checks.json`) | Supported as stated. Robust to dropping the null Llama cell (H_gain 0.174→0.208), only strengthens. Fine as a verdict. |
| **RQ2** — KL-SFT is a *free* improvement (preserves transfer at no represented cost) | **No** | H_preserve LCB +0.035 > 0 (transfer preserved) BUT H_cost LCB **−0.060 fails the −0.02 margin**; mean represented cost −0.035 (up to −0.057/checkpoint) | **NOT supported.** Must be stated plainly as a failed non-inferiority test, not softened. |
| **Row 1** — a guard's ranking flips across benchmarks (Qwen3-4B worst transfer specializer yet best zero-shot mortgage guard) | Directional only | Act III, "directional/small split" (line 305); ExpGuard top two within CI | Correctly hedged *in the row*, but the abstract states it as a resolved flip (see Row-1 abstract issue below). Keep the row's hedge; propagate to abstract. |
| **Row 2** — ordinary SFT buys represented ranking, not transfer; specialization in 15/15 seeds | Yes (effect); **No (the "15/15" number)** | Rep Δ (LCB) vs transfer (UCB); real fraction is **15/20** (`results_macros_gen.tex`: SpecializationSeedCount=15, TotalSeedCount=20, 5 uniform-gain seeds) | Effect supported; **denominator is wrong** — "15/15" overstates to 100%. Fix to 15/20. |
| **Row 3** — base-anchored KL penalty recovers most transfer, "free at inference," "keeps specialization" | Partially / overclaimed | KL(β=0.5) transfer gain vs SFT at a *represented cost* `\KLBetaHalfRepDelta` = **−0.035** (up to −0.057); n=4, **no interval**, retrospective | Overclaimed. "Free"/"keeps specialization" contradicts RQ2. Restate as "no extra *inference* cost but a represented-source cost"; drop "keeps specialization"; mark retrospective/interval-free (n=4). |
| **Row 4** — output-space composition repairs lost transfer at inference, not retraining | Yes (as retrospective estimation) | Base+adapter average beats equal-cost SFT+SFT ensemble (`tab:sftsft`) | Supported at its (retrospective) tier. No change needed to the claim itself. |
| **Row 5** — domain change hides violations; general-safety score ≠ compliance | Directional only | Rankings/margins shift on mortgage split; marked "directional" (line 329) | Correctly hedged in the row; abstract states it without the qualifier — propagate hedge to abstract. |
| **Row 6** — small single-token guard is fast/cheap/in-house (~10–50 ms/A100) | Yes (measured) | `tab:latency`, one forward pass | Supported. No change. |
| **Row 7** — fine-tuning a released guard specializes it too; KL-SFT keeps transfer but at a represented cost | **Yes** | Confirmatory 10-checkpoint: SFT rep AP `\AdaHGain` (LCB `\AdaHGainLCB`); KL-SFT preserves transfer (LCB `\AdaHPreserveLCB`) but represented cost (LCB **`\AdaHCostLCB` = −0.060**) fails −0.02 margin | Supported and correctly stated. This is the *authoritative* KL-SFT verdict; Row 3 must be reconciled to it. |
| **Abstract headline** — "the winner flips with the benchmark" | Overstated | ExpGuard top two tied (disclosed inline); mortgage leg is directional/small split (NOT disclosed in abstract) | The ExpGuard tie *is* disclosed; the **mortgage leg's directional caveat is missing**. Soften the "flip" to "the numerically top-ranked guard differs (directional, small split)." |
| **Abstract status** — "All evidence here is retrospective and estimation-only" | **No (internally false)** | Contradicts `sec:adaptation` line 6 and `tab:guidelines` caption line 293–294 | Must carve out the preregistered adaptation study. |

## 3. Inconsistencies & factual/number errors

1. **[high] "All evidence here is retrospective" contradicts the confirmatory study.** `unified_report.tex:127–129` (abstract) vs `act-adaptation.tex:6` ("the one *confirmatory* piece") and `tab:guidelines` caption `unified_report.tex:293–294` ("only the adaptation study is preregistered"). Since `act-adaptation` is `\input` at line 207, it *is* "here." (consistency-1, flow-4)

2. **[medium] "15/15 seeds."** `unified_report.tex:311` writes `\SpecializationSeedCount/\SpecializationSeedCount` → 15/15. Every other occurrence is 15/20 (`act1.tex:126,139`; `limitations-validation.tex:35,62`). Overstates specialization to 100%, contradicting the 5 documented uniform-gain seeds. (numbers-1)

3. **[medium] KL-SFT "free at inference" (Row 3) vs "not a free upgrade" (Row 7).** `unified_report.tex:315` vs `:342`. Different populations (general checkpoints vs released guards) and two senses of "free" (no extra forward pass vs no accuracy cost), but the scannable headlines collide and read as self-contradiction. (overclaim-1, flow-5)

4. **[medium] Intro frames output as estimation-only, omits the confirmatory study.** `unified_report.tex:195–197` ("the output is a reproducible, estimation-only characterization… plus one external domain replication"); the three-question roadmap (167–171) and five contributions (180–194) never mention the preregistered adaptation study. (consistency-2, flow-1)

5. **[medium] Unified evidence ledger omits the adaptation study.** `limitations-validation.tex:4` claims it records "*each* body of evidence," but `tab:ledger` (60–75) has only 4 rows (Act I/II/III-mortgage/III-ExpGuard) and its flavor axis (line 50) has no confirmatory slot; closing takeaway (line 102) again labels the whole report estimation-only. (consistency-3, flow-3)

6. **[low] Abstract drops the mortgage "small split / directional" qualifier** that the Synthesis (line 282) and Rows 1 & 5 attach; the closing disclaimer covers retrospective/LLM-judge but not sample-size fragility. `unified_report.tex:107–111`. (overclaim-2, overclaim-3)

7. **[low] "bound-confirmed transfer cost" without the bound shown.** `act-adaptation.tex` RQ1 paragraph quotes held-out SFT change `\AdaHheldSFT` = −0.065 and calls it "bound-confirmed," but only H_gain/H_conc LCBs are displayed; the H_held_sft one-sided UCB (−0.047 < 0, predicate true in `results.json`) is never surfaced. (numbers-2)

8. **[low] Conclusion under-credits the confirmatory work.** `unified_report.tex:450–451` calls "a prospectively locked evaluation" the unmet next step without distinguishing preregistered-analysis (already done in `sec:adaptation`) from genuinely-uninspected-data (the true remaining gap, `limitations-validation.tex:90`). (consistency-4)

9. **[minor, out of scope but worth a pass] "Acts I–II" vs "Acts I–III".** Synthesis caption (line 292) says "Acts I–II are retrospective"; `act-adaptation.tex:4` says "Acts I–III." Reconcile the act-count phrasing.

## 4. Methodological weaknesses & missing evidence

1. **Degenerate null cell retained in confirmatory means without disclosed sensitivity.** Llama-Guard-3-1B has represented Δ ≈ 5.5e-17 (mechanically zero: pruned, embedding-tied head unmovable by LoRA). It is disclosed as a null cell but silently averaged into the equal-family H means as a hard zero, diluting H_gain (0.174 all-six → 0.208 drop-Llama) and pulling H_cost toward the margin (−0.036 → −0.043). No leave-one-family-out is reported, though `background-setup.tex:172,455` advertise LOO checks as standing methodology. *Direction is conservative for both verdicts, so no conclusion is fragile — but the omission is a transparency gap in the one confirmatory section.* (stats-1)

2. **"Family" is overloaded in the statistical description.** In `act-adaptation.tex` (≈lines 24–25, 54–55) "family" means 6 *model* families (the equal-weight unit), 2140 *evaluation-row* clusters (the bootstrap resampling unit), and *RQ hypothesis* families — in adjacent sentences. The "family-aware" bootstrap holds model identities fixed, so the LCBs carry **no between-model-family variance** — exactly the quantity the equal-model-family mean is most sensitive to. The dominant-family caveat is present but unquantified: dropping the smollm family (2 checkpoints, +0.421) moves H_gain 0.174 → 0.124 (~0.05). (stats-4)

3. **KL Act I control is n=4, point-estimate only, no interval,** yet Row 3 presents an unqualified verdict. Per-checkpoint represented costs range −0.008 to −0.057. (overclaim-1)

4. **General-vs-purpose is descriptive (blocked, not randomized);** within-checkpoint deltas are causal only for this-recipe-on-this-checkpoint; all bounds are panel-conditional. This is stated in `sec:adaptation` but should be surfaced in the ledger (see §3.5).

## 5. Organization, flow & clarity

1. **The confirmatory study has no signposting in the framing spine.** Absent from the abstract "What is new" (114–119), the intro roadmap (167–171) and contributions (180–194), the evidence ledger (`tab:ledger`), and the conclusion. A reader primed for "three retrospective acts" hits a large, differently-framed confirmatory section with zero warning. (flow-1, flow-3)

2. **`sec:adaptation` is mis-placed / forward-referential.** It is `\input` at `unified_report.tex:207`, immediately after Act I and *before* Acts II–III, yet its opener ("Acts I–III… so they are retrospective estimation… This section is the one confirmatory piece," `act-adaptation.tex:4–6`) assumes the reader has finished all three retrospective acts. Meanwhile Act II's opener (`act3.tex:4`, "Act I left us with…") reads straight past it back to Act I. The confirmatory study is sandwiched and both neighbors ignore it. (flow-2)

3. **Four figures float without callouts.** `fig:act1-bars, fig:act3-bars, fig:quadrant, fig:mortgage-bars` (`unified_report.tex:208–244`) are never `\Cref`'d in prose (only their `\label` lines exist), unlike `fig:plane/attractor/prevalence/expguard-domains/pipeline`. Worse, `fig:act1-bars` (a per-checkpoint summary *of Act I*) is emitted at line 208, *after* the adaptation `\input` at 207, so it renders past its own narrative. (flow-6)

4. **The two "free" senses in the guidelines table are not scoped at a skim.** Rows 3 and 7 never signal that they concern different populations. (flow-5)

## 6. Prioritized, actionable recommendations

1. **[high] Fix the abstract status sentence to exempt the confirmatory study.** In `unified_report.tex:127–129`, change "All evidence here is *retrospective and estimation-only*… so no causal or fair-lending claim is licensed" to scope the estimation-only claim to Acts I–III and carve out the preregistered adaptation study (`\Cref{sec:adaptation}`), retaining the panel-conditional/no-fair-lending clause. (Resolves consistency-1, flow-4, and part of flow-1.)

2. **[high/medium] Reconcile the KL-SFT claim with its own confirmatory verdict.** In `unified_report.tex:315–319` (Row 3): replace "free at inference" with "at no extra inference cost," delete "it keeps specialization," add that the represented cost is quantified in Row 7, and mark Row 3 retrospective/interval-free (n=4). Ensure Row 3 and Row 7 headlines each name their population ("on general checkpoints" / "on a released guard"). (Resolves overclaim-1, flow-5.)

3. **[medium] Fix the "15/15 seeds" number.** In `unified_report.tex:311`, change the second `\SpecializationSeedCount` to `\TotalSeedCount` → 15/20. (Resolves numbers-1.)

4. **[medium] Add the confirmatory study to the evidence ledger and correct the closing takeaway.** In `limitations-validation.tex`: add a fifth `tab:ledger` row for the adaptation study (flavor: preregistered/confirmatory but panel-conditional; RQ1 supported, RQ2 not supported; limits: general-vs-purpose descriptive/not-randomized, panel-conditional bounds, Llama-Guard null cell); add a confirmatory slot to the flavor framing (line 50); and amend the closing takeaway (line 102) so it no longer labels the whole report estimation-only. (Resolves consistency-3, flow-3.)

5. **[medium] Signpost the confirmatory study in the intro.** In `unified_report.tex`: add a sixth contribution item (preregistered 10-checkpoint/6-family study testing RQ1 [supported] and RQ2 [not supported]) after line 193, and a one-line roadmap pointer after 171; add a clause to abstract "What is new" (117). Adjust the "estimation-only characterization… plus one external domain replication" phrasing (195–197) to acknowledge the confirmatory study. (Resolves consistency-2, flow-1.)

6. **[medium] Add a drop-degenerate (leave-one-family-out) sensitivity for the four H bounds.** In `act-adaptation.tex` "What this does and does not license," state explicitly that the Llama null cell is retained as a ~0 contribution and report drop-Llama point estimates (H_gain 0.174→0.208, H_conc 0.239→0.286, H_held_sft −0.065→−0.078, H_preserve 0.049→0.059, H_cost −0.036→−0.043), noting removal only strengthens RQ1 and makes RQ2's failure cleaner. Emit these as generated macros (extend `experiments/emit_adaptation_tex.py`). (Resolves stats-1.)

7. **[medium] Disambiguate "family" and quantify the dominant-family caveat.** In `act-adaptation.tex:24–25,54–55`, use "model families" vs "evaluation-row clusters" vs "RQ hypothesis groups," state that between-model-family variance is excluded from the bounds, and quantify the smollm-family sensitivity (H_gain +0.174 → +0.124 when dropped). (Resolves stats-4.)

8. **[low] Reposition the adaptation section or rewrite its opener.** Preferred: reword `act-adaptation.tex:4–6` to reference only Act I + the KL-SFT control (`sec:actI-klsft`) and drop the "Acts I–III" forward reference. Alternative: move `\input{sections/act-adaptation}` from `unified_report.tex:207` to just before Synthesis (after the Act III/ExpGuard block), restoring the Act I→Act II handoff. (Resolves flow-2.)

9. **[low] Propagate the mortgage directional qualifier to the abstract.** In `unified_report.tex:107–111`, add "(on a small, directional split)" to the mortgage/fairness clause and soften "the winner flips with the benchmark" to "the numerically top-ranked guard differs across benchmarks (directional)"; leave the already-disclosed ExpGuard tie as is. (Resolves overclaim-2, overclaim-3.)

10. **[low] Surface the held-out SFT bound.** In `act-adaptation.tex` RQ1, emit and show the H_held_sft one-sided UCB (−0.047 < 0) beside `\AdaHheldSFT` (−0.065), or reword "bound-confirmed" to point only to the displayed H_gain/H_conc LCBs. (Resolves numbers-2.)

11. **[low] Credit the confirmatory step in the conclusion.** In `unified_report.tex:450–451`, note that the preregistered adaptation study already locks its estimands/decision rules but still scores on the inspected Paper A manifest, so the remaining step is a prospective cohort on genuinely uninspected data (matching `limitations-validation.tex:90`). (Resolves consistency-4.)

12. **[low] Add figure callouts and fix figure order.** Add `\Cref` references for `fig:act1-bars` (Act I opposing-signs discussion), `fig:act3-bars` (Act II composition recovery), `fig:quadrant` and `fig:mortgage-bars` (Act III); and relocate the `fig:act1-bars` float (`unified_report.tex:208–212`) into/adjacent to Act I, before the adaptation `\input`. (Resolves flow-6.)

13. **[trivial] Reconcile "Acts I–II" vs "Acts I–III"** between the Synthesis caption (line 292) and `act-adaptation.tex:4`.