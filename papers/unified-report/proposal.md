# Proposal: Act II — Purpose-built guards and benchmark fragility

## Reviewer verdict and required changes

*Added by an independent multi-perspective review (5 fact-verifiers + 3 adversarial reviewers). The proposal's factual foundation verified as unusually solid; the changes below are conditions on adding the act, not a rejection.*

**Verdict: ADD — but NARROW and sequenced behind a governance decision.** The act closes a real, self-flagged gap: the report cites five released guards, scores none, and lists scoring them as a roadmap item. Its defensible core — RQ3 (ranking fragility) + RQ4 (contract dependence) + the exposure audit — directly hardens the "benchmark chooses the winner" thesis, which today rests on a single near-tie flip. But (a) it collides with a preregistered study for the Act II slot; (b) the full six-RQ program is paper-sized and re-broadens a deliberately narrowed report; and (c) by the proposal's own admission the vendor arms are *descriptive lineage contrasts*, not the report's signature paired-causal estimand.

**Do first (blocking) — resolve the Act II-slot collision with a recorded decision.** STATUS.md and `docs/unified-report-plan.md` assign Act II to the (preregistered, unrun) DPO/GRPO objective study, and the committed manuscript has *separately* already renumbered composition to displayed Act II. Reconcile all three surfaces (manuscript labels, planning docs, prereg) explicitly — default: keep the objective axis (Paper C) as Act II and place specialists as a later act (e.g. 5-act: Specialize / Objective axis / Compose / Purpose-built guards / Domains) or a companion paper; record the decision, including a dated revision in `paper-c-prereg.md` to preserve the timestamp chain. Do not silently evict the prereg.

**Ship this first (MVV, retrospective):** 3 ungated families — Qwen3Guard-Gen-0.6B, Qwen3Guard-Gen-4B, Granite Guardian 3.1-2B — under their native contracts on the existing locked panels, with RQ3 as the headline plus RQ4 and the exposure matrix. Defer to v2: gated models (Llama Guard, WildGuard), the five-seed controlled bridges (RQ1), the common-contract grid, and the fresh sealed cohort. GPU is not the binding cost (~30–60 A100-hr); the long poles are ~6 native-contract scorers and the fresh cohort's blinded human annotation.

**Frame it as** an external-validity / benchmark-sensitivity test of the thesis, NOT an extension of the paired-causal estimand. Name the exposure matrix and the native-vs-common contract-interaction estimand `I` as the methodological deliverables; cite one-universal-prompt guard comparisons (GuardBench) as the explicit contrast.

**Required methodology fixes before preregistration** (details inline):
- Resolve the product-comparison contradiction (Sec 2 vs Sec 6.1): define two named comparisons — common-contract and native-contract — each with its confound stated. *(fixed inline below)*
- Split the claim gate (Sec 2): a released-guard geometry claim needs only the native-contract + two-axis-exposure conditions; the SFT-preflight/five-seed conditions gate only the controlled bridge (RQ1).
- Give `S_i` and `I_{i,R}` paired bootstrap CIs; compute `S_i` on prevalence-matched AP; keep `S_i` exploratory-only (Sec 6.2, 7).
- Specify the prevalence-reweighting estimator; predeclare a minimum effective-positive count below which the 1% cell is "unstable / not claim-bearing" (Sec 11).
- Define the single bootstrap replicate over the mixed family-aware (Paper A) / shared-row (ExpGuard) surface; build a family graph for ExpGuard; disclose the mixed-independence assumption on every rank statistic (Sec 11).
- Label all rank-fragility summaries descriptive-only with comparison counts; derive headline reversals only from the predeclared primary contrast set (Sec 11).
- Bind each interpretation-matrix row (Sec 12) to a quantitative predicate + priority order in `claim_registry.json`, byte-check the selected row, and add an alternative-explanation guard to row 1 (concentration must survive the native-policy-coverage view and prevalence control).
- Fold specialist domain results (RQ5) into the existing Act IV rather than a parallel domain arm.

**Blocking prerequisites, independent of this act:** complete-or-remove the KL-SFT scaffold (still committed as all-`[pending]` placeholders rendered by live prose) and harden `reproduce.py` to fail closed even without `--check` (offenders: `expguard()` discards the subprocess return code; `mortgage()` gates on `src.exists()` rather than return codes; the `paper_a` pinned-env skip and `PENDING` pass the gate).

**Factual corrections applied inline below** (all low/med — the proposal is otherwise accurate): renumbering breaks no `\Cref` (Sec 3); the prereg file itself does not use the label "Act II" (Sec 3, 16.9); ShieldGemma's parent is "Gemma 2 2B", not "Gemma 2 IT 2B" (Sec 5.1/8); the five guards contribute no claim-bearing *numbers* but Llama Guard/WildGuard appear as declared-skipped gated footnotes in mortgage `tab:baseline` (Sec 1); the harness exit-status bullet refined to the real gap (Sec 14).

---

## Working research question

> **Do purpose-built guard checkpoints show the same specialization geometry and benchmark-dependent ranking fragility as guards produced by the report's common LoRA-SFT recipe?**

## Status and decision

This document proposes a new act for the unified report. It is a study protocol, not evidence that the answer is yes. No result from this proposal should enter the abstract, conclusion, or headline tables until the locked scoring, analysis, and release gates below pass.

**Recommendation: add the act.** It would materially strengthen the report because it tests whether the central finding survives beyond four general instruction checkpoints, two model lineages, and one researcher-controlled SFT recipe. If executed successfully, it would replace the current missing-baseline caveat with a practical comparison against guard products a practitioner might actually deploy.

The act must not become a conventional leaderboard. The scientific contribution is the combination of:

1. lineage-paired comparisons;
2. native scoring contracts rather than one invalid universal prompt;
3. an explicit model-by-benchmark exposure audit;
4. the report's existing family-aware metrics and calibration discipline;
5. rank-instability analysis across benchmark, policy, prevalence, and operating point; and
6. a new lock → scores → generated tables → manuscript evidence chain modeled on the strict Paper A release contract.

The safest manuscript title is:

> **Act II — Purpose-built guards: do released specialist checkpoints escape benchmark fragility?**

The user-facing question may ask whether purpose-built training shows the same specialization, but the manuscript must always describe what is actually observed: **released specialist checkpoints and their descriptive lineage contrasts**. Even three complete bridges cannot identify a vendor training effect because vendor data, compute, seeds, selection, and transformation details remain uncontrolled. Complete bridges permit stronger statements about recurring *geometry*, never causal vendor-training language.

---

## 1. Why this makes the report stronger

The current report establishes a careful result for one fixed panel:

- four compact instruction checkpoints;
- two lineages, Qwen and SmolLM;
- one frozen 1,200-row training manifest;
- one common LoRA-SFT recipe over five seeds; and
- one common `safe`/`unsafe` decision head.

That design is strong for estimating what *our* SFT recipe changed, but it leaves an obvious external-validity question: do guards explicitly built by Google, Qwen, IBM, Meta, and AI2 escape the same represented-source concentration, threshold drift, and benchmark-dependent ranking?

The report currently cites ShieldGemma, Llama Guard, Granite Guardian, WildGuard, and Qwen3Guard in [related work](sections/related-work.tex), but none contributes a claim-bearing result number. Where a released guard is named in a claim-bearing table (the mortgage `tab:baseline`), Llama Guard 3-1B and WildGuard-7B appear only as declared-skipped, gated (not-scored) footnote entries. (Note: the WildGuardTest and WildJailbreak *rows* that do appear in the transfer tables are benchmark datasets, not the WildGuard guard model.) Legacy ShieldGemma scripts exist under `legacy/` but do not have a current locked per-row evidence bundle. Those historical traces may guide implementation, but they are not admissible results for the new act.

A rigorous specialist act closes four current gaps at once:

- **Lineage breadth:** it adds Gemma, Granite, Llama, and Mistral-derived guards.
- **Training-recipe breadth:** it asks whether released specialist checkpoints show the same geometry as the common SFT recipe.
- **Practical relevance:** it compares models built specifically for moderation rather than only instruction checkpoints prompted as guards.
- **Title-level evidence:** it directly measures whether “the benchmark chooses the winner” even among purpose-built guards.

This does **not** turn the report into a population study. The expanded panel remains fixed and purposively selected.

---

## 2. The central validity rule: same methodology, different interpretation

The new act should reuse the report's rows, identity keys, family graph, tie-aware metrics, calibration split, operating-point protocol, bootstrap discipline, and artifact locks. It should **not** reuse the claim that every comparison isolates a controlled fine-tune.

There are three distinct quantities:

| Evidence arm | Comparison | Interpretation |
|---|---|---|
| Controlled bridge | declared parent checkpoint → our five-seed LoRA-SFT | Same-checkpoint effect of the report's frozen recipe, when the architecture passes the common-recipe preflight. |
| Released-lineage contrast | declared parent checkpoint → published specialist guard, under one held-fixed native contract | Descriptive lineage contrast. Vendor data, compute, selection, seeds, and sometimes exact parent revision are not controlled. |
| Product comparison | our SFT guard versus released specialist — reported in two named forms: **common-contract** (both under one locked contract) and **native-contract** (each under its own intended contract); see §6.1 | Practical comparison of two deployable products, not a training-algorithm comparison. The two forms are not pooled. |

These arms may appear side by side, but their deltas must never be pooled or described with the same causal language.

### Pairing tiers

Every specialist receives one locked pairing tier:

- **Tier A — documented public ancestor:** the model card identifies a public parent checkpoint and the native interface can be applied to both parent and guard.
- **Tier B — family-level ancestry:** the family is documented, but the exact parent revision or transformation is not recoverable.
- **Tier C — no defensible parent:** the model is an absolute external comparator only.

Even Tier A is not automatically causal. Llama Guard 3-1B, for example, includes pruning and distillation in addition to fine-tuning. “Documented ancestry” is not the same as “only one controlled intervention changed.”

### Claim gate

Describe a recurring specialization-like geometry only if all of the following hold:

1. at least three families have Tier A ancestry;
2. their parent checkpoints pass the unchanged Paper A prompt/token/LoRA preflight;
3. each controlled arm completes five seeds;
4. each parent and released guard is scored under the exact same native contract within its lineage contrast; and
5. the exposure analysis supports the terminology used for each benchmark regime.

If these conditions are not met, title the act **“Benchmark sensitivity of released purpose-built guards.”** Training-effect language is prohibited in either case.

---

## 3. Recommended four-act narrative

Insert the specialist act between the present Act I and composition:

1. **Act I — Controlled SFT specializes.** What does one common fine-tune change?
2. **Act II — Purpose-built guards.** Does the same geometry recur outside our recipe, and does the winner remain benchmark-dependent?
3. **Act III — Composition.** Can keeping the base recover transfer after specialization?
4. **Act IV — Regulated domains.** Do general and purpose-built guards cover domain policy, and does the winner flip again?

This ordering makes the argument cumulative: establish the phenomenon, test its breadth, test a repair, then test domain relevance.

The current source has a numbering gap: it links `sec:actI`, `sec:actIII`, and `sec:actIV`, but no Act II section or `sec:actII` label exists. Composition is displayed as Act II but labeled `sec:actIII`, and mortgage is displayed as Act III but labeled `sec:actIV`. Renumbering breaks **no** `\Cref`: `sec:actIII` (composition) and `sec:actIV` (mortgage) MUST be retained (every existing `\Cref{sec:actIII}`/`\Cref{sec:actIV}` still resolves), and a new `\label{sec:actII}` is simply added — after which the labels finally match the displayed numbers. The only real edits are the new section plus the many literal `Act~II`→`Act~III` / `Act~III`→`Act~IV` prose strings and comments (checklist item 10 covers this: `sections/act3.tex`, `related-work.tex`, `background-setup.tex`, `limitations-validation.tex`, `unified_report.tex`).

Two active planning tracks conflict with this numbering: [STATUS.md](STATUS.md) and [the unified-report plan](../../docs/unified-report-plan.md) label the objective axis Act II, and [the Paper C preregistration](../../docs/paper-c-prereg.md) is the committed preregistration of that same unrun DPO/GRPO objective study (the prereg file itself does not use the label "Act II"). Separately, the committed manuscript has already silently renumbered composition to displayed Act II without updating any of these — so this is a three-way inconsistency (manuscript labels vs. planning docs vs. prereg) that the governance decision must reconcile, not just insert a new section. Before editing the manuscript, make and record an explicit governance decision: defer/renumber the objective study, place the specialist study later, or keep it outside the unified report. Do not silently cancel or overwrite a preregistered study.

---

## 4. Research questions

### RQ1 — Controlled bridge

When the exact Paper A recipe is applied to additional eligible parent lineages, does the controlled base → our-SFT movement reproduce Act I's represented-source gain and heterogeneous transfer change?

This is the only arm that can inherit the existing same-checkpoint language.

### RQ2 — Released specialist geometry

For each documented parent/specialist lineage, what is the released guard's paired movement on the Act-I source panel, Act-I transfer panel, stress sets, and a fresh post-lock cohort?

This is a descriptive released-lineage contrast unless the entire vendor transformation is independently reproducible, which is not expected.

### RQ3 — Ranking fragility

Does the ordering of purpose-built guards change across:

- individual benchmarks;
- the Act-I source and transfer panels;
- documented-exposure versus no-documented-exposure views;
- native versus common scoring contracts;
- unsafe prevalence assumptions;
- native versus common-FPR operating points; and
- general-safety versus regulated-domain labels?

This is the act's strongest and most defensible headline question.

### RQ4 — Policy-contract dependence

How much of a specialist guard's apparent advantage depends on its native taxonomy, policy text, output schema, or strictness mapping rather than weights alone?

The scoring contract is an experimental factor, not an implementation detail.

### RQ5 — Domain-policy coverage

Do purpose-built general guards improve mortgage general-safety ranking (`G`), and do policy-configurable guards improve mortgage-policy ranking (`D`) when given one locked domain policy definition?

Out-of-box general safety and policy-conditioned mortgage evaluation are different arms and must not be pooled.

### RQ6 — Optional composition bridge

For Tier A specialist lineages, does a calibrated parent + released-guard average recover low-exposure or domain transfer, as parent + our-SFT composition does in the current report?

This is secondary. It has no five-seed vendor arm and cannot support the existing SFT+SFT mechanism control.

---

## 5. Proposed model panel

### 5.1 Launch panel: five families, six specialist checkpoints

| Specialist checkpoint | Declared parent or family | Native prompt-only output | Access/license | Role |
|---|---|---|---|---|
| [`google/shieldgemma-2b`](https://huggingface.co/google/shieldgemma-2b) | Gemma 2 2B (the card/tech report state only "Gemma 2"; base-vs-IT variant is not documented); a chosen comparator revision must be locked, and the vendor-used parent revision is not assumed | Policy-conditioned `Yes`/`No`; one score per harm policy | Gemma license; Hugging Face access gate | Core |
| [`Qwen/Qwen3Guard-Gen-0.6B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | `Qwen/Qwen3-0.6B`; exact vendor-used revision is not disclosed | `Safety: Safe/Unsafe/Controversial` plus categories | Apache-2.0; ungated | Core efficiency point |
| [`Qwen/Qwen3Guard-Gen-4B`](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) | `Qwen/Qwen3-4B`; overlaps the report's Qwen3-4B anchor, but vendor-used revision remains to be evidenced | Same three-level structured output | Apache-2.0; ungated | Core bridge |
| [`ibm-granite/granite-guardian-3.1-2b`](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b) | documented public ancestor ID `ibm-granite/granite-3.1-2b-instruct`; exact vendor-used revision is not disclosed | `Yes` risk / `No` safe under `risk_name=harm` or a custom risk | Apache-2.0; ungated | Core documented-public-ancestor arm |
| [`meta-llama/Llama-Guard-3-1B`](https://huggingface.co/meta-llama/Llama-Guard-3-1B) | Llama-3.2-1B pretrained; guard also uses pruning/distillation | `safe`/`unsafe` then hazard categories | Llama 3.2 Community License; gated | Core if access is accepted |
| [`allenai/wildguard`](https://huggingface.co/allenai/wildguard) | `mistralai/Mistral-7B-v0.3` | `Harmful request: yes/no` plus response/refusal fields | Apache-2.0 weights, AI2 access terms; gated | Extended core / home-benchmark stress |

Why both Qwen sizes: 4B supplies the cleanest bridge to the existing report, while 0.6B tests whether the ranking story changes under a much tighter deployment budget.

Why Granite 3.1 rather than the newer compact 3.2 MoE: Granite Guardian 3.1 names a public instruct ancestor ID, although not the exact revision used by IBM. The 3.2 3B-A800M model is useful as a deployment sensitivity but does not currently provide the same public-parent bridge.

Why keep WildGuard despite 7B scale and benchmark-family overlap: its “home-field” relationship to the distinct WildGuardTest held-out split is scientifically useful if it is exposed rather than hidden. WildGuard training also uses related WildGuardMix/WildJailbreak material and XSTest-inspired benign categories. The low-exposure sensitivity must exclude every documented training/derivative or constructionally related source, not only WildGuardTest.

The paper should use marketed checkpoint names but lock and report the loaded parameter count, active parameter count where applicable, peak memory, and latency. Hugging Face metadata currently reports approximately 3B loaded parameters for the models marketed as ShieldGemma-2B and Granite Guardian 3.1-2B, and approximately 0.8B for Qwen3Guard-Gen-0.6B; names must not be treated as exact resource measurements.

### 5.2 Optional references

- [`nvidia/Nemotron-Content-Safety-Reasoning-4B`](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B): useful ungated policy-conditioned contemporary reference under the NVIDIA Open Model License plus Gemma terms, but its reasoning/structured-output path is not identical to the compact one-token classifiers.
- Granite Guardian 3.2 3B-A800M: deployment sensitivity only, not a primary lineage pair.
- [`meta-llama/Llama-Guard-4-12B`](https://huggingface.co/meta-llama/Llama-Guard-4-12B): gated Llama 4 scale/multimodal ceiling only; outside the report's compact text-only core.
- [`meta-llama/Llama-Prompt-Guard-2-86M`](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M): gated Llama 4 injection/jailbreak-only positive control, never an overall content-moderation competitor.
- Qwen3Guard-Stream: streaming token-classification study, not part of the prompt-level core act.
- [`google/shieldgemma-2-4b-it`](https://huggingface.co/google/shieldgemma-2-4b-it): image safety, not a substitute for the text ShieldGemma model.

Optional models cannot enter the primary panel after results are visible. Promotion requires a protocol amendment with a new hash and an explicit exploratory label.

---

## 6. Study architecture

### 6.1 The lineage bridge

For eligible lineage `i`, define:

- `B_i`: the chosen public parent comparator;
- `A_i,r`: our frozen Paper A LoRA-SFT recipe at seed `r ∈ {42,…,46}`; and
- `G_i`: the published purpose-built guard.

The report may display all three products together, but it must calculate and interpret two internally paired contrasts separately:

1. **Common-recipe contrast:** `B_i → A_i,r` under the unchanged Paper A classifier contract.
2. **Released-lineage contrast:** `B_i → G_i` under the specialist's native contract, held byte-identical across the parent and guard.

The direct `A_i` versus `G_i` gap is a **product comparison**, reported in two explicitly named forms, each with its confound stated (this resolves the apparent tension with the Section 2 table): a **common-contract** product comparison (both scored under the same locked contract — mechanically standardized, but the generic contract may be out of scope for, and can disadvantage, the released guard) and a **native-contract** product comparison (each under its own intended contract — preserves intended use, but confounds product quality with the scoring contract, so it is descriptive-only). Neither is a training-recipe comparison, and the two forms must not be pooled.

### 6.2 Full interface grid

Where mechanically valid, score:

`{parent, our SFT, released guard} × {common contract, native contract}`.

The native contract is primary for practical guard ranking. The common contract is a sensitivity analysis showing whether a result survives prompt/policy standardization. A poor released-guard result under an unsupported generic prompt is not evidence that the released product is weak.

The contract interaction for a released lineage is:

`I_i,R = [M_R(G_i,native) − M_R(B_i,native)] − [M_R(G_i,common) − M_R(B_i,common)]`.

A large interaction means the apparent specialist advantage depends materially on the prompt/policy interface. That directly supports the report's thesis that the benchmark and scoring contract co-produce the winner.

The native-contract parent score is itself subject to an eligibility gate. If a parent checkpoint—especially a pretrained Llama or Mistral parent—cannot produce nondegenerate, contract-faithful candidate scores with acceptable generation/likelihood concordance, the released model remains an absolute product comparator; a failed parent adapter is not interpreted as a large training gain.

### 6.3 Common-recipe eligibility preflight

Before training any new parent, fail closed unless all checks pass:

- exact model and tokenizer revisions resolve;
- the full rendered Paper A prompt is hashable and stable;
- `safe` and `unsafe` are distinct single tokens under the exact Paper A convention;
- the unchanged LoRA target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` exist;
- completion-only labels and EOS behavior match training/scoring;
- the wrapper survives the 1,024-token truncation budget;
- a 32-row train/eval smoke run produces finite, nonconstant margins;
- base and adapter scoring are identity-complete; and
- the license permits the experiment and release of text-free scores.

Do not change the locked Paper A artifacts or release-bound modules to add these models. Build a separate specialist pipeline that imports reusable metric logic without mutating the Paper A release contract.

If decision tokens or target modules are incompatible, the checkpoint is ineligible for the **unchanged** bridge. An architecture-adapted recipe may be proposed under a separately named and locked protocol variant, but it cannot inherit Paper A's unchanged-recipe claim. Failure under one unswept recipe is not evidence that a lineage cannot be tuned.

---

## 7. Exact estimands and claim vocabulary

Let `M_i,k,c` be tie-aware AP for checkpoint `i`, benchmark `k`, and scoring contract `c`.

### Controlled common-recipe movement

For regime `R`:

`Δ_ours(i,R) = mean_r [M(A_i,r,R,common) − M(B_i,R,common)]`.

This can be called a same-checkpoint fine-tuning estimate when the base revision is identical and all existing controls remain fixed.

### Released specialist movement

`Δ_released(i,R) = M(G_i,R,native) − M(B_i,R,native)`.

Call this an **observed released-lineage contrast**, not a causal fine-tuning effect.

### Specialization geometry

For the controlled common-recipe arm, keep the Paper A contrast as a two-dimensional vector:

`θ_ours,i = (Δ_represented, Δ_transfer)`.

For a released guard, the Paper A source panel is **not** automatically represented and cannot define vendor specialization. A released-specialist concentration claim requires a model-specific, locked exposure view:

`θ_released,i = (Δ_documented-training-or-construction-related, Δ_low-exposure-or-chronologically-post-weight)`.

The exposure matrix must supply defensible rows on both axes. If it cannot, report only benchmark sensitivity and omit specialization-like language for that released guard. Paper A source-panel versus transfer-panel results remain a comparability view, not the vendor-specialization estimand.

A secondary exploratory concentration sensitivity, only where both model-specific axes exist, is:

`S_i = Δ_high-exposure − Δ_low-exposure`.

Never rank models by `S_i` or replace the two-dimensional movement plane with it. The report's primary scientific object remains the movement vector; this scalar only helps inspect one predeclared sensitivity.

For our SFT arm, “represented” and “transfer” retain the Paper A definitions. For vendor guards, use “Act-I source panel” and “Act-I transfer panel” unless vendor exposure has been independently established.

### Ranking fragility

For models `i,j` and benchmark `k`:

`d_i,j,k = AP_i,k − AP_j,k`.

Report:

- paired bootstrap intervals for every preregistered `d_i,j,k`;
- Kendall's tau-b between benchmark-specific rankings;
- each model's maximum rank displacement;
- pairwise sign-reversal rate across benchmark pairs;
- top-1 turnover across benchmark/policy/prevalence views;
- bootstrap probability that each model is top-ranked; and
- a predeclared practical tie band, proposed as `|ΔAP| < 0.01`, with sensitivity at `0.005` and `0.02`.

Do not infer a tie merely because two separate marginal confidence intervals overlap. Use the paired interval for the difference.

---

## 8. Model-specific scoring contracts

One universal `safe`/`unsafe` prompt would invalidate the comparison. Every adapter must preserve the official prompt structure and score semantics.

ShieldGemma and Granite document continuous token-probability scoring. Qwen3Guard-Gen, Llama Guard 3-1B, and WildGuard officially document generated verdicts; the continuous candidate-likelihood scores proposed below are **research-derived adapters**, not vendor-published native scores. Each derived adapter therefore needs a locked definition and a generation-versus-likelihood fidelity gate before its AP can be claim-bearing.

### ShieldGemma-2B

- Use the official policy-conditioned prompt.
- Run the four advertised prompt policies separately: dangerous content, harassment, hate, and sexually explicit content.
- For each policy, store `logit(Yes) − logit(No)` and the normalized two-label probability.
- Predeclare union-risk aggregation as the maximum normalized `P(Yes)` across policies, matching the ShieldGemma report; retain each normalized probability and raw margin separately.
- Hash exact policy text and aggregation rule.
- Never silently replace its native policies with the report's one-line generic definition.

### Qwen3Guard-Gen

- Preserve its structured `Safety: Safe|Unsafe|Controversial` contract.
- Under one locked forced prefix ending in `Safety:`, compute complete candidate-sequence log likelihoods `ℓ_S`, `ℓ_U`, and `ℓ_C`, including a predeclared EOS rule; this is a research-derived continuous score, not an official scalar.
- Define strict ranking score `logsumexp(ℓ_U,ℓ_C) − ℓ_S`; define the permissive sensitivity as `ℓ_U − logsumexp(ℓ_S,ℓ_C)`.
- Primary mapping: strict, `Unsafe + Controversial` counts as unsafe.
- Sensitivity mapping: `Unsafe` only.
- Fix this mapping globally before scoring; never choose the better mapping separately for each benchmark.
- Validate likelihood argmax against deterministic generation on a locked development sample and set a preregistered concordance threshold before full scoring.
- Use the real repository IDs `Qwen3Guard-Gen-*`; do not copy the reversed-ID typo present in one response example on the model card.

### Granite Guardian 3.1-2B

- Use its native chat template with `guardian_config={"risk_name":"harm"}` for the general-safety arm.
- Store `logit(Yes) − logit(No)` and normalized risk probability.
- For domain-policy experiments, use exactly one separately locked custom risk definition.
- Do not mix RAG groundedness, function-calling, or response-only risk modes into the prompt-only primary table.

### Llama Guard 3-1B

- Use the official chat template applied to a single-user conversation; the same model also supports response classification, which remains outside the core act.
- Store a research-derived normalized first-verdict `unsafe` versus `safe` margin after verifying the complete token sequences.
- Record generated hazard categories as diagnostics, not as replacements for the binary primary score.
- Record that the released model includes pruning/distillation; do not call its parent contrast pure SFT.

### WildGuard-7B

- Use the full official structured prompt, not the simplified prototype currently in the mortgage scorer.
- Score the first `Harmful request: yes/no` field for the prompt-only task.
- Store research-derived forced candidate likelihoods and deterministic generated output using the official prompt-harm package path.
- Treat malformed fields as explicit errors/abstentions; never default them to safe.
- Report a full documented-overlap exclusion sensitivity covering WildGuardTest, WildJailbreak-related material, and any other source classified as training/derivative or constructionally related in the exposure matrix.

### Universal scorer requirements

Every scorer must emit:

- raw candidate log likelihoods;
- derived continuous unsafe margin;
- native generated verdict;
- parse/concordance status;
- prompt/contract hash;
- policy ID and mapping version;
- token count, truncation status, and wrapper-preservation flag;
- per-row latency and batch configuration; and
- model/tokenizer revision and runtime fingerprint.

Store raw margins, not only sigmoids. A mathematical sigmoid is monotone and does not change AP in exact arithmetic, but finite-precision or rounded serialization can saturate extreme values into ties; storing raw margins prevents that reproducibility failure.

---

## 9. Benchmark roles and exposure control

### 9.1 Do not call the current transfer panel vendor-unseen

Vendor training and evaluation exposure is model-specific and partly opaque. Existing documented overlap includes examples such as ToxicChat, HarmBench, XSTest-related evaluation, Qwen evaluation on WildGuardTest, and WildGuard's own home benchmark family. “Not disclosed” is not evidence of non-exposure.

Exposure evidence must preserve task and variant. Qwen and Granite documentation may expose XSTest on response classification while this report uses prompt rows; that is developer familiarity, not automatically row overlap. Likewise, WildGuardTest is a distinct held-out home split rather than demonstrated training-row leakage. The matrix must encode these distinctions instead of collapsing every related name to “seen.”

Create a locked `model_benchmark_exposure_matrix.json` with one row per model × benchmark and these fields:

- `model_id` and revision;
- benchmark identifier/revision/hash;
- `exposure_status`;
- `exposure_task` (`prompt`, `response`, `mixed`, or `unknown`);
- `exposure_variant` (for example, XSTest prompt versus response, or WildGuard train versus held-out test);
- `row_overlap_status` (`checked_none`, `detected`, `not_checkable`, or `unknown`);
- source URL and source snapshot hash;
- evidence note;
- pinned-weight publication timestamp, row collection/creation provenance, and benchmark publication date; and
- reviewer sign-off.

Allowed statuses:

1. `documented_training_or_derivative`;
2. `constructionally_related_or_home_benchmark`;
3. `documented_author_evaluation`;
4. `no_documented_overlap` — explicitly not proof of cleanliness;
5. `unknown`; and
6. `chronologically_post_pinned_weights`.

Track researcher exposure separately:

- `development_visible`; or
- `sealed_until_protocol_lock`.

Publication after a model release is not enough: rows may have been collected or circulated earlier. “Model-unseen by chronology” requires auditable row-creation provenance after the pinned weight timestamp. Researcher prospectivity is a separate condition requiring that rows remain sealed until after preregistration. Only rows satisfying both conditions can support a genuinely prospective confirmation.

### 9.2 Required benchmark views

1. **Paper A comparability view:** the original represented-source, transfer, OR-Bench, and HarmBench groupings. These labels describe our common-recipe arm only.
2. **Vendor exposure view:** documented/related, author-evaluated, no-documented-overlap, and unknown.
3. **Native-policy view:** benchmark label is covered, ambiguous, or outside the guard's declared policy.
4. **Domain view:** ExpGuard finance/health/law and mortgage `G`/`D`, kept in their existing evidence tiers.
5. **Prospective view:** a newly sealed cohort, required before any confirmatory generalization claim.

### 9.3 Fresh cohort requirement

The current rows and legacy ShieldGemma comparisons have already been inspected. Locking a new script does not make those rows prospective.

The strongest version of the act should therefore add a fresh prompt-only cohort whose:

- collection and annotation plan is fixed before text is inspected;
- auditable row-creation timestamps postdate the pinned guard weights where possible;
- labels come from blinded, independent human/expert annotators appropriate to the target policy;
- item selection and exclusions occur without viewing candidate-model outputs;
- adjudication rules, exclusions, and inter-annotator agreement are locked and reported;
- family/near-duplicate graph is created before splitting;
- sample size is set by a pre-run precision/power report rather than convenience;
- calibration and test partitions are sealed separately; and
- text redistribution follows the source license.

Without this cohort, the act remains a valuable retrospective, estimation-only characterization.

---

## 10. General-safety and mortgage-policy evaluation must be separate

The mortgage benchmark's `G` and `D` labels answer different questions. A default ShieldGemma or Llama Guard miss on `D` may simply mean mortgage compliance is outside its native taxonomy.

Run two explicit arms:

### Arm A — Out-of-box native policy

All specialist guards use their published general-safety policy. Report:

- AP·G;
- AP·D as a **scope-mismatch diagnostic**, not an expected product target;
- `AP(D | G=0)` across G0/D1 positives and G0/D0 negatives, plus the G0/D1 score distribution and recall at the locked operating point;
- protected-pair score gaps; and
- native-policy coverage annotations.

### Arm B — Locked mortgage-policy conditioning

Only models that officially support supplied policy/risk definitions participate. Derive one consolidated mortgage policy solely from the pre-existing policy cards, audit its coverage independently, and hash it before scoring. This arm remains retrospective because the benchmark rows and cards have already been inspected. Do not feed row-specific policy-card IDs, gold categories, or rationales into the model; that would leak benchmark metadata.

Report separately:

- AP·D and paired difference versus native mode;
- G0/D1 recall at a calibration-selected threshold;
- benign false-positive cost;
- protected-pair mean and maximum score gap; and
- policy-prompt sensitivity under a predeclared paraphrase set; this is mandatory for ShieldGemma because its model card warns that results are sensitive to policy wording.

Never pool native-policy and mortgage-conditioned results. ExpGuard remains external expert-labeled breadth; mortgage remains LLM-judge, policy-card-consistent depth.

---

## 11. Metrics and statistical protocol

### Primary metrics

- Tie-aware, non-interpolated AP per benchmark.
- Macro-AP across predeclared benchmark groups.
- Paired AP differences on identical rows.
- The two-dimensional movement vector `(Δ_source-panel, Δ_transfer-panel)`.
- Pairwise rank reversals and Kendall tau-b across benchmarks.

### Secondary metrics

- AUROC.
- Temperature-calibrated Brier score and log loss.
- TPR at the existing calibration-selected 5% FPR target.
- Realized macro and pooled FPR.
- OR-Bench benign FPR.
- HarmBench recall.
- Mortgage G0/D1 recall and protected-pair invariance.
- P50/P90/P99 latency, peak accelerator memory, and throughput under two separately fixed protocols: forced-score mode and official generation mode. Each must bind warmup, hardware, dtype, batch size, input-length buckets, and generated-token budget.
- AP reweighted to predeclared unsafe prevalences of 50%, 10%, and 1%.

### Bootstrap

- Reuse family-aware resampling and identical family weights across all models being compared in one replicate **where a locked family graph exists**.
- Construct and lock family/near-duplicate graphs for new external cohorts before analysis. Current ExpGuard artifacts do not contain such a graph; until one is built, use a shared-row paired bootstrap and explicitly disclose the weaker independence assumption.
- Use the same row/family draw for every paired model difference within an evaluation surface.
- Resample five training seeds only for our common-recipe SFT arms.
- Do not invent seed variation for released vendor checkpoints; their intervals cover evaluation-family uncertainty only.
- Keep model identities fixed; this is not a sampled population of models.
- Report per-lineage results before any equal-weight fixed-panel average.
- Use paired difference intervals for ranking claims.

The full model × benchmark matrix is exploratory. Predeclare a small primary contrast set before scoring so the report cannot select whichever ranking reversal looks most dramatic afterward.

### Suggested primary contrast set

The preregistration must expand each item below into an executable record containing exact model and tokenizer revisions, dataset/split hashes, scoring contract, policy mapping, prevalence, statistic, bootstrap unit/seed/count, and direction. Side-by-side movement vectors from different contracts are descriptive; subtract them only when both systems are scored under the same locked contract.

1. Qwen3 parent checkpoint → our SFT and Qwen3 parent checkpoint → Qwen3Guard-Gen-4B, reported as separate common- and native-contract panels.
2. Gemma 2 parent checkpoint → our SFT and Gemma 2 parent checkpoint → ShieldGemma-2B, reported as separate common- and native-contract panels.
3. Granite parent checkpoint → our SFT and Granite parent checkpoint → Granite Guardian 3.1-2B, reported as separate common- and native-contract panels.
4. Specialist-panel rank concordance: Act-I source panel versus Act-I transfer panel.
5. Specialist-panel rank concordance: Act-I transfer versus ExpGuard.
6. Native general-safety versus mortgage-policy-conditioned ranking for configurable guards.
7. WildGuard result with versus without every documented training/derivative or constructionally related source.

Call a pairwise result **unresolved** when its paired interval crosses zero. Call it **practically equivalent** only when a predeclared equivalence interval lies wholly inside the locked practical tie margin; these are not synonyms.

---

## 12. Interpretation matrix fixed before results

| Observed pattern | Allowed interpretation |
|---|---|
| Released guards move up in a model-specific documented training/derivative or construction-related regime and down/flat on low-exposure or chronologically post-weight data | “A specialization-like concentration pattern appears in these fixed released lineages.” |
| Released guards improve both panels | “Released purpose-built guards show broader gains on these rows; the common-recipe specialization pattern does not generalize uniformly.” |
| Results are mixed by lineage | “There is no common released-specialist signature; lineage/policy matters.” |
| Rankings reverse across benchmarks/contracts | “Released purpose-built guards do not eliminate benchmark or policy dependence.” |
| Rankings remain stable with paired differences resolved | “The ranking-fragility extension was not supported on this fixed specialist panel.” |
| Native guards do well on `G` but poorly on mortgage `D` | “General moderation scope does not cover mortgage policy by default,” not “the guard is bad.” |
| Policy conditioning improves `D` but increases benign FPR or fairness gaps | “Policy coverage trades off against operating-point or invariance costs.” |
| No fresh sealed cohort is completed | All conclusions remain retrospective and estimation-only. |

Null and contradictory results remain reportable. The act is not contingent on reproducing the current headline.

---

## 13. Reproducibility and artifact contract

Create a new namespace; do not append files to the finalized Paper A release.

```text
configs/specialist_guards_v1.yaml
configs/specialist_guards_v1_release_anchor.json
requirements-specialist.txt
docs/specialist-guards-prereg.md
experiments/specialist_guard_common.py
experiments/preflight_specialist_guards.py
experiments/lock_specialist_guards.py
experiments/run_specialist_controls.py
experiments/eval_specialist_guards.py
experiments/analyze_specialist_guards.py
experiments/package_specialist_guards.py
tests/test_specialist_guards.py

artifacts/specialist_guards_v1/
  LOCK.json
  RELEASE.json
  protocol/
    primary_contract.json
    model_registry.json
    scoring_contracts.json
    policy_crosswalk.json
    model_benchmark_exposure_matrix.json
    primary_contrasts.json
    claim_registry.json
  public_manifests/
  manifests/                 # local-only, gitignored: licensed/raw text
  runs/                      # local-only, gitignored: adapters and run metadata
  base_scores/               # local-only, gitignored: rebuild caches
  smoke/                     # local-only, gitignored: nonfinal smoke outputs
  downloads/                 # local-only, gitignored: gated parquet/model caches
  scores/
    scores.parquet
    metadata.json
  analysis/
    results.json
    claim_checks.json
    sensitivity.json
    tables/
  provenance/
    execution-evidence.json
    execution-source-snapshot.json
```

`protocol/primary_contract.json` is the single normative source for the model panel, interfaces, policies, exposure schema, estimands, and primary contrasts. The YAML file is an authoring input only. All other `protocol/*.json` views must be generated from or validated against the normative contract hash so duplicated definitions cannot drift while remaining individually well-formed.

Add explicit `.gitignore` rules for every local-only path before the first smoke run. The release packager must use a positive allowlist; raw manifests, downloaded gated files, adapters, base caches, smoke outputs, symlinks, and credentials are forbidden even if they accidentally become unignored.

### LOCK.json must bind

- the exact preregistration bytes/hash;
- the normative `primary_contract.json` bytes/hash and every derived `protocol/*.json` hash;
- model and tokenizer IDs and immutable revisions;
- declared parent IDs, revisions, and pairing tiers;
- model-card/source snapshots supporting ancestry and exposure claims;
- license/terms URI and snapshot hash, without credentials;
- exact native and common prompt templates;
- every policy string, strictness mapping, and aggregation rule;
- candidate verdict sequences and score equations;
- dataset revisions, file hashes, row/family identities, and split roles;
- exposure matrix and researcher-visibility status;
- calibration and threshold protocol;
- primary contrasts, bootstrap seed/count, and tie band;
- source commit, every specialist execution-source file hash, and clean execution state;
- Python/model-library/CUDA environment; and
- score schema and expected matrix cardinality.

### Score schema

At minimum, each row should contain:

- `sample_id`, `content_sha256`, nullable `family_id`, explicit resampling unit, source, split, and gold label;
- `model_key`, model revision, parent key/revision, and pairing tier;
- condition (`parent`, `our_sft`, `released_guard`) and training seed where applicable;
- contract (`common`, `native`, `domain_policy`) and contract hash;
- policy ID and strictness mapping;
- candidate verdict strings and candidate log likelihoods;
- derived raw margin and calibrated probability;
- generated verdict, parse status, and likelihood/generation concordance;
- original/scored token counts and truncation status;
- latency, dtype, device, batch size, and runtime fingerprint; and
- benchmark-exposure status, task, variant, and row-overlap status.

The score release must include every calibration/dev score needed to reproduce thresholds, including mortgage dev scores. Saving only the reported test-split scores is insufficient for operating-point reproduction.

### Release rule

Release text-free row IDs/hashes, scores, metadata, contracts, and analysis outputs. Do not redistribute gated prompts, model weights, Hugging Face tokens, user credentials, or license-acceptance identity. Re-scoring may require gated access; analysis reproduction must require only committed scores.

Use a separate fully pinned `requirements-specialist.txt` and environment for specialist training/scoring. Do not modify Paper A-bound requirements, prompts, metrics, or core execution files merely to accommodate a specialist model; import stable interfaces or add new specialist modules so the finalized Paper A verification chain does not drift.

---

## 14. Required code hardening before claim-bearing runs

The existing specialist paths are useful prototypes but are not publication-ready:

- the mortgage `ChatYesNoGuard` reads the first sub-token without a complete sequence check;
- its WildGuard prompt is simplified rather than model-card exact;
- specialist entries in `baseline_guards.json` do not pin revisions;
- legacy parsers can turn malformed output into a hard label;
- ShieldGemma is absent from the canonical mortgage scorer;
- Qwen3Guard's three-level output has no locked continuous-score adapter;
- ExpGuard's current dataset download is not revision-pinned inside a dedicated external-validation lock; and
- the specialist artifacts do not yet have a release anchor.

The unified report reproduction harness should also be hardened before adding another evidence source:

- `--check` must fail when a committed generated destination is missing rather than generating it;
- `PENDING` must be a failure for a publication build;
- figure-generation, Tectonic, and subprocess failures currently fail the process **only** under `--check` (via the drift flag); a plain `reproduce.py` run returns `0` regardless of failure — the harness must fail closed even without `--check`;
- every scoring/analysis subprocess return code must be checked before stale outputs can be accepted (concrete current offenders: `expguard()` discards the subprocess return code; `mortgage()` gates on `src.exists()` rather than the return code; the `paper_a` pinned-env skip and any `PENDING` status silently pass the gate);
- a failed Tectonic `--build` must produce a nonzero process exit;
- expected model/table cardinality must be validated rather than relying on file existence; and
- every claim-bearing table must be byte-compared against the generated output.

The current KL-SFT scaffold is also pending and is not dispatched by the unified reproduction harness. Complete and reproduce that control, or remove its manuscript scaffold, before the report can satisfy a publication-wide “no `PENDING`” gate.

These are prerequisites, not optional cleanup.

---

## 15. Report outputs

### Generated tables

1. `specialist_panel.tex` — model, parent, pairing tier, license, policy, scorer, exposure summary.
2. `specialist_primary.tex` — parent, our-SFT mean/interval, released guard, source-panel and transfer-panel movements.
3. `specialist_per_benchmark.tex` — AP/AUROC by benchmark with exposure flags.
4. `specialist_rank_fragility.tex` — tau-b, rank range, top-1 turnover, pairwise reversals.
5. `specialist_operating_points.tex` — calibration, FPR/TPR, OR-Bench, HarmBench.
6. `specialist_domains.tex` — ExpGuard and mortgage results, evidence tiers separated.
7. `specialist_latency.tex` — latency/memory/throughput under one fixed protocol.

### Figures

1. **Lineage-paired movement plane:** facet common and native contracts; never subtract or draw a single shared arrow between systems scored under different contracts.
2. **Benchmark × guard rank heatmap:** cells annotated with exposure status.
3. **Winner map:** benchmark × contract × prevalence, with unresolved top sets rather than forced winners.
4. **Policy-contract interaction plot:** native versus common interface.
5. **Domain policy plot:** mortgage native-G versus conditioned-D performance and invariance.
6. **Cost-performance frontier:** quality, latency, and memory, with 7B/optional models clearly outside the compact core where applicable.

No result number should be typed manually into the TeX source. Result tables and figures must be generated from LOCK-bound scores; panel, policy, license, and exposure tables must be generated from LOCK-bound protocol artifacts.

---

## 16. Manuscript integration checklist

Only after the evidence release passes:

1. Add `sections/act2-specialists.tex` after Act I.
2. Change the displayed composition title to Act III and the displayed domain title to Act IV.
3. Expand the introduction from three to four questions.
4. Revise “the same four checkpoints recur across all three acts” into two explicit panels:
   - the controlled four-checkpoint panel for Acts I and III; and
   - the separately fixed specialist-lineage panel for Act II, with overlap into Act IV.
5. Add native scoring contracts and exposure terminology to the shared methods.
6. Update related work from a list of guard families to a design comparison.
7. Add Act II to the synthesis, practitioner guide, evidence ledger, limitations, roadmap, and reproducibility section.
8. Update the abstract and conclusion only with generated results that pass claim checks.
9. Record the explicit governance decision for the preregistered objective-axis study in `STATUS.md`, `docs/unified-report-plan.md`, and the Paper C preregistration; defer or renumber it rather than silently declaring it superseded.
10. Update every literal act reference in generated table captions, figure code, Graphviz sources, comments, README/status tables, and section prose—not only section headings.
11. Run an adversarial prose pass for causal language, “held-out” misuse, forced winner claims, and policy mismatch.

Suggested synthesis sentence if the observed results support it:

> Released purpose-built guards differ in performance level and taxonomy, but their scores are not intrinsic: the winning specialist still depends on the benchmark, policy contract, prevalence, and operating point.

That sentence is a placeholder claim template, not a conclusion to write before results.

---

## 17. Execution sequence

### Phase 0 — Freeze scope and access

- Accept required model licenses manually.
- Re-verify and pin every model/tokenizer revision.
- Snapshot model cards and terms used for ancestry, license, policy, and exposure claims.
- Select the launch panel and mark all optional models exploratory.
- Resolve the Qwen strictness mapping, ShieldGemma aggregation, Granite risk definition, and WildGuard prompt-only contract.

**Gate:** no GPU scoring until these choices are hashed.

### Phase 1 — Build and test native adapters

- Implement one adapter per output schema.
- Add deterministic fixtures for prompt rendering and candidate-token scoring.
- Add sign, constant-score, truncation, parse, and generation-concordance tests.
- Run small CPU/mock tests and a GPU smoke set.

**Gate:** every scorer produces finite, correctly oriented continuous scores and complete metadata.

### Phase 2 — Preflight the lineage bridge

- Reuse existing Qwen3-4B committed scores only as a replication reference for already-scored Paper A rows under the common contract.
- Locate and verify archived Qwen adapter bytes if available; otherwise plan to retrain all five Qwen adapters in the new locked namespace before native-contract, ExpGuard, mortgage, or fresh-cohort scoring.
- Run common-recipe preflight for Gemma, Granite, Llama, and Mistral parents.
- Run only explicitly nonfinal smoke training before the lock.
- Record ineligibility rather than altering the recipe post hoc to rescue a model.

**Gate:** at least three bridge candidates pass preflight; no claim-bearing training starts yet.

### Phase 2b — Preregister, collect, annotate, and seal the fresh cohort

- Commit a cohort-specific preregistration covering collection frame, policy, inclusion/exclusion rules, family construction, annotation, sample-size justification, and planned contrasts.
- Collect and annotate without showing candidate-model outputs to selectors or annotators.
- Measure inter-annotator agreement, adjudicate under the locked rule, and record every exclusion.
- Freeze calibration/test assignments, content hashes, family graph, and text redistribution status.
- Keep test text and labels sealed from model scoring and analysis code until the final study contract is ready.

**Gate:** the cohort manifest and annotation report are immutable inputs to the final lock. If this phase is omitted, mark the fresh cohort optional and keep every new-act claim retrospective.

### Phase 3 — Lock before full scoring

- Finalize the normative `primary_contract.json`, `LOCK.json`, exposure matrix, policy crosswalk, executable primary contrasts, claim registry, and expected score cardinality.
- Hash the canonical Paper A manifests, mortgage release, ExpGuard file/index, and any fresh cohort.
- Commit the lock before claim-bearing training or evaluation.

### Phase 4 — Train and score under the lock

- Train five common-recipe seeds for every eligible parent, including Qwen if reusable adapter bytes are unavailable. Any retrained adapter set must be rescored under every claim-bearing contract and dataset; never combine old common-contract scores with new native/domain scores as if they came from the same adapter instances.
- Score one model per process to avoid allocator and multi-load instability.
- Score calibration first, then tests without refitting.
- Preserve raw candidate likelihoods and margins.
- Benchmark latency under a separately fixed serving protocol.
- Re-run a cross-device subset to quantify numerical stability.

Derive the compute budget from measured smoke throughput rather than a guessed fixed cost. Record rows/second separately for native forced scoring, official generation, and each multi-policy pass; estimate the full matrix; set a hard GCP budget and wall-clock cap; and launch the full run only after that estimate is reviewed. An L4 may be sufficient for some scoring cells, while an A100-class GPU is the safer training/throughput baseline; the lock records the actual hardware rather than treating devices as interchangeable.

### Phase 5 — Analyze and release

- Run family-aware paired bootstrap where locked family graphs exist and shared-row paired bootstrap on surfaces that lack them, with the independence limitation disclosed.
- Run all exposure, contract, strictness, prevalence, and leave-home-benchmark-out sensitivities.
- Validate the complete score matrix and every claim rule.
- Package a text-free release with `RELEASE.json` and external anchor.
- Verify CPU-only reproduction from the release cache.

### Phase 6 — Edit the report
- Generate tables/figures.
- Add Act II and renumber displayed acts.
- Update evidence ledger and caveats.
- Run `pytest`, specialist release verification, `make -C papers/unified-report reproduce-check`, and Tectonic.
- Perform an independent technical-review pass against every headline sentence.

### Phase 7 — GCP teardown

If GCP is used, all resources must carry a study label and a deletion checklist. After artifacts are copied locally and hashes are independently verified:

- stop and delete VMs;
- delete attached and orphaned disks;
- remove snapshots and reserved IPs created for the run;
- verify no labeled compute resources remain; and
- save the teardown evidence in provenance.

Do not leave a VM running merely because report writing continues locally.

---

## 18. Promotion and stop gates

The act is publishable only when:

- at least three specialist families have complete, finite, continuous score matrices;
- any recurring specialization-like geometry wording has at least three eligible lineage bridges;
- all revisions, contracts, policies, strictness mappings, and exposure statuses are locked;
- every expected row/model/contract cell is present exactly once;
- no malformed output is silently mapped to safe;
- native and common-interface results are separated;
- vendor and controlled deltas are separated;
- known/home-benchmark overlap has an exclusion sensitivity;
- no “vendor-unseen” or “held-out” claim relies on unknown exposure;
- all generated outputs regenerate from LOCK-bound scores and protocol artifacts;
- `reproduce-check` fails closed on missing, pending, or drifting outputs;
- the TeX build has no undefined citations/references and no stale act numbering;
- the release contains no raw gated text, model weights, credentials, or secrets; and
- an adversarial reviewer signs off on the claim registry.

Stop or narrow the act if:

- fewer than three specialist models can produce a defensible continuous score;
- parent ancestry cannot be supported;
- the native interface cannot be reproduced faithfully;
- model-card exposure makes every claimed “transfer” source invalid and no fresh cohort is available;
- license terms prevent score-artifact publication; or
- the only available result is a hard-label leaderboard with large ties.

In those cases, retain a smaller **released-guard benchmark-sensitivity** section and explicitly defer the specialization question.

---

## 19. Claims the final report may and may not make

### Allowed, if supported

- “On these fixed declared-lineage pairs, released specialist checkpoints showed a specialization-like movement vector.”
- “The winning purpose-built guard changed across benchmark, scoring contract, prevalence, or operating point.”
- “Purpose-built guard checkpoints did not eliminate benchmark dependence.”
- “The common SFT specialization pattern did/did not recur on the additional controlled parent lineages.”
- “A native general-safety policy did/did not cover mortgage-policy violations out of the box.”
- “Supplying one locked domain policy changed performance for models designed to accept policy definitions.”

### Not allowed

- “Purpose-built training causes specialization.”
- “These vendor pairs isolate fine-tuning” without exact controlled evidence.
- “The Act-I transfer panel was unseen by the vendor guard” when exposure is unknown.
- “Model X is generally the best guard.”
- “Vendor training is better or worse than our LoRA recipe.”
- “ShieldGemma measures mortgage compliance by default.”
- “These models represent a population of purpose-built guards.”
- Any deployment-safety, legal-compliance, or fair-lending certification.

---

## 20. Bottom line

This act is worth doing. It can turn the report from a careful study of one common fine-tuning recipe into a broader, practitioner-relevant test of whether purpose-built guard products escape the same benchmark dependence.

The strongest contribution is **not** “we added ShieldGemma to a table.” It is:

> We subjected released specialist guards to the report's paired, exposure-aware, policy-explicit, reproducible methodology and measured whether their specialization geometry and rank survived changes in benchmark, policy contract, prevalence, operating point, and regulated domain.

If the rankings still flip, the report's thesis becomes substantially stronger. If they do not, that null is equally valuable because it precisely bounds the original claim. Either outcome improves the report, provided the controlled SFT evidence, descriptive vendor contrasts, and policy/domain evidence remain visibly separate.
