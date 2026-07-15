# ARCHIVED/HISTORICAL — Paper B: Two Guards, One Risk Budget

> **Superseded direction:** The entire body below is archived mortgage rationale. The later
> mortgage checkpoint plan is also archived; the current near-term direction is
> [`paper-b-compose-dont-tune-plan.md`](paper-b-compose-dont-tune-plan.md). The “joint
> calibration,” universal mortgage `BLOCK`, old E3 allocation-grid baseline, six-lock chain,
> and test-access commands below must not be implemented or cited as the current protocol.

## Document purpose

This was the original research sketch for a joint general-safety and mortgage-compliance guard stack. It is not executable or authoritative; current science, commands, and gates live only in the replacement document above. Historically, it separated three levels of evidence:

1. **Existing-data pilot:** implement and debug component scoring, score joins, optimizer mechanics, and decision replay using artifacts already in this repository. The current separately labeled datasets cannot fit or validate a joint policy.
2. **Publishable joint-stack study:** add family-disjoint, dual-labeled four-quadrant development and untouched test cohorts, plus expert re-adjudication.
3. **High-compliance claim:** add enough benign, expert-reviewed traffic to estimate a low system false-positive rate with useful precision, plus response/tool data if the paper claims a complete agent boundary.

Recommended title:

> **Two Guards, One Risk Budget: Joint Calibration of General-Safety and Mortgage-Compliance Guardrails**

Recommended thesis:

> A regulated-domain guardrail is a system rather than a classifier. General-safety and policy-specific guards jointly consume false-positive, latency, and deferral budgets; system-level calibration may choose a different deployment than optimizing either layer alone.

The novel target is not another mortgage LoRA. It is a dual-policy interaction benchmark plus a versioned, auditable system-level selection method evaluated under one risk budget. Abstention, base retention, and consolidated training are secondary ablations, not bundled parts of the primary claim.

**Direct call:** proceed conditionally. The strongest feasible primary comparison is independently calibrated `G OR D` versus jointly calibrated binary `G OR D`, using the same frozen components on fully dual-labeled data. The paper earns a positive result only if joint calibration improves benign blocking or prespecified policy cost while remaining noninferior on severe unsafe-ALLOW risk. A negative result is publishable if the benchmark and artifact chain are strong.

---

## 1. Direct implementation decision

### 1.1 Runtime v0

Use one frozen SmolLM3-3B base with named PEFT adapters:

- `B`: untuned base scored with the general-safety prompt;
- `G`: existing general-safety adapter scored with the general-safety prompt;
- `D`: existing mortgage adapter scored with the mortgage prompt.

The current local model revision is:

```text
HuggingFaceTB/SmolLM3-3B
a07cc9a04f16550a088caea529712d1d335b0ac1
```

Current local adapter artifacts:

| Adapter      | Path                                                                                  | SHA-256                                                              |
| ------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| general      | `notebooks/outputs/nb-smollm3-guard/adapter/adapter_model.safetensors`              | `c4076b7fa123281df3f7c1aa866321ac37fdabf0816bfa073d19153a14d8c1e6` |
| mortgage SFT | `notebooks/outputs/nb-smollm3-guard/mortgage_sft/adapter/adapter_model.safetensors` | `253e65d9a57bd1ba0e29a9346c257593902f2acf94c922452aa7d95b26a74f51` |

Recompute these hashes and fail on mismatch when building the experiment lockfile.

These adapters, `frozen_eval_rows.json`, and the current score caches are local ignored artifacts rather than a released, fresh-clone evidence package. They establish machine-local feasibility only. Before publication, preserve them under immutable public artifact IDs or publish byte-identical replacements with hashes and acquisition instructions.

For a mortgage product, always execute the mortgage guard. Do not begin with a learned router: the application scope is already known, and a router creates an avoidable bypass. A future deterministic selector may add jurisdiction/product policy packs, but it may not disable the mandatory mortgage layer.

### 1.2 Decision semantics

Use three actions:

- `ALLOW`: every mandatory layer is below the allow boundary;
- `REVIEW`: uncertainty, disagreement, missing context, scorer failure, or an intermediate score;
- `BLOCK`: a deterministic hard rule or a high-confidence violation.

Rules:

- Domain-safe never overrides general-unsafe.
- General-safe never overrides domain-unsafe.
- A cheap stage may early-block; no stage may early-allow and skip the mortgage layer.
- Timeout, NaN, adapter-load error, or missing policy context becomes `REVIEW`; if review is unavailable, fail closed to `BLOCK`.
- The same decision policy must be deterministic and replayable from stored scores and artifact hashes.

### 1.3 Scope discipline

Version 1 of the paper may claim only input-prompt screening. The repository currently has no response or tool-call mortgage evaluation. Add those surfaces before using “complete agent compliance stack.”

---

## 2. Feasibility from current repository data

### 2.1 Existing mortgage assets

| Artifact                                                                                                  | Verified content                                                                             | Immediate use                                     |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| [`mortgage_redteam_benchmark.jsonl`](../notebooks/data/benchmarks/full/mortgage_redteam_benchmark.jsonl) | 1,000 rows: 938 violations, 62 benign controls; 764 rows have nonempty regulatory references | recover policy provenance and annotation criteria |
| [`guard_benchmark.jsonl`](../notebooks/data/benchmarks/full/guard_benchmark.jsonl)                       | 1,563 rows: 938 flag, 625 allow                                                              | current mortgage train/dev/test source            |
| [`mortgage_split.json`](../notebooks/data/benchmarks/full/mortgage_split.json)                           | family-clustered train 1,094, dev 236, test 233                                              | training and pilot calibration                    |
| [`guard_benchmark_hard.jsonl`](../notebooks/data/benchmarks/full/guard_benchmark_hard.jsonl)             | 334 rows: 195 flag, 139 allow; 30 minimal-pair groups/60 paired rows; 15 multi-turn rows     | external hard pilot and re-adjudication seed      |
| hard-set caches                                                                                           | aligned base/general/mortgage score arrays, 334 each                                         | optimizer smoke test without inference            |

The 1,563-row and 334-row mortgage files have zero normalized exact-text overlap, producing 1,897 unique current mortgage prompts.

Policy-provenance recovery is practical rather than theoretical:

- exact prompt matching joins 995 of the 1,000 original red-team rows into `guard_benchmark.jsonl`;
- 759 joined rows carry nonempty regulatory references;
- five benchmark rows need manual reconciliation;
- the 563 generated-benign rows correctly have no originating violation citation, but should receive an explicit `not_applicable` or benign-control provenance record.

The current transformed training file discarded the rich fields. Rejoin them; do not regenerate them from memory.

### 2.2 Existing general assets

[`frozen_eval_rows.json`](../notebooks/outputs/frozen_eval_rows.json) provides:

- general dev: 1,102 rows, 563 unsafe and 539 safe;
- general test: 2,018 rows, 998 unsafe and 1,020 safe;
- general source-shift/OOD: 2,020 balanced rows plus 200 HarmBench unsafe-only rows.

This is enough to score every current component on both regimes. It is not enough for layer-attribution because general rows lack mortgage labels and mortgage rows lack general-safety labels.

### 2.3 Existing-data diagnostic manifests

Before collecting new data, build the following component-diagnostic bundles:

| Bundle                            |                                 Rows | What is actually known                                                          |
| --------------------------------- | -----------------------------------: | ------------------------------------------------------------------------------- |
| component development diagnostics | 1,102 general + 236 mortgage = 1,338 | one policy label per row; the other policy and usually final action are unknown |
| component ID diagnostics          | 2,018 general + 233 mortgage = 2,251 | one policy label per row; the other policy and usually final action are unknown |
| component hard/OOD diagnostics    | 2,220 general + 334 mortgage = 2,554 | one policy label per row; the other policy and usually final action are unknown |

For existing general rows:

```text
general_gold = existing label
domain_gold = null
final_gold = 1 if general_gold == 1 else null
```

For existing mortgage rows:

```text
general_gold = null
domain_gold = existing label
final_gold = 1 if domain_gold == 1 else null
```

If either known policy label is unsafe, final unsafe is logically known. If the known label is safe, final action remains unknown because the missing policy may still be unsafe. Setting a safe row's `final_gold` to zero would silently impute the missing policy as safe.

These bundles support scorer engineering, cache joins, per-layer positive-coverage diagnostics, and exploratory cross-trigger rates. They **cannot** estimate joint benign FPR, final risk, quadrant effects, expected system cost, or select joint thresholds. Therefore they are not a valid joint development or final test set.

---

## 3. Current-data optimizer smoke test

A retrospective mechanics check was run on the existing 334-row hard set using the three local score caches. All three caches were produced under the mortgage prompt. They do not implement the proposed runtime, where `B` and `G` use the general-safety prompt and `D` uses the mortgage prompt.

Protocol:

- 10 repeated 50/50 splits;
- minimal-pair rows kept together;
- thresholds selected on each development half;
- development FPR cap 30%, because the set is too small and current models are too overblocking for a credible 1–5% pilot;
- evaluation on the corresponding held-out half;
- values below are means over the 10 splits.

| System                                 | Test recall | Test FPR |
| -------------------------------------- | ----------: | -------: |
| untuned base under mortgage prompt     |       0.795 |    0.294 |
| general adapter under mortgage prompt  |       0.655 |    0.279 |
| mortgage adapter under mortgage prompt |       0.936 |    0.281 |
| adapter-score OR mechanics             |       0.930 |    0.287 |
| three-score OR mechanics               |       0.926 |    0.300 |
| calibrated two-score fusion mechanics  |       0.936 |    0.281 |

Score correlations on all 334 rows:

```text
corr(B,G) = 0.640
corr(B,D) = 0.533
corr(G,D) = 0.390
```

Interpretation:

1. Array alignment, score combination, repeated family splits, and finite-search mechanics are feasible now.
2. The three mortgage-prompt score vectors are correlated but not identical.
3. On domain-only labels, the optimizer selects the mortgage adapter alone; composition does not automatically improve the result.
4. The values cannot characterize the planned components or cross-policy errors because the prompt and labels are domain-only.
5. Rescore `B/G` with the frozen general prompt and `D` with the frozen mortgage prompt on identical rows.
6. The paper must obtain dual labels before fitting a joint policy or claiming that a joint stack is better.

This smoke test belongs in the implementation log or appendix, not the final headline table.

---

## 4. Research questions and claim gates

### RQ1 — Does joint calibration beat independent calibration? **Primary**

Independent baseline:

- on the same fully dual-labeled development rows, select each layer threshold separately against its own policy label and preregistered share of the system FPR budget;
- combine decisions by OR.

Joint method:

- hold `G` and `D` fixed and select their binary decision thresholds together on one fully dual-labeled system-level development objective;
- enforce system-level constraints directly.

The primary comparison must change only threshold selection: independent `G OR D` versus jointly calibrated binary `G OR D`. Base fusion and abstention are separate ablations. The joint-calibration claim requires lower expected policy cost or lower benign hard-blocking while satisfying a preregistered paired noninferiority margin for severe unsafe ALLOW. Choose that margin and sample size from a power/precision analysis; the current 60-family proposal cannot support a one-percentage-point margin.

### RQ2 — Does composition preserve both policy spaces?

Compare general-only, domain-only, naïve replacement, and composed systems on examples labeled for both general safety \(Y_g\) and mortgage compliance \(Y_d\).

Required quadrants:

| Quadrant  | Meaning                                               |
| --------- | ----------------------------------------------------- |
| `G0/D0` | benign under both policies                            |
| `G1/D0` | general harm, not a mortgage violation                |
| `G0/D1` | mortgage violation not captured by the general policy |
| `G1/D1` | violation under both policies                         |

The joint-system claim is allowed only if it improves or is noninferior to the best single layer on both `G1/D0` and `G0/D1` non-ALLOW coverage at the same system benign-block and deferral budgets.

### RQ3 — Does explicit review improve the deployable frontier?

Compare binary block/allow with ternary allow/review/block.

Primary outcomes:

- severe unsafe ALLOW rate;
- benign BLOCK rate;
- REVIEW rate;
- residual unsafe risk among ALLOW;
- p95/p99 latency.

Review must not be counted as a successful block. Report BLOCK and REVIEW separately. Unless actual reviewer accuracy, turnaround time, capacity, and final dispositions are measured, call this **selective deferral**, not evidence that human review improves end-to-end safety.

### RQ4 — Can a consolidated guard approach the modular stack?

This is secondary. Train a constrained single adapter using provenance-reviewed mortgage rows and a newly frozen, globally decontaminated general replay manifest, then compare it with the modular stack. The consolidated model is useful only if it meets both policy constraints and provides a material latency benefit.

---

## 5. Mathematical system formulation

### 5.1 Component scores and calibration

For component \(k\in\{B,G,D\}\), use the actual decision logits:

\[
z_k(x)=\ell_{unsafe,k}(x)-\ell_{safe,k}(x),
\quad
p_k(x)=\sigma(z_k(x)/T_k).
\]

Fit \(T_B\) and \(T_G\) on general-policy calibration labels. Fit \(T_D\) on mortgage-policy calibration labels. Use binary NLL with LBFGS and retain Brier/ECE diagnostics. Component temperature fitting may use policy-specific labels, but system threshold selection requires a separate, fully dual-labeled development cohort. Do not calibrate or select on final test or hard/OOD data.

### 5.2 Base retention in the general layer

Use a convex logit fusion:

\[
z_{gen,\alpha}(x)=
\alpha\operatorname{logit}(p_B(x))
+(1-\alpha)\operatorname{logit}(p_G(x)),
\]

\[
p_{gen,\alpha}(x)=\sigma(z_{gen,\alpha}(x)),
\quad
\alpha\in\{0,0.05,\ldots,1\}.
\]

This reuses the base-versus-tuned observation without assuming the current PIT ensemble result is valid.

Domain candidates:

1. untuned base under the mortgage prompt;
2. mortgage-SFT adapter;
3. optional constrained consolidated adapter.

Do not select the domain model on the 334 hard test.

### 5.3 Primary binary policy and secondary deferral policy

For the primary fixed-component comparison, use layer-specific thresholds:

\[
a_{bin}(x)=
\begin{cases}
BLOCK,&p_G(x)\ge\tau_G\;\lor\;p_D(x)\ge\tau_D,\\
ALLOW,&\text{otherwise}.
\end{cases}
\]

The independent baseline chooses \(\tau_G\) and \(\tau_D\) separately on the same frozen dual-labeled development cohort, using \(Y_g\) and \(Y_d\), respectively, and preregistered per-layer budget allocations. The proposed method selects the same two thresholds together on those same rows under one final-system constraint. This isolates the selection objective rather than confounding it with different development data.

Base retention is a later ablation that substitutes \(p_{gen,\alpha}\) for \(p_G\); it must not be part of the primary E3-to-E4 contrast.

For the secondary deferral arm, construct a fixed joint risk score after component calibration:

For a fixed component choice and alpha:

\[
p_{joint}(x)=\max(p_{gen,\alpha}(x),p_D(x)).
\]

For \(\tau_A\le\tau_B\):

\[
a(x)=
\begin{cases}
ALLOW,&p_{joint}(x)<\tau_A\\
REVIEW,&\tau_A\le p_{joint}(x)<\tau_B\\
BLOCK,&p_{joint}(x)\ge\tau_B.
\end{cases}
\]

Hard deterministic rules may override to `BLOCK`. Scorer/system failures override to `REVIEW`.

### 5.4 Weighted policy cost

Let \(y_i=Y_{g,i}\lor Y_{d,i}\), \(v_i\) be severity, and \(w_i\) be a prespecified deployment/source weight.

\[
J(\theta)=
C_{FN}\sum_i w_i v_i\mathbf 1[y_i=1,a_i=ALLOW]
+C_{FP}\sum_i w_i\mathbf 1[y_i=0,a_i=BLOCK]
+C_R\sum_i w_i\mathbf 1[a_i=REVIEW].
\]

For the primary binary arm, select \(\theta=(\tau_G,\tau_D)\). Base-retention and deferral ablations extend \(\theta\) with \(\alpha\) and \((\tau_A,\tau_B)\), respectively. Select parameters subject to:

\[
UCB_{95}(FNR_{severe,ALLOW})\le\epsilon,
\]

\[
UCB_{95}(FPR_{benign,BLOCK})\le\alpha_{sys},
\]

\[
ReviewRate\le\rho,
\quad p99Latency\le L_{max},
\]

plus a worst-category unsafe-ALLOW constraint.

Use lexicographic selection:

1. satisfy every constraint;
2. minimize \(J\);
3. minimize review rate;
4. prefer the wider review margin;
5. prefer the simpler architecture.

If no candidate is feasible, emit `NO_GO`. Never silently relax constraints.

### 5.5 OR-system identities

Let \(A_G\) and \(A_D\) be the events that the two layers intervene. On jointly benign traffic:

\[
FPR_{sys}
=P(A_G\cup A_D\mid Y_g=0,Y_d=0)
\]

\[
=FPR_G+FPR_D-P(A_G\cap A_D\mid Y_g=0,Y_d=0).
\]

Therefore:

\[
\max(FPR_G,FPR_D)
\le FPR_{sys}
\le \min(1,FPR_G+FPR_D).
\]

For any fixed positive target population \(Y=1\):

\[
FNR_{sys}=P(A_G^c\cap A_D^c\mid Y=1)
\le\min(FNR_G,FNR_D).
\]

These follow directly from set union/intersection. They prove why OR cannot have a lower FPR than both components and cannot have a higher miss rate than both on the same labeled population. They do not determine the actual rates; correlation must be measured on dual-labeled rows.

### 5.6 Exact finite-grid optimizer

For the primary binary arm, decisions change only when \(\tau_G\) crosses a unique observed `G` score or \(\tau_D\) crosses a unique observed `D` score. Therefore enumerate:

\[
\mathcal T_G=\{-\infty,+\infty\}\cup\{p_G(x_i)\},\qquad
\mathcal T_D=\{-\infty,+\infty\}\cup\{p_D(x_i)\}.
\]

Every continuous threshold pair has an equivalent representative in \(\mathcal T_G\times\mathcal T_D\), so exhaustive finite search returns a global empirical optimum for the fixed components and frozen development rows. Use vectorized masks, bitsets, or cumulative tables and verify the optimized implementation against brute force on small arrays.

The following analogous result applies to the secondary scalar-risk deferral arm.

For a fixed domain candidate and alpha, `p_joint` is fixed. Decisions change only when a threshold crosses an observed unique score. Therefore any continuous threshold pair has an adjacent representative drawn from:

\[
\mathcal T=\{-\infty,+\infty\}\cup\{p_{joint}(x_i)\}_{i=1}^n
\]

that produces identical decisions and empirical cost.

**Finite-grid optimality proposition.** Enumerating all ordered \((\tau_A,\tau_B)\in\mathcal T^2\), \(\tau_A\le\tau_B\), returns a global empirical optimum for the fixed score vector.

Proof: unique observed scores partition the real line. Moving a threshold inside one partition changes no comparison, action, cost, or constraint. Every equivalence class has a boundary representative in \(\mathcal T\).

Implementation:

1. Sort `p_joint` once.
2. Build prefix sums for weighted positives, benign rows, severity, review cost, and categories.
3. Evaluate each threshold pair in O(1).
4. Loop over 21 alpha values and the discrete domain candidates.

Do not use the current `n=1,338` partial-label bundle for this optimization. Complexity must be reported using the eventual fully dual-labeled development size. Learned logistic fusion is an ablation, not the primary selector.

---

## 6. Data schema and construction

### 6.1 Canonical row schema

Add `guard_stack/schema.py` with validation for:

```text
schema_version
sample_id
content_sha256
surface                       # input | response | tool_args | tool_result
scope                         # general | mortgage | interaction
source
source_revision
source_row_id
split
family_id
pair_id
mutation_id
text_or_event
general_gold                  # 0 | 1 | null
domain_gold                   # 0 | 1 | null
final_gold                    # 0 | 1 | null
category
subcategory
severity
persona
workflow_stage
jurisdiction
product
general_policy_version
domain_policy_version
general_policy_control_ids[]
domain_policy_control_ids[]
regulatory_refs[]
protected_class_proxy
label_provenance
annotator_ids[]
adjudication_status
ambiguity_flag
```

Rows with `ambiguity_flag=true` may evaluate REVIEW behavior but must not be silently forced into binary training labels.

### 6.2 Policy registries

Add both:

```text
data/policies/general/v1/controls.jsonl
data/policies/mortgage/us_v1/controls.jsonl
```

Each control requires:

```text
control_id
title
jurisdiction
product
workflow_surface
authoritative_source_url
source_snapshot_sha256
citation
effective_from
effective_to
prohibited_or_required_behavior
severity
default_action
owner
review_status
```

The mortgage registry must be reviewed by a qualified mortgage-compliance expert. The general registry must freeze a coherent safety taxonomy and adjudication rubric rather than treating heterogeneous Paper A source labels as one policy. Do not let a model invent specific citations at runtime. A control-specific reason requires a validated category/control head or deterministic rule match.

### 6.3 Dual-label interaction development and test cohorts

Start with 60 semantic scenario families × four variants = 240 prompts as an annotation and implementation **pilot**. Each family contains all four quadrants. Do not treat this pilot as the publishable final test.

Composition:

- 20 families: fair-lending controls;
- 20 families: compliance/bad-advice controls;
- 20 families: security misuse;
- 30 consumer and 30 loan-officer scenario families;
- quartet members share vocabulary and context where possible.

The family/quartet is the split and bootstrap unit. No family, template skeleton, or near-duplicate may cross development and test.

Each row receives:

- two independent general-safety judgments;
- two independent mortgage-compliance judgments;
- expert adjudication;
- severity, control IDs, rationale, and ambiguity flag.

The 240-row pilot is 960 primary binary judgments plus adjudication. At approximately two minutes per judgment, plan ~32 reviewer-hours plus 6–10 adjudication hours.

Before scaling, perform a power/precision analysis using the prespecified severe-risk rate, paired discordance expected between E3 and E4, noninferiority margin, and family-level design effect. Then collect:

1. a fully dual-labeled development cohort used for calibration, cost/threshold selection, and ablation selection; and
2. a family-disjoint, untouched, fully dual-labeled final test cohort.

The re-adjudicated 334-row hard set may serve as development/stress evidence because it has already been inspected repeatedly; it is not a pristine final test. As a minimum precision sanity check, roughly 100 independent severe units with zero failures are needed for a one-sided 95% binomial upper bound near 3%, and each critical quadrant needs its own adequate count. A paired one-percentage-point noninferiority claim will usually require far more than 60 families. Final sample size must come from the power analysis, not convenience.

### 6.4 Re-adjudicate the hard mortgage set

Have two mortgage SMEs independently assign the mortgage-policy label and two trained general-safety annotators independently assign the general-policy label for all 334 hard rows, with policy-specific adjudication. Reuse the trap types, rationales, pairs, persona, category, and conversation fields. Add policy control IDs/sources, expert confidence, adjudication status, and family IDs for all rows.

This requires approximately 45 primary-reviewer hours at two minutes per policy judgment plus adjudication.

### 6.5 Safe-sample requirement for a high-compliance FPR claim

The current two mortgage sets contain 764 rows labeled benign under their existing mortgage-only annotations. They are not yet independent, expert-validated, dual-safe units. For FPR near \(p=0.01\), an approximate 95% margin \(e\) requires:

\[
n_0\approx\frac{1.96^2p(1-p)}{e^2}.
\]

For \(e=0.005\), this is approximately 1,521 independent benign units.

Therefore:

- the 240-row interaction pilot tests annotation and layer-interaction mechanics, not low FPR;
- count existing benign rows only after family deduplication, expert validation, and both-policy labeling;
- target at least approximately 1,521 independent dual-benign units in total for the stated 1% ±0.5% approximation rather than assuming only 750 more are needed;
- use more if subgroup FPR is a primary claim.

Generate them from the existing hard-negative taxonomy—legitimate compliance training, lawful underwriting, fraud prevention, protected-class mentions, secure operations, and benign wrapper attacks. Do not achieve low FPR by filling the test with trivial textbook questions.

### 6.6 Response and tool extension

If the manuscript claims an agent boundary, add held-out sets for unsafe responses, dangerous tool arguments/results, and benign actions that resemble violations. Otherwise state “input-prompt guard stack” everywhere.

---

## 7. Code implementation

### 7.1 Package layout

Add:

```text
guard_stack/__init__.py
guard_stack/schema.py
guard_stack/prompts.py
guard_stack/multi_adapter_scorer.py
guard_stack/calibration.py
guard_stack/decision.py
guard_stack/optimize_policy.py
guard_stack/policy_registry.py
guard_stack/metrics.py
guard_stack/audit.py
guard_stack/replay.py

experiments/build_joint_manifest.py
experiments/score_joint_stack.py
experiments/fit_joint_policy.py
experiments/eval_joint_stack.py
experiments/benchmark_joint_latency.py
experiments/train_mortgage_constrained.py
experiments/analyze_paper_b.py
```

Producing programs write under `artifacts/paper_b/`; ignored notebook caches may be used only as pilot-migration inputs.

### 7.2 Multi-adapter scorer

PEFT 0.19.1 supports named adapters. Load the base once:

```python
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=model_revision,
    dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(
    base,
    general_adapter,
    adapter_name="general",
)
model.load_adapter(mortgage_adapter, adapter_name="mortgage")

with model.disable_adapter():
    z_base = score_decision_logits(model, general_prompt)

model.set_adapter("general")
z_general = score_decision_logits(model, general_prompt)

model.set_adapter("mortgage")
z_domain = score_decision_logits(model, mortgage_prompt)
```

Requirements:

- one tokenizer/base instance;
- byte-identical, versioned prompt templates;
- no sampled generation;
- distinct one-token label assertion;
- raw safe/unsafe logits, not probabilities alone;
- synchronized timing;
- error-to-review behavior;
- no cache accepted without row/model/adapter/prompt hashes.

Adapter switching must be measured. One early-block cascade arm may skip the mortgage pass after a definitive general block because the final action cannot become ALLOW.

### 7.3 Policy optimizer API

`guard_stack/optimize_policy.py` exposes:

```python
def fit_policy(
    dev_rows,
    component_scores,
    arm,
    risk_constraints,
    cost_matrix,
    deployment_weights,
    alpha_grid=None,
) -> DecisionPolicy | NoGo:
    ...
```

The result records:

```text
policy_id
training_manifest_hash
score_artifact_hashes
calibration_artifact_hashes
component_choices
arm
tau_general
tau_domain
alpha|null
tau_allow|null
tau_block|null
constraints
development_metrics
selection_trace_hash
created_at
code_git_sha
```

Fit code must not access test paths. Unit tests should alter test labels and prove the policy artifact is unchanged.
The primary arm must reject any development row for which either policy label is null.

### 7.4 Constrained consolidated adapter

This optional arm can use the 1,094 mortgage training rows only after provenance review, plus a newly frozen and globally decontaminated general replay manifest. Do not call Paper A's current 3,279-row pool clean: its immutable manifest and complete cross-source overlap audit are not available.

Use:

\[
\mathcal L=
\mathcal L_{mortgage}
+\lambda\mathcal L_{general-replay}
+\beta KL(q_0\|q_\theta)
\]

on a 1:1 mortgage/general batch mixture. Regenerate the two-token base logits under pinned model, tokenizer, prompt, row, and code hashes; do not reuse unaudited length-keyed caches.

Pilot grid:

```text
lambda in {0.25, 0.5, 1.0}
beta in {0.0, 0.1, 0.5, 1.0}
```

Twelve 250-step runs are manageable. Select on mortgage development utility subject to no more than 0.020 general-development macro-AP regression relative to the general adapter. Retrain only the locked winner for three or five seeds.

### 7.5 Required tests

Add:

```text
tests/test_multi_adapter_scorer.py
tests/test_prompts.py
tests/test_decision.py
tests/test_optimizer.py
tests/test_manifest.py
tests/test_metrics_ties.py
tests/test_calibration.py
tests/test_audit.py
tests/test_replay.py
tests/test_failure_policy.py
```

Properties:

- adapter-disabled score equals separately loaded base within tolerance;
- switching adapters does not change their weights;
- prompt hashes are stable;
- neither safe layer can override the other's unsafe decision;
- errors/NaNs produce REVIEW;
- exact optimizer equals brute force on small arrays;
- test labels cannot alter a fitted policy;
- score-tie permutations do not alter AP/AUROC;
- score joins are one-to-one by hash;
- replay returns the same action/scores within `1e-6` CPU / `1e-3` BF16;
- audit events validate against schema.

---

## 8. Experiment arms

Score every arm on identical rows.

| Arm | System                                      | Purpose                                                                     |
| --- | ------------------------------------------- | --------------------------------------------------------------------------- |
| E0  | untuned base under general prompt           | base reference                                                              |
| E1  | general LoRA only                           | general single layer                                                        |
| E2a | untuned base under mortgage prompt          | domain zero-shot reference                                                  |
| E2b | mortgage-SFT only                           | domain single layer                                                         |
| E3  | independently calibrated`G OR D`          | naïve layered baseline                                                     |
| E4  | jointly calibrated`G OR D`, binary        | **primary proposed method; only threshold selection changes from E3** |
| E5  | jointly calibrated`(B ⊕ G) OR D`, binary | base-retention ablation                                                     |
| E6  | E4 with REVIEW band                         | selective-deferral ablation                                                 |
| E7  | learned logistic fusion                     | fusion ablation                                                             |
| E8  | safe early-block cascade                    | latency ablation; never early-allows                                        |
| E9  | constrained consolidated adapter            | optional single-pass tradeoff                                               |

Define E3 before results. For example, allocate half the development system-FPR budget to each layer. Do not retrofit the baseline to make E4 look better. E3 and E4 must use the same fully dual-labeled development cohort, components, prompts, scores, and binary action space; only independent per-policy versus joint final-system threshold selection may differ.

---

## 9. Evaluation and statistics

### 9.1 Primary metrics

- severe unsafe ALLOW rate;
- non-ALLOW coverage on `G1/D0`, `G0/D1`, and `G1/D1`;
- benign BLOCK rate on `G0/D0`;
- REVIEW rate for the selective-deferral arm;
- unsafe prevalence among ALLOW;
- block precision;
- expected weighted policy cost;
- p50/p95/p99 latency;
- peak memory and adapter-switch cost.

### 9.2 Supporting metrics

- tie-aware AP/AUROC;
- Brier, log loss, ECE, reliability;
- per-category/worst-category rates;
- trap and minimal-pair consistency;
- counterfactual flip/block-rate gaps;
- coverage-risk curves;
- B/G/D error-overlap matrices;
- threshold/base-rate/cost sensitivity.

### 9.3 Inference units

- interaction data: resample semantic quartets/families in the powered dual-labeled cohort;
- hard mortgage data: resample semantic families/minimal pairs;
- general data: resample source and semantic-family clusters;
- training comparisons: pair matching seeds.

Use direct paired intervals. Keep conditional row/family uncertainty separate from training uncertainty.

For E3 versus E4, preregister a paired noninferiority test or interval based on family-level differences. Report the actual discordant-family counts that determine its precision. Do not infer a one-percentage-point margin from 60 families.

### 9.4 Production prevalence

Balanced challenge sets are not traffic. Report unweighted challenge results, at least three prespecified traffic mixtures, and a grid over prevalence and FN:FP:REVIEW costs. Freeze weights in the lockfile and call them scenarios unless measured from actual traffic.

---

## 10. Audit and policy lifecycle

### 10.1 Audit event

Do not store raw regulated text by default. Use a keyed HMAC rather than an unkeyed content hash.

```text
schema_version, event_id, occurred_at, trace_id,
request_hmac_sha256, encrypted_payload_ref|null,
surface, jurisdiction, product,
code_git_sha, model_id, model_revision, tokenizer_revision,
adapters[{name,revision,sha256}], prompt_template_hashes,
calibration_id/hash, policy_pack_id/version/hash,
decision_policy_id/hash,
raw_logits{base,general,domain},
probabilities{base,general,domain,joint},
alpha, tau_allow, tau_block, action, triggering_layer, reason_codes[],
latency_ms{total,base,general,domain,adapter_switch},
error_code|null, fallback_action,
human_review_status, supersedes_event_id|null
```

Human-review outcomes are immutable linked events, not mutations.

### 10.2 Policy update

```text
expert-reviewed registry diff
-> golden regression cases
-> shadow scoring
-> calibration/policy fit
-> signed release
-> replay verification
-> rollback pointer
```

If policy-updateability becomes a headline claim, hold out complete controls and introduce them through registry text only. Compare static SFT, policy-conditioned scoring, deterministic rules, and adapter retraining.

---

## 11. Commands after implementation

```bash
source .venv/bin/activate

python experiments/build_joint_manifest.py \
  --general notebooks/outputs/frozen_eval_rows.json \
  --mortgage-split notebooks/data/benchmarks/full/mortgage_split.json \
  --mortgage-hard notebooks/data/benchmarks/full/guard_benchmark_hard.jsonl \
  --regulatory-source notebooks/data/benchmarks/full/mortgage_redteam_benchmark.jsonl \
  --dual-dev data/paper_b/dual_policy_dev.jsonl \
  --dual-test data/paper_b/dual_policy_test.jsonl \
  --out artifacts/paper_b/manifests

python experiments/score_joint_stack.py \
  --manifest-dir artifacts/paper_b/manifests \
  --model HuggingFaceTB/SmolLM3-3B \
  --model-revision a07cc9a04f16550a088caea529712d1d335b0ac1 \
  --general-adapter notebooks/outputs/nb-smollm3-guard/adapter \
  --mortgage-adapter notebooks/outputs/nb-smollm3-guard/mortgage_sft/adapter \
  --batch-size 16 \
  --out artifacts/paper_b/scores

python experiments/fit_joint_policy.py \
  --dev-scores artifacts/paper_b/scores/dual_policy_dev.parquet \
  --arm joint-gd-binary \
  --max-severe-allow-ucb 0.03 \
  --max-benign-block-ucb 0.05 \
  --out artifacts/paper_b/decision_policy.json

python experiments/eval_joint_stack.py \
  --scores-dir artifacts/paper_b/scores \
  --policy artifacts/paper_b/decision_policy.json \
  --locked \
  --family-bootstrap 5000 \
  --out artifacts/paper_b/evaluation

python experiments/benchmark_joint_latency.py \
  --policy artifacts/paper_b/decision_policy.json \
  --batch-size 1 --warmup 50 --requests 1000

pytest -q tests/test_multi_adapter_scorer.py \
  tests/test_prompts.py tests/test_decision.py tests/test_optimizer.py \
  tests/test_manifest.py tests/test_metrics_ties.py tests/test_calibration.py \
  tests/test_audit.py tests/test_replay.py tests/test_failure_policy.py
```

Pilot scoring requires no new training. The first milestone is a corrected multi-prompt score artifact plus deterministic replay on the existing diagnostic bundles. A scientifically valid `decision_policy.json` requires the fully dual-labeled development cohort.

---

## 12. Acceptance gates

These are research gates, not production authorization.

### 12.1 Integrity

- [ ] Exact model, adapter, data, prompt, code, calibration, and policy hashes.
- [ ] Zero sample/family overlap across train/calibration/final test.
- [ ] Fit code cannot resolve or open test paths.
- [ ] Missing layer labels remain null and are never imputed.
- [ ] System thresholds are fit only on a fully dual-labeled development cohort.
- [ ] Final system claims use a family-disjoint, untouched, fully dual-labeled test cohort.
- [ ] All AP/AUROC code passes tie-order invariance tests.
- [ ] Every decision can be replayed from frozen artifacts.

### 12.2 Safety

- [ ] Severe final-unsafe ALLOW point estimate ≤1%.
- [ ] One-sided 95% upper bound for severe unsafe ALLOW ≤3%.
- [ ] `G0/D1` and `G1/D1` non-ALLOW coverage ≥95% point estimate.
- [ ] Lower 95% bound for those coverages ≥90%.
- [ ] No category with at least 30 unsafe examples has ALLOW rate >10%.

### 12.3 Utility and review

- [ ] Benign hard-BLOCK ≤5% with confidence interval.
- [ ] REVIEW ≤15% under the prespecified traffic mixture.
- [ ] BLOCK and REVIEW are reported separately.
- [ ] No threshold is selected on test.

### 12.4 Reliability and efficiency

- [ ] Every scorer failure becomes REVIEW or, if unavailable, BLOCK.
- [ ] Audit completeness is 100%.
- [ ] Replay tolerance ≤`1e-6` CPU / `1e-3` BF16.
- [ ] Three-pass M4 Max p95 ≤700 ms and p99 ≤1 s, or do not position it as inline.
- [ ] Adapter-switch p95 ≤10 ms, measured rather than assumed.

### 12.5 Proposed-method gate

Compared with naïve independent OR, E4 must:

1. not worsen severe unsafe ALLOW beyond the preregistered, power-supported noninferiority margin at the upper paired 95% interval; and
2. reduce benign hard-blocking or total expected policy cost under frozen weights.

If it does not, publish the negative result.

---

## 13. Paper structure and claim mapping

### 13.1 Section plan

1. Introduction: two policy spaces, one system budget.
2. Threat model and mortgage policy scope.
3. Policy-grounded dual-label benchmark.
4. Shared-base multi-adapter architecture.
5. Fixed-component system-level joint calibration.
6. Experiment design and baselines.
7. Layer interaction and joint-calibration results.
8. Deferral, latency, and failure analysis.
9. Limitations, legal scope, responsible release.
10. Conclusion.

### 13.2 Claim gates

| Proposed statement                          | Current evidence                                               | Publication requirement                                                                  |
| ------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Mortgage adaptation beats the general guard | exploratory corrected hard-set AP ~0.895 vs 0.820              | expert test, frozen thresholds, cluster-paired interval                                  |
| Mortgage adaptation beats the untuned base  | unsupported; corrected gap ~0.005 with interval crossing zero  | new evidence or state inconclusive                                                       |
| Layers require different choices            | not established by separate datasets                           | four-quadrant dual-label set and actual selection                                        |
| Joint calibration improves the stack        | no current composed-stack result                               | E3 vs E4 on fully dual-labeled dev/test under identical budgets                          |
| System is high compliance                   | current small-guard hard FPRs are 27–74% at oracle thresholds | ≥1,500 independent expert-validated dual-benign units, frozen policy, confidence bounds |
| Stack covers agent actions                  | current evidence is prompt-only                                | add response/tool tests or remove claim                                                  |
| Controls are traceable                      | rich metadata exists but was stripped                          | restored join, expert registry, versioned snapshots                                      |
| Protected-class behavior is controlled      | name probe is exploratory                                      | prespecified counterfactual set and family inference                                     |

### 13.3 Novelty boundary

[FinGuard](https://arxiv.org/abs/2605.29427) already supplies regulation-grounded financial compliance data and an expert-annotated guard. [ACL 2026 APT](https://aclanthology.org/2026.acl-long.748/) already studies unseen-policy generalization. [On Calibration of LLM-based Guard Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a99f732df9b668284b449da0214a3286-Abstract-Conference.html) already studies post-hoc guard calibration. Layering, calibration, abstention, and named adapters are not individually novel.

The defensible contribution is:

> a dual-labeled US-mortgage interaction benchmark and an exact, auditable joint-selection method showing how general and domain SLM guards consume one false-positive, review, and latency budget.

Avoid “first layered guardrail” unless a refreshed review proves it.

---

## 14. Compute, labor, and schedule

### 14.1 Existing-data v0

The 6,143 existing dev/evaluation rows require 18,429 component forwards. At measured one-pass p50 124 ms / p90 188 ms, naïve batch-1 execution implies approximately 38–58 minutes of pure inference. Budget 1–2 hours on M4 Max including tokenization/loading/I/O. Exact threshold search is minutes on CPU.

The base is roughly 6 GB BF16 and both adapters total ~462 MB. Expect ~7–10 GB working memory at batch 1, but report measured peak allocation.

### 14.2 Expert effort

| Work                                                            |                                                                                                           Estimate |
| --------------------------------------------------------------- | -----------------------------------------------------------------------------------------------------------------: |
| 240-row interaction annotation pilot, two labels, two reviewers |                                                                                                          ~32 hours |
| pilot adjudication                                              |                                                                                                        6–10 hours |
| powered dual-labeled development and final-test cohorts         |                                           determine after power analysis; likely substantially more than the pilot |
| 334 hard rows, two reviewers per policy                         |                                                                                         ~45 primary-reviewer hours |
| hard-set adjudication/provenance                                |                                                                                                        8–15 hours |
| dual-benign collection/validation to the powered total          |                                                         determine after reusable-row validation and power analysis |
| policy registry review                                          |                                                                                                       10–20 hours |
| **Total high-compliance evidence**                        | **cannot be estimated honestly until the powered cohort size and validated reusable benign count are known** |

### 14.3 Optional training

- no adapter retraining is required for E0–E8;
- 12 constrained-adapter pilots plus 3–5 final seeds should fit in roughly 5–10 A100-equivalent GPU-hours;
- expert annotation, not GPU work, is the dominant cost.

### 14.4 Schedule

1. Week 1: provenance recovery, scorer, optimizer, tests, existing-data v0.
2. Week 2: policy registry/manual; 20-family annotation pilot.
3. Weeks 3–4: complete the annotation pilot, hard re-adjudication, and power/precision analysis.
4. Weeks 5 onward: collect family-disjoint powered dual-labeled development/test and an adequate dual-benign set; duration depends on the power result.
5. First week after data lock: final scoring, latency, failure, and replay audits.
6. Optional next two weeks: consolidated/policy-update ablations, only if the primary comparison is complete.
7. Final three weeks: analysis, release, and paper.

---

## 15. Final go/no-go decision

Paper B can begin immediately as an engineering/system prototype because the shared base, two adapters, source rows, and score caches exist. Current data can prove scorer, join, optimizer, and replay mechanics; it cannot validly compare final-action behavior because safe rows have an unknown label for the other policy and the current caches use one mortgage prompt for all three score vectors.

It cannot prove composition is better because no current set has trustworthy labels for both policy layers on the same rows. The 240-row four-quadrant set is an annotation/engineering pilot, not a powered final experiment. Publication requires separate dual-labeled development and untouched test cohorts sized from the paired noninferiority and risk-bound calculations. A true low-FPR high-compliance statement additionally needs approximately 1,500 or more independent, expert-validated dual-benign units for the stated precision target.

Proceed to a positive-claim submission only if locked E4 improves the frontier over E3 under the prespecified budget and power-supported noninferiority margin. If it does not, publish the benchmark and negative system result honestly: evidence against unnecessary layering is useful and is not a reason to change the test after seeing it.
