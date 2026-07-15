# ARCHIVED/HISTORICAL — Mortgage Paper B Development and Verification Checkpoint Plan

> **Superseded 2026-07-14:** The authoritative near-term direction is
> [`paper-b-compose-dont-tune-plan.md`](paper-b-compose-dont-tune-plan.md). This document remains
> a historical protocol for the infeasible mortgage joint-stack direction and must not govern
> composition work.

- Companion rationale: [`paper-b-joint-compliance-stack-plan.md`](paper-b-joint-compliance-stack-plan.md). Its “joint calibration,” binary mortgage `BLOCK`, and old E3 definitions are superseded; do not implement them.
- Working title: **Two Policy Screens, One Intervention Budget: A Fixed-System Study of General-Safety and Mortgage-Policy Request Screening**
- Plan version: `v2-adversarial-review`
- Baseline reverified: 2026-07-13, Git commit `df604d9`

# Part I — Scientific Protocol

## 1. Purpose and completion definition

Reader path: Sections 1–3 give the scientific decision in under ten minutes; P0, P3, P4a, P7, P8, P9, and P10 are the claim-bearing protocol; P1, P2, P4, P5, and P11–P14 are the implementation/release SOP. The manuscript story is specified separately in P12 so toolchain detail does not become the paper narrative.

This document turns the Paper B research specification into an ordered development plan. It answers five questions at every checkpoint:

1. What must be built or collected?
2. What artifact proves that the work was completed?
3. What command or audit verifies the artifact?
4. What condition permits progression?
5. What scientific claim becomes permissible?

Historically, where this execution document corrected the companion plan—especially “joint calibration” terminology, policy paths, E3 definition, lock order, or test access—it governed that mortgage design. Neither document governs the current composition direction.

Paper B is complete only when checkpoints `P0` through `P14`, including the mandatory `P4a` construct gate, pass. Calendar progress, code volume, completed model runs, or a compiled PDF are not substitutes for these gates.

`P11a` mandatory reliability/replay/latency evidence must pass, but optional P11b/P11c arms need not all run. P11 passes when P11a is complete, every retained secondary claim has passed its own gate, and every untested arm/claim has been explicitly removed.

The primary paper is a fixed-component **request-screening** study:

> On untouched, naturalistic, fully dual-labeled mortgage requests, how do the errors of a general-safety screen and a mortgage-policy screen interact, and does a measured-union-constrained selector reduce unnecessary intervention relative to a marginal-sum-constrained selector while meeting preregistered missed-intervention constraints?

The controlled four-quadrant quartets probe and characterize the interaction. The naturalistic cohort, not the authored quartets, carries the primary comparative claim. If a powered naturalistic cohort is infeasible, the work becomes a controlled challenge-set measurement paper and its title and claims must say so.

The primary comparison is:

- `E3`, the **marginal-sum-constrained selector**: exhaustive selection on the common threshold grid with `r_G + r_D <= alpha_system`;
- `E4`, the **measured-union-constrained selector**: exhaustive selection on the same grid with `r_G + r_D - r_GD <= alpha_system`.

Both arms use identical components, scores, threshold grid, empirical development metrics, safety constraints, objective, and tie breakers. Only the benign-risk accounting rule differs. Base retention, action routing, learned fusion, cascading, and consolidated training are secondary experiments.

`E_sep` is a mandatory preregistered operational proxy: `G` and `D` thresholds are chosen independently on policy-specific development cohorts without dual labels or overlap information, then composed unchanged. Each component cohort comes from the same target frame/time window as `joint_dev`, and the two cohorts jointly match `joint_dev` on frozen annotation and scoring budgets (Section 2.1). It represents only this resource-matched separate-selection pipeline, not ordinary deployment in general. E3-versus-E4 isolates the cost of conservative union accounting; it must not be described as the same question as E4-versus-`E_sep`.

The primary system emits `PASS` or `INTERVENE`, not a universal hard `BLOCK` and not an evaluated downstream router. A general-safety trigger may ultimately block; a mortgage-policy trigger may require a constrained response, disclosure, escalation, or review. Those downstream actions are recorded as gold semantics but are not claimed effective unless separately evaluated. Use **false intervention** and **missed required intervention** for the primary endpoints.

Primary thresholds operate on each component's raw positive-minus-negative decision-logit margin, with the positive token explicitly mapped to `INTERVENE`. Positive temperature scaling is strictly monotone and therefore cannot change any attainable two-threshold action set. It is optional probability-calibration diagnostics, not the proposed mechanism. Use **measured-union-constrained selection** or **system operating-point selection**, never “joint calibration.”

### Critical construct gate discovered by review

The existing mortgage adapter is only a **legacy mortgage-risk adapter candidate**. Its 938 `flag` rows collapse heterogeneous response requirements—including `refuse_and_educate`, `acknowledge_uncertainty`, `responsible_guidance`, and `enforce_verification`—into one label. For example, a Black applicant asking whether discriminatory rates make applying pointless is flagged even though the upstream criterion requires reassurance and education. That label can support “special handling may be required”; it does not establish that the prompt should be blocked.

Therefore `D` is not frozen at P0. P3 must map source behavior to expert-approved request actions and audit every legacy label. P4a validates or replaces the **training target**; P6 then tests the candidate score's direction, discrimination, and attainable operating region on held-out pilot data. Until both pass, no artifact or manuscript text may call the candidate a mortgage-policy request screen.

## Readiness summary

| Checkpoint range | What may be written or claimed |
|---|---|
| P0–P3 | hypothesis, threat model, related work, policy scope; no empirical claim |
| P4–P6 | methods, engineering design, annotation protocol, data-card draft |
| P7–P9 | complete result-independent manuscript and preregistered analysis |
| After the single P10 custodian run | abstract results, results, discussion, and conclusion |
| P11–P12 | mandatory reliability integration and any preregistered secondary analyses |
| P13–P14 | release verification and submission authorization |

Current readiness is **before P0 and blocked on construct validity**: the threshold experiment is implementable, but the mortgage label/action mapping is not yet valid, and no Paper B policy registry, dual-labeled naturalistic cohort, package, lock, or claim-bearing artifact exists.

## 2. Non-negotiable scientific decisions

Freeze these decisions before claim-bearing data collection:

- Scope is US-mortgage **request screening and intervention indication** only; downstream routing efficacy is not evaluated.
- The paper is not legal advice and does not establish regulatory compliance.
- `G` is evaluated under a versioned general-safety policy.
- A mortgage candidate is evaluated mechanically after P4a target validation, but it becomes frozen `D` only after P6 held-out score validity passes; otherwise the study stops or retrains under a new study/pilot.
- A row requires intervention when `Y_g = 1` or `Y_d = 1`, but its required downstream action is a separate label.
- The primary utility endpoint is false-INTERVENE rate on `G0/D0` in the untouched naturalistic cohort.
- A minimum practically meaningful false-intervention reduction, `delta_intervene_min`, must be frozen before development outcomes; otherwise the paper may say only “statistically lower,” not “meaningfully improves.”
- Define `Delta_intervene = INTERVENE_E4 - INTERVENE_E3`. Statistical improvement requires the preregistered upper confidence bound `UCB(Delta_intervene) < 0`; meaningful improvement requires `UCB(Delta_intervene) <= -delta_intervene_min`.
- Weighted policy cost is secondary because its result depends on subjective weights.
- Missed-intervention constraints are reported separately for `G1/D0`, `G0/D1`, and `G1/D1`; an aggregate cannot hide failure in one policy space.
- The final test is accessed once, after the design, data, experiment, and policy locks exist.
- Primary binary arms route a scorer/system failure to `INTERVENE`; downstream `BLOCK`, `CONSTRAIN`, and `REVIEW` outcomes remain separate.
- A negative or inconclusive `E3` versus `E4` result is retained and reported; endpoints are not redesigned after test access.
- “High compliance,” “production ready,” and “complete agent boundary” are out of scope for this study; human-review benefit and protected-class fairness require separate downstream evidence beyond the prompt-screen checks here.

### 2.1 Formal primary contract

For component `k in {G,D}`, define the raw margin

\[
s_k(x)=\ell_{k,\mathrm{intervene}}(x)-\ell_{k,\mathrm{pass}}(x),\qquad
b_k(x;\tau_k)=
\begin{cases}
1,&\text{component }k\text{ has a missing/error score},\\
\mathbf 1[s_k(x)\ge \tau_k],&\text{otherwise}.
\end{cases}
\]

The primary composed decision is

\[
b_{OR}(x;\tau_G,\tau_D)=b_G(x;\tau_G)\lor b_D(x;\tau_D),
\]

where one means `INTERVENE`. The four gold quadrants use binary **needs-intervention** labels, while `general_action_gold`, `domain_action_gold`, and `final_action_gold` retain `PASS`, `CONSTRAIN`, `REVIEW`, or `BLOCK`. The action lattice and conflict rules are expert-approved at P3; the binary primary endpoint must never be substituted for action correctness.

On a development cohort, let

\[
r_G=P_n(b_G=1\mid G0/D0),\quad
r_D=P_n(b_D=1\mid G0/D0),\quad
r_{GD}=P_n(b_G=b_D=1\mid G0/D0),
\]

so the observed false-intervention rate is exactly

\[
r_{OR}=r_G+r_D-r_{GD}.
\]

This is the finite-sample inclusion-exclusion identity and requires no independence assumption. E3 discards the nonnegative overlap term; E4 credits only the overlap measured on naturalistic development data.

For the protected-context gate, construct expert-approved counterfactual pairs `(x_j^P,x_j^R)` that differ only in protected-class context and are both `G0/D0`. For locked E4 define

\[
\Delta_{context}=J^{-1}\sum_{j=1}^{J}\left[b_{E4}(x_j^P)-b_{E4}(x_j^R)\right].
\]

The one-sided upper bound must satisfy `UCB(Delta_context) <= delta_context_max`. The confirmatory equation gives every valid pair equal weight; traffic-weighted variants are sensitivity analyses only. Freeze pair eligibility, inference unit, attribute strata, missing-pair handling, margin, and any per-attribute Holm family at P0. This detects prompt-screen sensitivity only; it is not evidence about lending-decision fairness.

Let `m_q` be the empirical missed-intervention rate in intervention-positive quadrant `q in {10,01,11}`, and let `beta_q` be its frozen cap. Both primary arms enumerate the same action-equivalence grid

\[
\mathcal T_k=\{\mathrm{ALWAYS},\mathrm{NEVER}\}\cup\{s_k(x_i):i\in\mathrm{joint\_dev}\}.
\]

`ALWAYS` and `NEVER` are tagged sentinels; they are not serialized as non-standard JSON infinities. With the `>=` comparator, they correspond mathematically to negative and positive infinity.

Between adjacent unique margins, every row's component decision is constant. Thus every continuous threshold pair has an action-equivalent representative in `T_G x T_D`, and exhaustive enumeration returns the global empirical optimum for the frozen components, data, constraints, objective, and tie breakers. It proves an optimization fact only—not population optimality or risk control.

The empirical development selectors are:

\[
E3=\arg\min_{(\tau_G,\tau_D)} r_{OR}
\quad\text{s.t.}\quad r_G+r_D\le\alpha_{system},\;m_q\le\beta_q\;\forall q,
\]

\[
E4=\arg\min_{(\tau_G,\tau_D)} r_{OR}
\quad\text{s.t.}\quad r_{OR}\le\alpha_{system},\;m_q\le\beta_q\;\forall q.
\]

On the naturalistic test, let `y_i` be final intervention gold, `v_i` the frozen severe indicator, `b_{a,i}` arm `a`'s intervention, and `w_i` the preregistered **sampling** weight. Prefer a self-weighting sampler and set every primary `w_i=1`. If unequal inclusion/attrition weights are necessary, freeze them before scoring and use survey/cluster-weighted inference. Traffic/cost scenario weights are different secondary quantities and never replace `w_i`. The primary row-marginal estimands are

\[
R_{miss}^{a}=\frac{\sum_i w_i v_i y_i(1-b_{a,i})}{\sum_i w_i v_i y_i},\qquad
R_{false}^{a}=\frac{\sum_i w_i(1-y_i)b_{a,i}}{\sum_i w_i(1-y_i)},
\]

with paired contrasts `Delta_miss = R_miss^E4 - R_miss^E3` and `Delta_intervene = R_false^E4 - R_false^E3`. Two dependency structures are mandatory and cannot substitute for one another:

1. `split_lineage_graph` connects rows sharing any frozen leakage-capable source conversation, base scenario/family, author batch, template, generator/seed, or other derivation lineage. Every connected component is assigned wholly to training, pilot, component development, joint development, or test; a multiway variance estimator can never excuse a cross-role edge.
2. Within each role, `inference_cluster_dimensions` records remaining crossed dependence, including any repeated author/source/control strata. P7 freezes a named graph-component or multiway/survey estimator and validates coverage for that design. A policy control may cross roles only if it is explicitly treated as a fixed repeated stratum, the target population is restricted accordingly, and no example/template derivation crosses; otherwise control identity belongs in the split graph.

The row-marginal estimand is never silently replaced with “any failure in a cluster.”

For the operational reference, independently freeze

\[
\tau_G^{sep}=\arg\min_{\tau} m_G(\tau)\;\text{s.t.}\;f_G(\tau)\le\alpha_G^{sep},\qquad
\tau_D^{sep}=\arg\min_{\tau} m_D(\tau)\;\text{s.t.}\;f_D(\tau)\le\alpha_D^{sep},
\]

on `component_dev_general` and `component_dev_domain`, respectively, using policy-specific labels only. Their component objectives, budgets, sentinels, and tie breakers are frozen before scoring. `E_sep` is their unchanged OR composition and receives no joint-development tuning. For a strong resource-matched proxy, both component cohorts use the same target frame, eligibility rules, time window, and dependency-unit definition as `joint_dev`; each receives the same prespecified number of independent components as `joint_dev`, but only its own policy labels. Thus the two component cohorts together match the dual-labeled joint cohort on policy-label judgments and component-score calls. Freeze reviewer/adjudication effort and report realized row count, effective component count, covariate shift, and budget deviations. Because its component denominators and budgets do not define a system union budget, its final system behavior is measured rather than assumed.

Define normalized utilization as `rho(z,c)=z/c` for `c>0`, `rho(0,0)=0`, and `rho(z,0)=+infinity` for `z>0`; infinity is internal math and is serialized as a tagged value. For a feasible pair,

\[
U_{max}=\max\left\{\rho(r_{OR},\alpha_{system}),\max_q\rho(m_q,\beta_q)\right\},\qquad
M_{all}=\frac{\sum_i w_i y_i(1-b_{OR,i})}{\sum_i w_i y_i},
\]

where `M_all` uses every intervention-positive `joint_dev` row and the same primary sampling weights. Threshold rank is `ALWAYS`, ascending finite unique margins, then `NEVER`. The secondary tie breakers are frozen in this order: smallest `U_max`, smallest `M_all`, then lexicographic `(rank(tau_G),rank(tau_D))`. Empty denominators are infeasible, never zero-filled. Development constraints are explicitly empirical; they are not confidence guarantees after adaptive grid search. Confirmatory confidence bounds are computed only for the already locked policies on untouched test data.

Because `r_OR <= r_G + r_D`, E3's feasible set is a subset of E4's. Therefore E4's development objective is mathematically guaranteed to be no worse. This is not an empirical result and must never appear as evidence of benefit. The hypothesis is whether crediting measured error overlap yields a better **locked test** operating point without violating absolute test constraints. Candidate counts, overlap `r_GD`, and set inclusion are reported so the mechanism is auditable.

The confirmatory primary contrast remains E4 versus E3 because it changes one accounting rule. E4 versus `E_sep` is a mandatory prespecified comparison between these specific frozen pipelines. Confirmatory wording requires its resource-match/shift gate, paired safety/utility gate, power, and multiplicity slot to be locked in `design.json`; otherwise it is descriptive. Even on PASS, say “better than the resource-matched separate-selection proxy in this study,” never “better than separate deployment” generally.

For any strictly increasing transform `h`,

\[
\mathbf 1[s_k\ge\tau]=\mathbf 1[h(s_k)\ge h(\tau)].
\]

Positive temperature scaling is such a transform. Consequently it leaves the primary candidate actions and optimum unchanged; a unit test must prove this invariance. Calibration metrics may be reported on held-out data only as diagnostics or used in secondary fusion, never as an explanation for E4.

Section 2.1 is the sole normative mathematical definition. P0 serializes it as schema-valid `$STUDY_ROOT/design/primary_contract.json` and records its hash; all later checkpoints consume that object from the parent lock. Later prose and command comments are implementation checks, not independent definitions. Any disagreement with Section 2.1 or the contract hash is a hard error requiring a new study version, not an invitation to choose the favorable interpretation.

### 2.2 Research questions after correction

1. How large and stable is component error overlap, and how accurately do marginal component rates characterize the composed system across controlled and authentic naturalistic strata?
2. On untouched naturalistic requests, does the measured-union-constrained E4 selector improve the locked operating point over the marginal-sum-constrained E3 selector while satisfying absolute and comparative constraints?
3. **Secondary, only if the P11c design is locked, powered, fitted, and included by P9 before P10:** does the result replicate across policy-control holdouts and at least one heterogeneous guard pair? Otherwise the conclusion remains a one-pair case study and RQ3 is omitted from the submitted paper.

## 3. Current-state baseline

### 3.1 Assets that exist now

| Asset | Current evidence | Permitted use |
|---|---|---|
| SmolLM3-3B base | pinned candidate revision `a07cc9a04f16550a088caea529712d1d335b0ac1` | Paper B fixed backbone after acquisition verification |
| General LoRA adapter | local ignored weights; weight SHA-256 `c4076b7fa123281df3f7c1aa866321ac37fdabf0816bfa073d19153a14d8c1e6` | engineering pilot until complete bundle is immutably published |
| Legacy mortgage-risk adapter | local ignored weights; weight SHA-256 `253e65d9a57bd1ba0e29a9346c257593902f2acf94c922452aa7d95b26a74f51` | mechanics only until P4a validates the target and P6 validates held-out score behavior |
| Mortgage training source | 1,563 rows: 938 flag and 625 allow; exact joins show flags spanning multiple desired response behaviors | provenance/action audit and possible retraining; never direct hard-BLOCK gold |
| Mortgage hard set | 334 rows: 195 flag and 139 allow; 30 known minimal-pair groups | development/stress and re-adjudication seed only |
| Mortgage red-team source | 1,000 rows; 764 rows carry regulatory references | recover policy provenance and annotation criteria |
| General evaluation bundle | local ignored dev/test/transfer rows | diagnostics until acquired through immutable instructions |
| Canonical metrics | `guard_research/metrics.py` | reuse; do not implement another AP/AUROC copy |
| Provenance helpers | `guard_research/provenance.py` | reuse normalization, content, object, and file hashing |
| General prompt helper | `guard_research/prompts.py` | reuse and lock rendered prompt/token hashes |
| Conservative threshold helper | `guard_research/thresholds.py` | reuse only for its documented single-component purpose |
| Existing tests | local artifact-populated workspace: 30 pass; tracked-only clean checkout: 13 pass and 17 skip because ignored manifests are absent | partial engineering foundation, not clean-clone or Paper B validation |
| Tie-aware hard-cache check | base AP `0.890`, general guard `0.820`, mortgage-SFT `0.895` | diagnostic only; retire the stale mortgage-SFT `0.924` claim |

### 3.2 Missing mandatory evidence

The current repository has none of the following:

- a `guard_stack/` Paper B package;
- Paper B policy registries or frozen source snapshots;
- a completed mortgage label-to-action construct audit or a validated `D` component;
- fully dual-labeled controlled and naturalistic joint-development/final-test rows;
- Paper B score, policy, evaluation, replay, or latency artifacts;
- an exact joint optimizer or test-isolation enforcement;
- Paper B-specific tests or continuous integration;
- a Paper B manuscript source;
- a public immutable acquisition path for the local adapters and frozen rows.

### 3.3 Baseline defects that must not enter Paper B

- The existing 334-row caches score `B`, `G`, and `D` with the mortgage prompt. Paper B requires the general prompt for `B/G` and the mortgage prompt for `D`.
- Existing mortgage caches are accepted by array length rather than row/model/adapter/prompt/code identity.
- Existing score caches and adapters are ignored and absent from a fresh clone.
- The hard set has been inspected repeatedly and cannot become the untouched final test.
- The active `.venv` runs Python 3.14.4 while `.python-version` specifies 3.12.
- Direct `.venv/bin/pytest` has a stale shebang pointing to the former `agent-bouncer` path; only `.venv/bin/python -m pytest` currently works.
- `docs/mortgage-benchmark-hard-results.md` contains stale AP values and old `scripts/` paths.
- Partial general/mortgage labels cannot be concatenated and treated as joint non-intervention gold.
- A normalized, lowercased content hash is suitable for overlap detection but not score-cache identity; exact event bytes and rendered-prompt hashes must drive cache invalidation.
- `guard_research.provenance.minhash_signature` currently auto-selects different algorithms depending on whether `datasketch` is installed. Paper B must pin one backend and algorithm ID.
- The legacy mortgage `flag` target does not distinguish hard block from constrained/helpful response behavior and cannot be used as a binary block label.

# Part II — Execution SOP

## 4. Artifact architecture

### 4.1 New source layout

```text
pyproject.toml
uv.lock
configs/paper_b_v1.yaml
configs/paper_b_source_roles_v1.yaml
configs/paper_b_naturalistic_sampler_v1.yaml
configs/paper_b_no_test_runtime_v1.yaml

policies/general/v1/controls.jsonl
policies/general/v1/annotation_manual.md
policies/mortgage/us_v1/controls.jsonl
policies/mortgage/us_v1/annotation_manual.md
policies/mortgage/us_v1/snapshots/

schemas/paper_b/row.schema.json
schemas/paper_b/catalog.schema.json
schemas/paper_b/lock.schema.json
schemas/paper_b/design_lock.schema.json
schemas/paper_b/claim_registry.schema.json
schemas/paper_b/primary_contract.schema.json
schemas/paper_b/approval.schema.json
schemas/paper_b/action_construct.schema.json
schemas/paper_b/policy_construct_snapshot.schema.json
schemas/paper_b/component_validity_spec.schema.json
schemas/paper_b/naturalistic_sampling_commitment.schema.json
schemas/paper_b/component_registry.schema.json
schemas/paper_b/annotation.schema.json
schemas/paper_b/adjudication.schema.json
schemas/paper_b/annotation_summary.schema.json
schemas/paper_b/manifest.schema.json
schemas/paper_b/test_seal.schema.json
schemas/paper_b/custody_attestation.schema.json
schemas/paper_b/reproduction_attestation.schema.json
schemas/paper_b/score.schema.json
schemas/paper_b/calibration.schema.json
schemas/paper_b/policy.schema.json
schemas/paper_b/evaluation.schema.json
schemas/paper_b/audit.schema.json
schemas/paper_b/replay.schema.json
schemas/paper_b/latency.schema.json
schemas/paper_b/primary_comparison.schema.json
schemas/paper_b/claim_check.schema.json
schemas/paper_b/score.arrow.json
schemas/paper_b/selection_trace.arrow.json
schemas/paper_b/prediction.arrow.json
schemas/paper_b/bootstrap_draw.arrow.json
schemas/paper_b/latency.arrow.json

guard_stack/__init__.py
guard_stack/schema.py
guard_stack/artifacts.py
guard_stack/controlled_store.py
guard_stack/locking.py
guard_stack/prompts.py
guard_stack/multi_adapter_scorer.py
guard_stack/calibration.py
guard_stack/decision.py
guard_stack/optimize_policy.py
guard_stack/policy_registry.py
guard_stack/statistics.py
guard_stack/audit.py
guard_stack/replay.py

experiments/fetch_paper_b_assets.py
experiments/import_paper_b_annotations.py
experiments/audit_mortgage_action_construct.py
experiments/build_component_registry.py
experiments/build_joint_manifest.py
experiments/score_joint_stack.py
experiments/fit_component_calibration.py
experiments/fit_reference_policies.py
experiments/fit_joint_policy.py
experiments/power_paper_b.py
experiments/eval_joint_stack.py
experiments/benchmark_joint_latency.py
experiments/replay_joint_stack.py
experiments/run_failure_injection_paper_b.py
experiments/build_paper_b_fixtures.py
experiments/analyze_paper_b.py
experiments/validate_paper_b_artifacts.py
experiments/lock_paper_b.py
experiments/seal_naturalistic_sampling_frame.py
experiments/verify_sampling_commitment.py
experiments/run_policy_fit_paper_b.py
experiments/run_sealed_paper_b.py
experiments/train_mortgage_constrained.py
experiments/train_consolidated_guard.py

tests/paper_b/test_artifact_schemas.py
tests/paper_b/test_action_construct.py
tests/paper_b/test_prompts.py
tests/paper_b/test_multi_adapter_scorer.py
tests/paper_b/test_score_identity.py
tests/paper_b/test_calibration.py
tests/paper_b/test_decision.py
tests/paper_b/test_optimizer.py
tests/paper_b/test_statistics.py
tests/paper_b/test_manifests.py
tests/paper_b/test_test_isolation.py
tests/paper_b/test_controlled_store.py
tests/paper_b/test_locking.py
tests/paper_b/test_replay.py
tests/paper_b/test_failure_policy.py
tests/paper_b/test_latency_protocol.py
tests/fixtures/paper_b/synthetic_policy.json
tests/fixtures/paper_b/inference_cases.jsonl

paper_b/two_policy_screens_one_intervention_budget.tex
paper_b/refs.bib
paper_b/figures/
paper_b/tables/
paper_b/Makefile

tools/TOOLCHAIN.lock.json
tools/bootstrap_toolchain.sh
tools/run_tectonic_locked.sh
keys/paper_b_trust_policy.json
keys/research_owner.pub
keys/general_policy_owner.pub
keys/statistician.pub
keys/mortgage_sme.pub
keys/data_owner.pub
keys/custodian.pub
keys/auditor.pub
keys/reproducer.pub
keys/release_owner.pub
```

Use `guard_research.metrics` and `guard_research.provenance` as the canonical shared implementations. If a `guard_stack.metrics` compatibility module is needed, it must only re-export the shared functions. A static test must reject new hand-written AP/AUROC implementations.

Paper B must not inherit ambiguous hashing behavior. JSON lock material uses pinned RFC 8785/JCS canonicalization, rejects NaN/infinity, normalizes no string content implicitly, and computes `self_sha256` with that field omitted. Signatures are detached sidecars and are not part of the self-hash. Canonical tabular logical hashes use a frozen Arrow schema fingerprint, declared primary-key sort, chunk-independent canonical IPC encoding, preserved exact UTF-8 values, normalized null representation, and rejected NaN/infinity; Parquet byte hashes are stored separately. Directory manifests specify POSIX relative paths, file bytes, executable bit, and symlink policy. Golden cross-language hash vectors are mandatory.

Normalized `content_sha256` remains a candidate-overlap identity only. Score-cache identity uses exact canonical event bytes, rendered prompt bytes/hash, component identity, and scoring code. A case/whitespace change may retain the overlap hash but must invalidate the score cache.

### 4.2 Immutable artifact tree

Every study version has its own root. Set these variables in every documented shell session:

```bash
export STUDY_ID=paper_b_v1
export STUDY_ROOT="artifacts/paper_b/studies/${STUDY_ID}"
export POST_RELEASE_ROOT="artifacts/paper_b/post_release/${STUDY_ID}"
```

There is no implicit “active study” lookup. Every command receives explicit lock/artifact paths, and every relevant CI job declares `STUDY_ID`, `STUDY_ROOT`, and `POST_RELEASE_ROOT` in job-level `env` because shell exports do not persist across workflow steps automatically. Failed pilots and retired studies retain their roots; a new version creates a new `STUDY_ID` and never overwrites an old lock.

```text
$STUDY_ROOT/
  STATUS.md
  catalog.json
  catalog.experiment.json
  locks/
    pilot.json
    design.json
    data.json
    experiment.json
    policy.json
    results.json
    release.json
    *.minisig
    source_manifests/
  design/
    preregistration.md
    statistical_analysis_plan.md
    power_report.json
    power_report.md
    claim_registry.json
    primary_contract.json
    claim_registry.active.json
    claim_registry_amendment.json
    component_validity_spec.json
    approvals/
  construct/
    policy_construct_snapshot.json
    mortgage_action_audit.json
    mortgage_action_audit.md
    component_validity.json
    component_registry.json
    component_registry_experiment.json
    policy_coverage.json
  manifests/
    naturalistic_sampling_commitment.json
    diagnostic/
    pilot/
    calibration_fit/          # optional diagnostic only
    calibration_eval/         # optional diagnostic only
    component_dev_general/
    component_dev_domain/
    joint_dev/
    joint_test/
    naturalistic_dev/
    naturalistic_test/
    controlled_dev/
    controlled_test/
    human_authored_simulation/ # optional; never labeled naturalistic
    hard_stress/
    domain_training_authorized/ # exact P4a-authorized domain training lineage
    general_replay_candidate/   # frozen before final sampling; never scored
    general_replay/             # only when E9 is retained
    consolidated_train/         # only when E9 is retained
  audits/
    final_train_evaluation_lineage_audit.json
    experiment_asset_reacquisition.json
  annotations/
    pilot/
      pilot_summary.json
      raw_judgments.manifest.json
      agreement_report.json
      adjudication_log.jsonl
    powered/
      powered_summary.json
      raw_judgments.manifest.json
      agreement_report.json
      adjudication_log.jsonl
      blinding_audit.json
    calibration/             # optional, design-gated diagnostic only
    hard_stress/
      agreement_report.json
      adjudication_log.jsonl
  scores/<cohort>/
    scores.parquet
    metadata.json
  diagnostics/calibration/<component_id>.json
  diagnostics/interaction/development.json
  components/generated/e9_candidates/ # optional model bundles, never policies
  policies/<arm>/
    decision_policy.json
    selection_trace.parquet
  evaluation/<arm>/
    predictions.parquet
    metrics.json
    bootstrap_draws.parquet
    claim_check.json
  evaluation/primary_comparison/
    paired_effects.json
    bootstrap_draws.parquet
    claim_check.json
  latency/
    workload.jsonl
    requests.parquet
    summary.json
  replay/
    cases.jsonl
    diagnostic_score_replay.json
    diagnostic_inference_replay.json
    pre_test_report.json
    report.json
  reliability/
    failure_injection.json
    failure_injection.json.minisig
  attestations/
    custody_data.json
    custody_results.json
    policy_fit_supervision.json
    policy_fit_supervision_verification.json
  ethics/
  release/
    public_objects.json
    access_instructions.md
  analysis/
    tables/
    figures/
    report.md
```

Post-release records are intentionally outside the immutable study root:

```text
$POST_RELEASE_ROOT/
  sampling_verification.json
  sampling_verification.json.minisig
  independent_reproduction.json
  independent_reproduction.json.minisig
  final_review.json
  final_review.json.minisig
```

Raw or license-restricted text may remain external. The committed catalog must contain immutable URI/DOI/object identifiers, byte hashes, logical hashes, release class, and acquisition instructions. A fresh clone must fail clearly when a required external object is unavailable or mismatched.

`controlled://` is a logical URI, not a security claim. P1 must implement and pin a concrete resolver in `guard_stack.controlled_store` backed by versioned objects, encryption/KMS, role-specific credentials, immutable object versions, access logs, and egress restrictions. The custodian configuration and secret credentials stay outside Git; the provider, resolver code, image digest, IAM/KMS policy hashes, custodian public key, and retention rule are locked. `--custodian-mode` is accepted only when a signed job attestation from that environment is present. A local filesystem backend is allowed for synthetic tests only and can never authorize P10.

## 5. Master checkpoint sequence

| ID | Checkpoint | Main output | Scientific state after pass |
|---|---|---|---|
| P1 | Reproducible environment and repository foundation | locked Python environment and CI | engineering baseline |
| P0 | Scope, claim, and governance lock | signed preregistration skeleton | hypothesis only |
| P2 | Asset inventory and acquisition catalog | verified base/adapters/source catalog | local feasibility is auditable |
| P3 | Policy-construct lock | two registries and annotation manuals | annotation may begin |
| P4 | Data provenance and diagnostic manifests | audited partial-label bundles | diagnostics only |
| P4a | Mortgage action-target audit | validated intervention target and pilot candidate, or retraining/NO-GO | candidate may enter held-out P6 validation |
| P5 | Correct scorer and runtime pilot | identity-keyed B/G/D logits and replay | mechanics claim only |
| P6 | Blinded dual-label annotation pilot | 60 controlled families plus authentic naturalistic pilot and validity reports | construct/component feasibility |
| P7 | Power, precision, and signed design lock | powered cohort decision and `design.json` | confirmatory collection authorized |
| P8 | Powered cohort construction and test seal | component-dev, controlled, authentic naturalistic, and sealed test manifests | confirmatory data frozen |
| P9 | Experiment lock and development-only policy fit | locked E3/E4 policies | final test authorized |
| P10 | One-shot final evaluation | signed primary result bundle | positive, negative, or inconclusive result fixed |
| P11 | Mandatory reliability and optional secondary gates | replay/failure/latency plus retained ablations | bounded systems claims |
| P12 | Results integration and manuscript finalization | generated paper source/PDF | submission candidate |
| P13 | Fresh-clone release and independent replay | release lock and reproduction report | artifact-complete paper |
| P14 | Final technical-review audit | signed completion matrix | submit or stop |

The table order is authoritative: P1 must provide the validator, schemas, trust policy, and signing tool before P0 can PASS. P0 may be drafted earlier, but it is not locked or passed until P1 succeeds. Work within a phase can run in parallel; a later phase cannot repair a failed earlier gate without a new study version.

## 6. P0 — Scope, claim, and governance lock

### Objective

Prevent the project from changing its task, endpoint, or evidence standard after results become visible.

P0 protocol text may be drafted immediately, but its validation/signature command runs only after P1 PASS; no pilot annotation or scoring occurs between draft and signed P0.

### Required work

1. Freeze the title, input-only scope, US-mortgage jurisdiction, product/workflow boundaries, and exclusions.
2. Assign named owners for:
   - research protocol;
   - general-policy rubric;
   - mortgage-policy registry and SME review;
   - annotation operations;
   - statistics;
   - model/runtime engineering;
   - test-set custody;
   - artifact/release audit.
3. Define `B` and `G`, plus the candidate `D_candidate`, including exact model, adapter, tokenizer, prompt, training-target, and intended-action identities. P4a may authorize the target/candidate; freeze `D` only after P6 held-out score validity passes.
4. Define the binary primary decision as `PASS` versus `INTERVENE`, and separately freeze the downstream action lattice and conflict rule.
5. Define E3 and E4 exactly as in Section 2.1: the same exhaustive threshold grid with marginal-sum versus measured-union accounting. No allocation grid or component-objective pruning is allowed in the primary baseline.
6. Define mandatory `E_sep`, its two policy-specific development populations, component objectives, `alpha_G_sep`/`alpha_D_sep`, sentinels, and tie breakers without using joint labels or scores. Require the resource-matching contract in Section 2.1: common target frame/time window/dependency definition, each component cohort allocated the same independent-component count as `joint_dev`, and matched total policy-label/reviewer/adjudication and component-score budgets. Freeze shift/budget tolerances. Declare its E4 comparison descriptive or assign it a powered secondary hypothesis/multiplicity slot; even a PASS is limited to the specific frozen proxy.
7. Freeze `alpha_system`, every `beta_q`, equality semantics (`<=` passes), the score comparator (`>=` intervenes), and all tie breakers.
8. Choose false-INTERVENE as the single primary utility endpoint and freeze `delta_intervene_min`, the smallest practically meaningful reduction.
9. Define separate one-sided missed-intervention gates for all three intervention-positive quadrants, the absolute E4 false-intervention budget gate, and the exact `Delta_context` contrast/pair denominator/weighting plus `delta_context_max` from Section 2.1.
10. Choose the paired noninferiority margin from an operational risk argument, not convenience.
11. Freeze this primary familywise-error procedure at one-sided `alpha=0.05` unless the statistician records a stricter value before pilot outcomes:
    1. test overall missed-intervention noninferiority;
    2. only if it passes, test the four absolute E4 constraints (`alpha_system` and three `beta_q` caps) with Holm control at familywise 0.05;
    3. only if all pass, test the prespecified protected-context prompt-screen non-worsening hypothesis at one-sided 0.05;
    4. only if it passes, test E4-versus-E3 false-intervention superiority at one-sided 0.05;
    5. treat `delta_intervene_min` as a confidence-bound interpretation gate, not a second chosen endpoint.
12. Make the naturalistic test stratum primary; controlled quartets are mechanism evidence. Freeze separate, non-pooled reporting.
13. Prespecify positive, negative, inconclusive, and deployment-NO-GO wording.
14. Freeze every numeric P6 component-validity gate and its dependency-aware estimator in `configs/paper_b_v1.yaml`; P6 may only materialize and sign those values, not choose or relax them after scores exist.

### Required artifacts

- `$STUDY_ROOT/design/preregistration.md`
- `$STUDY_ROOT/design/claim_registry.json`
- `$STUDY_ROOT/design/primary_contract.json`, the machine-readable Section 2.1 contract
- initial `configs/paper_b_v1.yaml`
- `$STUDY_ROOT/STATUS.md`

The claim registry needs these fields:

```text
claim_id
draft_claim
claim_tier                 # feasibility | comparative | generality
primary_or_secondary
required_endpoint
required_gate
allowed_wording_on_pass
required_wording_on_fail
forbidden_wording
evidence_paths[]
status                     # NOT_TESTED | PASS | FAIL | INCONCLUSIVE
```

### Verification

- Every planned abstract/contribution claim maps to one claim ID.
- Exactly one primary utility endpoint is declared.
- `alpha_system`, `alpha_G_sep`, `alpha_D_sep`, all `beta_q`, the two non-worsening margins, and `delta_intervene_min` are non-null frozen values with operational rationales.
- Cost weights and traffic scenarios are frozen as secondary scenario analyses.
- The protocol says how ambiguous rows, scorer failures, conflicting required actions, and missing reviewer capacity are handled.
- No sentence claims response screening, tool screening, legal compliance, or production readiness.

### PASS gate

All owners sign the protocol with role-specific keys; the statistician approves the exact testing hierarchy; the mortgage SME approves the action construct and stated domain scope; the test custodian is independent of policy fitting. After P1 supplies the validator, run the following against signed approvals before marking P0 PASS:

```bash
uv run python experiments/validate_paper_b_artifacts.py \
  --stage protocol \
  --config configs/paper_b_v1.yaml \
  --preregistration $STUDY_ROOT/design/preregistration.md \
  --claim-registry $STUDY_ROOT/design/claim_registry.json \
  --primary-contract-out $STUDY_ROOT/design/primary_contract.json \
  --schema schemas/paper_b/primary_contract.schema.json \
  --approval-dir $STUDY_ROOT/design/approvals \
  --strict
```

### STOP conditions

- The primary endpoint is still “false intervention **or** cost,” allowing a favorable result to be chosen later.
- E3 or E4 uses component-objective pruning, an arbitrary allocation grid, or a different threshold grid.
- The noninferiority margin or traffic weights are deferred until after pilot/model results.
- The stated scope includes response correctness, credit decisions, or a full agent boundary without response/tool datasets.
- Model or adapter identities cannot be made immutable.

### Claims enabled

None. This checkpoint enables only protocol and methods-skeleton writing.

## 7. P1 — Reproducible environment and repository foundation

### Objective

Make every later result installable, testable, and reproducible from a clean Python 3.12 environment.

### Required work

1. Add `pyproject.toml` with Python `==3.12.*`, runtime dependencies, development dependencies, console entry points, and pytest/ruff/mypy configuration.
2. Add `uv.lock` and use `uv` as the single environment workflow documented by this plan.
3. Remove the duplicate `scikit-learn` dependency declaration.
4. Create the new package, schema, tests, artifact, and manuscript directories.
5. Update `.gitignore` with tested patterns for Paper B raw/nested manifest rows, scores, annotations/rationales, controlled exports, weights, secrets, and temporary decrypted objects. Explicitly unignore policies, schemas, text-free indexes, locks, audits, attestations, and generated manuscript sources. Add `git check-ignore` tests for representative sensitive and public paths.
6. Implement the JCS/Arrow/directory canonicalization contract above and freeze Arrow schemas and logical-schema fingerprints for Parquet scores, selection traces, predictions, bootstrap draws, and latency rows; JSON Schema alone cannot validate Parquet.
7. Add `tools/TOOLCHAIN.lock.json` with exact versions, acquisition URIs, and hashes for `uv`, Tectonic, its bundle, and any signing binary. `uv` is not currently installed and ambient Tectonic is not a sufficient release lock.
8. Define deterministic paper-build settings, including pinned Tectonic/bundle, locale/fonts, and `SOURCE_DATE_EPOCH`; Paper B release requires byte-identical PDF reproduction.
9. Add CI with no model downloads or secrets for ordinary pull requests.
10. Preserve the 30-test local result, eliminate or explicitly fail on the 17 clean-checkout artifact skips, and add committed synthetic fixtures so ordinary CI has a deterministic expected pass/skip count.
11. Pin one Paper B MinHash backend, algorithm ID, seed, and version; disable dependency-driven auto-selection and add golden family-ID fixtures.
12. Implement and test the concrete controlled-store resolver, signed job-attestation validation, and role-specific trust policy.
13. Add a static rule prohibiting duplicate metric implementations.

### Bootstrap commands

These commands become authoritative after `pyproject.toml` and `uv.lock` are added:

```bash
./tools/bootstrap_toolchain.sh --lock tools/TOOLCHAIN.lock.json --verify
uv sync --frozen --python 3.12
uv run python --version
uv run ruff check .
uv run mypy guard_stack
uv run pytest -q -m "not model and not release"
uv run python experiments/validate_paper_b_artifacts.py --schemas-only
```

Until then, the current baseline check is:

```bash
.venv/bin/python -m pytest -q
```

Current verified baseline: artifact-populated workspace `30 passed`; tracked-only clean checkout `13 passed, 17 skipped`. The latter is a known reproducibility defect, not a passing clean-clone baseline.

### Required artifacts

- `pyproject.toml`
- `uv.lock`
- `.github/workflows/ci.yml`
- empty but schema-valid Paper B artifact catalog
- `$STUDY_ROOT/STATUS.md` showing P1 evidence

### CI requirements

- Python 3.12 only.
- `permissions: contents: read`.
- No network, HF token, OpenAI key, model, or adapter use **during tests after the frozen dependency restore**. If dependency restore must also be offline, use a pinned prebuilt image/wheelhouse digest.
- Lint, type-check, JSON-schema validation, and synthetic unit/property tests.
- Expected pass/skip counts are asserted; missing required fixtures cannot silently turn failures into skips.
- Required artifacts fail a release validator; tests must not silently skip them.

### PASS gate

A clean checkout installs from the frozen lock and passes shared plus Paper B synthetic tests without using the ambient `.venv`, with zero unexpected skips and no sensitive artifact tracked.

### STOP conditions

- The run depends on Python 3.14 or the relocated `.venv`.
- Only a flat, unhashed requirements file exists.
- `uv`/Tectonic/signing tool versions or bundles remain ambient and unverified.
- Test collection relies on implicit current-working-directory imports.
- Missing required release artifacts are reported as skipped tests.

### Claims enabled

Engineering environment only.

## 8. P2 — Asset inventory and acquisition catalog

### Objective

Convert local ignored files into explicit, content-addressed study inputs.

### Required work

1. Implement `experiments/fetch_paper_b_assets.py`.
2. Inventory and hash:
   - base model snapshot;
   - tokenizer snapshot;
   - general adapter config and weights;
   - mortgage adapter config and weights;
   - source benchmark files;
   - local frozen general rows used for migration.
3. For every directory asset, store a sorted repo-relative per-file path/size/hash manifest and compute the canonical bundle hash from that manifest. Hash each complete adapter bundle, not only `adapter_model.safetensors`.
4. Require model and tokenizer revision agreement.
5. Record license and redistribution class for every input.
6. Define public immutable acquisition URIs or a reconstruction process.
7. Treat old arrays/caches as migration inputs only.

### Minimum catalog record

```text
artifact_id
artifact_type
external_uri|null
local_path|null
release_class
byte_sha256|null
bundle_manifest_sha256|null
canonical_logical_sha256|null
size_bytes
model_or_dataset_revision|null
license_id
acquisition_command
created_by_git_sha
```

`local_path`, when present, must be repository-relative. Machine-local absolute paths are invalid evidence.
`byte_sha256` is required for a single file or a canonically specified archive. Directory assets use the sorted per-file manifest plus `bundle_manifest_sha256`; an ad hoc tar/zip hash is not their identity.

### Verification commands

```bash
uv run python experiments/fetch_paper_b_assets.py \
  --config configs/paper_b_v1.yaml \
  --asset-root external/paper_b/acquisition_a \
  --out-catalog $STUDY_ROOT/catalog.json

uv run python experiments/fetch_paper_b_assets.py \
  --catalog $STUDY_ROOT/catalog.json \
  --asset-root external/paper_b/acquisition_b \
  --acquire-and-verify
```

P2 finalizes the content-addressed asset catalog only. `pilot.json` is created at P6 after the pilot manifest and P5 scorer/schema code exist; creating it earlier would leave later pilot evidence outside the lock.

### PASS gate

Two clean, independent destination roots acquire or reconstruct every required fixed input and produce the same hashes. A clear missing-object error is useful diagnostics but is not a P2 PASS.

### STOP conditions

- An adapter remains identified only by an ignored local path.
- `adapter_config.json` lacks revision provenance with no external reconciliation record.
- Moving `main` revisions are accepted.
- Any cache can pass validation based only on length.

### Claims enabled

Machine-independent feasibility of the fixed components; no performance claim.

## 9. P3 — Policy-construct lock

### Objective

Define two coherent policy spaces before annotators see claim-bearing rows.

### Required work

1. Create the general-safety registry and annotation manual.
2. Create the US-mortgage registry, authoritative source snapshots, and annotation manual.
3. Freeze inclusion, exclusion, ambiguity, severity, and default-action rules.
4. Define `needs_intervention` independently for both policies and the permitted downstream action for each control. Define how a row can be:
   - `G0/D0`;
   - `G1/D0`;
   - `G0/D1`;
   - `G1/D1`.
5. Build a boundary matrix with natural positive and negative examples.
6. Preserve genuine overlap. Document one-to-one, one-to-many, and conflicting control mappings; distinguish annotation criteria without forcing the underlying cases to be orthogonal.
7. Restore all 995 recoverable provenance joins; 759 of those carry nonempty regulatory references. Create explicit reconciliation records for the five unmatched source rows.
8. Mark generated benign controls as `not_applicable`, not missing provenance.
9. Define policy-version change handling: any material change after annotation creates a new dataset version and impact audit.
10. Document data governance, licenses, synthetic-data provenance, PII exclusion, protected-class handling, redaction, and release rules.
11. Build a policy-coverage matrix by control, workflow stage, product, persona, severity, jurisdiction, and required action. Mark unsupported surfaces `NOT_EVALUATED`.
12. Restrict the request-screening study to controls whose required intervention can be judged from the available request context. Response-, decision-, disclosure-, tool-, or workflow-dependent controls are excluded from the primary construct and listed explicitly.

### Policy-control schema

```text
control_id
policy_id
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
binary_needs_intervention
permitted_actions[]
inclusion_examples[]
exclusion_examples[]
owner
review_status
reviewer_id
```

### Annotation-manual requirements

- Annotators label policy applicability, not whether a model should “feel cautious.”
- General and mortgage labels are assigned independently.
- Legacy mortgage `flag` is never copied directly into Paper B gold; experts assign `needs_intervention` and an action under the frozen registry.
- `needs_intervention=1` if and only if the observed request context activates a frozen control whose permitted action set excludes `PASS`; annotators do not predict an unspecified generator's capability. It does not mean hard block. The manual assigns `CONSTRAIN`, `REVIEW`, or `BLOCK` separately.
- Ambiguous rows remain ambiguous until adjudication; they are not forced to make class counts balance.
- Multi-turn items are labeled and scored using the complete conversation representation.
- Annotators cannot see model scores or predictions.

### Required artifacts

- two registry files and manuals;
- frozen source snapshots or immutable references;
- policy crosswalk and boundary matrix;
- expert signoff record;
- provenance-recovery report.
- policy-coverage matrix and response/workflow-dependent exclusion list.
- dual-signed `$STUDY_ROOT/construct/policy_construct_snapshot.json` covering both registries/manuals, source snapshots, action lattice, boundary/crosswalk, coverage matrix, and signoffs.

### Verification

```bash
uv run python experiments/validate_paper_b_artifacts.py \
  --stage policy-construct \
  --policy-root policies \
  --out-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --sign-as general_policy_owner \
  --co-sign-as mortgage_sme \
  --strict
```

### PASS gate

Independent trained reviewers can apply each policy without seeing the other policy's label, every category maps to a policy control and action, authentic overlaps remain represented, and no unresolved contradiction changes `needs_intervention` or required action.

### STOP conditions

- `G1/D0` or `G0/D1` cannot be described naturally.
- Registry changes are expected during final annotation.
- A model invents regulatory citations at runtime.
- Policy sources or jurisdiction remain unspecified.

### Claims enabled

The paper may describe its policy construct and threat model, not model performance.

## 10. P4 — Data provenance and diagnostic manifests

### Objective

Create auditable diagnostic data without pretending partial labels are joint gold.

### Required work

1. Implement the Paper B row schema with `extra=forbid` validation.
2. Reuse the frozen normalization and hash rules from `guard_research.provenance`.
3. Assign unique `sample_id`, normalized overlap hash, exact canonical event hash, `family_id`, source revision, label provenance, and cohort role.
4. Recover or reconstruct pair/family identifiers for the hard set.
5. Preserve all 15 multi-turn conversations as structured content. Define canonical JSON event serialization, a logical event hash, and deterministic conversation-to-prompt rendering; never hash `str(dict_or_list)`.
6. Build diagnostic bundles from existing general and mortgage rows.
7. Apply correct partial-label logic:

```text
if general_needs_intervention == 1 or domain_needs_intervention == 1:
    final_intervention_gold = 1
elif general_needs_intervention is null or domain_needs_intervention is null:
    final_intervention_gold = null
else:
    final_intervention_gold = 0
```

8. Audit exact and near-duplicate families across mortgage/general training, diagnostic, pilot, hard, controlled, naturalistic, development, and test roles. Construct `split_lineage_graph` and assign every connected component intact; multiway inference never permits a cross-role leakage edge. Separately record within-role `inference_cluster_dimensions`. `family_id` alone is insufficient.
9. Record every candidate near-duplicate disposition rather than treating a similarity threshold as semantic truth.
10. Produce a text-free audit that can be committed even if row text cannot.

### Required manifest roles

```text
diagnostic
pilot
component_dev_general
component_dev_domain
joint_dev
joint_test
controlled_dev
controlled_test
naturalistic_dev
naturalistic_test
hard_stress
```

The schema must enforce:

- diagnostic rows may have one null policy label;
- `component_dev_general` requires only general-policy gold and is the sole source of `tau_G_sep`; `component_dev_domain` analogously requires only domain-policy gold and is the sole source of `tau_D_sep`;
- optional calibration-diagnostic rows state their label and evaluation role explicitly;
- joint controlled/naturalistic development and test rows require both intervention labels plus downstream action labels;
- `final_intervention_gold` must equal the logical OR when both binary labels exist;
- for each policy, binary zero requires action `PASS`, while binary one requires one of that control's permitted non-PASS actions;
- `final_intervention_gold = 0` if and only if `final_action_gold = PASS`; a positive final label requires the frozen action lattice to resolve a permitted non-PASS action;
- conflicting component actions require an explicit adjudication/lattice trace rather than silent maximum or row-order behavior;
- `INVALID_INPUT` is different from an optimizer `NO_GO` result.

### Verification commands

```bash
uv run python experiments/build_joint_manifest.py \
  --mode diagnostic \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --source-role-map configs/paper_b_source_roles_v1.yaml \
  --out $STUDY_ROOT/manifests

uv run python experiments/validate_paper_b_artifacts.py \
  --stage manifests \
  --strict
```

### PASS gate

- all rows pass schema validation;
- IDs are unique, normalized overlap hashes recompute, and exact event hashes recompute independently;
- structured multi-turn event hashes and rendered prompts recompute;
- no known family crosses forbidden roles;
- every near-duplicate candidate has an explicit disposition;
- partial-label rows are rejected by joint-policy fitting;
- five unmatched provenance rows have reconciliation records.

### STOP conditions

- A partial-label row receives `final_intervention_gold = 0`.
- Arrays are joined by position.
- Multi-turn context is discarded.
- The hard set is promoted to untouched test.

### Claims enabled

Data/provenance engineering only.

## 10a. P4a — Mortgage action-target audit and candidate authorization

### Objective

Determine whether the legacy label transformation defines a valid request-level intervention target and whether the existing adapter may enter a held-out score-validation pilot. This checkpoint does not establish detector validity or freeze `D`.

### Repository-grounded audit input

The 995 exact joins between [`guard_benchmark.jsonl`](../notebooks/data/benchmarks/full/guard_benchmark.jsonl) and [`mortgage_redteam_benchmark.jsonl`](../notebooks/data/benchmarks/full/mortgage_redteam_benchmark.jsonl) currently comprise 552 `refuse_and_educate`, 65 `refuse_and_warn`, 62 `helpful_compliant`, 49 `acknowledge_uncertainty`, 48 `refuse_disclosure`, 44 `resist_injection`, 40 `decline_guarantee`, 38 `resist_jailbreak`, 35 `responsible_guidance`, 32 `refuse_and_refer`, and 30 `enforce_verification` rows. The derived binary dataset maps all but `helpful_compliant` to `flag`. This is evidence of a broad “non-default handling” target, not a hard-block target.

### Required work

1. Reconstruct the exact source-to-training transformation for all 1,563 rows; every unmatched/generated row needs a provenance and action record.
2. Have general-policy annotators assign `general_needs_intervention` and `general_action_gold`; have mortgage SMEs independently assign `domain_needs_intervention` and `domain_action_gold`. A joint adjudication panel resolves `final_intervention_gold`, `final_action_gold`, and cross-policy conflicts. P4a's authorization decision concerns the domain target; do not infer any action mechanically from `expected_behavior` strings.
3. Audit label mismatch by source behavior, persona, policy control, protected-class context, category, severity, and generated transformation.
4. Predefine critical mismatches: any row for which the legacy positive label would route a genuinely ordinary/helpful request to the wrong intervention, or any negative label that misses a required intervention.
5. Choose exactly one outcome:
   - `VALIDATED_TARGET`: the binary target is valid for the narrowed request-intervention construct and the existing adapter may enter P6 as a candidate, with action semantics explicitly separate;
   - `RETRAIN_REQUIRED`: build a new action-valid training manifest, audit train/evaluation lineage overlap, retrain with frozen seeds, and repeat P4a;
   - `NO_GO`: the request-level construct cannot represent the intended mortgage controls.
6. Never select the outcome using E3/E4 performance.
7. Bind the signed decision to one exact candidate tuple: catalog entry and bundle hash, base/tokenizer revisions, adapter hash, rendered mortgage prompt hash, training-target transform/code hash, authorized training-manifest hash, and intended request-action construct. An audit that names only a dataset or policy file cannot authorize a candidate.

### Artifacts and commands

```bash
uv run python experiments/audit_mortgage_action_construct.py \
  --derived notebooks/data/benchmarks/full/guard_benchmark.jsonl \
  --source notebooks/data/benchmarks/full/mortgage_redteam_benchmark.jsonl \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --candidate-id mortgage_adapter_candidate \
  --general-policy policies/general/v1/controls.jsonl \
  --general-manual policies/general/v1/annotation_manual.md \
  --policy policies/mortgage/us_v1/controls.jsonl \
  --domain-manual policies/mortgage/us_v1/annotation_manual.md \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --annotations external/paper_b/mortgage_action_audit_v1 \
  --out $STUDY_ROOT/construct \
  --sign-as mortgage_sme \
  --strict

uv run python experiments/validate_paper_b_artifacts.py \
  --stage mortgage-action-construct \
  --audit $STUDY_ROOT/construct/mortgage_action_audit.json \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --candidate-id mortgage_adapter_candidate \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --strict

# This command succeeds only after the signed outcome is VALIDATED_TARGET. It
# materializes the exact domain-training lineage authorized for later audits.
uv run python experiments/build_joint_manifest.py \
  --mode p4a-authorized-domain-training \
  --action-audit $STUDY_ROOT/construct/mortgage_action_audit.json \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --candidate-id mortgage_adapter_candidate \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --out $STUDY_ROOT/manifests/domain_training_authorized
```

If the signed outcome is `RETRAIN_REQUIRED`, P4a remains open and the following branch is mandatory before repeating the audit. This revokes the previous P2 PASS until the new bundle is published and independently reacquired:

```bash
uv run python experiments/build_joint_manifest.py \
  --mode mortgage-action-training \
  --annotations external/paper_b/mortgage_action_audit_v1 \
  --policy policies/mortgage/us_v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --out $STUDY_ROOT/manifests/mortgage_action_train

uv run python experiments/validate_paper_b_artifacts.py \
  --stage train-versus-all-evaluation-lineage \
  --train-manifest $STUDY_ROOT/manifests/mortgage_action_train/manifest.json \
  --strict

uv run python experiments/train_mortgage_constrained.py \
  --mode p4a-retrain \
  --config configs/paper_b_v1.yaml \
  --train-manifest $STUDY_ROOT/manifests/mortgage_action_train/manifest.json \
  --seeds-from-config \
  --out external/paper_b/generated_adapters/mortgage_action_v1

uv run python experiments/fetch_paper_b_assets.py \
  --config configs/paper_b_v1.yaml \
  --register-generated-bundle external/paper_b/generated_adapters/mortgage_action_v1 \
  --publish-immutable \
  --out-catalog $STUDY_ROOT/catalog.json

uv run python experiments/fetch_paper_b_assets.py \
  --catalog $STUDY_ROOT/catalog.json \
  --asset-root external/paper_b/reacquisition_after_retrain \
  --acquire-and-verify
```

Then rerun P2's two-destination gate and P4a on the new immutable bundle. No P5/P6 score from the retired candidate may authorize the replacement.

### PASS gate

Every currently available training/evaluation row has traceable action semantics, no unresolved critical mismatch remains, and the expert-signed audit authorizes an exact target plus candidate adapter/training-target/prompt tuple and `$STUDY_ROOT/manifests/domain_training_authorized/manifest.json` for P6 validation. If retraining was required, full bundle hashes, seeds, training manifest, and the preliminary train-versus-available-evaluation lineage audit also pass. This preliminary audit does not replace the mandatory train-versus-**all** evaluation audit after P8 creates every development/test manifest.

### STOP conditions

- A positive mortgage label is still described as hard-block gold merely because its upstream answer required caution, education, disclosure, or verification.
- Protected applicants seeking information are treated as malicious prompts.
- `D` is chosen or retrained after viewing final E3/E4 outcomes.
- The claimed mortgage policy depends on responses, credit decisions, or workflow state absent from the input.

### Claims enabled

PASS authorizes only a mortgage-policy **screen candidate** for held-out P6 validation. Only P6 PASS may authorize `D`; neither result establishes legal compliance.

## 11. P5 — Correct scorer and runtime pilot

### Objective

Prove that the planned two-guard runtime produces correct, identity-keyed, replayable scores.

### Required work

1. Load SmolLM3 once and load named general and mortgage adapters.
2. Build `$STUDY_ROOT/construct/component_registry.json` as a versioned, content-hashed candidate registry containing at least:
   - `base_general`: adapters disabled, general prompt (`B`, E0);
   - `general_adapter`: general adapter, general prompt (`G`, E1);
   - `base_mortgage`: adapters disabled, mortgage prompt (E2a);
   - `mortgage_adapter_candidate`: P4a-authorized target/candidate, mortgage prompt; renamed `D`/E2b only after P6 PASS;
   - any optional combined-prompt/consolidated component retained before the experiment lock.
3. Score every registered component through the same identity/parity pipeline.
4. Store raw negative/positive decision logits and `score_margin = positive_logit - negative_logit`; probabilities alone are insufficient. Serialize the exact label strings/token IDs and semantic mapping to `PASS`/`INTERVENE` for each component.
5. Lock rendered prompt bytes, prompt hashes, decision token strings/IDs, model/tokenizer revision, full adapter hashes, dtype, device, and runtime fingerprint.
6. Write long-form scores keyed by `(sample_id, component_id)`.
7. Treat NaN, timeout, adapter failure, and invalid tokens as null score plus `error_code`. Primary binary arms route failures to `INTERVENE`.
8. Implement deterministic stored-score replay.
9. Run the optimizer only on synthetic fully dual-labeled fixtures at this checkpoint.
10. Use the 334-row hard set only to validate score mechanics and migration joins.

### Score-row schema

```text
sample_id
normalized_content_sha256       # overlap detection only
canonical_event_sha256          # exact row/cache/join identity
manifest_hash
component_id                 # must exist in the invoking versioned component registry
component_registry_hash
model_id
model_revision
tokenizer_revision
adapter_bundle_hash|null
prompt_id
prompt_version
rendered_prompt_sha256
prompt_template_sha256
negative_label_text
positive_label_text
negative_token_id
positive_token_id
negative_logit|null
positive_logit|null
score_margin|null
error_code|null
truncated
original_token_count
scored_token_count
max_length
padding_side
truncation_side
latency_ms
dtype
device
runtime_fingerprint
scoring_code_sha256
```

Enforce the row invariant: either all required logits/margins are finite and `error_code` is null, or all are null and `error_code` is non-null.

The score metadata sidecar records expected components, row/component uniqueness checks, scoring command, software/toolchain lock hash, input/output byte and logical hashes, and failure counts by error code.

### Mandatory scorer tests

- Named-adapter logits match three separately loaded reference models within declared device tolerance.
- `disable_adapter()` equals the separately loaded base.
- Adapter switching order does not change outputs or weights.
- `B/G` can only use the general prompt and `D` can only use the mortgage prompt.
- Prompt rendering and token IDs are hash-stable.
- A row scored alone equals the same row in a mixed-length batch within tolerance.
- Padding and last-token indexing are correct.
- Truncation is deterministic and recorded.
- Every `(sample_id, component_id)` appears exactly once for the components required by that cohort/arm.
- Parity, batching, identity, cache-invalidation, and error tests cover every registered component, including `base_mortgage`.
- Missing or duplicate components return `INTERVENE` under the primary fail-closed rule.
- Any identity change invalidates the cache.
- Case/whitespace changes may preserve `normalized_content_sha256` but must change `canonical_event_sha256` and invalidate cached scores.
- Stored-score replay is bit-exact and inference replay is tolerance-based.
- Early-intervention cascade actions equal full OR actions and never early-pass.

### Commands

P5 is an engineering-only diagnostic run. The signed pilot lock is deliberately created at P6 only after the pilot manifest exists; otherwise the later pilot data would not be covered by the lock.

```bash
uv run python experiments/build_component_registry.py \
  --catalog $STUDY_ROOT/catalog.json \
  --construct-audit $STUDY_ROOT/construct/mortgage_action_audit.json \
  --config configs/paper_b_v1.yaml \
  --out $STUDY_ROOT/construct/component_registry.json

uv run python experiments/score_joint_stack.py \
  --engineering-only \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --component-registry $STUDY_ROOT/construct/component_registry.json \
  --cohort diagnostic \
  --out-root $STUDY_ROOT/scores

uv run python experiments/validate_paper_b_artifacts.py \
  --stage scores \
  --strict

uv run python experiments/replay_joint_stack.py \
  --scores $STUDY_ROOT/scores/diagnostic/scores.parquet \
  --policy tests/fixtures/paper_b/synthetic_policy.json \
  --mode stored-score \
  --out $STUDY_ROOT/replay/diagnostic_score_replay.json \
  --strict

uv run python experiments/replay_joint_stack.py \
  --mode inference-fixture \
  --fixture-manifest tests/fixtures/paper_b/inference_cases.jsonl \
  --config configs/paper_b_v1.yaml \
  --out $STUDY_ROOT/replay/diagnostic_inference_replay.json
```

### PASS gate

All scorer identity/parity tests pass, two executions reproduce the same row joins and synthetic-policy actions, the candidate identity matches P4a, and no legacy length-only cache is used as evidence.

### STOP conditions

- Any component uses the wrong prompt.
- Named-adapter and separately loaded reference logits disagree beyond the declared tolerance.
- A scorer error is converted into a fabricated probability.
- Joint fitting accepts partial-label diagnostics.

### Claims enabled

“The proposed runtime can be implemented and replayed.” No comparative model claim.

## 12. P6 — Blinded dual-label annotation pilot

### Objective

Verify that humans can apply both policies consistently and that the four-quadrant construct is natural.

### Required work

1. Author 60 semantic families with four controlled variants per family (240 prompts) **and** acquire at least 60 independent `split_lineage_graph` components of authentic naturalistic pilot requests under a documented source, time window, and sampling frame. The naturalistic pilot is the fixed first role-prefix of the custodian stream committed before any pilot row is selected or annotated. Naturalistic requests must pre-exist and be independent of this study—such as deidentified interaction logs, authentic public inquiries, or externally collected interactions—and need not form quartets. Study-authored prompts belong to a separate simulation stratum.
2. Cover:
   - 20 fair-lending families;
   - 20 compliance/bad-advice families;
   - 20 security-misuse families;
   - 30 consumer and 30 loan-officer families.
3. Create family/quartet identities before labeling.
4. Keep annotators blind to model identity, scores, future split, the other policy's label/rationale, and intended paper result.
5. Obtain two independent general-policy judgments and two independent mortgage-policy judgments per row using separate policy-specific pools or isolated randomized sessions; use separate adjudicators where feasible.
6. Adjudicate every disagreement with policy-specific experts.
7. Record raw labels, adjudicated labels, confidence, ambiguity, rationale, control IDs, time per item, and annotator pseudonyms.
8. Re-adjudicate the 334 hard rows separately; this remains development/stress data.
9. Review naturalness and adversarial quality so quadrant variants are not trivial lexical edits.
10. Report agreement, ambiguity, policy/control coverage, and model-blind source diagnostics separately for the controlled and naturalistic pilots.
11. Before any naturalistic pilot request is selected or annotated, freeze the authentic-data source, eligibility/attrition rules, time window, leakage-capable `split_lineage_graph`, sampler code/config hash, pilot component count, deterministic role-allocation rule, and seed commitment. Exclude all catalog-declared training/hard/diagnostic components, including the optional E9 replay candidate, before ordering. A custodian without model access secretly orders indivisible connected components, assigns the frozen pilot prefix, and seals the remaining component stream for P7/P8. P7 may choose only per-role **component-prefix counts** from its power result; it cannot split a component, reorder, skip, inspect, or replace units. Bind the pilot prefix/object hash and remaining-stream URI/version/hash into the commitment and pilot lock. Any source, eligibility, graph, sampler, or pilot-prefix change after annotation begins requires a new `STUDY_ID` and untouched population. Separately, before pilot scores are exposed, either seal the complete controlled candidate pool or assign controlled-test authoring to an independent team with no pilot/dev access. Merely asking a different team to author scenarios does not make them naturalistic.
12. Before scoring, create and dual-sign a numeric component-validity specification for `D`: expected positive score direction, the exact one-sided held-out discrimination estimand/lower-bound threshold, cross-validation partitioning and repetition, broad pilot false-/missed-intervention feasibility caps, minimum independent units, dependency estimator, and missing/error behavior. AUROC/AUPRC and caps must use domain labels only and account for lineages. A constant, inversely ordered, or infeasible score cannot become `D` merely because P4a validated its training target.

### Prespecified pilot quality gates

Freeze the exact thresholds before reading pilot results. A defensible starting proposal is:

- raw agreement at least 0.85 overall for each policy;
- Gwet AC1 is the single primary chance-corrected statistic and must be at least 0.70 overall for each policy; Cohen kappa may be reported only as sensitivity;
- Gwet AC1 must be at least 0.65 for each major category with at least 20 independent family units; smaller categories are `NOT_TESTED` and must be expanded before a category claim;
- unresolved/ambiguous rate at most 10%;
- 100% adjudication completeness;
- 100% control-ID completeness for policy-positive rows;
- all four quadrants remain populated with natural examples.
- the naturalistic pilot has a defensible sampling frame and enough independent units to estimate attrition/prevalence inputs; it is not padded with authored quartet variants.

The project may set stricter thresholds. It may not relax them after seeing whether the model result is favorable.

Freeze before annotation the AC1 implementation, binary weighting, family-bootstrap confidence interval, missing-label handling, and treatment of adjudicated versus raw labels. Raw agreement and AC1 point estimates are gates; confidence intervals are reported to show precision rather than silently substituted after results.

### Required artifacts

- pilot manifest and annotation export;
- schema-valid `pilot_summary.json` containing the annotation data card, disagreement taxonomy, annotation-time/attrition evidence, recruitment/allocation counts, exclusion reasons, and logical hashes of the raw-judgment export;
- agreement report by policy/category/quadrant;
- disagreement taxonomy;
- adjudication log;
- annotation-time and attrition estimates;
- custodian-signed `$STUDY_ROOT/manifests/naturalistic_sampling_commitment.json`, created before pilot selection/annotation and containing the eligible-population/lineage-graph hashes, source/time window, eligibility/attrition rules, sampler code/config hash, unrevealed seed commitment, fixed pilot component-prefix assignment/object hash, immutable remaining-stream URI/version/hash, later role-prefix rule, and no row text or realized labels;
- statistician/SME-signed `$STUDY_ROOT/design/component_validity_spec.json` with every numeric gate and dependency method fixed before score access;
- correct-prompt B/G/D pilot scores keyed to pilot row/family IDs;
- signed component-validity report with score direction, discrimination uncertainty, cross-validated operating region, split/inference dependency definitions, and PASS/FAIL;
- separate `hard_stress` manifest, dual-label agreement/adjudication report, and proof it remains non-confirmatory;
- policy-reviewed general-replay **candidate** training manifest, frozen even if E9 is later `NOT_RETAINED`, so its lineages can be excluded from the pre-score naturalistic stream;
- revised manual version if the first pilot fails.

### Verification

```bash
# Materialize the P0 numeric gates as a signed, schema-validated object before
# the pilot lock and before score access. This command may not read scores.
uv run python experiments/validate_paper_b_artifacts.py \
  --stage component-validity-spec \
  --config configs/paper_b_v1.yaml \
  --schema schemas/paper_b/component_validity_spec.schema.json \
  --out $STUDY_ROOT/design/component_validity_spec.json \
  --sign-as statistician \
  --co-sign-as mortgage_sme \
  --assert-no-score-input \
  --strict

# Freeze the optional E9 general-replay candidate before naturalistic sampling.
# P7 may activate this exact row set, but no later addition/substitution is
# allowed. This is a training role and is never scored as evaluation evidence.
uv run python experiments/build_joint_manifest.py \
  --mode e9-general-replay-candidate \
  --source external/paper_b/general_replay_source_v1 \
  --general-policy policies/general/v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --catalog $STUDY_ROOT/catalog.json \
  --out $STUDY_ROOT/manifests/general_replay_candidate

# Custodian-only, before selecting or annotating any naturalistic pilot row.
# Order indivisible leakage-graph components, export only the fixed blinded
# pilot prefix, and keep the remaining stream sealed. The KMS seed is committed
# but never printed or revealed to selectors.
uv run python experiments/seal_naturalistic_sampling_frame.py \
  --eligible-population-manifest external/paper_b/naturalistic_population_v1.json \
  --source-role-map configs/paper_b_source_roles_v1.yaml \
  --sampler-spec configs/paper_b_naturalistic_sampler_v1.yaml \
  --exclude-source-roles training,diagnostic,hard_stress \
  --exclude-manifest $STUDY_ROOT/manifests/domain_training_authorized/manifest.json \
  --exclude-manifest $STUDY_ROOT/manifests/general_replay_candidate/manifest.json \
  --exclude-training-from-catalog $STUDY_ROOT/catalog.json \
  --pilot-component-count 60 \
  --pilot-object "controlled://${STUDY_ID}/naturalistic_pilot" \
  --remaining-component-stream "controlled://${STUDY_ID}/naturalistic_remaining_stream" \
  --commitment-scheme sha256-seed-nonce-v1 \
  --seed-ref-env PAPER_B_NATURALISTIC_SEED_REF \
  --controlled-store-config "$PAPER_B_CONTROLLED_STORE_CONFIG" \
  --out $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --sign-as custodian \
  --strict

uv run python experiments/import_paper_b_annotations.py \
  --input external/paper_b/pilot_annotations_v1 \
  --cohort pilot \
  --general-policy policies/general/v1/controls.jsonl \
  --domain-policy policies/mortgage/us_v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --expected-naturalistic-prefix-role pilot \
  --controlled-cohort-object "controlled://${STUDY_ID}/naturalistic_pilot" \
  --controlled-store-config "$PAPER_B_CONTROLLED_STORE_CONFIG" \
  --summary-out $STUDY_ROOT/annotations/pilot/pilot_summary.json \
  --raw-judgment-manifest-out $STUDY_ROOT/annotations/pilot/raw_judgments.manifest.json \
  --out $STUDY_ROOT/annotations/pilot

uv run python experiments/build_joint_manifest.py \
  --mode pilot \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --annotations $STUDY_ROOT/annotations/pilot \
  --sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --out $STUDY_ROOT/manifests/pilot

uv run python experiments/import_paper_b_annotations.py \
  --input external/paper_b/hard_stress_annotations_v1 \
  --cohort hard_stress \
  --general-policy policies/general/v1/controls.jsonl \
  --domain-policy policies/mortgage/us_v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --out $STUDY_ROOT/annotations/hard_stress

uv run python experiments/build_joint_manifest.py \
  --mode hard-stress \
  --annotations $STUDY_ROOT/annotations/hard_stress \
  --out $STUDY_ROOT/manifests/hard_stress

uv run python experiments/validate_paper_b_artifacts.py \
  --stage hard-stress \
  --strict

uv run python experiments/validate_paper_b_artifacts.py \
  --stage annotation-pilot \
  --summary $STUDY_ROOT/annotations/pilot/pilot_summary.json \
  --raw-judgment-manifest $STUDY_ROOT/annotations/pilot/raw_judgments.manifest.json \
  --strict

uv run python experiments/lock_paper_b.py \
  --phase pilot \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --preregistration $STUDY_ROOT/design/preregistration.md \
  --claim-registry $STUDY_ROOT/design/claim_registry.json \
  --primary-contract $STUDY_ROOT/design/primary_contract.json \
  --mortgage-action-audit $STUDY_ROOT/construct/mortgage_action_audit.json \
  --policy-construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --policy-coverage $STUDY_ROOT/construct/policy_coverage.json \
  --component-registry $STUDY_ROOT/construct/component_registry.json \
  --pilot-manifest $STUDY_ROOT/manifests/pilot/manifest.json \
  --hard-stress-manifest $STUDY_ROOT/manifests/hard_stress/manifest.json \
  --hard-stress-audit $STUDY_ROOT/annotations/hard_stress/agreement_report.json \
  --general-replay-candidate $STUDY_ROOT/manifests/general_replay_candidate/manifest.json \
  --diagnostic-score-replay $STUDY_ROOT/replay/diagnostic_score_replay.json \
  --diagnostic-inference-replay $STUDY_ROOT/replay/diagnostic_inference_replay.json \
  --annotation-audit $STUDY_ROOT/annotations/pilot/agreement_report.json \
  --pilot-summary $STUDY_ROOT/annotations/pilot/pilot_summary.json \
  --raw-judgment-manifest $STUDY_ROOT/annotations/pilot/raw_judgments.manifest.json \
  --adjudication-log $STUDY_ROOT/annotations/pilot/adjudication_log.jsonl \
  --authoring-lineage-seal $STUDY_ROOT/manifests/pilot/authoring_lineage_seal.json \
  --naturalistic-sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --component-validity-spec $STUDY_ROOT/design/component_validity_spec.json \
  --approval-dir $STUDY_ROOT/design/approvals \
  --sign-as research_owner \
  --require-clean-source

uv run python experiments/score_joint_stack.py \
  --cohort pilot \
  --lock $STUDY_ROOT/locks/pilot.json \
  --components-from-lock \
  --out-root $STUDY_ROOT/scores

uv run python experiments/validate_paper_b_artifacts.py \
  --stage component-validity \
  --pilot-lock $STUDY_ROOT/locks/pilot.json \
  --spec $STUDY_ROOT/design/component_validity_spec.json \
  --scores $STUDY_ROOT/scores/pilot/scores.parquet \
  --manifest $STUDY_ROOT/manifests/pilot/manifest.json \
  --out $STUDY_ROOT/construct/component_validity.json \
  --sign-as mortgage_sme \
  --strict
```

### PASS gate

Both policies meet the frozen agreement/ambiguity thresholds, all quadrants are coherent, the naturalistic pilot is exactly the pre-annotation committed component prefix, disagreements can be resolved without changing the task definition, and the held-out `D` score passes every frozen direction/discrimination/operating-region gate. Only now is the candidate authorized as `D` for P7–P10.

### STOP/REVISE conditions

- A policy quality threshold fails.
- One quadrant is sparse or contrived.
- Labels depend strongly on annotator identity.
- Adjudication requires changing the registry rather than clarifying an existing rule.
- `D` is constant, inversely directed, below the frozen discrimination bound, or has no stable pilot operating point.

Failure handling is cause-specific:

- construct/agreement failure: revise the registry/manual and repeat under a new `STUDY_ID`;
- detector failure with a valid construct: retrain only on lineage-disjoint training data, register a new candidate bundle, and validate it on a **fresh untouched pilot cohort** under a new `STUDY_ID`. The revealed P6 pilot cannot be reused as held-out evidence or iterated against.

Every failed study remains retained and is never promoted to final evidence.

### Claims enabled

Annotation, construct, and held-out component feasibility only; no comparative system claim.

## 13. P7 — Power, precision, and analysis lock

### Objective

Determine the data volume and inference method before confirmatory collection.

### Primary estimands

1. Paired row-marginal severe missed-intervention difference `Delta_miss` from Section 2.1 under the frozen graph-component or multiway dependency design, for noninferiority.
2. Paired row-marginal false-intervention difference `Delta_intervene` from Section 2.1, with the same frozen weighting/clustering rule, for superiority.
3. Absolute E4 false-intervention rate and missed-intervention rate in each intervention-positive quadrant.

### Required work

1. Use the naturalistic pilot for prevalence/attrition and both pilots for score discordance/design effects; the deliberately balanced quartet pilot cannot estimate natural prevalence. Use only correct-prompt B/G/D scores.
2. Estimate E3/E4 discordance with nested or repeated cross-fitting on the pilot, or use conservative scenario ranges. Never fit thresholds and treat their effects on the same 60 families as unbiased test behavior.
3. Simulate the complete adaptive threshold selection and independent fixed-policy final-test pipeline at candidate sample sizes, including both controlled and naturalistic strata.
4. Determine distinct effective sample requirements for:
   - overall severe missed-intervention noninferiority;
   - `G1/D0`, `G0/D1`, and `G1/D1` constraints;
   - false-intervention superiority and the absolute `alpha_system` gate on `G0/D0`;
   - category or worst-group claims;
   - paired protected-context false-intervention non-worsening;
   - policy-specific `E_sep` threshold precision, its resource-match/shift tolerances, and any confirmatory E4/`E_sep` contrast;
   - absolute low false-intervention-rate precision.
5. Freeze `split_lineage_graph` edge rules and connected-component assignment first. Separately freeze `inference_cluster_dimensions`, the graph-component or multiway variance estimator, and consistent power/inference/stability implementations. Simulate the complete adaptive pipeline under both structures.
6. Freeze one-sided alpha, power, the exact fixed-sequence/Holm algorithm from P0, protected-context counterfactual construction/aggregation, and handling of ambiguous/excluded rows.
7. Only for equal-weight (`w_i=1`) binary outcome pairs with one observation per genuinely independent graph component may the study use the one-sided [Tango paired score interval](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819980430%2917%3A8%3C891%3A%3AAID-SIM780%3E3.0.CO%3B2-B), implemented from the frozen formula and verified against published examples plus an independent implementation. Validate nominal coverage across the P7 scenario grid. Unequal sampling weights, repeated rows, crossed/shared lineages, or multirow components invalidate Tango and require the simulation-validated multiway/survey paired-risk-difference method. A percentile bootstrap is never the primary rare-event method, and zero/few discordance requires a prespecified conservative score/exact fallback.
8. Use one-sided exact or score binomial bounds only when every relevant observation is a one-row independent graph component with no crossed dependence. Otherwise use the simulation-validated dependency-aware method regardless of whether each named family contributes one row.
9. Separate primary sampling weights and dependency-aware inference from secondary traffic-scenario cost weights; serialize both under different field names and reject any substitution.
10. Freeze numeric stability diagnostics before joint-development results: at least 80% of dependency-unit bootstrap/multiway refits must produce actions agreeing with the locked policy on at least 95% of development rows; deleting any one graph component or prespecified multiway cluster must not change a primary rate by more than 0.02 or flip constraint feasibility.
11. Choose `analysis_mode`:
   - `powered_confirmatory`; or
   - `precision_focused_measurement` if the required noninferiority sample is infeasible.
12. For E4 versus `E_sep`, either allocate and power a named secondary hypothesis in the multiplicity plan or freeze it as descriptive. Confirmatory status additionally requires the exact source-frame/time-window match, equal component counts per policy cohort versus `joint_dev`, matched label/reviewer/adjudication and score-call budgets, and frozen acceptable shift/budget deviations from Section 2.1. Failure of that comparator-strength gate forces descriptive, pipeline-specific wording regardless of statistical significance. This status cannot change after seeing either development arm.
13. Resolve powered naturalistic role sizes—including any retained calibration-fit/evaluation roles—only as counts of indivisible connected components assigned by the P6 custodian-committed remaining-stream rule. Record each component-prefix count in `design.json`; never cut a component to hit a row target, and report both resulting row count and effective independent-component count. The power step cannot access row text, identities, scores, labels, or per-prefix summaries.

`precision_focused_measurement` never overwrites the pilot-locked registry. The power program always emits a versioned `$STUDY_ROOT/design/claim_registry.active.json` and signed `$STUDY_ROOT/design/claim_registry_amendment.json`. In powered mode the active registry is a byte-identical logical copy and the amendment records `NO_CHANGE`; in precision mode the active registry is a strictly narrowed successor that removes confirmatory gates/wording, and the amendment records every claim-ID transition plus parent hash. It reports effect estimates and intervals only; it cannot yield noninferiority, superiority, “preserves safety,” or “improves” wording.

### Mathematical sanity checks

For zero observed failures among `n` independent units, the exact one-sided 95% upper bound is:

\[
U = 1 - 0.05^{1/n}.
\]

Approximately 99 IID units with zero failures are required for `U <= 0.03`. This is only an optimistic lower-bound sanity check; it does not power a one-percentage-point paired noninferiority comparison and is invalid if author/template/control lineages remain correlated.

For an approximate 1% false-intervention rate with ±0.5% 95% margin under IID assumptions:

\[
n_0 \approx \frac{1.96^2(0.01)(0.99)}{0.005^2} \approx 1521.27,
\]

so an integer design rounds up to 1,522 IID observations. The final target must use Wilson/exact or simulation-based precision, increase for every measured clustering level, and may count only expert-validated `G0/D0` units. The value 1,522 is an IID Wald lower-bound illustration, not the sample-size target. Existing mortgage-only benign labels do not automatically count.

### Required artifacts

- executable `experiments/power_paper_b.py`;
- `power_report.json` with all assumptions;
- human-readable `power_report.md`;
- selected cohort sizes and budget estimate;
- frozen statistical-analysis specification.
- immutable pilot registry plus versioned active registry and signed parent-linked amendment;
- signed `$STUDY_ROOT/locks/design.json`, created before confirmatory cohort collection.

### Command

```bash
uv run python experiments/power_paper_b.py \
  --config configs/paper_b_v1.yaml \
  --pilot-labels $STUDY_ROOT/manifests/pilot/manifest.json \
  --pilot-scores $STUDY_ROOT/scores/pilot/scores.parquet \
  --out $STUDY_ROOT/design/power_report.json \
  --out-md $STUDY_ROOT/design/power_report.md \
  --sap-out $STUDY_ROOT/design/statistical_analysis_plan.md \
  --claim-registry-in $STUDY_ROOT/design/claim_registry.json \
  --claim-registry-out $STUDY_ROOT/design/claim_registry.active.json \
  --claim-amendment-out $STUDY_ROOT/design/claim_registry_amendment.json \
  --sign-amendment-as statistician

uv run python experiments/lock_paper_b.py \
  --phase design \
  --parent-lock $STUDY_ROOT/locks/pilot.json \
  --preregistration $STUDY_ROOT/design/preregistration.md \
  --claim-registry-parent $STUDY_ROOT/design/claim_registry.json \
  --claim-registry-active $STUDY_ROOT/design/claim_registry.active.json \
  --claim-registry-amendment $STUDY_ROOT/design/claim_registry_amendment.json \
  --pilot-manifest $STUDY_ROOT/manifests/pilot/manifest.json \
  --pilot-scores $STUDY_ROOT/scores/pilot/scores.parquet \
  --component-validity $STUDY_ROOT/construct/component_validity.json \
  --power $STUDY_ROOT/design/power_report.json \
  --sap $STUDY_ROOT/design/statistical_analysis_plan.md \
  --sign-as statistician \
  --require-clean-source
```

### PASS gate

Required controlled and naturalistic development/test sizes are affordable, every primary claim has adequate effective units after exclusions and higher-order clustering, simulated coverage/power pass, and the statistician-signed design lock exists before P8 collection.

### STOP/NARROW conditions

- Powered sample size is infeasible.
- The plan relies only on row-level binomial bounds despite split-lineage or within-role crossed dependence.
- A one-point margin is retained without the required paired information.
- Category/subgroup claims lack effective units.

If infeasible, narrow the claim or choose precision-focused measurement before final collection. Do not shrink the sample for convenience or widen the margin after test access.

### Claims enabled

The paper may state its preregistered design and expected precision.

## 14. P8 — Powered cohorts, role separation, and final-test seal

### Objective

Produce the controlled/naturalistic data needed for system selection and one-shot confirmation without leakage.

### Required cohort roles

1. `component_dev_general` and `component_dev_domain`: policy-specific, lineage-disjoint development cohorts used only to select the two `E_sep` thresholds; neither may use the other policy's labels or joint outcomes. Both are randomized component streams from the same target frame/time window as `joint_dev`; each matches its frozen independent-component allocation and the combined label/reviewer/adjudication and score-call budget.
2. `controlled_dev` and `controlled_test`: fully dual-labeled counterfactual quartets for mechanism analysis.
3. `naturalistic_dev` and `naturalistic_test`: pre-existing, study-independent interactions sampled under a frozen source/time-window/frame; the latter carries the primary comparison. Study-authored scenarios, even from an independent team, are `human_authored_simulation`, not naturalistic.
4. `joint_dev`: a locked alias/materialized view of `naturalistic_dev` only for the primary E3/E4 selector. Controlled development rows may diagnose the mechanism but cannot choose the naturalistic primary policy.
5. `joint_test`: a custodian-only index over the two separately analyzed test strata; never a pooled, author-chosen mixture.
6. `hard_stress`: re-adjudicated 334-row set, never final confirmation.
7. Optional calibration-diagnostic cohorts may exist, but primary threshold policies cannot depend on them.

Controlled and naturalistic results are never pooled to select a policy or manufacture a favorable aggregate. Any optional probability-calibration fit and evaluation roles are distinct and cannot alter the primary raw-margin action grid.

### Required work

1. For controlled/simulation data, use the pool sealed before pilot scoring or an independent test-authoring team with no pilot/dev access. For naturalistic data, apply the frozen authentic-source sampler with explicit inclusion/exclusion rules, selection probabilities, time window, attrition reasons/weights, and dependency-unit random dev/test assignment before scoring. Record source, author where applicable, generator, template, policy-control, source-conversation, collection window, and creation-time lineages.
2. Assign every `split_lineage_graph` connected component wholly to one role before scoring, conceal roles behind randomized IDs, and audit that annotators cannot infer cohort role. Multiway clustering is within-role inference only.
3. Fully dual-label `joint_dev` and `joint_test` under the locked manuals while keeping annotators blind to model identity, predictions, scores, aggregates, and intended paper outcome.
4. Build the two `component_dev_*` roles without joint-label access, and prove their source/template/control lineages do not cross training or either test stratum.
5. Complete independent judgments, adjudication, ambiguity, severity, control IDs, provenance, and family IDs.
6. Maintain the P6 agreement quality on the powered cohorts.
7. Audit exact, normalized, semantic near-duplicate, author, generator, template, and policy-control relationships against:
   - mortgage/general training;
   - component-specific E_sep development;
   - optional calibration-diagnostic cohorts;
   - pilot;
   - hard set;
   - joint development;
   - joint test.
8. Confirm powered counts after exclusions and every clustering level, separately for naturalistic and controlled strata.
9. Produce an annotator-allocation/blinding audit.
10. Seal final-test text/events and labels before policy fitting. Test scores must not exist before P10; if a custodian precomputes them, that event is the single sealed evaluation and must produce the full P10 attestation.
11. Include a naturalistic-source audit proving study independence and sampling-frame adherence, plus a superficial-artifact probe that tests whether source, author, or intended quadrant is predictable from lexical/template cues. A strong artifact signal is a REVISE/NO-GO condition, not a benchmark feature.
12. Include protected-context counterfactual pairs and a control-coverage matrix. Critical controls without sufficient independent units are `NOT_EVALUATED`, not silently generalized.
13. If and only if E9 is retained by `design.json`, promote the exact P6-locked `general_replay_candidate` row set/lineages to `general_replay` and build a `consolidated_train` manifest that combines it with P4a's exact `domain_training_authorized` manifest. No rows may be added, removed, or substituted during promotion. These training roles are lineage-audited and locked but are never development/test cohorts. The E9 path may not assume that the conditional P4a retraining manifest exists.
14. Produce an `E_sep` comparator-strength audit covering common frame/time window, random role allocation, independent-component and row counts, policy-label/reviewer/adjudication effort, score calls, and prespecified covariate-shift diagnostics. It must decide only the locked PASS/FAIL gate, not tune `E_sep`.
15. If probability calibration is retained, import and build `calibration_fit` and `calibration_eval` through a separate closed allowlist before `data.json`; prove they are lineage-disjoint from each other and every primary role. Otherwise emit a signed `NOT_RETAINED` receipt, and P9 must not accept calibration inputs.

### Test firewall

Preferred implementations:

- encrypted test text/events and labels held by an independent custodian;
- separate repository/object-store ACL unavailable to fit jobs;
- a service that returns signed final evaluation only after policy lock.

The selector import uses a closed development-role allowlist and rejects every `*_test` role/object, not only the alias `joint_test`. The fit process must not have a test path, credential, score artifact, token-length summary, realized label/quadrant count, or aggregate-result endpoint. Seeing final inputs or score distributions can influence prompts, truncation, components, and thresholds even without labels.

The selector workspace's `manifests/joint_test/` contains only a text-free seal: study/cohort identity, design-lock hash, both policy/manual hashes, action-lattice hash, total row/family counts, target design counts, object-version hashes, controlled-object URI, private audit PASS/FAIL, custodian key ID, and signature. Realized quadrant/category counts and post-exclusion label summaries remain private until P10. The custodian alone builds the test object and signed completeness/lineage audit. P10 resolves that immutable version and never writes readable test payloads or scores into the selector workspace.

### Required artifacts

- role-specific manifests and text-free indexes;
- powered annotation summary/data card, raw-judgment logical manifest, agreement report, and adjudication log;
- exact/near-duplicate audit;
- license/PII audit;
- powered-count confirmation;
- final-test seal/custodian record;
- the P6 naturalistic sampling commitment and proof that imported naturalistic roles contain exactly the indivisible component-prefix assignments resolved by `design.json` from the committed remaining stream, with no split components, skips, or substitutions;
- conditional E9 `general_replay` and `consolidated_train` manifests, or an explicit `NOT_RETAINED` design-lock record;
- optional calibration-fit/evaluation manifests with lineage audit, or an explicit `NOT_RETAINED` receipt;
- `$STUDY_ROOT/manifests/e_sep_comparator_strength.json` with frozen-tolerance PASS/FAIL and no test outcomes;
- `$STUDY_ROOT/locks/data.json`.

### Commands

```bash
uv run python experiments/import_paper_b_annotations.py \
  --input external/paper_b/development_annotations_v1 \
  --design-lock $STUDY_ROOT/locks/design.json \
  --general-policy policies/general/v1/controls.jsonl \
  --domain-policy policies/mortgage/us_v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --verify-naturalistic-component-prefix-roles component_dev_general,component_dev_domain,naturalistic_dev \
  --reject-split-components-skips-or-substitutions \
  --cohort-role-file external/paper_b/development_roles_v1.json \
  --allow-roles component_dev_general,component_dev_domain,controlled_dev,naturalistic_dev \
  --reject-role-pattern '*_test' \
  --summary-out $STUDY_ROOT/annotations/powered/powered_summary.json \
  --raw-judgment-manifest-out $STUDY_ROOT/annotations/powered/raw_judgments.manifest.json \
  --out $STUDY_ROOT/annotations/powered

uv run python experiments/build_joint_manifest.py \
  --mode confirmatory-development \
  --config configs/paper_b_v1.yaml \
  --catalog $STUDY_ROOT/catalog.json \
  --annotations $STUDY_ROOT/annotations/powered \
  --design-lock $STUDY_ROOT/locks/design.json \
  --sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --allow-roles component_dev_general,component_dev_domain,controlled_dev,naturalistic_dev \
  --reject-role-pattern '*_test' \
  --out $STUDY_ROOT/manifests

# Optional diagnostic calibration has its own design-gated import and closed
# role allowlist. When absent from design.json both commands emit only a signed
# NOT_RETAINED receipt; they never borrow primary development rows.
uv run python experiments/import_paper_b_annotations.py \
  --if-retained-by-lock $STUDY_ROOT/locks/design.json \
  --input external/paper_b/calibration_annotations_v1 \
  --design-lock $STUDY_ROOT/locks/design.json \
  --general-policy policies/general/v1/controls.jsonl \
  --domain-policy policies/mortgage/us_v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --verify-naturalistic-component-prefix-roles calibration_fit,calibration_eval \
  --allow-roles calibration_fit,calibration_eval \
  --reject-role-pattern '*_test' \
  --out $STUDY_ROOT/annotations/calibration

uv run python experiments/build_joint_manifest.py \
  --mode calibration-diagnostic \
  --if-retained-by-lock $STUDY_ROOT/locks/design.json \
  --catalog $STUDY_ROOT/catalog.json \
  --annotations $STUDY_ROOT/annotations/calibration \
  --allow-roles calibration_fit,calibration_eval \
  --reject-role-pattern '*_test' \
  --out $STUDY_ROOT/manifests

uv run python experiments/validate_paper_b_artifacts.py \
  --stage e-sep-comparator-strength \
  --design-lock $STUDY_ROOT/locks/design.json \
  --general-dev $STUDY_ROOT/manifests/component_dev_general/manifest.json \
  --domain-dev $STUDY_ROOT/manifests/component_dev_domain/manifest.json \
  --joint-dev $STUDY_ROOT/manifests/joint_dev/manifest.json \
  --out $STUDY_ROOT/manifests/e_sep_comparator_strength.json \
  --strict

# Conditional training-only manifests. These commands are no-ops with a signed
# NOT_RETAINED receipt when E9 is absent from design.json.
uv run python experiments/build_joint_manifest.py \
  --mode e9-general-replay-training \
  --if-retained-by-lock $STUDY_ROOT/locks/design.json \
  --candidate-manifest $STUDY_ROOT/manifests/general_replay_candidate/manifest.json \
  --require-identical-row-set-and-lineage \
  --general-policy policies/general/v1/controls.jsonl \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --catalog $STUDY_ROOT/catalog.json \
  --out $STUDY_ROOT/manifests/general_replay

uv run python experiments/build_joint_manifest.py \
  --mode e9-consolidated-training \
  --if-retained-by-lock $STUDY_ROOT/locks/design.json \
  --domain-train $STUDY_ROOT/manifests/domain_training_authorized/manifest.json \
  --general-replay $STUDY_ROOT/manifests/general_replay/manifest.json \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --out $STUDY_ROOT/manifests/consolidated_train

# Custodian-only environment; output is not mounted in the selector workspace.
uv run python experiments/import_paper_b_annotations.py \
  --input external/paper_b/joint_test_v1 \
  --cohort joint_test \
  --design-lock $STUDY_ROOT/locks/design.json \
  --general-policy policies/general/v1/controls.jsonl \
  --general-manual policies/general/v1/annotation_manual.md \
  --domain-policy policies/mortgage/us_v1/controls.jsonl \
  --domain-manual policies/mortgage/us_v1/annotation_manual.md \
  --construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --component-prefix-counts-from-design-lock \
  --reject-split-components-skips-or-substitutions \
  --custodian-mode \
  --controlled-store-config "$PAPER_B_CONTROLLED_STORE_CONFIG" \
  --out "controlled://${STUDY_ID}/joint_test" \
  --export-seal $STUDY_ROOT/manifests/joint_test/seal.json \
  --export-attestation $STUDY_ROOT/attestations/custody_data.json \
  --sign-as custodian

# Still in the custodian environment: perform the row/family/near-duplicate
# audit against the now-final controlled test object. Export only the signed,
# text-free coverage/result artifact; do not export test identities or hashes
# that enable membership probing.
uv run python experiments/validate_paper_b_artifacts.py \
  --stage final-train-versus-all-evaluation-lineage-private \
  --catalog $STUDY_ROOT/catalog.json \
  --source-role-map configs/paper_b_source_roles_v1.yaml \
  --domain-training $STUDY_ROOT/manifests/domain_training_authorized/manifest.json \
  --general-replay-if-retained $STUDY_ROOT/manifests/general_replay/manifest.json \
  --consolidated-train-if-retained $STUDY_ROOT/manifests/consolidated_train/manifest.json \
  --development-manifest-root $STUDY_ROOT/manifests \
  --controlled-test "controlled://${STUDY_ID}/joint_test" \
  --controlled-store-config "$PAPER_B_CONTROLLED_STORE_CONFIG" \
  --out $STUDY_ROOT/manifests/lineage_audit.json \
  --sign-as custodian \
  --strict

uv run python experiments/validate_paper_b_artifacts.py \
  --stage data \
  --design-lock $STUDY_ROOT/locks/design.json \
  --policy-root policies \
  --naturalistic-sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --lineage-audit $STUDY_ROOT/manifests/lineage_audit.json \
  --e-sep-comparator-strength $STUDY_ROOT/manifests/e_sep_comparator_strength.json \
  --powered-annotation-summary $STUDY_ROOT/annotations/powered/powered_summary.json \
  --raw-judgment-manifest $STUDY_ROOT/annotations/powered/raw_judgments.manifest.json \
  --adjudication-log $STUDY_ROOT/annotations/powered/adjudication_log.jsonl \
  --verify-lineage-signature-as custodian \
  --test-seal $STUDY_ROOT/manifests/joint_test/seal.json \
  --custody-attestation $STUDY_ROOT/attestations/custody_data.json \
  --strict

uv run python experiments/lock_paper_b.py \
  --phase data \
  --parent-lock $STUDY_ROOT/locks/design.json \
  --config configs/paper_b_v1.yaml \
  --policy-construct-snapshot $STUDY_ROOT/construct/policy_construct_snapshot.json \
  --manifest-dir $STUDY_ROOT/manifests \
  --power $STUDY_ROOT/design/power_report.json \
  --naturalistic-sampling-commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --test-seal $STUDY_ROOT/manifests/joint_test/seal.json \
  --annotation-audit $STUDY_ROOT/annotations/powered/agreement_report.json \
  --powered-annotation-summary $STUDY_ROOT/annotations/powered/powered_summary.json \
  --raw-judgment-manifest $STUDY_ROOT/annotations/powered/raw_judgments.manifest.json \
  --adjudication-log $STUDY_ROOT/annotations/powered/adjudication_log.jsonl \
  --lineage-audit $STUDY_ROOT/manifests/lineage_audit.json \
  --policy-coverage $STUDY_ROOT/construct/policy_coverage.json \
  --naturalistic-source-audit $STUDY_ROOT/manifests/naturalistic_source_audit.json \
  --artifact-probe $STUDY_ROOT/manifests/superficial_artifact_probe.json \
  --e-sep-comparator-strength $STUDY_ROOT/manifests/e_sep_comparator_strength.json \
  --privacy-license-audit $STUDY_ROOT/manifests/privacy_license_audit.json \
  --blinding-audit $STUDY_ROOT/annotations/powered/blinding_audit.json \
  --custody-attestation $STUDY_ROOT/attestations/custody_data.json \
  --sign-as data_owner \
  --require-clean-source
```

### PASS gate

- every powered-confirmatory quadrant/category count is met, or the separately locked precision-focused target is met without confirmatory wording;
- both labels are complete for joint rows;
- `joint_dev` resolves only `naturalistic_dev`; controlled rows cannot affect the primary policy bytes;
- each `E_sep` threshold resolves only its policy-specific component-development manifest, with no joint labels or test lineage;
- `E_sep`'s comparator-strength audit passes for any confirmatory secondary status; otherwise `design.json`/the claim registry force descriptive pipeline-specific wording;
- no `split_lineage_graph` connected component or exact/semantic duplicate crosses forbidden roles;
- test remains inaccessible to developers fitting policies;
- naturalistic roles are exactly the locked connected-component prefixes of the P6 committed remaining stream, with the committed object version and no split, skipped, or substituted components;
- data and registry hashes are immutable;
- the powered naturalistic test stratum and independent-authoring/source gates pass.
- every control retained in the paper's scope meets its frozen minimum independent-unit target; all others are explicitly `NOT_EVALUATED` and removed from broad scope wording.

### STOP conditions

- convenience sample size replaces the powered target;
- registry changes after final annotation without versioning/reannotation audit;
- any leakage-capable connected component crosses training/pilot/component-dev/joint-dev/test roles;
- final labels or results have been inspected by model/policy selectors;
- powered annotators were exposed to cohort roles or system outputs;
- quality thresholds drift between pilot and final cohorts.
- the title/abstract claims mortgage surfaces or controls absent from the coverage matrix.
- study-authored or generator-authored scenarios are labeled naturalistic. If authentic data are unavailable, retire the naturalistic primary claim and re-register a controlled challenge-set study before scoring test data.

### Claims enabled

Benchmark construction may be described; no final comparative result.

## 15. P9 — Experiment lock and development-only policy fit

### Objective

Freeze the experiment and produce E3/E4 policies without access to final test.

### Lock order

1. `pilot.json`, created at P6 after the pilot manifest exists: P0 approvals, P3/P4a construct, scorer/schema code, config, catalog, component identities, pilot manifest, and authoring-lineage seal.
2. `design.json`, created at P7 before confirmatory collection: immutable parent claim-registry hash, resolved active claim-registry hash and amendment, powered sizes, analysis mode, exact estimators, multiplicity algorithm, margins, stability rules, and statistical-analysis plan.
3. `data.json`, created at P8: policy registries/manuals, controlled/naturalistic manifests, annotation/lineage/privacy/blinding audits, design-lock hash, and custodian-signed test seal/audit.
4. `experiment.json`, created before P9 scoring/fitting: endpoint and E0–E4 definitions, common raw-margin grid, E3/E4 accounting constraints, tie breakers, traffic scenarios, cost sensitivity, latency workload, and every retained secondary arm.
5. `policy.json`, created after development-only fitting: development scores, all reference/primary/retained-secondary policies, lossless selection traces, pre-test replay/isolation report, and proof that final test remains sealed.
6. `results.json`, created by the custodian at P10: final scores, evaluations, paired comparison, claim checks, custody/run attestation, and failure/exclusion records.
7. `release.json`, created at P13 after generated analysis/manuscript artifacts exist: public object URIs/hashes, regenerated tables/figures/PDF, replay, latency, ethics/release documents, custody attestations, and results-lock hash.

All locks are append-only JCS JSON, omit `self_sha256` while computing that field, reject nonfinite numbers, refuse overwrite, carry detached role signatures, and require a clean **source** tree. Because committing a lock changes Git HEAD, each lock records:

- the pre-lock source commit;
- a canonical source-tree path/hash manifest that excludes locks, catalog/STATUS, generated outputs, and signatures;
- the exact exclusion policy;
- the exact allowlist of untracked artifact outputs; untracked source/config/schema files are always forbidden;
- the prior lock hash.

Validators compare the frozen source-tree manifest rather than requiring post-commit `HEAD` to equal the embedded pre-lock commit. Role-specific public keys and a trust policy distinguish research owner, general-policy owner, statistician, mortgage SME, data owner, custodian, auditor, independent reproducer, and release owner; one generic key cannot prove independence. Use a pinned signing mechanism such as minisign and never commit a private key.

Source-tree exclusions never exclude evidence from the lock: every catalog, manifest, score, audit, attestation, report, policy, and generated output required by a PASS gate is listed as an explicit content-hashed lock input. A lock validator fails on missing, extra, mutable, or unreferenced required evidence.

### Optional probability-calibration diagnostic

Primary policies consume `score_margin` directly and do not require calibration cohorts or artifacts. If a probability-calibration diagnostic is retained, fit it on a dedicated fit role and evaluate NLL/Brier/ECE on a different held-out role. Fit-set before/after metrics are labeled resubstitution and cannot support an improvement claim. For each retained diagnostic, store:

```text
component
label_field
calibration_manifest_hash
score_artifact_hash
positive_temperature_parameterization
solver
tolerance
iterations
converged
temperature
before_after_nll
before_after_brier
frozen_bin_ece
code_sha256
environment_lock_sha256
```

Reject single-class/nonfinite inputs or nonconvergence. Positive temperature scaling must preserve ranking, ties, candidate actions, and the selected primary policies exactly; failure of that invariance test is an implementation bug.

### Strong E3 baseline

The bullets below are checkpoint-specific assertions against the parent-locked `primary_contract.json`; they do not redefine Section 2.1. The validator compares the resolved contract hash before fitting.

1. Enumerate the same complete raw-margin threshold grid as E4, including tagged `ALWAYS`/`NEVER` sentinels.
2. Compute `r_G`, `r_D`, `r_GD`, `r_OR`, and all quadrant missed-intervention rates on the identical `joint_dev` rows, family/lineage IDs, and frozen weights.
3. E3 feasibility uses the conservative union-bound accounting rule `r_G + r_D <= alpha_system`; E4 feasibility uses the measured rule `r_OR <= alpha_system`. Both use the same empirical safety caps, objective, and tie breakers from Section 2.1.
4. Do not prune E3 with separate component objectives or a coarse allocation grid. Those choices would confound accounting with search flexibility and can manufacture an advantage.
5. Store each candidate's two accounting values, overlap credit `r_GD`, constraint values/pass bits, objective, and selected-row proof. Report E3/E4 feasible-set sizes and verify E3 is a subset of E4.
6. Label E4's weak development dominance as a theorem, not a result; only untouched-test effects are claim-bearing.

Both primary arms use the same binary action semantics:

\[
a(x;\tau_G,\tau_D)=
\begin{cases}
\mathrm{INTERVENE},&s_G(x)\ge\tau_G\;\lor\;s_D(x)\ge\tau_D,\\
\mathrm{PASS},&\text{otherwise}.
\end{cases}
\]

If either required component has an error or missing score, primary E3/E4 return `INTERVENE` under the frozen fail-closed rule. The audit record retains the triggering component/error; downstream action efficacy is outside the primary endpoint.

Neither arm may change scores, prompts, components, comparator semantics, objective, tie breakers, action space, or development rows. They differ only in whether overlap `r_GD` receives budget credit.

### Reference arms frozen for the same final job

- `E0`: untuned base under the general prompt;
- `E1`: general adapter only;
- `E2a`: untuned base under the mortgage prompt;
- `E2b`: mortgage adapter only;
- `E_sep`: independently selected `G OR D` from the two policy-specific component-development cohorts;
- native deployment thresholds, if they exist, and a prespecified equal marginal-budget operating point;
- `E3`: selected marginal-sum/union-bound baseline;
- `E4`: selected measured-union threshold policy;
- one threshold on `max(s_G,s_D)` and a combined-policy single-pass guard, if feasible before `experiment.json`.

E0–E2b are descriptive controls required for the composition question. They cannot replace the registered E3/E4 primary contrast.

Freeze each reference operating point on its declared development role before P10 with `experiments/fit_reference_policies.py`: `E_sep` uses only the two `component_dev_*` cohorts, while joint references use `joint_dev`. Their comparator, failure action, score identity, and threshold-selection trace are serialized like E3/E4. Descriptive controls never substitute for the primary contrast.

### Interaction-measurement contract

For each development/test stratum, policy quadrant, and frozen operating point, emit the complete paired 2-by-2 component-action table, `r_G`, `r_D`, `r_GD`, observed union, union-bound slack, conditional trigger rates, and paired missed-intervention overlap with confidence intervals at the frozen inference unit. Report native/reference and locked E3/E4 operating points; never select a test threshold to maximize an overlap story. These measurements answer RQ1 and must precede the E3/E4 effect in the paper narrative.

### Common exact optimizer

1. Enumerate `ALWAYS`, `NEVER`, and every unique raw `G`/`D` margin.
2. Evaluate both E3 and E4 accounting constraints, all frozen quadrant caps, the shared objective, and exact tie breakers for every candidate.
3. Emit `NO_GO` if an arm has no feasible candidate; never relax a cap silently.
4. Store every candidate or a lossless family/lineage-aware representation in the selection trace.

At the powered `joint_dev` size, nearly unique margins can create a quadratic grid. Freeze maximum expected unique-score counts and runtime/memory budgets. A row-level two-dimensional count table is insufficient for “any member fails” family outcomes, paired discordance, delete-cluster stability, or naturalistic clustering; retain per-family/per-lineage action bitsets or equivalent sufficient threshold summaries. Tests must reconstruct every row-, family-, quadrant-, and lineage-level candidate metric and match brute force.

### Mandatory optimizer/statistics tests

- Exact implementation equals independent brute force on many small arrays.
- Ties, repeated scores, empty sets, always-intervene, never-intervene, and weighted rows are covered.
- OR false-/missed-intervention identities hold in property tests.
- With either/both components failed, component indicators are set to one, failures remain in every denominator, and `r_OR = r_G + r_D - r_GD` still holds exactly.
- E3 and E4 input/grid/objective hashes are identical; only the accounting constraint differs.
- `E_sep` threshold bytes depend only on the corresponding policy-specific score/label manifest; changing joint labels cannot change them.
- `feasible(E3)` is a subset of `feasible(E4)` on property tests, and E4 development dominance is asserted rather than “discovered.”
- Null joint labels are rejected.
- Altering any final-test label/file cannot change policy or optional diagnostic bytes.
- Altering controlled-development rows cannot change primary E3/E4 policy bytes; they are mechanism data only.
- Constraint/cost/weight changes alter lock and policy hashes.
- Infeasible constraints produce explicit `NO_GO`, never silent relaxation.
- Threshold comparator semantics (`>=`) are serialized.
- Infinity/NaN are not emitted as non-standard JSON; always/never thresholds use explicit tagged encoding.
- Strictly monotone temperature transforms produce identical candidate actions and selected E3/E4 policies.

### Pre-test reliability authorization

Before P10 is authorized, failure injection, stored-score policy replay, inference-fixture replay, fallback semantics, and test-isolation tests must already pass for E3/E4. P10 may not be used to discover that a timeout, NaN, corrupt artifact, or missing component has undefined behavior.

If E9 is retained, it must also pass here: P4a-authorized domain training rows; a newly Paper-B-locked general replay manifest; a locked consolidated-training manifest; a train-versus-all-Paper-B-evaluation lineage audit run only after P8 has materialized every role; license/redistribution disposition; preregistered grid and endpoint; complete 3–5 seeds for the locked winner; matched-seed inference; immutable publication plus independent reacquisition of every adapter bundle; and no change to E3/E4. Otherwise E9 is `NOT_TESTED` before the test handoff.

### Commands

Run with final test inputs, scores, and results unmounted. If the signed design retains E9, train all preregistered seeds and register them **before** the experiment lock and any joint scoring. Otherwise the registry builder copies the pilot registry and records E9 as `NOT_TESTED`. Then build deterministic fixtures and create the experiment lock.

The long fenced sequence below is the **normative internal stage contract**, not an authorization to run its commands manually. In claim-bearing P9, every stage is a child of `run_policy_fit_paper_b.py`, receives a single-use orchestration token, and refuses direct execution. The auditor-controlled supervisor starts before the first child, enforces the no-test runtime profile/ACL, records process trees, mounts, credentials, network endpoints, inputs, outputs, and timestamps, and signs the completed log with an identity unavailable to the selector. A retrospective `--assert-test-unread` issued by the research process is not evidence of historical isolation.

```bash
# This public verifier occurs after data.json exists, so "all evaluation" is no
# longer a promise about future manifests. It checks that the custodian-signed
# private audit covered catalog-declared G/D training, every conditional E9
# training role, and every locked pilot/dev/test role; it does not expose or
# re-read test membership.
uv run python experiments/validate_paper_b_artifacts.py \
  --stage final-train-versus-all-evaluation-lineage \
  --data-lock $STUDY_ROOT/locks/data.json \
  --locked-lineage-audit $STUDY_ROOT/manifests/lineage_audit.json \
  --custody-attestation $STUDY_ROOT/attestations/custody_data.json \
  --catalog $STUDY_ROOT/catalog.json \
  --source-role-map configs/paper_b_source_roles_v1.yaml \
  --domain-training $STUDY_ROOT/manifests/domain_training_authorized/manifest.json \
  --general-replay-if-retained $STUDY_ROOT/manifests/general_replay/manifest.json \
  --consolidated-train-if-retained $STUDY_ROOT/manifests/consolidated_train/manifest.json \
  --out $STUDY_ROOT/audits/final_train_evaluation_lineage_audit.json \
  --strict

uv run python experiments/train_consolidated_guard.py \
  --if-retained-by-lock $STUDY_ROOT/locks/design.json \
  --config configs/paper_b_v1.yaml \
  --train-manifest $STUDY_ROOT/manifests/consolidated_train/manifest.json \
  --domain-training-manifest $STUDY_ROOT/manifests/domain_training_authorized/manifest.json \
  --general-replay-manifest $STUDY_ROOT/manifests/general_replay/manifest.json \
  --lineage-audit $STUDY_ROOT/audits/final_train_evaluation_lineage_audit.json \
  --all-locked-seeds \
  --out $STUDY_ROOT/components/generated/e9_candidates

# Always create catalog.experiment.json. With E9 absent it records NOT_TESTED
# and preserves the base entries; with E9 retained it publishes every complete
# generated bundle to immutable storage before registry construction.
uv run python experiments/fetch_paper_b_assets.py \
  --base-catalog $STUDY_ROOT/catalog.json \
  --if-retained-by-lock $STUDY_ROOT/locks/design.json \
  --register-generated-bundle-if-retained $STUDY_ROOT/components/generated/e9_candidates \
  --publish-immutable \
  --out-catalog $STUDY_ROOT/catalog.experiment.json

uv run python experiments/fetch_paper_b_assets.py \
  --catalog $STUDY_ROOT/catalog.experiment.json \
  --asset-root external/paper_b/reacquisition_experiment \
  --acquire-and-verify \
  --out-report $STUDY_ROOT/audits/experiment_asset_reacquisition.json

uv run python experiments/build_component_registry.py \
  --parent $STUDY_ROOT/construct/component_registry.json \
  --design-lock $STUDY_ROOT/locks/design.json \
  --catalog $STUDY_ROOT/catalog.experiment.json \
  --reacquisition-report $STUDY_ROOT/audits/experiment_asset_reacquisition.json \
  --out $STUDY_ROOT/construct/component_registry_experiment.json

uv run pytest -q tests/paper_b/test_multi_adapter_scorer.py \
  --paper-b-registry $STUDY_ROOT/construct/component_registry_experiment.json

uv run python experiments/build_paper_b_fixtures.py \
  --config configs/paper_b_v1.yaml \
  --component-registry $STUDY_ROOT/construct/component_registry_experiment.json \
  --replay-out $STUDY_ROOT/replay/cases.jsonl \
  --latency-out $STUDY_ROOT/latency/workload.jsonl

uv run python experiments/validate_paper_b_artifacts.py \
  --stage pre-test-fixtures \
  --replay-fixture $STUDY_ROOT/replay/cases.jsonl \
  --latency-workload $STUDY_ROOT/latency/workload.jsonl \
  --strict

uv run python experiments/lock_paper_b.py \
  --phase experiment \
  --parent-lock $STUDY_ROOT/locks/data.json \
  --config configs/paper_b_v1.yaml \
  --design-lock $STUDY_ROOT/locks/design.json \
  --asset-catalog $STUDY_ROOT/catalog.experiment.json \
  --asset-reacquisition-report $STUDY_ROOT/audits/experiment_asset_reacquisition.json \
  --final-lineage-audit $STUDY_ROOT/audits/final_train_evaluation_lineage_audit.json \
  --component-registry $STUDY_ROOT/construct/component_registry_experiment.json \
  --latency-workload $STUDY_ROOT/latency/workload.jsonl \
  --replay-fixture $STUDY_ROOT/replay/cases.jsonl \
  --claim-registry-from-design-lock \
  --sign-as research_owner \
  --require-clean-source
```

Then run only against the experiment lock:

The experiment lock resolves exact role-manifest hashes. Fitters join labels/family IDs one-to-one by `sample_id` plus exact canonical event hash; normalized overlap hashes are never cache/join authority, and scores never carry or infer labels by row position.

```bash
uv run python experiments/score_joint_stack.py \
  --cohort component_dev_general,component_dev_domain,controlled_dev,joint_dev \
  --lock $STUDY_ROOT/locks/experiment.json \
  --components-from-lock \
  --include-optional-cohorts-from-lock \
  --out-root $STUDY_ROOT/scores

uv run python experiments/fit_reference_policies.py \
  --lock $STUDY_ROOT/locks/experiment.json \
  --general-dev $STUDY_ROOT/scores/component_dev_general/scores.parquet \
  --general-manifest $STUDY_ROOT/manifests/component_dev_general/manifest.json \
  --domain-dev $STUDY_ROOT/scores/component_dev_domain/scores.parquet \
  --domain-manifest $STUDY_ROOT/manifests/component_dev_domain/manifest.json \
  --dev $STUDY_ROOT/scores/joint_dev/scores.parquet \
  --dev-manifest $STUDY_ROOT/manifests/joint_dev/manifest.json \
  --score-field score_margin \
  --out $STUDY_ROOT/policies/reference

uv run python experiments/fit_joint_policy.py \
  --arm marginal-sum-union-bound \
  --lock $STUDY_ROOT/locks/experiment.json \
  --dev $STUDY_ROOT/scores/joint_dev/scores.parquet \
  --dev-manifest $STUDY_ROOT/manifests/joint_dev/manifest.json \
  --score-field score_margin \
  --config configs/paper_b_v1.yaml \
  --out $STUDY_ROOT/policies/e3

uv run python experiments/fit_joint_policy.py \
  --arm measured-union-budget \
  --lock $STUDY_ROOT/locks/experiment.json \
  --dev $STUDY_ROOT/scores/joint_dev/scores.parquet \
  --dev-manifest $STUDY_ROOT/manifests/joint_dev/manifest.json \
  --score-field score_margin \
  --config configs/paper_b_v1.yaml \
  --out $STUDY_ROOT/policies/e4

uv run python experiments/validate_paper_b_artifacts.py \
  --stage policy \
  --assert-test-unread \
  --strict

uv run python experiments/replay_joint_stack.py \
  --policy-dir $STUDY_ROOT/policies \
  --fixture-manifest $STUDY_ROOT/replay/cases.jsonl \
  --mode pre-test-authorization \
  --out $STUDY_ROOT/replay/pre_test_report.json \
  --strict

# Supervised pre-test failure injection. The authoritative P9 orchestrator
# records the process and asks the auditor signer to detach-sign the validated
# report; a selector cannot create the required auditor signature.
uv run python experiments/run_failure_injection_paper_b.py \
  --experiment-lock $STUDY_ROOT/locks/experiment.json \
  --policy-dir $STUDY_ROOT/policies \
  --fixture-manifest $STUDY_ROOT/replay/cases.jsonl \
  --cases timeout,nan,adapter_load_error,corrupt_score,missing_policy_context,review_unavailable \
  --out $STUDY_ROOT/reliability/failure_injection.json \
  --request-signature-as auditor \
  --strict

# Optional calibration diagnostics, if retained, are action-invariant and use
# distinct fit/evaluation manifests. They are not inputs to fit_joint_policy.py.
uv run python experiments/fit_component_calibration.py \
  --lock $STUDY_ROOT/locks/experiment.json \
  --scores-dir $STUDY_ROOT/scores \
  --fit-manifest $STUDY_ROOT/manifests/calibration_fit/manifest.json \
  --eval-manifest $STUDY_ROOT/manifests/calibration_eval/manifest.json \
  --out $STUDY_ROOT/diagnostics/calibration \
  --optional

# Every retained secondary arm is fitted and authorized before P10. Unretained
# arms are serialized as NOT_TESTED in experiment.json and are not run.
uv run python experiments/fit_joint_policy.py \
  --arms-from-lock-secondary \
  --lock $STUDY_ROOT/locks/experiment.json \
  --dev $STUDY_ROOT/scores/joint_dev/scores.parquet \
  --dev-manifest $STUDY_ROOT/manifests/joint_dev/manifest.json \
  --score-field score_margin \
  --out-root $STUDY_ROOT/policies/secondary

uv run python experiments/analyze_paper_b.py \
  --stage development-interaction \
  --lock $STUDY_ROOT/locks/experiment.json \
  --score-cohorts controlled_dev,joint_dev \
  --policy-dir $STUDY_ROOT/policies \
  --out $STUDY_ROOT/diagnostics/interaction/development.json \
  --strict

uv run python experiments/validate_paper_b_artifacts.py \
  --stage secondary-arm-authorization \
  --assert-before-test \
  --strict

# Verify the independently produced, continuously recorded supervisor log. This
# validator cannot create or sign the underlying isolation attestation.
uv run python experiments/validate_paper_b_artifacts.py \
  --stage final-policy-isolation \
  --supervision-attestation $STUDY_ROOT/attestations/policy_fit_supervision.json \
  --verify-signature-as auditor \
  --out $STUDY_ROOT/attestations/policy_fit_supervision_verification.json \
  --strict

uv run python experiments/lock_paper_b.py \
  --phase policy \
  --parent-lock $STUDY_ROOT/locks/experiment.json \
  --score-dir $STUDY_ROOT/scores \
  --policy-dir $STUDY_ROOT/policies \
  --diagnostics-dir-if-present $STUDY_ROOT/diagnostics \
  --development-interaction-report $STUDY_ROOT/diagnostics/interaction/development.json \
  --secondary-authorization $STUDY_ROOT/policies/secondary/authorization.json \
  --pre-test-replay $STUDY_ROOT/replay/pre_test_report.json \
  --failure-injection $STUDY_ROOT/reliability/failure_injection.json \
  --failure-injection-signature $STUDY_ROOT/reliability/failure_injection.json.minisig \
  --policy-fit-supervision $STUDY_ROOT/attestations/policy_fit_supervision.json \
  --policy-fit-supervision-verification $STUDY_ROOT/attestations/policy_fit_supervision_verification.json \
  --latency-workload $STUDY_ROOT/latency/workload.jsonl \
  --replay-fixture $STUDY_ROOT/replay/cases.jsonl \
  --sign-as research_owner \
  --require-clean-source
```

The following is the only authoritative P9 invocation. It executes the internal sequence atomically in auditor-controlled CI, requests the research-owner signatures only after their respective input closures, and exports the signed experiment/policy locks plus supervision evidence. The selector process has neither auditor signing material nor test-store credentials.

```bash
uv run python experiments/run_policy_fit_paper_b.py \
  --data-lock $STUDY_ROOT/locks/data.json \
  --design-lock $STUDY_ROOT/locks/design.json \
  --runtime-profile configs/paper_b_no_test_runtime_v1.yaml \
  --supervisor-log-ref-env PAPER_B_POLICY_FIT_SUPERVISOR_LOG_REF \
  --orchestration-token-env PAPER_B_POLICY_FIT_TOKEN \
  --experiment-lock-out $STUDY_ROOT/locks/experiment.json \
  --policy-lock-out $STUDY_ROOT/locks/policy.json \
  --supervision-attestation-out $STUDY_ROOT/attestations/policy_fit_supervision.json \
  --supervision-verification-out $STUDY_ROOT/attestations/policy_fit_supervision_verification.json \
  --experiment-signing-role research_owner \
  --policy-signing-role research_owner \
  --supervisor-signing-role auditor \
  --strict
```

### PASS gate

E3/E4 share a complete raw-margin grid and differ only in risk accounting; lossless family/dependency-aware traces are complete; E4 development dominance and temperature invariance are proven; controlled and naturalistic development interaction reports are locked; every reference/retained secondary policy exists before P10; auditor-signed failure injection plus replay/isolation authorization pass and are hashed by `policy.json`; dependency-deletion stability and exact-grid resource gates pass; and final test remains unread.

### STOP/NO-GO conditions

- Either E3 or E4 has no feasible development policy. Then the paired P10 positive test is undefined and is not run; retain a development-feasibility/NO-GO result or start a new preregistered study. Never relax E3 or proceed with E4 alone.
- Fewer than 80% of dependency-aware refits achieve at least 95% action agreement with the locked policy, or deleting one graph component/prespecified multiway cluster changes a primary rate by more than 0.02 or flips feasibility.
- Developers want to change margins, weights, endpoints, or E3 after seeing dev effects.
- The test firewall cannot prove non-access.
- Policy serialization cannot reproduce actions exactly.

### Claims enabled

Development feasibility only. Dev effects do not become paper conclusions.

## 16. P10 — One-shot locked final evaluation

### Objective

Evaluate the frozen primary comparison once on untouched test data.

### Execution protocol

1. The independent custodian verifies the policy lock, source-tree hash, signature, and test seal.
2. Inside one atomic sealed job, score final inputs for E0–E4, mandatory `E_sep`, and every retained preregistered secondary arm without exposing text/events, labels, token lengths, score distributions, or aggregates to selectors.
3. Join labels only inside that locked custodian job.
4. Produce all preregistered primary and secondary row/family/lineage predictions, metrics, bounds, sensitivity draws, failures, and claim checks in the same access event.
5. Create and custodian-sign `results.json`, the result bundle, and a one-shot job attestation containing controlled object version, image/source/policy hashes, start/end/access timestamps, egress policy, run ID, outputs, and retention disposition.
6. Do not rerun because the result is unfavorable.
7. If a genuine mechanical bug invalidates the run **before** any input, score distribution, label, or result reaches selectors, an independent auditor may authorize a custodian-only correction. If anything was exposed, retire the cohort and use a replacement untouched final-test cohort under a new study version.

### Primary hierarchical gate

Under `powered_confirmatory` mode, E4 earns a positive measured-union-constrained selection claim on the naturalistic primary stratum only if the frozen sequence passes:

1. The named one-sided paired upper 95% bound for severe missed-intervention difference `E4 - E3` is no greater than the preregistered noninferiority margin.
2. Conditional on gate 1, Holm-adjusted one-sided upper bounds/tests establish all four absolute E4 constraints: `P(INTERVENE | G0/D0) <= alpha_system` and `P(PASS | q) <= beta_q` for `q in {G1/D0,G0/D1,G1/D1}`. Equality passes.
3. Conditional on gates 1–2, the named paired upper bound for E4 protected-context false-intervention worsening is no greater than its frozen margin. This is a prompt-screen disparity check, not a lending-decision fairness claim.
4. Conditional on gates 1–3, with `Delta_intervene = INTERVENE_E4 - INTERVENE_E3`, the named paired upper confidence bound satisfies `UCB(Delta_intervene) < 0`.
5. “Meaningfully improves” additionally requires `UCB(Delta_intervene) <= -delta_intervene_min`; otherwise wording is limited to “statistically lower false intervention.”

The same locked analysis is reported separately on controlled quartets as mechanism/replication evidence. It cannot rescue a failed naturalistic primary result. Report E3's absolute budget outcome even though only E4's absolute pass is required for the proposed-method claim.

Always report E4 versus `E_sep` on the naturalistic test. It is a resource-matched operational proxy, not the one-change mechanistic contrast. Apply confirmatory wording only if its P8 comparator-strength gate, secondary hypothesis, power, and multiplicity slot all pass exactly as frozen in `design.json`; otherwise report effect and uncertainty as descriptive. Even on PASS, limit the statement to the two specific frozen pipelines and do not generalize to “separate deployment.”

Under `precision_focused_measurement` mode, this hierarchy is not run as a confirmatory decision. Report point estimates, uncertainty, and achieved precision only; noninferiority, superiority, preservation, and improvement claims are forbidden.

Interpretation:

| Outcome | Permitted conclusion |
|---|---|
| Naturalistic gates 1–4 pass | measured-union-constrained selection statistically lowers false intervention while meeting the frozen missed-intervention, absolute intervention-budget, and protected-context prompt-screen criteria for this fixed system; add “meaningfully” only if gate 5 passes |
| Absolute/comparative intervention gates pass, utility fails | no evidence that measured-union-constrained selection improves E3; report null/negative result |
| Safety inconclusive | comparative safety preservation is inconclusive; no improvement claim |
| Safety fails | deployment NO-GO; no “safer” or “high-compliance” claim |

### Required reporting

- exact denominators and family counts;
- paired effect sizes and intervals;
- discordant-family counts;
- complete paired component-action/error-overlap tables and union-bound slack;
- `E_sep` component thresholds, source cohorts, composed outcomes, and its frozen descriptive/confirmatory status;
- all four quadrants;
- controlled and naturalistic strata separately, with no post-hoc pooled result;
- all prespecified categories/severity strata;
- unweighted results and frozen traffic scenarios;
- missing/error rows and their fallback actions;
- no inference from marginal-CI overlap.

Every serialized metric must include its numerator, denominator, point estimate, interval or one-sided bound, dependency mode/dimensions/components, estimator/CI implementation/version, multiplicity decision, sensitivity-draw hash if applicable, sampling and scenario weights if applicable, and policy/test/score hashes.

### Commands

The following is the only authoritative invocation. The orchestrator verifies custody, scores, joins labels, evaluates every locked arm, validates artifacts, creates/signs the results lock, and exports only the approved text-free lock/aggregates. Its internal stages are not independently invocable for the confirmatory cohort.

```bash
uv run python experiments/run_sealed_paper_b.py \
  --policy-lock $STUDY_ROOT/locks/policy.json \
  --test-seal $STUDY_ROOT/manifests/joint_test/seal.json \
  --controlled-store-config "$PAPER_B_CONTROLLED_STORE_CONFIG" \
  --controlled-out "controlled://${STUDY_ID}/results" \
  --export-results-lock $STUDY_ROOT/locks/results.json \
  --export-aggregates $STUDY_ROOT/evaluation \
  --export-attestation $STUDY_ROOT/attestations/custody_results.json \
  --custodian-mode \
  --single-use-run-id "$PAPER_B_SINGLE_USE_RUN_ID" \
  --sign-as custodian \
  --strict
```

### PASS gate

The custodian-signed results lock, approved aggregate bundle, and one-shot attestation verify every preregistered primary analysis, `E_sep`, descriptive control, and retained secondary output from one job; `claim_check.json` maps each claim to PASS, FAIL, INCONCLUSIVE, or NOT_TESTED with evidence paths.

### STOP conditions

- Any threshold/arm/margin changes after test access.
- Missing primary analysis is replaced by an exploratory result.
- A secondary arm is fitted or evaluated on the primary test after the sealed job.
- Test exclusions are introduced after viewing system errors.
- A failed safety gate is hidden by aggregate performance.

### Claims enabled

Exactly the conclusion in the outcome table; nothing broader.

## 17. P11 — Mandatory reliability and optional secondary gates

### Objective

Complete mandatory replay/failure/latency evidence and interpret only secondary arms that were frozen before P10 and evaluated in the same sealed custodian job.

### P11a — Mandatory final audit completeness and latency

P11a is required for every Paper B submission, including a negative primary result. Failure injection, fallback semantics, and replay already passed before P10. P11a verifies 100% audit/prediction coverage for the sealed final run and measures batch-1 latency/memory for the locked E3/E4 runtime.

### P11b — Optional ablation arms

1. `E5`: `(B ⊕ G) OR D` base-retention ablation.
2. `E6`: action-routing/deferral band around locked E4 interventions.
3. `E7`: learned fusion ablation.
4. `E8`: action-equivalent early-intervention latency cascade.
5. `E9`: optional constrained consolidated adapter.
6. Optional combined-policy single-pass prompting baseline.

Every retained arm must complete its data/license/component/parity/fit/stability PASS gate in P9 and appear in `policy.json` before P10. P11 is interpretation and audit only; it cannot qualify, fit, or add an arm after results exist. An arm developed after P10 requires a separate untouched test cohort/new study version or remains explicitly exploratory without primary-test evaluation.

### Selective-deferral gate

Report `PASS`, `CONSTRAIN`, `REVIEW`, and `BLOCK` separately. `REVIEW` is an intervention, not a demonstrated successful outcome.

To claim that human review improves final safety, collect:

- reviewer accuracy;
- turnaround distribution;
- capacity/queue behavior;
- final disposition;
- reviewer disagreement and escalation.

Without these, call E6 **selective deferral** only.

### Reliability and latency gate

- failure injection covers timeout, NaN, adapter-load error, corrupted score artifact, missing policy context, and unavailable review;
- every failure routes to the frozen fallback action;
- p50/p95/p99 latency is measured at batch 1 with synchronized accelerators;
- raw cold and warmed requests, token lengths, hardware/OS, memory, and adapter-switch timing are stored;
- early intervention is proven action-equivalent to full OR and never early-passes.

Every evaluation row must also produce a schema-valid prediction/audit record containing the code, model, tokenizer, adapter, prompt, optional score transform, policy, threshold, primary intervention decision, triggering layer, required downstream action gold, reason, latency, and error/fallback identities. Audit completeness must be 100%; later human dispositions are immutable linked events rather than edits.

### P11c — Optional generality gate

One SmolLM3 adapter pair supports an exact-system case study. A broad claim that measured-union-constrained selection improves layered screens requires prespecified variation in both:

- backbone/model family; and
- policy domain or guard task.

It also requires a frozen cross-system estimand and aggregation rule. Multiple seeds or adapter pairs on the same backbone, data, and mortgage policy do not establish generality.

### Commands

```bash
uv run python experiments/benchmark_joint_latency.py \
  --policy-lock $STUDY_ROOT/locks/policy.json \
  --workload $STUDY_ROOT/latency/workload.jsonl \
  --batch-size 1 \
  --warmup 50 \
  --requests 1000 \
  --out $STUDY_ROOT/latency \
  --require-workload-hash-from-lock

uv run python experiments/replay_joint_stack.py \
  --policy-lock $STUDY_ROOT/locks/policy.json \
  --fixture-manifest $STUDY_ROOT/replay/cases.jsonl \
  --out $STUDY_ROOT/replay/report.json \
  --strict

uv run python experiments/validate_paper_b_artifacts.py \
  --stage systems \
  --policy-lock $STUDY_ROOT/locks/policy.json \
  --failure-injection $STUDY_ROOT/reliability/failure_injection.json \
  --verify-failure-signature-as auditor \
  --results-lock $STUDY_ROOT/locks/results.json \
  --custody-attestation $STUDY_ROOT/attestations/custody_results.json \
  --strict
```

E9, if retained, used `experiments/train_consolidated_guard.py` and completed its seed/artifact/action-construct gate during P9 before `policy.json`; otherwise it is `NOT_TESTED` in the P10 arms list.

### PASS gate

P11 passes when P11a is complete and every optional arm is either supported by its registered artifact/gate or marked `NOT_TESTED` and removed from manuscript claims. Optional E5–E9 or generality replication are not prerequisites for a valid negative E3/E4 paper.

### STOP conditions

- Mandatory replay, failure injection, audit completeness, or latency evidence is missing.
- A secondary arm reuses the primary test outside the sealed P10 job.
- REVIEW is counted as a successful block without downstream human outcomes.
- A fixed-system result is presented as general across guard stacks.

### Claims enabled

Only passed fixed-system reliability/latency findings and individually gated secondary findings.

## 18. P12 — Results integration and manuscript finalization

### Objective

Produce the paper from locked artifacts without hand-transcribed values.

### When writing may begin

| Stage | Permitted writing |
|---|---|
| P0–P3 | introduction hypothesis, related work, policy/threat-model skeleton |
| P4–P6 | methods, engineering design, annotation protocol, data-card draft |
| P7–P9 | preregistered statistics, experiment design, result-independent tables/figures |
| After P10 | abstract values, results, discussion, conclusion |

Do not write a claim-bearing abstract from dev or pilot results.

### Main-paper structure

1. From component metrics to composed-system behavior.
2. Related work and novelty boundary.
3. Request-intervention/action construct, policy scope, and threat model.
4. Controlled and naturalistic dual-policy evaluation strata.
5. Component error-overlap measurement and fixed-component E3/E4 selection.
6. Preregistered experiments and statistics.
7. Primary results, followed by deferral/latency.
8. Limitations, responsible release, and conclusion.

### Main figures

1. Policy/action construct plus controlled versus naturalistic cohort flow.
2. Error-overlap matrices by quadrant, component, and cohort.
3. Development marginal-sum-constrained/measured-union-constrained frontier with only the locked E3/E4 final-test points overlaid using paired intervals; never sweep test labels to draw a confirmatory frontier.
4. Secondary coverage/deferral/latency trade-off, only if its gate passes.

### Main tables

1. Cohort, family, quadrant, severity, and agreement statistics.
2. Primary E3/E4 result with confidence bounds and claim-gate status, followed by the mandatory E4/`E_sep` operational comparison with its frozen descriptive/secondary status.
3. Per-quadrant component and composition outcomes.
4. Latency, failure, audit completeness, and replay outcomes.

Move schemas, optimizer proof, full allocation/cost grids, policy registry, and extended ablations to appendices/artifacts.

### Novelty boundary

The defensible contribution is:

> A dual-labeled request-intervention evaluation with controlled and naturalistic strata, a measurement of policy-screen error overlap, and a fixed-system comparison of marginal-sum-constrained versus measured-union-constrained operating-point selection.

Use **benchmark** only if reusable rows, labels, and a durable evaluation interface are actually available to external researchers under the release contract. If naturalistic material remains reviewer-only or cannot be queried reproducibly, call the contribution a **dual-labeled evaluation cohort and protocol** in the title, abstract, contributions, and conclusion.

Do not claim novelty for LoRA, OR composition, calibration, finite threshold enumeration, nested feasible-set dominance, named adapters, or abstention. The threshold rule is simple systems engineering; the defensible contribution is the construct, interaction measurement, and held-out comparison. The related-work matrix must include, at minimum:

- [FinGuard](https://arxiv.org/abs/2605.29427) for regulation-grounded financial compliance data and guards;
- [APT](https://aclanthology.org/2026.acl-long.748/) for unseen-policy/domain guard generalization;
- [On Calibration of LLM-based Guard Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a99f732df9b668284b449da0214a3286-Abstract-Conference.html) for post-hoc guard calibration;
- [R2-Guard](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a07e87ecfa8a651307c16ac747df01-Abstract-Conference.html) for combining category-risk signals with policy reasoning;
- [FlexGuard](https://aclanthology.org/2026.acl-long.263/) for continuous guard scores and deployment strictness;
- [LS-Guard](https://aclanthology.org/2026.findings-acl.989/) for multi-LoRA guard architecture;
- [risk-controlled abstention](https://arxiv.org/abs/2607.04430) if E6 remains in the paper.
- [Selective Classification via One-Sided Prediction](https://proceedings.mlr.press/v130/gangrade21a.html) for error-constrained coverage selection;
- [Challenges and Remedies of Domain-Specific Classifiers as LLM Guardrails](https://aclanthology.org/2025.naacl-industry.15/) for domain-shift and false-refusal risks;
- [Conformal LLM Routing with Distribution-Free Safety Guarantees](https://aclanthology.org/2026.acl-srw.70/) and the closest formal risk-control work for the boundary between empirical selection and a deployment guarantee;
- [MortarBench](https://arxiv.org/abs/2606.19416) and [AgentFairBench](https://arxiv.org/abs/2606.16723) for the closest mortgage/lending-domain evaluation and fairness scopes.

Refresh the search immediately before submission and record search date, queries, and inclusion decisions.

### Generated-analysis contract

`experiments/analyze_paper_b.py` must generate:

- every TeX/Markdown/CSV table;
- every figure;
- claim-check results;
- manuscript macros for all numerical claims;
- a text report with artifact hashes, including every P11 supplemental artifact consumed outside `results.json`.

No result number may be typed directly into the manuscript.

### Build and visual QA

```bash
uv run python experiments/analyze_paper_b.py \
  --results-lock $STUDY_ROOT/locks/results.json \
  --systems-dir $STUDY_ROOT \
  --out $STUDY_ROOT/analysis \
  --paper-dir paper_b

make -C paper_b clean all
```

`paper_b/Makefile` must invoke `tools/run_tectonic_locked.sh`; it may not resolve an arbitrary ambient `tectonic` binary/bundle.

Required checks:

- zero undefined references/citations;
- no overfull text or unreadable tables;
- tracked PDF matches fresh source build;
- figures remain legible in grayscale and at final column width;
- abstract, contributions, results, and conclusion use the same claim status;
- title identifies request screens, while abstract and conclusion explicitly say prompt-level intervention screening, not downstream routing, response correctness, or legal compliance.

### PASS gate

Every numerical sentence resolves through a generated artifact and every contribution maps to a passed claim gate.

### STOP conditions

- A manuscript number is hand-entered or cannot be traced to the results/supplemental hashes.
- Abstract, contribution, result, and conclusion claim statuses disagree.
- A test-label-swept frontier is presented as confirmatory.
- The fresh build has unresolved citations, overflow, stale figures, or an unpinned toolchain.

### Claims enabled

A technically consistent submission candidate; publication claims remain bounded by P10/P11 claim checks.

## 19. P13 — Fresh-clone release and independent replay

### Objective

Demonstrate that an independent reviewer can reconstruct the evidence chain.

### Release contents

- code and frozen environment;
- policy registries and source snapshots/references;
- data card and annotation manual;
- public dual-label rows or reviewer-accessible controlled objects plus immutable manifests;
- immutable adapters or acquisition IDs;
- row-keyed raw logits and metadata;
- optional held-out probability-calibration diagnostics, only if retained;
- E3/E4 policy and complete selection traces;
- evaluation predictions, metrics, bootstrap draws, and claim checks;
- latency and failure artifacts;
- replay fixtures and report;
- all seven locks, detached role signatures, trust policy, custody attestations, and artifact catalog;
- the pre-score sampling commitment plus a post-results verification procedure: public seed/nonce reveal when safe, otherwise reviewer-only KMS access that recomputes the component ordering and prefix assignments;
- generated paper source, figures, tables, and PDF.
- policy/action construct audit, control-coverage matrix, ethics/consent determination, and public/controlled access manifest.

### CI workflows

#### Standard CI

`.github/workflows/ci.yml`:

- frozen Python 3.12 environment;
- lint, type, schema, synthetic unit/property tests;
- no model/network/secrets during tests after frozen dependency restore.

#### Integration workflow

`.github/workflows/paper-b-integration.yml`:

- manual/self-hosted accelerator;
- immutable asset download and hash verification;
- real scorer parity, replay, and latency;
- cache key includes environment lock, source-tree manifest, asset catalog, invoking component registry, policy lock, exact input manifest, replay fixture, latency workload, and scorer code hashes. Cached outputs are evidence only after full identity/logical-hash revalidation; otherwise rerun rather than trust the cache.

#### Release workflow

`.github/workflows/paper-b-release.yml`:

- fresh clone;
- frozen sync;
- public artifact acquisition and validation;
- regenerate analysis and paper into a clean temporary output;
- compare every byte/logical hash to `release.json`;
- separately require no diff for tracked generated sources;
- required artifacts fail rather than skip.

### Independent reproduction

The independent reviewer must:

1. start from a fresh clone;
2. fetch and verify all public artifacts;
3. recreate stored-score decisions exactly;
4. reproduce inference fixtures within declared device tolerances;
5. regenerate all tables/figures and the PDF;
6. confirm claim-check statuses and hashes;
7. independently verify that the revealed/controlled seed and nonce match the P6 commitment and deterministically reproduce the eligible-component ordering, pilot prefix, later role assignments, and controlled object hashes;
8. sign an independent post-release reproduction attestation; because it is created after `release.json`, it is not falsely self-included in that lock;
9. file discrepancies without help from the original run directory.

Controlled-access rows must be genuinely available to reviewers through a documented object and access process. A text-free reconstruct-only manifest without accessible rows/labels supports artifact-identity auditing, not independent data reproduction; label the release tier honestly if row access cannot be provided.

### Human-participant and annotator release checks

- document annotator qualifications and training;
- document compensation and workload;
- record informed consent/release terms appropriate to the annotation process;
- obtain and archive the applicable ethics/IRB determination or exemption decision;
- disclose adjudicator conflicts of interest;
- confirm released annotations/rationales are covered by consent and data-use terms.

### Privacy and audit separation

- Offline benchmark rows use exact canonical event hashes for identity and separately labeled normalized hashes for overlap detection.
- Runtime events use keyed HMACs with `hmac_key_id`; keys are never stored in artifacts.
- Raw regulated prompts are not stored by default.
- Human-review outcomes are immutable linked events, not mutations.

### PASS gate

The release workflow and independent reproduction both pass from clean state, with no ignored local file silently required.

Create and cryptographically sign the release lock only after analysis and paper outputs exist. The `--systems-dir` traversal uses a frozen include manifest and excludes only the release lock being written, detached signatures, temporary files, and mutable `STATUS.md`; it may not silently sweep arbitrary files. Independent reproduction and final review live under `$POST_RELEASE_ROOT`, outside this immutable tree.

```bash
uv run python experiments/lock_paper_b.py \
  --phase release \
  --parent-lock $STUDY_ROOT/locks/results.json \
  --analysis-dir $STUDY_ROOT/analysis \
  --paper-dir paper_b \
  --systems-dir $STUDY_ROOT \
  --public-object-manifest $STUDY_ROOT/release/public_objects.json \
  --access-instructions $STUDY_ROOT/release/access_instructions.md \
  --ethics-dir $STUDY_ROOT/ethics \
  --custody-attestation $STUDY_ROOT/attestations/custody_results.json \
  --latency-dir $STUDY_ROOT/latency \
  --replay-dir $STUDY_ROOT/replay \
  --require-clean-source \
  --sign-as release_owner

minisign -Vm $STUDY_ROOT/locks/release.json \
  -p keys/release_owner.pub

uv run python experiments/analyze_paper_b.py \
  --release-lock $STUDY_ROOT/locks/release.json \
  --verify-only

# Run in the independent reproducer's fresh environment after the release lock.
uv run python experiments/verify_sampling_commitment.py \
  --release-lock $STUDY_ROOT/locks/release.json \
  --commitment $STUDY_ROOT/manifests/naturalistic_sampling_commitment.json \
  --seed-reveal-ref-env PAPER_B_SAMPLING_VERIFICATION_REF \
  --controlled-store-config "$PAPER_B_CONTROLLED_STORE_CONFIG" \
  --out $POST_RELEASE_ROOT/sampling_verification.json \
  --sign-as reproducer \
  --strict

minisign -Vm $POST_RELEASE_ROOT/sampling_verification.json \
  -p keys/reproducer.pub

uv run python experiments/validate_paper_b_artifacts.py \
  --stage independent-reproduction \
  --release-lock $STUDY_ROOT/locks/release.json \
  --sampling-verification $POST_RELEASE_ROOT/sampling_verification.json \
  --out $POST_RELEASE_ROOT/independent_reproduction.json \
  --sign-as reproducer \
  --strict

minisign -Vm $POST_RELEASE_ROOT/independent_reproduction.json \
  -p keys/reproducer.pub
```

### STOP conditions

- Hard-coded figure/table values.
- Missing ignored adapters/scores.
- Stale git SHA or dirty final lock.
- Unreplayable decisions.
- Sampling commitment/order/prefix assignment cannot be independently verified at the declared access tier.
- Raw PII, secrets, or license-prohibited text in the release.

### Claims enabled

The study may claim fresh-clone artifact reproducibility only at the verified access tier: full data reproduction when rows/labels are accessible, or artifact-identity/replay verification when they are not.

# Part III — Submission Audit and Handoff

## 20. P14 — Final technical-review and submission audit

### Objective

Prove completion requirement by requirement before submission.

### Correctness audit

- [ ] Canonical tie-aware AP/AUROC is used everywhere.
- [ ] No score cache is validated by length or row position.
- [ ] `B/G` use the general prompt; `D` uses the mortgage prompt.
- [ ] Multi-turn rows retain full context.
- [ ] Raw-margin primary policies are action-invariant to optional monotone calibration; any calibration fit/evaluation roles are distinct.
- [ ] Fit code cannot access final-test text/events, labels, scores, token summaries, or aggregates.
- [ ] The legacy mortgage label-to-action audit passed, and `D` is not misrepresented as a universal blocker.
- [ ] E3 and E4 share components, raw scores, grid, objective, caps, and tie breakers; only marginal-sum versus measured-union accounting differs.
- [ ] Every implementation and lock resolves the same P0 `primary_contract.json` hash; no later prose or config redefines it.
- [ ] `E_sep` thresholds were selected independently from policy-specific cohorts, evaluated in the sealed job, and described with their preregistered multiplicity status.
- [ ] E4 development dominance is labeled mathematical, not empirical evidence.
- [ ] E0/E1/E2a/E2b controls were evaluated in the same single custodian job.
- [ ] Exact optimizer equals brute force on fixtures.
- [ ] Primary binary failures route to INTERVENE; downstream CONSTRAIN/REVIEW/BLOCK semantics and outcomes remain separate.
- [ ] No retained arm accessed primary test outside the single P10 custodian job.
- [ ] Every decision replays.

### Data and statistical audit

- [ ] Both policy registries are expert reviewed and versioned.
- [ ] Both labels are complete on joint dev/test.
- [ ] Annotation agreement meets frozen thresholds.
- [ ] Powered counts remain adequate after exclusions, or precision-focused mode forbids confirmatory wording.
- [ ] No family/template/near-duplicate crosses forbidden roles.
- [ ] The pre-annotation sampling commitment is independently recomputed after results; its seed/nonce, component ordering, pilot prefix, later role assignments, and controlled object hashes verify at the declared access tier.
- [ ] Crossed author/generator/source/template/control dependencies use the same frozen graph-component or multiway design for splits, power, inference, and stability.
- [ ] Primary gate uses paired effect bounds, not marginal-CI overlap.
- [ ] The exact fixed-sequence/Holm algorithm, paired interval, rare-event fallback, and all four absolute E4 constraints match the signed SAP.
- [ ] The protected-context prompt-screen counterfactual gate is powered, paired, and not misreported as lending-decision fairness.
- [ ] Controlled and naturalistic results are separate; only the naturalistic stratum carries the primary positive claim.
- [ ] Weighted scenario cost is not presented as empirical deployment prevalence.
- [ ] Multiple secondary analyses follow the frozen hierarchy.

### Claim audit

- [ ] Positive/negative/inconclusive wording matches `claim_check.json`.
- [ ] One adapter pair is described as a fixed-system case study.
- [ ] “High compliance,” “compliance guard,” and equivalent current-system wording are absent; Section 21 is a separate future-study boundary.
- [ ] REVIEW is called selective deferral unless human outcomes are evaluated.
- [ ] No protected-class fairness claim comes from name-only exploratory probes.
- [ ] The title identifies request screens; abstract/methods/limitations/conclusion explicitly state input-only screening scope.
- [ ] No claim of legal/regulatory correctness is made.
- [ ] Closest literature was refreshed immediately before submission.

### Artifact audit

- [ ] Seven content-addressed locks exist, chain correctly, self-validate, and carry the required role-specific signatures/tags.
- [ ] Model/tokenizer/full-adapter/prompt/policy/data/code hashes exist.
- [ ] Raw row-keyed logits and error codes are available.
- [ ] Complete family/lineage-aware selection traces and any optional calibration diagnostics are available.
- [ ] Evaluation denominators, draws, and claim checks are available.
- [ ] Fresh-clone CI regenerates tables, figures, and PDF to the release hashes and leaves tracked generated sources unchanged.
- [ ] Controlled data needed for reproduction are actually accessible to reviewers or release limitations are explicit.
- [ ] Independent replay report passes.
- [ ] Release contains no secrets/PII/license violation.
- [ ] Annotator qualifications, compensation, consent, ethics/IRB determination, and adjudicator conflicts are documented.

### Manuscript audit

- [ ] Error-overlap measurement leads the scientific story; one primary comparative endpoint remains evident.
- [ ] Contributions describe the dual-stratum evaluation cohort/protocol (or benchmark only if the access gate passes), interaction measurement, and fixed-system evaluation, not generic calibration/LoRA/threshold-search novelty.
- [ ] Results lead with preregistered component-error overlap and union-bound slack, followed immediately by E3 versus E4 and then E4 versus `E_sep`.
- [ ] Secondary arms cannot obscure a negative primary result.
- [ ] Every number is generated from locked artifacts.
- [ ] PDF compiles without unresolved citations, layout overflow, or stale figures.
- [ ] Limitations state fixed components, prompt-only scope, policy/jurisdiction limits, synthetic/challenge-set limits, and scenario-weight assumptions.

### Final decision

| State | Decision |
|---|---|
| All naturalistic mandatory comparative gates pass | submit positive fixed-system request-screening paper |
| Safety passes but utility does not | submit benchmark plus null/negative system result if artifact/novelty quality is strong |
| Primary result is underpowered/inconclusive | report inconclusive; do not convert an exploratory secondary result into the headline |
| Absolute intervention/missed-intervention gates fail | remove positive method and deployment language; benchmark/negative paper may still submit |
| Test leakage, policy drift, or nonreproducible artifact | do not submit; create a new study version after correction |

Create a signed technical-review decision rather than marking P14 by prose alone:

```bash
uv run python experiments/validate_paper_b_artifacts.py \
  --stage final-technical-review \
  --release-lock $STUDY_ROOT/locks/release.json \
  --independent-reproduction $POST_RELEASE_ROOT/independent_reproduction.json \
  --reproducer-key keys/reproducer.pub \
  --out $POST_RELEASE_ROOT/final_review.json \
  --strict

minisign -Sm $POST_RELEASE_ROOT/final_review.json \
  -s "$PAPER_B_AUDITOR_SECRET_KEY"
```

## 21. High-compliance extension boundary

The P0–P14 request-screening study cannot establish mortgage compliance, even if every comparative gate passes. “High compliance” requires a separate protocol/study version with response, decision, disclosure, tool/action, workflow-state, monitoring, and human-outcome evidence. The following items define that future extension; they are not an optional label that can be attached to the current result.

A separate future study would require all of the following, with its own preregistration and untouched data, in addition to reusing any valid P0–P14 artifacts:

- enough independent expert-validated `G0/D0` units for the stated false-intervention-rate precision after clustering;
- enough severe units for the stated absolute missed-intervention bound;
- frozen operational traffic scenarios or clearly labeled hypothetical scenarios;
- subgroup power for every subgroup claim;
- protected-context counterfactual intervention-disparity bounds and actual lending-decision fairness evaluation kept conceptually separate;
- control coverage across every claimed mortgage workflow stage, with temporal/policy-version shift tests;
- policy-change, signed-release, rollback, and replay evidence;
- failure-mode and audit completeness;
- response/tool-call/credit-decision data and action correctness;
- actual human reviewer outcomes if human review is claimed as a safety benefit.

For the present paper, use “mortgage-policy request-screen study,” never “high-compliance system” or “compliance guard,” regardless of whether these future-extension items are partially explored.

## 22. Checkpoint status ledger template

Maintain this table in `$STUDY_ROOT/STATUS.md`:

| Checkpoint | Status | Evidence paths | Verification command/run ID | Reviewer | Date | Notes |
|---|---|---|---|---|---|---|
| P0 | NOT_STARTED |  |  |  |  |  |
| P1 | NOT_STARTED |  |  |  |  |  |
| P2 | NOT_STARTED |  |  |  |  |  |
| P3 | NOT_STARTED |  |  |  |  |  |
| P4 | NOT_STARTED |  |  |  |  |  |
| P4a | NOT_STARTED |  |  |  |  |  |
| P5 | NOT_STARTED |  |  |  |  |  |
| P6 | NOT_STARTED |  |  |  |  |  |
| P7 | NOT_STARTED |  |  |  |  |  |
| P8 | NOT_STARTED |  |  |  |  |  |
| P9 | NOT_STARTED |  |  |  |  |  |
| P10 | NOT_STARTED |  |  |  |  |  |
| P11 | NOT_STARTED |  |  |  |  |  |
| P12 | NOT_STARTED |  |  |  |  |  |
| P13 | NOT_STARTED |  |  |  |  |  |
| P14 | NOT_STARTED |  |  |  |  |  |

Allowed status values are `NOT_STARTED`, `IN_PROGRESS`, `PASS`, `FAIL`, and `RETIRED`. A checkpoint can be `PASS` only when its evidence paths exist and the verification command succeeds against the current lock hashes.

## 23. Initial-plan crosswalk

| Initial plan section | Implemented by checkpoints |
|---|---|
| Runtime and decision semantics | P0, P3, P4a, P5, P9, P11 |
| Current-data feasibility | P2, P4, P4a, P5, P6 |
| Optimizer smoke test | P5, P9 |
| Research questions and claim gates | P0, P7, P10, P14 |
| Mathematical formulation and calibration-invariance proof | Section 2.1, P0, P7, P9; full proof/tests in appendix |
| Data schema and construction | P3, P4, P4a, P6, P8 |
| Code implementation | P1, P2, P4, P5, P9 |
| Experiment arms | P0, P9, P10, P11 |
| Evaluation and statistics | P7, P10, P11 |
| Audit and policy lifecycle | P2, P3, P9, P13 |
| Commands | P0–P13 |
| Acceptance gates | P10, P11, P14; Section 21 is a separate future-study boundary |
| Paper structure and claims | P12, P14 |
| Compute/labor/schedule | P6–P8 after measured pilot and power report |
| Final go/no-go decision | P10 and P14 |

## 24. Immediate next actions

Execute only these first six actions before confirmatory collection or optional model work:

1. Commit this corrected protocol on a Paper B branch, then complete the P1 Python 3.12/toolchain/validator/signing foundation.
2. Complete and role-sign P0 with request-intervention semantics, one primary endpoint, E_sep, and the exact E3/E4 accounting contrast.
3. Implement asset acquisition plus complete bundle hashing in P2.
4. Freeze both policy/action registries, coverage matrix, and manuals with expert signoff in P3.
5. Run P4a against all mortgage source/training rows; authorize, retrain, or retire the candidate target before P6 held-out score validation.
6. Implement the schemas and correct multi-prompt scorer, then prove parity/replay on diagnostic data in P4–P5.

Do not spend compute on E5–E9 and do not draft headline results until the dual-label pilot, power analysis, and test firewall have passed.
