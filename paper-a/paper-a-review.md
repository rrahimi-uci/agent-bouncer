# Paper A deep review

**Paper:** The Benchmark Chooses the Winner: Measuring Fine-Tuning Specialization Across Safety-Guard Benchmarks

**Review date:** 2026-07-13

**Scope:** manuscript, Paper A data construction, training/evaluation code, statistical analysis, locked artifacts, clean-checkout reproducibility, PDF build, and current related work.

## Decision

**Implementation repaired; legacy-only scientific result. No-go for confirmatory submission until a clean v2 GPU rerun and a genuinely untouched evaluation cohort exist.**

The paper now reports the useful descriptive signal that the committed score table actually supports: all four adapted checkpoints improve on represented-source tests, while the fixed-panel transfer change is negative and heterogeneous. The score table is internally complete, its arithmetic checks out, and the manuscript labels the result retrospective, precision-focused, and estimation-only.

The repository implementation now repairs the fail-open lock, data-family, truncation, adapter, score, analysis, and paper-generation paths. It cannot retroactively repair the committed v1 models and scores: their locked Git revision is unrecoverable, two transfer cohorts came from a previously scored legacy subset, known benchmark pairs were not linked, calibration and ID leaked two families, and long-input preprocessing dropped the classifier instruction. Those limitations are now disclosed rather than hidden.

Accordingly, the corrected paper is a defensible report of a legacy artifact, not clean confirmatory evidence. Promoting the result requires retraining all 20 adapters, rescoring all 24 bundles under the v2 contract, and using a prospective uninspected cohort or benchmark for any confirmatory claim.

## Second-pass repair status (2026-07-13)

The review was re-audited against the code and artifacts before implementation. Two factual corrections were made: the direct XSTest relationship count is 58 pairs across eight blocks (not 45), and the manifest builder already reads the repository `.env` file directly. Hardware reporting was also narrowed from “absent” to “partial”: legacy metadata contains the A100 device, peak memory, timing, and core package versions, but not a complete OS/driver/runtime/container record.

The repository repair now addresses the defects that can be fixed without minting new scientific evidence:

- strict v2 locks are self-hashed, require clean source/prerequisite bindings, freeze the exact four-by-five seed/recipe/data/operating-point/resampling protocol, verify all files, and use `artifacts/paper_a_sft_v2/` so a new run cannot overwrite immutable v1 evidence;
- training, run validation, evaluation, and analysis fail closed on recipe, adapter, manifest, score-schema, matrix, identity, score-digest, calibrated-probability, threshold, and prediction drift;
- resolved-path guards keep final, nonfinal, v1, and v2 write namespaces disjoint even through absolute paths or symlinks, and final input verification rejects redirected config, manifest, audit, public-release, or execution-source files; score caches bind batch size and their actual producer runtime;
- pinned Hugging Face reconstruction is the default; the seed-7 frozen cache requires an explicit retrospective flag;
- authoritative upstream families now cover all 36 selected JailbreakBench pairs and 58 selected XSTest pairs, calibration is family-disjoint from every reported test/stress surface, and the versioned audit must contain exactly the defined 24 hard assertions;
- MinHash is one versioned NumPy implementation, independent of optional packages;
- a tracked recursively redacted public snapshot, policy crosswalk, and contradictory-label inventory contain no raw/normalized prompt text and explicitly identify themselves as legacy evidence; the clean contract additionally binds the audited public projection to the raw manifest commitments;
- completed runs bind the fixed data-order seed, rendered prompt identity, and protocol-relevant package versions; final scoring rechecks those versions and explicitly serializes a valid positive-infinity threshold as `PREDICT_NONE` without nonstandard JSON;
- the analyzer uses the observed estimator, labels two-sided intervals correctly, emits no formal precision-mode rejection/Holm result, reports macro and pooled RQ4 values plus both stress directions, generates narrative macros and a 20-cell seed appendix, binds both score and score-metadata digests, and produces byte-idempotent outputs;
- the manuscript and READMEs now use retrospective estimation language, surface the HarmBench decline and legacy provenance/truncation/family defects, and distinguish the clean v2 pipeline from explicit v1 reproduction;
- the paper/HTML build paths are wired after the nested move, the old broad-study explorer is unlinked and labeled archived, and tracked third-party prompt samples were removed.

These changes do **not** retroactively repair the committed model scores. The truncation fix changes training preprocessing and therefore requires retraining all 20 adapters and rescoring all 24 bundles. Corrected transfer selection changes 605 WildGuard/WildJailbreak rows; global calibration/ID assignment moves three rows. Moreover, the corrected transfer cohorts still retain 615 previously inspected rows, so a prospective confirmatory claim requires a genuinely uninspected cohort or benchmark. Until that rerun exists, the scientifically defensible deliverable is the explicitly labeled legacy, precision-focused estimate, and the publication decision remains major revision / no-go for confirmatory claims.

### Original audit finding map

The map and detailed P0/P1/P2 sections below preserve the state found in the original legacy implementation. Their present-tense evidence describes that reviewed baseline, not the repaired worktree. Closure status and post-repair verification are recorded above and in the reproducibility section; the scientific limitations that require new training/scoring evidence remain open.

| ID | Severity | Finding | Consequence |
|---|---|---|---|
| P0-1 | Blocker | Locked Git SHA predates the Paper A pipeline and records a dirty tree | Executed source is unrecoverable |
| P0-2 | Blocker | Lock hashes and prerequisites are recorded but not enforced | Inputs can drift without stopping a final stage |
| P0-3 | Blocker | Incomplete score tables can still pass every gate | Missing evidence becomes a positive claim |
| P0-4 | Blocker | Two transfer cohorts come from a legacy seed-7 subset previously scored during HPO | Declared selection and prospective-holdout story are false |
| P0-5 | Blocker | Headline uses bootstrap mean instead of the defined observed estimator | Reported +0.325 should be +0.333 on current artifacts |
| P0-6 | Blocker | Precision-focused lock is reported with formal gate/rejection language | Statistical governance contract is violated |
| P1-1/2 | High | Upstream pairs are not clustered and calibration/ID share global families | Dependence and held-back calibration claims are wrong |
| P1-3 | High | Left truncation removes the classifier system prompt on long rows | Some base/SFT comparisons use a different interface |
| P1-4/5 | High | Audit and run validation fail open | Corrupt families or stale adapters can be accepted |
| P1-9 | High | RQ4 table omits base TPR, delta TPR, intervals, and base stress values | Material HarmBench degradation is hidden from the paper |
| P1-12/13 | High | Manifests and required frozen cache are absent from Git | Fresh-clone artifact tests skip and manifest build fails |
| P1-15/16/17 | High | Paper sync, environment lock, and MinHash backend are not reproducible | A documented reproduction can use different inputs or leave stale paper output |

## What I checked

- paper-a/benchmark_chooses_the_winner.tex and the rendered PDF.
- paper-a/README.md, the root README.md, both Makefiles, generated tables, and the specialization figure.
- configs/paper_a_sft.yaml.
- All Paper A manifest, audit, lock, training, scoring, and analysis code.
- LOCK.json, all six local manifests, audit outputs, 20 run metadata files, scores.parquet, score metadata, analysis JSON/CSV/TeX outputs, and the committed paper copies.
- Current tests, a tracked-only clean archive, the cached analysis path, the synthetic analysis self-test, and a clean Tectonic build.
- Primary-source related work through July 2026.

Severity meanings:

- **P0 / blocker:** invalidates the present claim-bearing evidence chain or requires a new evaluation/rerun.
- **P1 / high:** must be fixed before submission; may not require retraining if the corrected artifacts remain unchanged.
- **P2 / medium:** reporting, reproducibility, or methodological defect that materially weakens the paper.
- **P3 / minor:** editorial/build issue.

## Original legacy-artifact baseline that checked out

These were the coherent parts of the v1 artifact found during the original audit and preserved by the repair:

- The local test suite passes 30/30 when the ignored manifests are present.
- The committed score table contains exactly 3,308 rows for each of 24 bundles, for 79,392 total rows.
- It has no duplicate bundle/sample keys, missing locked cells, non-finite raw scores, or current identity drift.
- Every current score row joins one-to-one to the local manifest on sample ID; gold label, source, split, content hash, and family ID agree.
- score_raw exactly equals unsafe_logit minus safe_logit, and probability_raw recomputes exactly.
- All 24 current temperature fits report success and all current thresholds report status ok.
- The canonical average-precision implementation is sklearn-backed and tie-aware.
- The Clopper-Pearson calculation and threshold candidate search are arithmetically correct for independent pooled calibration rows.
- The local manifest files match the hashes and row counts recorded in the lock.
- The lock self-hash recomputes exactly. The defect is that no final stage verifies it and its source provenance is unrecoverable.
- A clean cached analysis reproduces the committed metrics and claim JSON semantically.
- The manuscript compiles to an eight-page PDF with no undefined citations or references.

These checks establish that the committed score table is coherent. They do not establish that the data, code, adapters, and scores are cryptographically bound into the chain claimed by the paper.

## P0 publication blockers

### P0-1. The locked revision cannot reconstruct the executed Paper A code

**Evidence**

- artifacts/paper_a_sft/LOCK.json records Git SHA 1d00f9dc0e9c7b7a567299550d0d08b8ce01f8c8, git_tracked_dirty=true, one tracked dirty file, sixteen untracked entries, and dirty_state_policy=recorded_not_enforced.
- The same SHA is repeated in all 20 run metadata files.
- That commit does not contain configs/paper_a_sft.yaml, any focused Paper A experiment script, or guard_research/metrics.py. Those files first enter history in the later Paper A foundation commit.
- No patch, source bundle, or per-source-file hashes are stored in the lock.
- Makefile lines 33-34 do not pass --require-clean. lock_paper_a_sft.py lines 147-151 enforce cleanliness only when that optional flag is supplied.

**Why this matters**

The lock identifies neither the exact training code nor the exact scoring/analysis code. A reviewer cannot reconstruct the executed source from the recorded revision, and a later commit cannot prove that it is byte-identical to the dirty/untracked code used for the run. This defeats the central “auditable evidence chain” claim in README.md lines 102-119 and manuscript lines 100 and 367.

**Required change**

1. Put every execution input in a clean source commit.
2. Record the clean commit plus hashes of every execution file, environment/container image, prompt implementation, and dependency lock.
3. Make lock creation fail by default on any tracked or untracked execution-relevant file.
4. Add one central verify_lock function and call it before training, scoring, and analysis.
5. Rerun at least scoring and analysis under the clean lock. If the exact adapters cannot be proven to have been trained by source-identical code, retrain all 20 cells.

### P0-2. The lock is informational rather than an enforced gate

**Evidence**

- experiments/paper_a_common.py lines 473-479 only check that a lock file exists and then parse JSON.
- experiments/lock_paper_a_sft.py lines 116-145 record missing manifests, audit, power report, and tokenizer probes rather than requiring them.
- A lock can be created successfully with audit=null, train_manifest_sha256=null, all six split files missing, and zero tokenizer probes.
- Training records the observed train-manifest hash but does not compare it with the locked hash before training.
- Evaluation fingerprints the scoring manifests but does not compare those fingerprints with LOCK.json before scoring.
- Analysis reads a lock and a Parquet file directly; it does not read scores/metadata.json, validate the lock self-hash, compare the score metadata lock hash, or verify a score-file hash.
- scores/metadata.json itself does not record the SHA-256 of scores.parquet, and results.json does not record the score or score-metadata hash.
- The current config byte hash is 2e7c7bf1..., while LOCK.json records 076951ca.... The parsed YAML object is unchanged because the only difference is a comment, but no verifier distinguishes semantic from byte identity.

**Why this matters**

The paper says final stages refuse mismatched inputs. They do not. A present-but-invalid lock is accepted, and arbitrary scores can be analyzed under it. The current chain is a collection of provenance fields, not an enforced chain of custody.

**Required change**

- Require all lock prerequisites and an audit PASS.
- Verify both canonical-object and raw-byte config hashes, with an explicit policy for comment-only drift.
- Verify every manifest hash and expected row count.
- Verify the audit hash and its hard-assertion status.
- Bind training metadata to the lock and recompute adapter hashes.
- Bind score metadata to the lock, manifests, adapters, schema, and a SHA-256 of the combined Parquet.
- Bind analysis outputs to both the lock hash and score hash.
- Fail closed on every missing, unexpected, duplicated, or non-finite item.

### P0-3. Analysis accepts incomplete evidence and can still pass all gates

**Evidence**

- experiments/analyze_paper_a_sft.py lines 55-97 derive the available benchmarks from the input table.
- Line 536 silently restricts model_keys to models found in the input.
- There is no assertion for four models, five SFT seeds, seven two-class benchmarks, two stress sets, 24 bundles, 3,308 samples per bundle, or 79,392 rows.
- There is no schema, uniqueness, gold-label parity, prompt hash, adapter hash, or finite-value check.
- _sign_stable at lines 279-283 drops NaNs. Empty leave-one-out complements therefore pass vacuously.

**Executed counterexample**

An input restricted to Qwen3-4B and one transfer source still emitted:

    models accepted: [qwen3_4b]
    transfer benchmarks: [wildjailbreak]
    leave-one-base result: NaN; sign_stable=True
    leave-one-benchmark result: NaN; sign_stable=True
    gate_a=True; gate_b=True; specialization=True

A second run with every Qwen3-4B row removed also exited successfully and analyzed a three-model panel.

**Why this matters**

The core claim gate can turn missing evidence into a successful claim. This is a direct correctness failure, not merely missing validation.

**Required change**

Add a pre-analysis validator that asserts the exact locked cross-product:

- expected models, conditions, seeds, splits, sources, and sample IDs;
- exactly one base row and one row for every SFT seed per model/sample;
- exact schema and unique composite key;
- identical gold/source/split/content/family identity across bundles;
- finite logits, scores, probabilities, and valid predictions;
- prompt/model/adapter hashes equal locked metadata;
- nonempty two-class benchmark cells and nonempty leave-one-out complements.

In precision mode, a missing cell must produce a hard error, never an inconclusive or passing claim.

### P0-4. The WildGuard and WildJailbreak cohorts were not built by the declared rule

**Evidence**

- prepare_paper_a_manifests.py lines 199-216 load an ignored legacy frozen cache and replace upstream row identifiers with frozen/{key}/posN.
- Lines 375-384 then apply the claimed hash ranking to this already selected subset, not to the pinned upstream candidate pool.
- Reconstructing the declared ranking from the pinned upstream revisions gives:

    WildGuard expected 800 intersect frozen 800: 387; 413 rows differ
    WildJailbreak expected 420 intersect frozen 420: 228; 192 rows differ

- legacy/experiments/guard_eval_pipeline.py lines 35-37 built those frozen subsets using random.Random(seed=7), not the Paper A hash-ranking rule.
- The same frozen transfer rows were loaded and scored as “novel tracking” during prior HPO in legacy/experiments/hpo_guard.py lines 33-38 and 98-103.
- The HPO objective returned represented-development AP, not novel AP, so the code does not prove direct numerical optimization on transfer. The HPO path logs 48/48 trials with a novel-transfer metric and loads the current frozen-cache path, which is strong evidence of repeated inspection; the logs do not hash the cache bytes, so exact byte identity is not cryptographically proven.
- The manuscript acknowledges that benchmarks were previously inspected, but still calls the decision rules preregistered and the final scoring confirmatory.

**Why this matters**

The current study population is not the one specified by the config and manuscript. The ignored cache also lacks authoritative upstream IDs, so the selected gated rows cannot be independently reconstructed from the released artifacts. Repeated prior scoring creates an adaptive-analysis risk and rules out “untouched” or prospective-confirmatory language.

**Required change**

The defensible option is to:

1. Rebuild the complete candidate pools from pinned upstream revisions.
2. Preserve authoritative upstream identifiers.
3. Apply the declared hash-ranking rule once.
4. Recompute exact and family overlap before any new scoring.
5. Freeze and publish a text-free identifier/hash manifest.
6. Retrain if the corrected evaluation set changes training-side overlap removals or family assignments.
7. Score the corrected transfer cohort once under a new clean lock. The declared hash-ranked rebuild retains 615 of 1,220 previously inspected WildGuard/WildJailbreak rows, so a prospective confirmatory claim additionally needs a genuinely uninspected cohort or benchmark.

If no rerun is feasible, describe the legacy frozen rows and seed-7 selection exactly, label the study retrospective/precision-focused, and drop preregistration and confirmatory claims.

### P0-5. The manuscript reports the wrong primary point estimate

**Evidence**

- The manuscript defines the fixed-panel point estimator in lines 253-267 as the arithmetic mean of the four observed checkpoint deltas.
- experiments/analyze_paper_a_sft.py lines 116-129 compute that estimator correctly.
- The committed results contain:

    observed represented point estimate: +0.3326732237
    observed transfer point estimate:    -0.0503344458

- The bootstrap summary at lines 195-205 separately computes the mean of the 10,000 bootstrap replicates:

    bootstrap mean, represented: +0.3248815782
    bootstrap mean, transfer:    -0.0498049412

- claim_checks and the fixed-panel Table 3 row use the bootstrap mean, not the point estimate.
- Per-checkpoint rows use observed deltas, so Table 3 mixes estimators. Averaging the four displayed represented deltas gives about +0.333, not the displayed +0.325 aggregate.
- The specialization figure uses the observed point estimate, so its black fixed-panel marker disagrees with the table and prose.
- The wrong +0.325 value appears in the abstract, Results, conclusion, root README, paper table, and claim JSON.

**Why this matters**

This is a direct numerical error in the headline estimand. It does not reverse the current sign, but it makes the paper internally inconsistent and invalidates the “no number is hand-transcribed” claim.

**Required change**

Use the observed estimator as the point:

    represented: +0.333
    transfer:    -0.050

Retain the bootstrap distribution only for uncertainty. Regenerate every narrative macro/table from a machine-readable result macro file and add a test that the aggregate equals the mean of the four checkpoint deltas.

### P0-6. The paper violates its locked precision-focused analysis mode

**Evidence**

- LOCK.json records analysis_mode=precision_focused.
- power_report and seed_count_decision are null.
- claim_checks.json explicitly says estimation-only mode and “no formal specialization rejection is claimed.”
- docs/paper-a-minimal-refactor-plan.md lines 1377-1380 require estimation language and prohibit formal rejection claims in this mode.
- The manuscript nevertheless says “both intersection-union gates therefore hold,” calls Qwen2.5 “statistically unchanged,” says SFT “improved” and “reduced” AP, calls the trade-off “robust,” and presents a “final gated statement.”
- The root README shows checkmarks for Gate A and Gate B and says the claims are decided by intersection-union gates.
- claim_checks still emits passed=true, specialization=true, and Holm reject=true objects in precision mode.
- lock creation permits powered_confirmatory without requiring a power report.

**Why this matters**

The paper’s inferential language contradicts the locked governance rule. The local lock was made after pilot results and public benchmark inspection, so “preregistered” is also unsupported. At most, the rules were prespecified before this final scoring pass.

**Required change**

- In precision-focused mode, rename passed to descriptive_criterion_met or omit it.
- Do not emit rejection decisions or Holm results.
- Replace “preregistered” with “prespecified before the final scoring pass.”
- Replace “improved,” “reduced,” “statistically unchanged,” “established,” and “robust” with estimate-and-interval language.
- If a confirmatory paper is desired, create a prospective protocol, power/precision report, clean lock, and genuinely untouched evaluation before rerunning.

## P1 high-priority code and data defects

### P1-1. Authoritative pair/family dependencies are not implemented

**Evidence**

- The manuscript says rows sharing an upstream family, conversation, pair, or scenario ID are connected.
- make_row does not store a dedicated upstream_family_id.
- load_hf_source embeds an upstream ID inside a split-qualified source_row_id.
- The family builder at prepare_paper_a_manifests.py lines 434-445 groups on the entire source_row_id, so distinct rows almost never share a key.
- The audit reports n_upstream_edges=0 over 9,789 candidate rows.
- The current transfer manifest retains 36 JailbreakBench harmful/benign matched indexes; zero share a family.
- Across eight direct XSTest contrast blocks it retains 58 safe/unsafe pairs; zero share a family. The earlier count of 45 captured only a subset of the block-specific relationships.

**Impact**

The family bootstrap treats known paired items as independent, contrary to the stated dependence model. This can make uncertainty too narrow and makes the paper’s claim that one family weight preserves cross-label and cross-dataset dependence false.

**Required change**

Preserve a source-independent upstream_family_id, define source-specific extraction rules, connect those IDs before MinHash clustering, and rerun manifests and uncertainty. Add fixtures for JailbreakBench and XSTest pairs.

### P1-2. Calibration and ID are not globally family-disjoint

**Evidence**

- prepare_paper_a_manifests.py lines 515-539 calls split_calibration_id separately for each source.
- paper_a_manifest_lib.py lines 311-340 hashes the assignment with the source name.
- Two global near-duplicate families currently span calibration and ID: four calibration rows and six ID rows.
- The audit only tests train families against the union of evaluation families; it does not test calibration against ID.

**Impact**

Primary raw-score AP is unaffected, but temperature/threshold fitting is not fully held back from represented-source ID relatives. RQ4’s “held-back calibration” interpretation is compromised.

**Required change**

Assign globally connected components atomically to calibration or ID across all represented sources, hard-assert no shared global family, and recompute all thresholded metrics.

### P1-3. Left truncation can remove the classifier instruction

**Evidence**

- Training and evaluation set tokenizer.truncation_side=left after rendering the full system/user chat.
- Long examples are then truncated to 1,024 tokens from the left.
- For the longest inspected Qwen ID row, the full rendering is 2,277 tokens and the 1,024-token scored sequence no longer contains the system prompt.
- The score table records 21-25 truncated examples per model, representing 26 unique calibration/ID prompts. Most are Jailbreak-Classification rows.
- The training manifest also contains long examples; 16 Qwen training rows exceed the limit.

**Impact**

Those examples do not use the claimed prompt interface. Because the adapted model was trained on similarly truncated examples while the base was not trained for the task, this can disproportionately affect the represented-source base-to-SFT contrast.

**Required change**

Budget and truncate user content before rendering, then assert that the complete system message, role delimiters, and assistant-generation prefix remain. Report truncation by source/label and add exclusion/long-context sensitivity.

### P1-4. The hard audit can report PASS despite family leakage

**Evidence**

- audit_paper_a_splits.py computes schema completeness, join validity, role validity, and train/eval family sharing.
- Its hard-assertion list at lines 272-284 excludes family disjointness, unique sample IDs, recomputed content-hash equality, role validation, and locked count/balance checks.
- Injecting one shared train/eval family caused the audit to print train/eval shared families: 1 and still report ALL HARD ASSERTIONS PASS.

**Required change**

Promote every integrity condition to a hard assertion, including:

- exact schema and types;
- recomputed content hashes;
- unique sample IDs;
- exact per-split/source/label counts;
- allowed source roles;
- train/eval family disjointness;
- calibration/ID family disjointness;
- expected known-conflict dispositions;
- resolved licenses and redistribution policy.

The independent audit should recompute facts rather than trust stored fields.

### P1-5. Training and evaluation trust stale or mutable metadata

**Evidence**

- Training accepts --max-steps even though the recipe is described as authoritative.
- run metadata stores the locked recipe even when max steps are overridden.
- Existing completed cells are skipped based on status plus adapter presence without validating them against the current lock.
- validate-runs does not verify global steps, seed, model revision, full recipe, lock hash, or observed prompt-template hash.
- Evaluation completeness trusts status=completed and the adapter hash written in metadata. It does not recompute the adapter hash before scoring.
- lock_model_panel drops locked dtype, trust_remote_code, and attention implementation.
- Training/evaluation then hard-code trust_remote_code=True, even though the lock records false for three of four models.
- Evaluation’s dtype CLI can override the locked dtype.

**Current-artifact note**

The copied current run metadata reports all 20 cells at 300 steps. This finding is a fail-open pipeline defect, not proof that the current runs used the wrong step count.

**Required change**

Final-mode runners must reject all recipe overrides, recompute adapter hashes, validate every run field against the lock, and use all locked runtime settings. Development overrides need a separate non-final artifact namespace.

### P1-6. Residual near-duplicate sensitivity is underreported

The audit reports six train/evaluation pairs at estimated MinHash similarity at least 0.80, zero at 0.85, and zero at 0.90. The 0.85 threshold is prespecified, so zero at 0.85 is not itself a failure. However:

- the six 0.80 pairs are not listed or adjudicated;
- sensitivity is based on probabilistic LSH candidate retrieval;
- the paper and README use strong “decontaminated corpus” language.

Publish the six pairs by identifiers/hashes, manually adjudicate them before publication, and call the corpus “audited at the prespecified 0.85 MinHash threshold” rather than unqualified decontaminated.

## P1 statistical and reporting defects

### P1-7. Table 3 labels two-sided intervals as one-sided

The table generator prints ci95_two_sided endpoints at analyze_paper_a_sft.py lines 469-477. Manuscript lines 282 and 289 call those brackets one-sided 95% intervals. For the represented aggregate:

- displayed bracket: [0.2718, 0.3793], the 2.5th/97.5th percentiles;
- actual one-sided lower bound: 0.2805, the 5th percentile.

Label the brackets “two-sided 95% bootstrap interval” and display a separate one-sided bound only if the governance mode permits a gate.

### P1-8. Bootstrap crossing fractions are not valid p-values

claim_checks treats the proportion of ordinary bootstrap replicates crossing zero as one-sided p-values, applies Holm, and can report p=0.0. The ordinary bootstrap distribution estimates sampling variation around the observed statistic; its crossing fraction is not automatically a null-calibrated hypothesis-test p-value. With 10,000 replicates, zero crossings also means less than the simulation resolution, not mathematical zero.

Remove these p-values and Holm decisions in precision mode. If formal testing is later required, specify and validate a null-resampling/test-inversion method prospectively.

### P1-9. RQ4 and stress reporting omits material results

The manuscript promises base/SFT TPR, paired delta TPR, and realized-FPR intervals. Its lower-table caption also describes stress diagnostics without making their base/SFT comparison equally explicit. The generated lower table includes:

- SFT TPR only;
- base and SFT FPR without intervals;
- SFT OR-Bench FPR only;
- SFT HarmBench recall only.

The omitted current stress comparison is important:

    HarmBench recall: base 0.78375 -> SFT 0.57450
    OR-Bench benign FPR: base 0.126875 -> SFT 0.099625

The HarmBench loss is counterevidence to a simple “adaptation improves guarding” story and belongs in Results/Discussion, not only a JSON artifact. Report base, SFT, paired delta, uncertainty, sample size, and aggregation for every secondary metric.

### P1-10. The paper conflates benchmark agreement with transferable safety competence

The transfer macro-average mixes different native policies: harm, jailbreak attempt/type, contrast prompts, and prompt-harm annotations. The same normalized prompt even occurs as unsafe in represented Jailbreak-Classification ID and safe in WildJailbreak transfer.

Therefore, a negative transfer delta can reflect policy/label disagreement rather than degradation of one coherent latent safety capability. The manuscript already acknowledges heterogeneous policies, but its introduction and conclusion still use “competence,” “capability,” and “general moderation upgrade.”

Use “discrimination/agreement under these benchmark-specific binary mappings.” Publish a policy crosswalk and cross-benchmark contradictory-label inventory. Do not interpret the macro as a single safety construct.

### P1-11. The declared appendix is absent

Manuscript line 367 says the appendix contains:

- full five-seed values;
- full per-source tables;
- calibration diagnostics;
- per-source and macro FPR with family-weighted bounds;
- model/tokenizer/prompt hashes;
- implementation and hardware details.

None of those tables appears in the appendix. The text also claims a power/precision report hash even though power_report is null. Its prose renders the literal “[RESULT]” macro name while discussing placeholders; this is an editorial artifact, not a missing numeric result. Run metadata records the GPU model, device, peak memory, and wall time, but not the complete platform needed for reproduction, such as OS, driver, runtime CUDA, container identity, Accelerate, and TRL versions.

Either add the promised appendix or change every cross-reference to the released artifact files. Do not claim a family-weighted FPR bound: the current threshold uses pooled row-level Clopper-Pearson.

## P1 reproducibility and release defects

### P1-12. A fresh clone does not contain the released manifests

- .gitignore lines 30-34 say text-free manifest.json remains tracked, but line 46 ignores the entire manifests directory.
- git contains no artifacts/paper_a_sft/manifests files.
- The manuscript says the released artifact tree contains frozen manifests or identifiers.
- In a tracked-only archive, pytest reports 13 passed and 17 skipped because every manifest test calls pytest.skip when inputs are absent.
- README.md promises 30 tests.

Release text-free manifests with upstream IDs, revisions, hashes, labels, family IDs, split assignments, and dispositions. Raw third-party text can remain excluded. Required release-integrity tests must fail, not skip, when those artifacts are absent.

### P1-13. Clean manifest construction requires an ignored legacy file

prepare_paper_a_manifests.py unconditionally opens data/frozen_eval_rows.json. That file is ignored and absent in a fresh checkout. The README claims gated datasets fall back to Hugging Face when the cache is absent, but the program fails before any fallback decision.

Make the source mode explicit. Either provide a deterministic identifier-only cache plus resolver, or implement a real pinned-Hugging-Face fallback. Preserve upstream IDs in both paths and verify that both resolve the same rows.

### P1-14. The documented six-step scratch path blocks at lock creation

The repository ships LOCK.json. A fresh tracked archive running the documented make lock exits because lock_paper_a_sft.py refuses to overwrite it and the Makefile provides neither --force nor a verify/rotate workflow.

Provide separate make verify-lock and make relock targets. A full-from-scratch target should write to a new run root or require an explicit destructive flag.

### P1-15. Cached analysis is not connected to the manuscript

- make repro writes only artifacts/paper_a_sft/analysis.
- paper-a/Makefile consumes paper-local table and figure copies.
- No target copies or verifies those files.
- The paper-local primary table already differs from the canonical generated table because model labels were manually prettified.
- Numeric prose in the abstract, Results, and conclusion is literal TeX text.
- After a clean full analysis, make paper can report “Nothing to be done.”

The current table-4 and figure copies happen to match their canonical outputs, but this is convention, not enforcement.

Generate a TeX macro file for every narrative number, generate human-readable model labels in the canonical table producer, and make the paper depend on a sync-or-diff check. CI should fail on any divergence.

### P1-16. The exact-environment claim is false

requirements.txt and LOCK.json describe Python 3.14-era/current analysis dependencies, while every run metadata file and score metadata record the GPU run as:

    Python 3.10.12
    NumPy 2.2.6
    pandas 2.3.3
    pyarrow 25.0.0
    scikit-learn 1.7.2
    Torch 2.9.1+cu129
    Transformers 4.56.2

The package declares Python >=3.11, so the recorded result-producing environment is not even supported by the declared project metadata. The run metadata also omits CUDA, driver, OS, accelerate, and TRL versions.

Publish separate locked training/scoring and CPU-analysis environments, ideally containers with immutable digests.

### P1-17. The recommended install changes the MinHash backend

README.md recommends pip install -e ".[all]". pyproject.toml makes all include datasketch. The locked audit records minhash_backend=numpy_fallback, while guard_research/provenance.py dynamically switches to datasketch when importable and warns that signatures from the two backends must not be mixed.

Thus the recommended install can rebuild different family signatures and manifests from the locked study. Pin one implementation and algorithm version in config/lock; never select a data-defining algorithm from optional package availability.

## P2 methodological limitations and claim narrowing

### P2-1. Uncertainty is conditional on fixed datasets, not “over evaluation datasets”

The bootstrap resamples family weights within seven named datasets. It does not resample dataset identities. The abstract should say uncertainty over training seeds and evaluation families/rows conditional on these fixed benchmarks.

The four checkpoint identities are also fixed by design, so no result generalizes statistically to model families. The paper mostly says this correctly; retain that discipline everywhere.

### P2-2. Five seeds and fixed data order limit the training-uncertainty claim

Seed resampling uses only five observed runs per checkpoint, and the data order is fixed. The run variation therefore covers initialization/dropout and other nondeterminism, not sampling of training examples or example order. Describe it precisely and avoid strong tail-probability language.

### P2-3. AP applies to constructed balanced subsets, not native prevalence

AP is prevalence-dependent. The study balances and subsamples every primary benchmark, including the legacy-selected transfer cohorts. The limitation section notes this, but the headline still reads like native benchmark performance.

Add full/native-prevalence and AUROC sensitivity where licenses permit, and state “on the constructed balanced subsets” in the main result.

### P2-4. Calibration target and reported FPR use different aggregation

Threshold selection constrains pooled calibration-negative FPR, while the headline reports a benchmark-macro transfer FPR. Those are different estimands; the current base transfer FPR is about 8.3% macro but about 4.4% pooled.

Define both, report both, and do not imply that one is the realized version of the other without qualification.

### P2-5. Clopper-Pearson is row-independent while calibration contains families

The “exact” Clopper-Pearson bound assumes independent Bernoulli negatives. Calibration contains multi-row near-duplicate families. Treat the bound as a pooled-row diagnostic or use a family-aware procedure; do not call it a production guarantee.

### P2-6. The design does not identify a mechanism

Base-versus-SFT identifies the change caused by this complete adaptation intervention on this panel. It does not distinguish:

- benchmark-source specialization;
- generic continued-training forgetting;
- label-policy mismatch;
- prompt-interface learning;
- long-input truncation effects.

A generic-SFT control, shuffled-label/control-task adaptation, or retention objective would help. Without controls, keep the result descriptive and avoid mechanistic language such as “convergence toward specialization.”

## Original manuscript and README errors (now corrected or disclosed)

These were confirmed against the original manuscript. The revised manuscript, generated artifacts, and READMEs now correct or explicitly disclose each item; they remain here as the audit trail that motivated the repair.

1. **“No stage reads evaluation labels before final scoring” is false.** Manifest label mapping, balancing, conflict checks, calibration/ID assignment, calibration fitting, and analysis read labels. The defensible sentence is that training never reads evaluation rows or labels.
2. **“Preregistered” is unsupported.** Use “prespecified before the final scoring pass.”
3. **Qwen2.5 is not “statistically unchanged.”** Its interval includes zero; that is inconclusive, not evidence of equivalence.
4. **Figure caption exception is false.** Qwen2.5 seed 43 has transfer delta +0.0368. The correct quadrant count is 14/20 specialization and 6/20 uniform gain; SmolLM2 contributes five of the six.
5. **Leave-one-out deletions are called resampling.** They are sensitivity analyses.
6. **README says leave-one-family-out sign-stable.** The code implements leave-one-checkpoint/base and leave-one-benchmark, not leave-one-family.
7. **“Successful jailbreak/injection” is unsupported.** Most source labels identify prompt type/harm, not measured attack success.
8. **One shared rendered prompt template is inaccurate.** The semantic instruction is shared, but the lock contains three distinct model-native serialization hashes.
9. **The data table says “see manifest” for licenses that remain unresolved.** Prompt-Injections, Jailbreak-Classification, and OR-Bench are still unknown-verify-before-lock.
10. **Dataset papers are not cited in the data table/text.** ToxicChat, XSTest, JailbreakBench, WildTeaming/WildGuard, OR-Bench, and HarmBench entries exist in refs.bib but are not cited where used.
11. **The auditable-release claim is false in Git.** Manifests are absent, and no upstream IDs for the frozen gated rows are released.
12. **The “in seconds” reproduction claim is false on the reviewed machine.** The exact 10,000-replicate clean analysis took more than 11 minutes at roughly one CPU core with no progress output.
13. **make test does not run 30 tests in a release checkout.** It runs 13 and skips 17.
14. **pip install -e . is not enough for all claimed CPU tasks.** Analysis always imports matplotlib, but matplotlib is optional; make test also needs optional pytest.

## Novelty and related-work assessment

The defensible novelty is incremental but useful: an artifact-backed, same-checkpoint, base-versus-LoRA-SFT, multi-seed estimate on a fixed compact-model panel. The revised paper no longer presents benchmark specialization or unseen-policy weakness as a new discovery.

The original review required direct positioning against the following work; the revised related-work section now provides it:

- [Domain Generalizable AI Guardrails with Augmented Policy Training](https://aclanthology.org/2026.acl-long.748/) already studies policy overfitting and unseen-policy guard generalization.
- [Defenses Against Prompt Attacks Learn Surface Heuristics](https://aclanthology.org/2026.acl-long.502/) directly reports narrow correlations and safe-input rejection after supervised defense training.
- [GuardBench](https://aclanthology.org/2024.emnlp-main.1022/) shows that general instruction models can rival specialized guard models.
- [FlexGuard](https://aclanthology.org/2026.acl-long.263/) is relevant to mixed strictness/policy consistency, not only calibration.
- [LoRA-Guard](https://aclanthology.org/2024.emnlp-main.656/) is directly relevant to parameter-efficient guardrail adaptation and retaining the base model's generative path.
- [On Guardrail Models’ Robustness to Mutations and Adversarial Attacks](https://aclanthology.org/2025.findings-emnlp.922/) is an important omitted robustness boundary.
- [COMPASS](https://aclanthology.org/2026.acl-long.2139/) and [Policy Compliance of User Requests](https://aclanthology.org/2026.acl-industry.21/) are optional, more tangential policy-conditioned alternatives rather than submission-critical omissions.

The revised related-work section now says plainly that prior work establishes policy overfitting, surface-heuristic defense behavior, and guard fragility, while this paper contributes the particular paired fixed-panel measurement and repaired release contract.

Bibliography metadata was also repaired: ExpGuard now has authors and the cited 2026 ACL entries include proceedings metadata.

## Reproducibility/build observations

### Verified post-repair commands and outcomes

| Check | Outcome |
|---|---|
| Full workspace pytest without raw v2 manifests installed | 76 passed, 22 skipped; every skip is an explicitly gated raw-v2 integration check |
| Corrected v2 manifest integration against the isolated build | 23 passed; together the two modes exercise all 98 current tests |
| Artifact-contract regression suite | 34 passed, including external absolute-root, redirected-input, root/file-symlink, exact-audit-schema, fixed-protocol, software-version, metadata-digest, and development-override cases |
| Corrected pinned manifest build and audit | 1,200 train / 451 calibration / 677 ID / 1,580 transfer / 400 OR-Bench / 200 HarmBench; all 24 hard assertions passed |
| Analysis self-test and legacy-lock verification | passed |
| Full legacy analysis, 10,000 family-bootstrap replicates | completed twice over 2,235 families |
| Exact repeat comparison | all 11 generated files byte-identical; runtime/source attestation binds SHA-256 digests for all 10 scientific outputs |
| Observed fixed-panel estimates | represented +0.3327, 95% two-sided interval [+0.2718, +0.3793]; transfer -0.0503, interval [-0.0760, -0.0250] |
| Paper-consumed generated files | all four TeX inputs and the figure pass byte-for-byte `cmp` against the analysis directory |
| Tectonic PDF build | exited 0; 9 pages; no undefined citations/references; every page visually inspected |
| HTML build | 5 tables, 2 figures, 32 numbered labels, 10 cross-references, zero leftover brackets, and zero nested equations |

The repaired analyzer is byte-idempotent in the tested environment. Two complete runs with the same frozen execution sources and inputs produced identical JSON, CSV, TeX, and PDF-figure bytes. `analysis_metadata.json` records the actual execution-source aggregate, runtime attestation, input digests, and hashes of the ten scientific outputs; volatile timestamps and PDF creation metadata are excluded from those deterministic artifacts.

Tectonic still emits numerous benign font/underfull-box warnings, a 1.166-point overfull vertical box at the table/reference boundary, and its internal “rerun seems needed” warning after six passes. The PDF nevertheless has resolved references and citations, Table 5 precedes the bibliography, and visual inspection found no clipping, overlap, or unreadable content. Page 9 contains the natural continuation of the bibliography, so its remaining whitespace is ordinary end-of-paper space rather than a blank-page defect. The HTML build's structural checks also pass; browser automation was unavailable in this environment, so HTML verification was static rather than interactive.

## Current worktree integration note

The `paper-html` edition is integrated under `paper-a/paper-html`: repository/paper roots, generated-input copying, PDF links, Make targets, and the HTML build all use the nested paths. Pandoc's dropped TikZ body is replaced in HTML by an accessible checked-in SVG, display equations are normalized for MathJax, and the older broad-study explorer remains only as an explicitly archived, unlinked page. Its generated third-party prompt samples are local-only and gitignored, and the previously tracked raw sample assets were removed. This resolves the worktree wiring/scope issue but does not change any scientific-result limitation.

## Required repair sequence

### Phase 1: stop claim-bearing release

- Mark the current manuscript and README as retrospective, precision-focused, and not yet publication-ready.
- Remove gate checkmarks, rejection language, “preregistered,” and the wrong +0.325 point estimate.
- Surface the HarmBench base-to-SFT degradation.

### Phase 2: rebuild data lineage

- Preserve upstream IDs and explicit pair/family keys.
- Reconstruct WildGuard/WildJailbreak from pinned upstream revisions under the declared rule.
- Globally split calibration/ID by family.
- Resolve or disclose the six 0.80 similarity pairs.
- Resolve source licenses.
- Publish text-free manifests and a policy/label crosswalk.

### Phase 3: make the pipeline fail closed

- Implement and test verify_lock.
- Require clean lock provenance and all prerequisite artifacts.
- Validate adapters and the complete score matrix.
- Bind score and analysis hashes.
- Make missing release artifacts fail tests.
- Pin one MinHash implementation.
- Preserve system prompts under truncation.

### Phase 4: rerun

- Retrain all 20 adapters because the instruction-preserving truncation fix changes training preprocessing even though the corrected data-lineage build retains the same 1,200 training rows.
- Score the pinned, hash-ranked corrected cohort once under the new v2 lock.
- Recompute family-aware intervals and secondary metrics.

### Phase 5: regenerate the paper

- Use the direct observed point estimator.
- Label interval types correctly.
- Generate every narrative number, table, and figure from one result source.
- Add the promised appendix.
- Update related work and dataset citations.
- Build from a clean clone in CI and require a zero-diff reproduction check.

## Acceptance criteria

Paper A is ready for another scientific review only when all of the following are true:

- [ ] A newly executed v2 lock points to a clean commit containing every execution file. (Requires the clean rerun.)
- [x] Every v2 final stage verifies the lock and fails on any mismatch.
- [ ] Corrected text-free manifests are tracked with the clean lock. (The tracked public snapshot is intentionally labeled legacy and not clean-rerun-compatible.)
- [ ] The scored transfer cohort matches the corrected deterministic construction. (Code and audit pass; new scores do not yet exist.)
- [x] Prior exposure is disclosed; a new untouched cohort is still required for prospective confirmation.
- [x] Upstream JailbreakBench and XSTest pairs produce nonzero, verified family edges in the corrected builder/audit.
- [x] Train/eval and calibration/ID assignment are globally family-disjoint in the corrected builder/audit.
- [x] Long-input processing preserves the complete classifier instruction.
- [x] The audit fails on schema, hash, role, count, and family defects.
- [x] Analysis rejects every incomplete or extra score matrix.
- [x] Adapters, caches, scores, and analysis are hash-bound under the v2 lock contract.
- [x] The reported represented point estimate is the observed fixed-panel mean, not the bootstrap mean.
- [x] Interval labels match the generated two-sided percentile quantiles.
- [x] Precision mode contains no formal rejection, p-value, Holm, or gate-pass claim.
- [x] RQ4 reports base, SFT, deltas, pooled and macro values, and both stress directions; it is explicitly a point-estimate deployment diagnostic rather than an uncertainty-bearing confirmatory result.
- [x] Claims are framed as benchmark-policy discrimination on constructed subsets.
- [ ] The exact clean training/scoring environment is locked and reproduced. (The release now documents legacy versus current environments, but no new container/lock has been executed.)
- [x] Fresh-clone tests exercise the tracked redacted release snapshot instead of skipping all artifact checks.
- [x] `make repro-legacy` updates and verifies every paper-consumed table, figure, and narrative macro.
- [x] The appendix contains the promised per-seed results and provenance/statistical details.
- [x] Repeated legacy analysis is byte-idempotent, and the PDF/HTML builds resolve all references. (Tectonic still emits a non-fatal internal `.bbl` rerun warning.)

## Suggested interim wording

If the author needs a truthful description of the current cached result before the rerun, use:

> On the current fixed four-checkpoint panel and previously constructed benchmark subsets, the observed mean base-to-SFT change was +0.333 macro AP on represented-source tests and -0.050 on dataset-held-out tests. The descriptive two-sided percentile-bootstrap intervals were [+0.272, +0.379] and [-0.076, -0.025]. These are conditional, precision-focused estimates rather than prospective confirmatory rejections; transfer effects were heterogeneous by checkpoint and benchmark.

This is now the manuscript's operative wording. The family graph, calibration/ID split, transfer-row construction, and long-input prompt handling are corrected in code, but the committed v1 scores predate those fixes; only a clean rerun can replace the legacy estimates.
