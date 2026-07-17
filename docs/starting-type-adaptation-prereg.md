# Preregistration skeleton — Starting-type adaptation study

Status: dev / nonfinal. This is a preregistration *skeleton*, not evidence that any hypothesis is
true. The normative source of truth is
[`artifacts/starting_type_adaptation_v1/protocol/primary_contract.json`](../artifacts/starting_type_adaptation_v1/protocol/primary_contract.json);
the full protocol is [`papers/unified-report/proposal.md`](../papers/unified-report/proposal.md)
(Sections 3, 5, 6, 9, 10, 16, 17). The authoring input is
[`configs/starting_type_adaptation_v1.yaml`](../configs/starting_type_adaptation_v1.yaml).
No field is claim-bearing until `artifacts/starting_type_adaptation_v1/LOCK.json` binds it, and no
claim-bearing GPU work begins until this preregistration and scope are hashed (proposal Phase 0).

## Design

A 2 x 3 **blocked comparative** design (not a randomized factorial):
`starting_type in {general, purpose_built}` x `condition in {unmodified, sft, kl_sft}`. For each
starting checkpoint the KL reference is that same unmodified checkpoint. Within-checkpoint deltas are
causal only for this locked recipe on that specific checkpoint; cross-panel differences are
descriptive fixed-panel interactions, never a "purpose-built causes X" claim.

## Panel

- General (4, revisions pinned from Paper A): `qwen25_15b`, `smollm2_17b`, `smollm3_3b`, `qwen3_4b`.
- Purpose-built (up to 6, revisions `PIN_AT_PHASE0`): `shieldgemma_2b`, `qwen3guard_gen_06b`,
  `qwen3guard_gen_4b`, `granite_guardian_31_2b`, `llama_guard_3_1b`, `wildguard_7b`.
- Minimum claim-bearing purpose-built panel: **>= 3 families** with complete U/SFT/KL-SFT cells;
  both Qwen sizes count as **one** family. Only Qwen and Granite are cleanly ungated (two families),
  so Phase 0 must secure and snapshot at least one gated guard (Gemma / Llama / Mistral-WildGuard)
  and predeclare replacements. Models may not be added or promoted after result inspection.

## Confirmatory hypotheses (predicates in `protocol/claim_registry.json`)

- **RQ1 (further specialization)** supported iff `LCB(H_gain) > 0` and `LCB(H_conc) > 0`, where
  `H_gain = mean_f Delta_SFT(f, represented)` and
  `H_conc = mean_f [Delta_SFT(f, represented) - Delta_SFT(f, held_out)]`.
- **RQ2 (KL preservation)** supported iff `LCB(H_preserve) > 0` and `LCB(H_cost) > -m`, where
  `H_preserve = mean_f P(f, held_out)`, `H_cost = mean_f P(f, represented)`, and `P` is the
  seed-paired KL-SFT minus SFT contrast.
- Non-inferiority margin **`m = 0.02` AP** (two units of the 0.01 tie band; fixed, not estimated;
  sensitivities at 0.01/0.03 cannot replace the primary decision).
- LCBs are one-sided **97.5%** bootstrap lower bounds; the two families RQ1/RQ2 are Bonferroni-
  controlled at familywise `alpha = 0.05`. LCBs are conditional on the fixed model panel (eval-row +
  seed uncertainty only); report per-family movements and leave-one-family-out beside every bound.
- RQ3-RQ6 (interaction `Gamma`, ranking fragility, native retention, domains) are descriptive /
  secondary unless the prereg adds an explicit multiplicity procedure before results.

## Locked coefficients and selection rules

- Primary KL arm `beta = 0.5`; `beta = 1.0` is a locked sensitivity with separate cardinality.
- **No test-driven beta selection and no post-hoc verdict wording.** Interpretation language is
  chosen by the first matching predicate in the claim registry's interpretation matrix.

## Metrics and resampling

- Primary: tie-aware non-interpolated AP; macro-AP over locked groups; paired AP differences.
- Hierarchical paired Poisson family bootstrap; model identities fixed; SFT/KL-SFT paired by seed
  and data order; unmodified treated as one fixed realization (`seed = -1`), not five replicates.
- Store raw margins (`z_unsafe - z_safe`), not only sigmoid probabilities.

## Missing-cell rule

Expected cardinality per row x checkpoint x primary contract is `1 U + 5 SFT + 5 KL-SFT`
(the `beta = 1.0` sensitivity is counted separately). A failed or missing cell blocks the fixed-panel
aggregate for its family; support cannot be declared from a favorable subset. Cells below the locked
minimum effective-positive count are labeled unstable and are not claim-bearing.

## Scoring contracts (see `protocol/scoring_contracts.json`, `protocol/policy_crosswalk.json`)

Primary = study-native (each checkpoint's top-level verdict); retention = official-native;
secondary = common `paper_a_safe_unsafe`. Views are never pooled. ShieldGemma trains under one locked
broad study policy; its four official policies stay an official-native evaluation panel only.

## Stop / promote gates (proposal Section 17)

- **Promote** only when all four general checkpoints and >= 3 purpose-built families have complete
  study-native U/SFT/KL-SFT matrices; every KL reference is the exact unmodified checkpoint;
  revisions/recipes/contracts/policies/exposure are locked; cardinality is exact; no malformed
  verdict maps to safe; contract views are separated; fixed-panel interactions are not presented as
  randomized causal effects; result language is predicate-selected; reproduction fails closed on
  pending/missing/stale/failed outputs; and an adversarial reviewer signs the claim registry.
- **Narrow or stop** the cross-type claim if fewer than three purpose-built families pass preflight,
  most purpose-built checkpoints need architecture-specific module changes, adapter-disable KL
  reference recovery fails, native parsing prevents a valid top-level or retention score, licenses
  prevent score-artifact publication, or only hard labels with large ties are obtainable. Fallback:
  keep checkpoint-level case studies and retention results; omit the pooled interaction.

## Exposure and generalization

Fill `protocol/model_benchmark_exposure_matrix.json` (one row per checkpoint x benchmark) with
reviewer sign-off before the lock. Do not call the adaptation-held-out panel "vendor-unseen." Without
a fresh sealed cohort, all generalization claims remain retrospective and estimation-only.
