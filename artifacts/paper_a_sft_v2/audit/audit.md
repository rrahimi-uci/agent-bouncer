# Paper A — data audit

Manifest dir: `artifacts/paper_a_sft_v2/manifests`  
Provenance: `guard_research.provenance` (MinHash backend `numpy-blake2b-mersenne31-v1`, 256 perms, 5-gram, J>=0.85)

## Hard assertions

| assertion | result |
|---|---|
| `config_hash_matches == true` | PASS |
| `source_mode == huggingface_pinned_revisions` | PASS |
| `manifest_file_hashes_and_counts_match == true` | PASS |
| `declared_row_and_class_counts_match == true` | PASS |
| `exact_schema_and_types_valid == true` | PASS |
| `stored_content_hashes_recompute == true` | PASS |
| `sample_id_and_split_names_valid == true` | PASS |
| `source_revision_and_license_metadata_match == true` | PASS |
| `sample_ids_unique == true` | PASS |
| `or_bench_train_count == 0` | PASS |
| `beavertails_train_count == 0` | PASS |
| `exact_train_vs_eval_overlap == 0` | PASS |
| `conflicting_label_overlap == 0` | PASS |
| `every_row_has_source_revision == true` | PASS |
| `every_row_has_content_hash == true` | PASS |
| `every_near_duplicate_candidate_has_disposition == true` | PASS |
| `train_eval_and_calibration_id_families_disjoint == true` | PASS |
| `calibration_vs_all_reported_test_families_disjoint == true` | PASS |
| `all_observed_similarity_and_upstream_edges_share_family_id == true` | PASS |
| `calibration_has_both_classes_and_at_least_10_negatives_per_source == true` | PASS |
| `source_roles_valid == true` | PASS |
| `licenses_resolved_and_conflicts_recorded == true` | PASS |
| `minhash_backend_and_algorithm_pinned == true` | PASS |
| `public_text_free_release_complete == true` | PASS |

**All hard assertions pass: True**

## Row and class counts

| manifest | rows | safe | unsafe |
|---|---:|---:|---:|
| train | 1200 | 600 | 600 |
| calibration | 451 | 215 | 236 |
| id_test | 677 | 364 | 313 |
| transfer_test | 1580 | 790 | 790 |
| orbench_safe_stress | 400 | 400 | 0 |
| harmbench_positive_stress | 200 | 0 | 200 |

### Per-source counts

- **train**: jailbreak_classification (safe 200, unsafe 200); prompt_injections (safe 200, unsafe 200); toxicchat (safe 200, unsafe 200)
- **calibration**: jailbreak_classification (safe 43, unsafe 60); prompt_injections (safe 16, unsafe 29); toxicchat (safe 156, unsafe 147)
- **id_test**: jailbreak_classification (safe 80, unsafe 79); prompt_injections (safe 40, unsafe 27); toxicchat (safe 244, unsafe 207)
- **transfer_test**: jailbreakbench (safe 60, unsafe 60); wildguardtest (safe 400, unsafe 400); wildjailbreak (safe 210, unsafe 210); xstest (safe 120, unsafe 120)
- **orbench_safe_stress**: orbench (safe 400, unsafe 0)
- **harmbench_positive_stress**: harmbench (safe 0, unsafe 200)

## Exact train↔eval overlap

- total exact overlap: **0** (same-label 0, conflicting 0)

| eval split | exact | same-label | conflicting |
|---|---:|---:|---:|
| calibration | 0 | 0 | 0 |
| id_test | 0 | 0 | 0 |
| transfer_test | 0 | 0 | 0 |
| orbench_safe_stress | 0 | 0 | 0 |
| harmbench_positive_stress | 0 | 0 | 0 |

## Near-duplicate sensitivity (char-5gram MinHash)

| threshold | total pairs | cross train↔eval | within-train | within-eval |
|---|---:|---:|---:|---:|
| 0.80 | 113 | 6 | 35 | 72 |
| 0.85 | 72 | 0 | 21 | 51 |
| 0.90 | 48 | 0 | 15 | 33 |

> 0.85 is the prespecified candidate-generation threshold; the builder removes the train-side of every cross-split component at >=0.85, so cross_train_eval_pairs at 0.85 must be 0. Counts at 0.80/0.90 are reported for sensitivity only; dispositions are not changed after final scores are viewed.

## Family validation

- total families: 4350; train families 1180; eval families 3170
- **train↔eval shared families: 0** (0 expected: cross-split near-dup train members are removed)

## Disposition coverage

- exact overlaps removed at build: 60
- cross-split near-dup components removed at build: 44
- known WildJailbreak overlaps removed: 4
- undisposed cross-split near-dup pairs at 0.85 in final manifests: 0

## License inventory

| source | license(s) | redistribution class(es) |
|---|---|---|
| harmbench | MIT (200) | permissive (200) |
| jailbreak_classification | Apache-2.0 (662) | permissive_with_notice (662) |
| jailbreakbench | MIT (120) | permissive (120) |
| orbench | CC-BY-4.0 (400) | permissive_with_attribution (400) |
| prompt_injections | Apache-2.0 (512) | permissive_with_notice (512) |
| toxicchat | CC-BY-NC-4.0 (1154) | noncommercial_reconstruct_only (1154) |
| wildguardtest | ODC-BY-AI2-gated (800) | gated_no_redistribution (800) |
| wildjailbreak | ODC-BY-AI2-gated (420) | gated_no_redistribution (420) |
| xstest | CC-BY-4.0 (240) | permissive_with_attribution (240) |

## Role and join validation

- role validation ok: True
- OR-Bench stress labels: ['safe']; HarmBench stress labels: ['unsafe']
- one-to-one joins (unique sample_id): True
- OR-Bench in train: 0; BeaverTails in train: 0
