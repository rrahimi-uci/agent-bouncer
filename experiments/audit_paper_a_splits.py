#!/usr/bin/env python
"""Audit the Paper A manifests (plan sec 6.7) and enforce the hard assertions.

Computes (over the written manifests, independently of the builder's bookkeeping):
  1. normalized exact-text overlap, train vs each eval manifest;
  2. conflicting-label exact overlap (reported separately from same-label);
  3. char-5gram MinHash near-duplicate sensitivity at 0.80 / 0.85 / 0.90;
  4. source-family membership validation (train/eval family disjointness);
  5. class and row counts;
  6. source-revision presence;
  7. license inventory;
  8. family/cluster construction validation;
  9. train/calibration/test role validation;
 10. proof that OR-Bench and BeaverTails counts in training are zero.

Hard assertions (EXIT NONZERO if any fails):
  or_bench_train_count == 0
  beavertails_train_count == 0
  exact_train_vs_eval_overlap == 0
  conflicting_label_overlap == 0
  every_row_has_source_revision == true
  every_row_has_content_hash == true
  every_near_duplicate_candidate_has_disposition == true

Usage:
  .venv/bin/python experiments/audit_paper_a_splits.py \
    --config configs/paper_a_sft.yaml \
    --manifest artifacts/paper_a_sft_v2/manifests/manifest.json \
    --out artifacts/paper_a_sft_v2/audit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_manifest_lib as L  # noqa: E402
import paper_a_common as C  # noqa: E402
import prepare_paper_a_manifests as P  # noqa: E402

TRAIN_SPLIT = L.SPLIT_TRAIN
EVAL_SPLITS = [L.SPLIT_CALIBRATION, L.SPLIT_ID, L.SPLIT_TRANSFER, L.SPLIT_ORBENCH, L.SPLIT_HARMBENCH]
REPRESENTED_SOURCES = {"toxicchat", "prompt_injections", "jailbreak_classification"}
TRANSFER_SOURCES = {"jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"}
FORBIDDEN_TRAIN_SOURCES = {"orbench", "or_bench", "or-bench", "beavertails", "beaver_tails"}

STEM_TO_SPLIT = {
    "train": L.SPLIT_TRAIN,
    "calibration": L.SPLIT_CALIBRATION,
    "id_test": L.SPLIT_ID,
    "transfer_test": L.SPLIT_TRANSFER,
    "orbench_safe_stress": L.SPLIT_ORBENCH,
    "harmbench_positive_stress": L.SPLIT_HARMBENCH,
}

# Exact expectations for the pinned revisions and deterministic target rules.
EXPECTED_TRAIN_SOURCE_LABEL = {
    ("toxicchat", "safe"): 200, ("toxicchat", "unsafe"): 200,
    ("prompt_injections", "safe"): 200, ("prompt_injections", "unsafe"): 200,
    ("jailbreak_classification", "safe"): 200,
    ("jailbreak_classification", "unsafe"): 200,
}
EXPECTED_REPRESENTED_UNION_SOURCE_LABEL = {
    ("toxicchat", "safe"): 400, ("toxicchat", "unsafe"): 354,
    ("prompt_injections", "safe"): 56, ("prompt_injections", "unsafe"): 56,
    ("jailbreak_classification", "safe"): 123,
    ("jailbreak_classification", "unsafe"): 139,
}
EXPECTED_TRANSFER_SOURCE_LABEL = {
    ("jailbreakbench", "safe"): 60, ("jailbreakbench", "unsafe"): 60,
    ("xstest", "safe"): 120, ("xstest", "unsafe"): 120,
    ("wildguardtest", "safe"): 400, ("wildguardtest", "unsafe"): 400,
    ("wildjailbreak", "safe"): 210, ("wildjailbreak", "unsafe"): 210,
}
NULLABLE_SCHEMA_FIELDS = {"upstream_family_id"}
PUBLIC_BANNED_CONTENT_KEYS = {
    "text", "raw_text", "norm_text", "normalized_text",
    "text_or_download_reference", "prompt", "response", "completion", "adversarial",
}
CONFIG_KEY_EMITTED = {
    "toxicchat": "toxicchat", "toxicchat_test": "toxicchat",
    "prompt_injections": "prompt_injections",
    "prompt_injections_test": "prompt_injections",
    "jailbreak_classification": "jailbreak_classification",
    "jailbreak_classification_test": "jailbreak_classification",
    "jailbreakbench": "jailbreakbench", "xstest": "xstest",
    "wildguardtest": "wildguardtest", "wildjailbreak": "wildjailbreak",
    "orbench_hard": "orbench", "harmbench": "harmbench",
}


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_banned_public_keys(value, path="$"):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in PUBLIC_BANNED_CONTENT_KEYS:
                found.append(f"{path}.{key}")
            found.extend(find_banned_public_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found.extend(find_banned_public_keys(item, f"{path}[{i}]"))
    return found


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_manifests(manifest_dir):
    out = {}
    for stem in L.MANIFEST_FILES:
        path = os.path.join(manifest_dir, f"{stem}.jsonl")
        out[stem] = load_jsonl(path)
    return out


def family_edge_mismatches(rows, edges):
    """Return observed similarity edges whose endpoints carry different families."""
    mismatches = []
    for edge in edges:
        left, right = int(edge[0]), int(edge[1])
        if rows[left].get("family_id") != rows[right].get("family_id"):
            mismatches.append({
                "left_sample_id": rows[left].get("sample_id"),
                "right_sample_id": rows[right].get("sample_id"),
                "left_family_id": rows[left].get("family_id"),
                "right_family_id": rows[right].get("family_id"),
            })
    return mismatches


def audit(config_path, manifest_json_path, out_dir, public_manifest_path=None):
    os.makedirs(out_dir, exist_ok=True)
    config = load_config(config_path)
    manifest_dir = os.path.dirname(manifest_json_path)
    manifest_meta = json.load(open(manifest_json_path))
    M = read_manifests(manifest_dir)

    all_rows = [r for rows in M.values() for r in rows]
    train_rows = M["train"]
    eval_stems = ["calibration", "id_test", "transfer_test",
                  "orbench_safe_stress", "harmbench_positive_stress"]
    eval_rows = [r for s in eval_stems for r in M[s]]

    report = {"manifest_dir": os.path.relpath(manifest_dir, _ROOT),
              "config_sha256": manifest_meta.get("config_sha256"),
              "provenance": manifest_meta.get("provenance")}
    report["manifest_index"] = {
        "path": os.path.relpath(manifest_json_path, _ROOT),
        "observed_sha256": L.sha256_of_file(manifest_json_path),
    }

    # The --config argument is authoritative, not decorative.
    expected_config_sha = L.sha256_of_obj(config)
    report["config_validation"] = {
        "expected_sha256": expected_config_sha,
        "manifest_sha256": manifest_meta.get("config_sha256"),
        "matches": expected_config_sha == manifest_meta.get("config_sha256"),
        "source_mode": manifest_meta.get("source_mode"),
        "pinned_hf_default": manifest_meta.get("source_mode") == "huggingface_pinned_revisions",
    }

    # Verify the manifest index's byte commitments and row counts.
    file_integrity = {}
    for stem, rows in M.items():
        path = os.path.join(manifest_dir, f"{stem}.jsonl")
        declared = manifest_meta.get("files", {}).get(stem, {})
        observed_sha = L.sha256_of_file(path)
        file_integrity[stem] = {
            "declared_sha256": declared.get("sha256"),
            "observed_sha256": observed_sha,
            "sha256_matches": declared.get("sha256") == observed_sha,
            "declared_rows": declared.get("n_rows"),
            "observed_rows": len(rows),
            "row_count_matches": declared.get("n_rows") == len(rows),
        }
    report["file_integrity"] = file_integrity

    # ---- 5. class + row counts ---------------------------------------------
    counts = {}
    for stem, rows in M.items():
        counts[stem] = {
            "n_rows": len(rows),
            "class_counts": dict(Counter(r["label"] for r in rows)),
            "per_source": {s: dict(Counter(r["label"] for r in rows if r["source"] == s))
                           for s in sorted({r["source"] for r in rows})},
        }
    report["counts"] = counts
    train_observed = Counter((r["source"], r["label"]) for r in train_rows)
    represented_observed = Counter(
        (r["source"], r["label"]) for r in M["calibration"] + M["id_test"])
    transfer_observed = Counter((r["source"], r["label"]) for r in M["transfer_test"])
    count_validation = {
        "train_source_label_exact": dict(train_observed) == EXPECTED_TRAIN_SOURCE_LABEL,
        "represented_union_source_label_exact": (
            dict(represented_observed) == EXPECTED_REPRESENTED_UNION_SOURCE_LABEL),
        "transfer_source_label_exact": dict(transfer_observed) == EXPECTED_TRANSFER_SOURCE_LABEL,
        "orbench_exact_400_safe": (
            len(M["orbench_safe_stress"]) == 400
            and {r["label"] for r in M["orbench_safe_stress"]} == {"safe"}),
        "harmbench_exact_200_unsafe": (
            len(M["harmbench_positive_stress"]) == 200
            and {r["label"] for r in M["harmbench_positive_stress"]} == {"unsafe"}),
    }
    count_validation["all_expected_counts_match"] = all(count_validation.values())
    report["count_validation"] = count_validation

    # ---- 6/? every row has revision + content hash + all schema fields -----
    missing_rev = [r.get("sample_id") for r in all_rows if not r.get("source_revision")]
    missing_hash = [r.get("sample_id") for r in all_rows if not r.get("content_sha256")]
    schema_missing = defaultdict(list)
    schema_exact_problems = []
    type_problems = []
    expected_fields = set(L.ROW_SCHEMA_FIELDS)
    for r in all_rows:
        for fld in L.ROW_SCHEMA_FIELDS:
            if fld not in r or (fld not in NULLABLE_SCHEMA_FIELDS and r.get(fld) in (None, "")):
                schema_missing[fld].append(r.get("sample_id"))
        if set(r) != expected_fields:
            schema_exact_problems.append({
                "sample_id": r.get("sample_id"),
                "missing": sorted(expected_fields - set(r)),
                "extra": sorted(set(r) - expected_fields),
            })
        if r.get("label") not in {"safe", "unsafe"} or r.get("gold") not in {0, 1}:
            type_problems.append(f"{r.get('sample_id')}: invalid label/gold")
        elif r["gold"] != L.to_gold(r["label"]):
            type_problems.append(f"{r.get('sample_id')}: label/gold disagree")
        for fld in ("content_sha256", "family_id"):
            if re.fullmatch(r"[0-9a-f]{64}", str(r.get(fld, ""))) is None:
                type_problems.append(f"{r.get('sample_id')}: malformed {fld}")
    report["schema_completeness"] = {
        "n_rows_total": len(all_rows),
        "rows_missing_source_revision": len(missing_rev),
        "rows_missing_content_sha256": len(missing_hash),
        "fields_with_missing_values": {k: len(v) for k, v in schema_missing.items()},
        "exact_schema_problems": schema_exact_problems[:50],
        "n_exact_schema_problems": len(schema_exact_problems),
        "type_problems": type_problems[:50],
        "n_type_problems": len(type_problems),
    }

    # content hash recomputation check (does stored hash match normalization?)
    hash_mismatch = [r["sample_id"] for r in all_rows
                     if r.get("content_sha256") != L.content_sha256(r["text_or_download_reference"])]
    report["schema_completeness"]["content_hash_recompute_mismatch"] = len(hash_mismatch)

    sample_id_mismatch = [r.get("sample_id") for r in all_rows
                          if r.get("sample_id") != f"{r.get('source')}::{r.get('source_row_id')}"]
    split_mismatch = []
    for stem, rows in M.items():
        expected_split = STEM_TO_SPLIT[stem]
        split_mismatch.extend(r.get("sample_id") for r in rows if r.get("split") != expected_split)
    report["schema_completeness"]["sample_id_formula_mismatch"] = len(sample_id_mismatch)
    report["schema_completeness"]["split_name_mismatch"] = len(split_mismatch)

    role_for_split = {
        L.SPLIT_TRAIN: "train", L.SPLIT_CALIBRATION: "represented_test",
        L.SPLIT_ID: "represented_test", L.SPLIT_TRANSFER: "transfer",
        L.SPLIT_ORBENCH: "stress_safe", L.SPLIT_HARMBENCH: "stress_unsafe",
    }
    expected_source = {}
    for key, spec in config["sources"].items():
        expected_source[(CONFIG_KEY_EMITTED[key], spec["role"])] = {
            "revision": spec["revision"],
            "license": L.resolved_license_metadata(spec),
        }
    source_metadata_problems = []
    for r in all_rows:
        expected = expected_source.get((r.get("source"), role_for_split.get(r.get("split"))))
        if expected is None:
            source_metadata_problems.append(f"{r.get('sample_id')}: unexpected source/role")
            continue
        if r.get("source_revision") != expected["revision"]:
            source_metadata_problems.append(f"{r.get('sample_id')}: source revision drift")
        if r.get("license_id") != expected["license"]["license_id"]:
            source_metadata_problems.append(f"{r.get('sample_id')}: license id drift")
        if r.get("redistribution_class") != expected["license"]["redistribution_class"]:
            source_metadata_problems.append(f"{r.get('sample_id')}: redistribution class drift")
    report["source_metadata_validation"] = {
        "problems": source_metadata_problems[:50],
        "n_problems": len(source_metadata_problems),
        "ok": len(source_metadata_problems) == 0,
    }

    # ---- one-to-one join check (unique sample_id across all manifests) -----
    sid_counts = Counter(r["sample_id"] for r in all_rows)
    dup_sids = [s for s, c in sid_counts.items() if c > 1]
    report["join_validation"] = {
        "n_rows": len(all_rows), "n_unique_sample_id": len(sid_counts),
        "duplicate_sample_ids": dup_sids[:20], "n_duplicate_sample_ids": len(dup_sids),
        "one_to_one": len(dup_sids) == 0,
    }

    # ---- 1/2. exact normalized-text overlap, train vs each eval ------------
    train_hash = defaultdict(list)   # content_sha256 -> list of train rows
    for r in train_rows:
        train_hash[r["content_sha256"]].append(r)
    overlap_by_eval = {}
    total_exact = 0
    total_conflict = 0
    total_samelabel = 0
    conflict_records = []
    for stem in eval_stems:
        same = conflict = 0
        recs = []
        for r in M[stem]:
            for tr in train_hash.get(r["content_sha256"], []):
                if tr["label"] == r["label"]:
                    same += 1
                else:
                    conflict += 1
                    conflict_records.append({
                        "content_sha256": r["content_sha256"],
                        "train": {"source": tr["source"], "label": tr["label"],
                                  "sample_id": tr["sample_id"]},
                        "eval": {"split": stem, "source": r["source"], "label": r["label"],
                                 "sample_id": r["sample_id"]}})
                recs.append(r["content_sha256"])
        overlap_by_eval[stem] = {"exact_overlap": len(recs),
                                 "same_label": same, "conflicting_label": conflict}
        total_exact += len(recs); total_samelabel += same; total_conflict += conflict
    report["exact_overlap"] = {
        "total_exact_train_vs_eval": total_exact,
        "total_same_label": total_samelabel,
        "total_conflicting_label": total_conflict,
        "by_eval": overlap_by_eval,
        "conflict_records": conflict_records[:50],
    }
    all_by_hash = defaultdict(list)
    for row in all_rows:
        all_by_hash[row["content_sha256"]].append(row)
    cross_source_label_conflicts = []
    for content_sha, rows in all_by_hash.items():
        if len({r["source"] for r in rows}) < 2 or len({r["label"] for r in rows}) < 2:
            continue
        cross_source_label_conflicts.append({
            "content_sha256": content_sha,
            "rows": [{"sample_id": r["sample_id"], "source": r["source"],
                      "split": r["split"], "label": r["label"],
                      "family_id": r["family_id"]} for r in rows],
            "one_global_family": len({r["family_id"] for r in rows}) == 1,
        })
    report["exact_overlap"]["cross_source_label_conflicts"] = cross_source_label_conflicts
    report["exact_overlap"]["n_cross_source_label_conflicts"] = len(
        cross_source_label_conflicts)

    # ---- 3. near-dup MinHash sensitivity 0.80/0.85/0.90 --------------------
    texts = [r["text_or_download_reference"] for r in all_rows]
    sides = ["train" if r["split"] == TRAIN_SPLIT else "eval" for r in all_rows]
    sigs = L.build_minhash_signatures(texts)
    cand = L.lsh_candidate_pairs(sigs)
    sens = {}
    cross_pairs_by_thr = {}
    edges_by_thr = {}
    for thr in (0.80, 0.85, 0.90):
        edges = L.edges_at_threshold(sigs, cand, thr)
        edges_by_thr[f"{thr:.2f}"] = edges
        cross = [(i, j, e) for (i, j, e) in edges if sides[i] != sides[j]]
        within_train = sum(1 for (i, j, _) in edges if sides[i] == sides[j] == "train")
        within_eval = sum(1 for (i, j, _) in edges if sides[i] == sides[j] == "eval")
        sens[f"{thr:.2f}"] = {
            "total_near_dup_pairs": len(edges),
            "cross_train_eval_pairs": len(cross),
            "within_train_pairs": within_train,
            "within_eval_pairs": within_eval,
        }
        cross_pairs_by_thr[f"{thr:.2f}"] = cross
    # detail for the cross pairs that remain at the 0.85 candidate threshold
    remaining_085 = [
        {"est_jaccard": round(e, 4),
         "a": {"split": all_rows[i]["split"], "source": all_rows[i]["source"],
               "sample_id": all_rows[i]["sample_id"]},
         "b": {"split": all_rows[j]["split"], "source": all_rows[j]["source"],
               "sample_id": all_rows[j]["sample_id"]}}
        for (i, j, e) in cross_pairs_by_thr["0.85"]]
    report["near_dup_sensitivity"] = {
        "candidate_pairs_generated": len(cand),
        "by_threshold": sens,
        "remaining_cross_split_at_0.85": remaining_085[:50],
        "note": ("0.85 is the prespecified candidate-generation threshold; the "
                 "builder removes the train-side of every cross-split component at "
                 ">=0.85, so cross_train_eval_pairs at 0.85 must be 0. Counts at "
                 "0.80/0.90 are reported for sensitivity only; dispositions are not "
                 "changed after final scores are viewed."),
    }

    # ---- 4/8. family validation --------------------------------------------
    train_fams = {r["family_id"] for r in train_rows}
    eval_fams = {r["family_id"] for r in eval_rows}
    shared_fams = sorted(train_fams & eval_fams)
    calibration_fams = {r["family_id"] for r in M["calibration"]}
    id_fams = {r["family_id"] for r in M["id_test"]}
    shared_cal_id_fams = sorted(calibration_fams & id_fams)
    reported_test_fams = {
        r["family_id"]
        for stem in ("id_test", "transfer_test", "orbench_safe_stress",
                     "harmbench_positive_stress")
        for r in M[stem]
    }
    shared_cal_reported_fams = sorted(calibration_fams & reported_test_fams)
    fam_all = Counter(r["family_id"] for r in all_rows)
    exact_family_mismatch = [
        content_sha for content_sha, rows in all_by_hash.items()
        if len(rows) > 1 and len({row["family_id"] for row in rows}) != 1
    ]
    minhash_family_mismatch = family_edge_mismatches(all_rows, edges_by_thr["0.85"])
    all_by_upstream = defaultdict(list)
    for row in all_rows:
        if row.get("upstream_family_id"):
            all_by_upstream[row["upstream_family_id"]].append(row)
    global_upstream_family_mismatch = [
        family_id for family_id, rows in all_by_upstream.items()
        if len(rows) > 1 and len({row["family_id"] for row in rows}) != 1
    ]

    transfer_by_upstream = defaultdict(list)
    missing_required_upstream = []
    for row in M["transfer_test"]:
        if row["source"] in {"jailbreakbench", "xstest"}:
            if not row.get("upstream_family_id"):
                missing_required_upstream.append(row["sample_id"])
            else:
                transfer_by_upstream[row["upstream_family_id"]].append(row)
    jbb_pairs = [rows for fid, rows in transfer_by_upstream.items()
                 if fid.startswith("jailbreakbench:")
                 and {r["label"] for r in rows} == {"safe", "unsafe"}]
    xstest_pairs = [rows for fid, rows in transfer_by_upstream.items()
                    if fid.startswith("xstest:direct-contrast:")
                    and {r["label"] for r in rows} == {"safe", "unsafe"}]
    upstream_family_mismatch = []
    for fid, rows in transfer_by_upstream.items():
        if len(rows) > 1 and len({r["family_id"] for r in rows}) != 1:
            upstream_family_mismatch.append(fid)
    report["family_validation"] = {
        "n_families_total": len(fam_all),
        "n_train_families": len(train_fams),
        "n_eval_families": len(eval_fams),
        "train_eval_shared_families": len(shared_fams),
        "shared_family_ids_sample": shared_fams[:20],
        "calibration_id_shared_families": len(shared_cal_id_fams),
        "calibration_id_shared_family_ids_sample": shared_cal_id_fams[:20],
        "calibration_reported_test_shared_families": len(shared_cal_reported_fams),
        "calibration_reported_test_shared_family_ids_sample": shared_cal_reported_fams[:20],
        "all_rows_have_family_id": all(r.get("family_id") for r in all_rows),
        "required_rows_missing_upstream_family_id": len(missing_required_upstream),
        "jailbreakbench_selected_direct_pairs": len(jbb_pairs),
        "xstest_selected_direct_pairs": len(xstest_pairs),
        "upstream_groups_split_across_family_ids": len(upstream_family_mismatch),
        "upstream_group_mismatch_sample": upstream_family_mismatch[:20],
        "exact_content_groups_split_across_family_ids": len(exact_family_mismatch),
        "exact_family_mismatch_sample": exact_family_mismatch[:20],
        "minhash_085_edges_split_across_family_ids": len(minhash_family_mismatch),
        "minhash_family_mismatch_sample": minhash_family_mismatch[:20],
        "global_upstream_groups_split_across_family_ids": len(
            global_upstream_family_mismatch),
        "global_upstream_group_mismatch_sample": global_upstream_family_mismatch[:20],
        "family_stats_from_build": manifest_meta.get("family_stats"),
    }

    # ---- 7. license inventory ----------------------------------------------
    lic = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        lic[r["source"]][r.get("license_id", "unknown")] += 1
    redist = defaultdict(lambda: defaultdict(int))
    for r in all_rows:
        redist[r["source"]][r.get("redistribution_class", "unknown")] += 1
    report["license_inventory"] = {s: dict(v) for s, v in lic.items()}
    report["redistribution_inventory"] = {s: dict(v) for s, v in redist.items()}
    unresolved_license_rows = [
        r["sample_id"] for r in all_rows
        if str(r.get("license_id", "unknown")).lower().startswith("unknown")]
    source_license_metadata_problems = []
    for key, spec in config["sources"].items():
        expected = L.resolved_license_metadata(spec)
        observed = manifest_meta.get("sources", {}).get(key, {}).get("license_metadata")
        if observed != expected:
            source_license_metadata_problems.append(key)
    prompt_meta = manifest_meta.get("sources", {}).get(
        "prompt_injections", {}).get("license_metadata", {})
    prompt_conflict_recorded = (
        prompt_meta.get("status") == "canonical_card_value_with_recorded_metadata_conflict"
        and prompt_meta.get("metadata_values", {}).get("card_top_level") == "Apache-2.0"
        and prompt_meta.get("metadata_values", {}).get("dataset_info_nested") == "CC-BY-4.0"
    )
    report["license_validation"] = {
        "unresolved_license_rows": len(unresolved_license_rows),
        "source_metadata_problems": source_license_metadata_problems,
        "prompt_injections_conflict_recorded": prompt_conflict_recorded,
        "ok": (not unresolved_license_rows and not source_license_metadata_problems
               and prompt_conflict_recorded),
    }

    # ---- 9. role validation ------------------------------------------------
    role_problems = []
    for r in train_rows:
        if r["source"] not in REPRESENTED_SOURCES:
            role_problems.append(f"train row from non-train source {r['source']}")
    for r in M["calibration"] + M["id_test"]:
        if r["source"] not in REPRESENTED_SOURCES:
            role_problems.append(f"cal/id row from non-represented source {r['source']}")
    for r in M["transfer_test"]:
        if r["source"] not in TRANSFER_SOURCES:
            role_problems.append(f"transfer row from non-transfer source {r['source']}")
    orbench_labels = set(r["label"] for r in M["orbench_safe_stress"])
    harmbench_labels = set(r["label"] for r in M["harmbench_positive_stress"])
    if orbench_labels != {"safe"}:
        role_problems.append(f"orbench stress not single-class safe: {orbench_labels}")
    if harmbench_labels != {"unsafe"}:
        role_problems.append(f"harmbench stress not single-class unsafe: {harmbench_labels}")
    report["role_validation"] = {"problems": role_problems[:50], "ok": len(role_problems) == 0,
                                 "orbench_labels": sorted(orbench_labels),
                                 "harmbench_labels": sorted(harmbench_labels)}

    # ---- 10. forbidden-source train counts ---------------------------------
    train_src_counts = Counter(r["source"] for r in train_rows)
    or_bench_train_count = sum(v for k, v in train_src_counts.items()
                               if k in ("orbench", "or_bench", "or-bench"))
    beavertails_train_count = sum(v for k, v in train_src_counts.items()
                                  if k in ("beavertails", "beaver_tails"))
    report["forbidden_train_sources"] = {
        "train_source_counts": dict(train_src_counts),
        "or_bench_train_count": or_bench_train_count,
        "beavertails_train_count": beavertails_train_count,
    }

    # ---- disposition coverage of near-dup candidates -----------------------
    removals = manifest_meta.get("removals", {})
    n_exact_removed = removals.get("exact_train_vs_eval", {}).get("count", 0)
    n_cross_removed = removals.get("cross_split_near_dup", {}).get("count", 0)
    cross_pairs_085 = sens["0.85"]["cross_train_eval_pairs"]
    every_candidate_disposed = (cross_pairs_085 == 0)
    report["disposition_coverage"] = {
        "build_exact_overlaps_removed": n_exact_removed,
        "build_cross_split_components_removed": n_cross_removed,
        "undisposed_cross_split_pairs_at_0.85_in_final_manifests": cross_pairs_085,
        "every_near_duplicate_candidate_has_disposition": every_candidate_disposed,
        "known_wildjailbreak_overlaps_in_build": manifest_meta.get(
            "known_wildjailbreak_overlaps", {}).get("count"),
    }

    expected_provenance = {
        "provenance_source": L.PROVENANCE_SOURCE,
        "minhash_backend": L.MINHASH_BACKEND,
        "minhash_algorithm_version": L.MINHASH_ALGORITHM_VERSION,
        "minhash_seed": L.MINHASH_SEED,
        "minhash_num_perm": L.NUM_PERM,
        "minhash_ngram": L.NGRAM,
        "minhash_jaccard_threshold": L.MINHASH_JACCARD_THRESHOLD,
        "lsh_bands": L.LSH_BANDS,
        "lsh_rows": L.LSH_ROWS,
    }
    observed_provenance = manifest_meta.get("provenance", {})
    provenance_mismatch = {
        key: {"expected": value, "observed": observed_provenance.get(key)}
        for key, value in expected_provenance.items()
        if observed_provenance.get(key) != value
    }
    report["provenance_validation"] = {
        "expected": expected_provenance,
        "mismatch": provenance_mismatch,
        "ok": not provenance_mismatch,
    }

    if public_manifest_path is None:
        public_dir = os.path.join(os.path.dirname(manifest_dir), "public_manifests")
        public_manifest_path = os.path.join(public_dir, "manifest.json")
    else:
        public_manifest_path = os.path.abspath(public_manifest_path)
        public_dir = os.path.dirname(public_manifest_path)
    public_problems = []
    public_conflict_inventory_ok = False
    public_manifest_sha256 = None
    if not os.path.exists(public_manifest_path):
        public_problems.append(f"missing {public_manifest_path}")
    else:
        public_meta = json.load(open(public_manifest_path, encoding="utf-8"))
        public_manifest_sha256 = L.sha256_of_file(public_manifest_path)
        public_problems.extend(find_banned_public_keys(public_meta))
        if public_meta.get("source_contract") != "pinned_hf_v2":
            public_problems.append("public source_contract is not pinned_hf_v2")
        if public_meta.get("clean_rerun_compatible") is not True:
            public_problems.append("public snapshot is not marked clean-rerun-compatible")
        raw_commitment = public_meta.get("raw_artifact_commitment") or {}
        if raw_commitment.get("manifest_sha256") != L.sha256_of_file(manifest_json_path):
            public_problems.append("public raw manifest commitment mismatch")
        effective_license_by_source = {}
        for source_meta in manifest_meta.get("sources", {}).values():
            emitted = source_meta.get("emitted_source")
            effective_license_by_source[emitted] = L.resolved_license_metadata(source_meta)
        for stem in L.MANIFEST_FILES:
            path = os.path.join(public_dir, f"{stem}.jsonl")
            if not os.path.exists(path):
                public_problems.append(f"missing {path}")
                continue
            declared = public_meta.get("files", {}).get(stem, {})
            if declared.get("sha256") != L.sha256_of_file(path):
                public_problems.append(f"{stem}: public sha256 mismatch")
            rows = load_jsonl(path)
            if declared.get("n_rows") != len(rows):
                public_problems.append(f"{stem}: public row-count mismatch")
            public_problems.extend(find_banned_public_keys(rows))
            raw_path = os.path.join(manifest_dir, f"{stem}.jsonl")
            raw_declared = (raw_commitment.get("splits") or {}).get(stem, {})
            if raw_declared.get("sha256") != L.sha256_of_file(raw_path):
                public_problems.append(f"{stem}: public raw-split commitment mismatch")
            if raw_declared.get("n_rows") != len(M[stem]):
                public_problems.append(f"{stem}: public raw-split row-count mismatch")
            expected_public_rows = [
                P.project_public_row(raw, effective_license_by_source) for raw in M[stem]]
            if rows != expected_public_rows:
                public_problems.append(f"{stem}: public rows differ from deterministic raw projection")
        for name in ("policy_label_crosswalk", "contradictory_label_inventory"):
            declared = public_meta.get("supplemental_files", {}).get(name, {})
            path = declared.get("path")
            if not path:
                public_problems.append(f"{name}: missing supplemental declaration")
                continue
            path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
            if not os.path.exists(path):
                public_problems.append(f"{name}: missing {path}")
                continue
            if declared.get("sha256") != L.sha256_of_file(path):
                public_problems.append(f"{name}: sha256 mismatch")
            supplemental = json.load(open(path, encoding="utf-8"))
            public_problems.extend(find_banned_public_keys(supplemental))
            if name == "contradictory_label_inventory":
                public_conflict_inventory_ok = (
                    len(supplemental.get("final_manifest_cross_source_conflicts", []))
                    == len(cross_source_label_conflicts))
    report["public_release_validation"] = {
        "path": os.path.relpath(public_manifest_path, _ROOT),
        "problems": public_problems[:50],
        "n_problems": len(public_problems),
        "ok": len(public_problems) == 0 and public_conflict_inventory_ok,
        "contradictory_label_inventory_matches": public_conflict_inventory_ok,
        "manifest_sha256": public_manifest_sha256,
    }

    # ---- HARD ASSERTIONS ---------------------------------------------------
    calibration_by_source = {
        source: Counter(r["label"] for r in M["calibration"] if r["source"] == source)
        for source in REPRESENTED_SOURCES
    }
    calibration_has_required_classes = all(
        counts_for_source.get("safe", 0) >= 10
        and counts_for_source.get("unsafe", 0) > 0
        for counts_for_source in calibration_by_source.values()
    )
    file_integrity_ok = all(
        item["sha256_matches"] and item["row_count_matches"]
        for item in file_integrity.values())
    family_validation_ok = (
        len(shared_fams) == 0
        and len(shared_cal_id_fams) == 0
        and len(shared_cal_reported_fams) == 0
        and not missing_required_upstream
        and len(jbb_pairs) == 36
        and len(xstest_pairs) == 58
        and not upstream_family_mismatch
        and not exact_family_mismatch
        and not minhash_family_mismatch
        and not global_upstream_family_mismatch
        and int(manifest_meta.get("family_stats", {}).get("n_upstream_edges", 0)) >= 94
    )
    assertions = {
        "config_hash_matches == true": report["config_validation"]["matches"],
        "source_mode == huggingface_pinned_revisions": report["config_validation"]["pinned_hf_default"],
        "manifest_file_hashes_and_counts_match == true": file_integrity_ok,
        "declared_row_and_class_counts_match == true": count_validation["all_expected_counts_match"],
        "exact_schema_and_types_valid == true": (
            not schema_missing and not schema_exact_problems and not type_problems),
        "stored_content_hashes_recompute == true": len(hash_mismatch) == 0,
        "sample_id_and_split_names_valid == true": (
            not sample_id_mismatch and not split_mismatch),
        "source_revision_and_license_metadata_match == true": (
            not source_metadata_problems),
        "sample_ids_unique == true": len(dup_sids) == 0,
        "or_bench_train_count == 0": or_bench_train_count == 0,
        "beavertails_train_count == 0": beavertails_train_count == 0,
        "exact_train_vs_eval_overlap == 0": total_exact == 0,
        "conflicting_label_overlap == 0": total_conflict == 0,
        "every_row_has_source_revision == true": len(missing_rev) == 0,
        "every_row_has_content_hash == true": len(missing_hash) == 0,
        "every_near_duplicate_candidate_has_disposition == true": every_candidate_disposed,
        "train_eval_and_calibration_id_families_disjoint == true": family_validation_ok,
        "calibration_vs_all_reported_test_families_disjoint == true": (
            len(shared_cal_reported_fams) == 0),
        "all_observed_similarity_and_upstream_edges_share_family_id == true": (
            not exact_family_mismatch
            and not minhash_family_mismatch
            and not global_upstream_family_mismatch),
        "calibration_has_both_classes_and_at_least_10_negatives_per_source == true": (
            calibration_has_required_classes),
        "source_roles_valid == true": len(role_problems) == 0,
        "licenses_resolved_and_conflicts_recorded == true": report["license_validation"]["ok"],
        "minhash_backend_and_algorithm_pinned == true": report["provenance_validation"]["ok"],
        "public_text_free_release_complete == true": report["public_release_validation"]["ok"],
    }
    if set(assertions) != set(C.AUDIT_HARD_ASSERTION_KEYS):
        raise RuntimeError("audit implementation does not match the locked hard-assertion schema")
    report["audit_contract_version"] = C.AUDIT_CONTRACT_VERSION
    report["hard_assertions"] = assertions
    all_pass = all(assertions.values())
    report["all_hard_assertions_pass"] = all_pass

    # ---- write audit.json + audit.md ---------------------------------------
    json_path = os.path.join(out_dir, "audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)

    md = build_markdown(report)
    with open(os.path.join(out_dir, "audit.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # ---- console summary ---------------------------------------------------
    print("== HARD ASSERTIONS ==")
    for k, v in assertions.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nexact train<->eval overlap: {total_exact} "
          f"(same-label {total_samelabel}, conflicting {total_conflict})")
    print(f"near-dup cross-split pairs @0.80/0.85/0.90: "
          f"{sens['0.80']['cross_train_eval_pairs']}/"
          f"{sens['0.85']['cross_train_eval_pairs']}/"
          f"{sens['0.90']['cross_train_eval_pairs']}")
    print(f"train/eval shared families: {len(shared_fams)}")
    print(f"or_bench_train={or_bench_train_count}  beavertails_train={beavertails_train_count}")
    print(f"one-to-one joins: {report['join_validation']['one_to_one']}")
    print(f"\nwrote {json_path}")
    print(f"wrote {os.path.join(out_dir, 'audit.md')}")

    if not all_pass:
        print("\n!! ONE OR MORE HARD ASSERTIONS FAILED", file=sys.stderr)
        raise SystemExit(1)
    print("\nALL HARD ASSERTIONS PASS")
    return report


def build_markdown(report):
    a = report["hard_assertions"]
    lines = []
    lines.append("# Paper A — data audit\n")
    lines.append(f"Manifest dir: `{report['manifest_dir']}`  ")
    lines.append(f"Provenance: `{report['provenance'].get('provenance_source')}` "
                 f"(MinHash backend `{report['provenance'].get('minhash_backend', 'n/a')}`, "
                 f"{report['provenance'].get('minhash_num_perm')} perms, "
                 f"{report['provenance'].get('minhash_ngram')}-gram, "
                 f"J>={report['provenance'].get('minhash_jaccard_threshold')})\n")

    lines.append("## Hard assertions\n")
    lines.append("| assertion | result |")
    lines.append("|---|---|")
    for k, v in a.items():
        lines.append(f"| `{k}` | {'PASS' if v else '**FAIL**'} |")
    lines.append(f"\n**All hard assertions pass: {report['all_hard_assertions_pass']}**\n")

    lines.append("## Row and class counts\n")
    lines.append("| manifest | rows | safe | unsafe |")
    lines.append("|---|---:|---:|---:|")
    for stem, c in report["counts"].items():
        cc = c["class_counts"]
        lines.append(f"| {stem} | {c['n_rows']} | {cc.get('safe', 0)} | {cc.get('unsafe', 0)} |")

    lines.append("\n### Per-source counts\n")
    for stem, c in report["counts"].items():
        lines.append(f"- **{stem}**: " +
                     "; ".join(f"{s} (safe {v.get('safe', 0)}, unsafe {v.get('unsafe', 0)})"
                               for s, v in c["per_source"].items()))

    lines.append("\n## Exact train↔eval overlap\n")
    eo = report["exact_overlap"]
    lines.append(f"- total exact overlap: **{eo['total_exact_train_vs_eval']}** "
                 f"(same-label {eo['total_same_label']}, conflicting {eo['total_conflicting_label']})")
    lines.append("\n| eval split | exact | same-label | conflicting |")
    lines.append("|---|---:|---:|---:|")
    for stem, v in eo["by_eval"].items():
        lines.append(f"| {stem} | {v['exact_overlap']} | {v['same_label']} | {v['conflicting_label']} |")

    lines.append("\n## Near-duplicate sensitivity (char-5gram MinHash)\n")
    lines.append("| threshold | total pairs | cross train↔eval | within-train | within-eval |")
    lines.append("|---|---:|---:|---:|---:|")
    for thr, v in report["near_dup_sensitivity"]["by_threshold"].items():
        lines.append(f"| {thr} | {v['total_near_dup_pairs']} | {v['cross_train_eval_pairs']} | "
                     f"{v['within_train_pairs']} | {v['within_eval_pairs']} |")
    lines.append(f"\n> {report['near_dup_sensitivity']['note']}\n")

    lines.append("## Family validation\n")
    fv = report["family_validation"]
    lines.append(f"- total families: {fv['n_families_total']}; train families {fv['n_train_families']}; "
                 f"eval families {fv['n_eval_families']}")
    lines.append(f"- **train↔eval shared families: {fv['train_eval_shared_families']}** "
                 f"(0 expected: cross-split near-dup train members are removed)")

    lines.append("\n## Disposition coverage\n")
    dc = report["disposition_coverage"]
    lines.append(f"- exact overlaps removed at build: {dc['build_exact_overlaps_removed']}")
    lines.append(f"- cross-split near-dup components removed at build: "
                 f"{dc['build_cross_split_components_removed']}")
    lines.append(f"- known WildJailbreak overlaps removed: "
                 f"{dc['known_wildjailbreak_overlaps_in_build']}")
    lines.append(f"- undisposed cross-split near-dup pairs at 0.85 in final manifests: "
                 f"{dc['undisposed_cross_split_pairs_at_0.85_in_final_manifests']}")

    lines.append("\n## License inventory\n")
    lines.append("| source | license(s) | redistribution class(es) |")
    lines.append("|---|---|---|")
    for s in sorted(report["license_inventory"]):
        lic = ", ".join(f"{k} ({v})" for k, v in report["license_inventory"][s].items())
        rd = ", ".join(f"{k} ({v})" for k, v in report["redistribution_inventory"][s].items())
        lines.append(f"| {s} | {lic} | {rd} |")

    lines.append("\n## Role and join validation\n")
    lines.append(f"- role validation ok: {report['role_validation']['ok']}")
    lines.append(f"- OR-Bench stress labels: {report['role_validation']['orbench_labels']}; "
                 f"HarmBench stress labels: {report['role_validation']['harmbench_labels']}")
    lines.append(f"- one-to-one joins (unique sample_id): {report['join_validation']['one_to_one']}")
    lines.append(f"- OR-Bench in train: {report['forbidden_train_sources']['or_bench_train_count']}; "
                 f"BeaverTails in train: {report['forbidden_train_sources']['beavertails_train_count']}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public-manifest", default=None,
                    help="text-free public manifest index (default: sibling public_manifests)")
    args = ap.parse_args()
    audit(args.config, args.manifest, args.out, args.public_manifest)


if __name__ == "__main__":
    main()
