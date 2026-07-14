"""Focused regressions for Paper A's data/provenance repair path."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
for path in (ROOT, EXP):
    if path not in sys.path:
        sys.path.insert(0, path)

from guard_research import provenance  # noqa: E402
import paper_a_manifest_lib as L  # noqa: E402
import prepare_paper_a_manifests as P  # noqa: E402
import audit_paper_a_splits as A  # noqa: E402


def test_minhash_is_one_versioned_backend_with_golden_signature(monkeypatch):
    # Merely making a datasketch-looking module importable must have no effect.
    monkeypatch.setitem(sys.modules, "datasketch", object())
    signature = provenance.minhash_signature("The quick brown fox")
    digest = hashlib.sha256(signature.tobytes()).hexdigest()
    assert provenance.MINHASH_BACKEND == "numpy-blake2b-mersenne31-v1"
    assert provenance.MINHASH_ALGORITHM_VERSION == 1
    assert digest == "d8eefc98bb4e465ea8ca05182a7d224cf3e54c729a4883da107c23731a6a3f2f"


def test_global_calibration_split_never_splits_cross_source_family():
    rows = [
        {"source": "a", "family_id": "shared"},
        {"source": "b", "family_id": "shared"},
        {"source": "a", "family_id": "a-only-1"},
        {"source": "a", "family_id": "a-only-2"},
        {"source": "b", "family_id": "b-only-1"},
        {"source": "b", "family_id": "b-only-2"},
    ]
    _, assignment = L.split_calibration_id_global(rows, data_seed=42, cal_frac=0.4)
    assert set(assignment) == {r["family_id"] for r in rows}
    assert assignment["shared"] in {L.SPLIT_CALIBRATION, L.SPLIT_ID}
    for source in {r["source"] for r in rows}:
        source_rows = [r for r in rows if r["source"] == source]
        _, local = L.split_calibration_id(source_rows, source, data_seed=42, cal_frac=0.4)
        for family_id in {r["family_id"] for r in source_rows} - {"shared"}:
            assert assignment[family_id] == local[family_id]


def test_calibration_families_present_on_reported_tests_route_to_id():
    assignment = {
        "cal-only": L.SPLIT_CALIBRATION,
        "shared-transfer": L.SPLIT_CALIBRATION,
        "already-id": L.SPLIT_ID,
    }
    corrected, routed = L.route_calibration_conflicts_to_id(
        assignment, {"shared-transfer", "stress-only"})
    assert routed == ["shared-transfer"]
    assert corrected["cal-only"] == L.SPLIT_CALIBRATION
    assert corrected["shared-transfer"] == L.SPLIT_ID
    assert assignment["shared-transfer"] == L.SPLIT_CALIBRATION


def test_family_edge_audit_detects_corrupted_component_identity():
    rows = [
        {"sample_id": "a", "family_id": "family-a"},
        {"sample_id": "b", "family_id": "family-b"},
    ]
    mismatches = A.family_edge_mismatches(rows, [(0, 1, 0.99)])
    assert len(mismatches) == 1
    rows[1]["family_id"] = "family-a"
    assert A.family_edge_mismatches(rows, [(0, 1, 0.99)]) == []


def _synthetic_xstest_rows():
    rows = []
    types = [
        "homonyms", "contrast_homonyms",
        "figurative_language", "contrast_figurative_language",
        "safe_targets", "contrast_safe_targets",
        "safe_contexts", "contrast_safe_contexts",
        "definitions", "contrast_definitions",
        "nons_group_real_discr", "real_group_nons_discr", "contrast_discr",
        "historical_events", "contrast_historical_events",
        "privacy_public", "privacy_fictional", "contrast_privacy",
    ]
    pos = 0
    for row_type in types:
        for ordinal in range(25):
            rows.append({
                "id": f"v2-{pos + 1}",
                "type": row_type,
                "prompt": f"{row_type} synthetic prompt {ordinal}",
            })
            pos += 1
    return rows


def test_xstest_preserves_all_eight_direct_contrast_blocks(monkeypatch):
    monkeypatch.setattr(P, "_hf", lambda *args, **kwargs: _synthetic_xstest_rows())
    spec = {
        "revision": "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
        "hf_config": "default", "split": "prompts", "text_field": "prompt",
        "hf_path": "natolambert/xstest-v2-copy", "role": "transfer",
        "label_rule": "type prefix contrast -> unsafe; else safe",
        "license_id": "CC-BY-4.0", "redistribution_class": "permissive",
    }
    rows = P.load_xstest_labeled(None, "xstest", spec)
    paired = {}
    for row in rows:
        family = row["upstream_family_id"]
        if family.startswith("xstest:direct-contrast:"):
            paired.setdefault(family, []).append(row)
    assert len(paired) == 8 * 25
    assert all(len(members) == 2 for members in paired.values())
    assert all({r["label"] for r in members} == {"safe", "unsafe"}
               for members in paired.values())


def test_pinned_hf_is_default_and_legacy_cache_requires_named_opt_in():
    params = inspect.signature(P.build).parameters
    assert params["allow_legacy_frozen_cohorts"].default is False


def test_pinned_license_metadata_and_prompt_conflict_are_recorded():
    config = P.load_config(os.path.join(ROOT, "configs", "paper_a_sft.yaml"))
    prompt = L.resolved_license_metadata(config["sources"]["prompt_injections"])
    jailbreak = L.resolved_license_metadata(config["sources"]["jailbreak_classification"])
    orbench = L.resolved_license_metadata(config["sources"]["orbench_hard"])
    assert prompt["license_id"] == "Apache-2.0"
    assert prompt["metadata_values"] == {
        "card_top_level": "Apache-2.0", "dataset_info_nested": "CC-BY-4.0"}
    assert jailbreak["license_id"] == "Apache-2.0"
    assert orbench["license_id"] == "CC-BY-4.0"
    xstest = L.resolved_license_metadata(config["sources"]["xstest"])
    assert xstest["license_id"] == "CC-BY-4.0"
    assert xstest["redistribution_class"] == "permissive_with_attribution"


def test_public_release_is_recursive_and_contains_crosswalks(tmp_path):
    raw_dir = tmp_path / "manifests"
    public_dir = tmp_path / "public_manifests"
    raw_dir.mkdir()
    raw_row = {
        "sample_id": "toxicchat::row-1", "source": "toxicchat",
        "source_config": "cfg", "source_revision": "a" * 40,
        "source_row_id": "row-1", "upstream_family_id": None,
        "source_origin": "hf:test", "split": "train", "label": "safe", "gold": 0,
        "label_provenance": "rule", "text_or_download_reference": "SECRET RAW TEXT",
        "content_sha256": "b" * 64, "family_id": "c" * 64,
        "license_id": "Apache-2.0", "redistribution_class": "permissive",
        "known_overlap_disposition": "none",
    }
    files = {}
    for stem in L.MANIFEST_FILES:
        path = raw_dir / f"{stem}.jsonl"
        path.write_text(json.dumps({**raw_row, "sample_id": f"toxicchat::{stem}",
                                    "source_row_id": stem, "split": stem}) + "\n")
        files[stem] = {"path": str(path), "n_rows": 1, "sha256": L.sha256_of_file(path)}
    raw_manifest = {
        "schema_version": 1, "files": files,
        "sources": {"toxicchat": {
            "emitted_source": "toxicchat", "hf_path": "lmsys/toxic-chat",
            "hf_config": "toxicchat0124", "revision": "a" * 40, "role": "train",
            "license_id": "Apache-2.0",
            "nested": {"normalized_text": "SECOND SECRET"},
        }},
        "removals": {"exact_train_vs_eval": {"records": []}},
    }
    raw_manifest_path = raw_dir / "manifest.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest))
    result = P.write_public_release(raw_manifest_path, public_dir)
    assert os.path.exists(result["manifest_path"])
    serialized = "\n".join(path.read_text() for path in public_dir.iterdir())
    assert "SECRET RAW TEXT" not in serialized
    assert "SECOND SECRET" not in serialized
    assert "text_or_download_reference" not in serialized
    public_meta = json.loads((public_dir / "manifest.json").read_text())
    assert public_meta["source_contract"] == "legacy_v1_snapshot"
    assert public_meta["clean_rerun_compatible"] is False
    assert public_meta["known_integrity_limitations"]
    assert set(public_meta["supplemental_files"]) == {
        "policy_label_crosswalk", "contradictory_label_inventory"}
    assert P.recursively_redact_public({"a": [{"prompt": "secret", "ok": 1}]}) == {
        "a": [{"ok": 1}]}
    public_train = json.loads((public_dir / "train.jsonl").read_text().splitlines()[0])
    assert public_train == P.project_public_row({
        **raw_row, "sample_id": "toxicchat::train", "source_row_id": "train",
        "split": "train",
    }, {
        "toxicchat": L.resolved_license_metadata(raw_manifest["sources"]["toxicchat"])
    })
