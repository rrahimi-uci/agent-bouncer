"""Manifest decontamination tests (plan sec 13.4 test_manifests).

Guards the reviewer's central "contamination + no auditable artifacts" blocker:
  * no exact train/evaluation overlap;
  * the 4 known jailbreak-classification <-> WildJailbreak overlap rows are absent
    from training;
  * OR-Bench and BeaverTails counts in training are zero;
  * every row carries provenance, family, and hash fields;
  * joins are one-to-one (unique sample_id across all manifests).

These run against corrected manifests at artifacts/paper_a_sft_v2/manifests. Build
them first with:
  .venv/bin/python experiments/prepare_paper_a_manifests.py \
    --config configs/paper_a_sft.yaml --out artifacts/paper_a_sft_v2/manifests
"""

import json
import os
import sys
from collections import Counter

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_manifest_lib as L  # noqa: E402

# Never interpret the immutable v1 raw cache as corrected v2 data.  The synthetic
# regressions and tracked public-release test still run in a clean checkout; these
# integration checks activate only after `make manifests` creates the v2 namespace.
MANIFEST_DIR = os.path.join(_ROOT, "artifacts", "paper_a_sft_v2", "manifests")
PUBLIC_MANIFEST_DIR = os.path.join(_ROOT, "artifacts", "paper_a_sft", "public_manifests")
REPRESENTED_SOURCES = {"toxicchat", "prompt_injections", "jailbreak_classification"}
TRANSFER_SOURCES = {"jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"}


def _load(stem):
    path = os.path.join(MANIFEST_DIR, f"{stem}.jsonl")
    if not os.path.exists(path):
        pytest.skip(f"manifest {path} not built; run prepare_paper_a_manifests.py first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def manifests():
    return {stem: _load(stem) for stem in L.MANIFEST_FILES}


@pytest.fixture(scope="module")
def manifest_meta():
    path = os.path.join(MANIFEST_DIR, "manifest.json")
    if not os.path.exists(path):
        pytest.skip("manifest.json not built")
    return json.load(open(path))


@pytest.fixture(scope="module")
def train_rows(manifests):
    return manifests["train"]


@pytest.fixture(scope="module")
def eval_rows(manifests):
    stems = ["calibration", "id_test", "transfer_test",
             "orbench_safe_stress", "harmbench_positive_stress"]
    return [r for s in stems for r in manifests[s]]


@pytest.fixture(scope="module")
def all_rows(manifests):
    return [r for rows in manifests.values() for r in rows]


# --------------------------------------------------------------------------
# no exact train/evaluation overlap
# --------------------------------------------------------------------------
def test_no_exact_train_eval_overlap(train_rows, eval_rows):
    train_hashes = {r["content_sha256"] for r in train_rows}
    eval_hashes = {r["content_sha256"] for r in eval_rows}
    overlap = train_hashes & eval_hashes
    assert overlap == set(), f"{len(overlap)} exact train/eval content-hash overlaps"


def test_no_exact_train_eval_overlap_recomputed(train_rows, eval_rows):
    """Recompute content hashes from text (do not trust the stored field)."""
    train_hashes = {L.content_sha256(r["text_or_download_reference"]) for r in train_rows}
    eval_hashes = {L.content_sha256(r["text_or_download_reference"]) for r in eval_rows}
    assert train_hashes.isdisjoint(eval_hashes)


def test_no_conflicting_label_overlap(train_rows, eval_rows):
    train_by_hash = {}
    for r in train_rows:
        train_by_hash.setdefault(r["content_sha256"], set()).add(r["label"])
    conflicts = 0
    for r in eval_rows:
        labs = train_by_hash.get(r["content_sha256"])
        if labs and r["label"] not in labs:
            conflicts += 1
    assert conflicts == 0


# --------------------------------------------------------------------------
# the 4 known overlap rows are absent from training
# --------------------------------------------------------------------------
def test_four_known_wildjailbreak_overlaps_recorded(manifest_meta):
    known = manifest_meta.get("known_wildjailbreak_overlaps", {})
    assert known.get("count") == 4, f"expected 4 known jc<->wjb overlaps, got {known.get('count')}"
    for rec in known["records"]:
        assert rec["train_source"] == "jailbreak_classification"
        assert "wildjailbreak" in rec["eval_sources"]


def test_four_known_overlaps_absent_from_train(manifest_meta, train_rows):
    known = manifest_meta["known_wildjailbreak_overlaps"]["records"]
    assert len(known) == 4
    train_hashes = {r["content_sha256"] for r in train_rows}
    train_srids = {r["source_row_id"] for r in train_rows}
    for rec in known:
        assert rec["content_sha256"] not in train_hashes, \
            f"known overlap {rec['content_sha256'][:12]} present in train"
        assert rec["train_source_row_id"] not in train_srids, \
            f"known overlap row {rec['train_source_row_id']} present in train"


# --------------------------------------------------------------------------
# OR-Bench and BeaverTails counts in training are zero
# --------------------------------------------------------------------------
def test_no_orbench_or_beavertails_in_train(train_rows):
    src = Counter(r["source"] for r in train_rows)
    for forbidden in ("orbench", "or_bench", "or-bench", "beavertails", "beaver_tails"):
        assert src.get(forbidden, 0) == 0, f"{forbidden} present in train"


def test_train_sources_are_exactly_the_three_represented(train_rows):
    assert {r["source"] for r in train_rows} == REPRESENTED_SOURCES


# --------------------------------------------------------------------------
# every row has provenance, family, and hash fields
# --------------------------------------------------------------------------
def test_every_row_has_all_schema_fields(all_rows):
    for r in all_rows:
        for field in L.ROW_SCHEMA_FIELDS:
            assert field in r, f"row {r.get('sample_id')} missing field {field}"


def test_every_row_has_provenance_family_hash(all_rows):
    for r in all_rows:
        assert r.get("source_revision"), f"row {r.get('sample_id')} missing source_revision"
        assert r.get("label_provenance"), f"row {r.get('sample_id')} missing label_provenance"
        assert r.get("family_id"), f"row {r.get('sample_id')} missing family_id"
        assert r.get("content_sha256"), f"row {r.get('sample_id')} missing content_sha256"


def test_stored_content_hash_matches_normalized_text(all_rows):
    for r in all_rows:
        assert r["content_sha256"] == L.content_sha256(r["text_or_download_reference"]), \
            f"content hash mismatch for {r.get('sample_id')}"


# --------------------------------------------------------------------------
# joins are one-to-one
# --------------------------------------------------------------------------
def test_sample_ids_are_unique_across_all_manifests(all_rows):
    counts = Counter(r["sample_id"] for r in all_rows)
    dups = [s for s, c in counts.items() if c > 1]
    assert dups == [], f"{len(dups)} duplicate sample_ids (e.g. {dups[:5]})"


def test_family_ids_join_one_to_one(all_rows):
    """A family_id must never map to two distinct connected-component identities.

    family_id is defined as the sha256 of the component's smallest content hash,
    so it is a stable per-component key; here we assert every row that carries a
    family_id carries a well-formed 64-hex-char digest (joinable across tables).
    """
    for r in all_rows:
        fid = r["family_id"]
        assert isinstance(fid, str) and len(fid) == 64 and all(c in "0123456789abcdef" for c in fid)


def test_train_and_eval_families_are_disjoint(train_rows, eval_rows):
    train_fams = {r["family_id"] for r in train_rows}
    eval_fams = {r["family_id"] for r in eval_rows}
    assert train_fams.isdisjoint(eval_fams), \
        "train and eval share family ids (cross-split near-dup leak)"


def test_calibration_and_id_families_are_globally_disjoint(manifests):
    calibration = {r["family_id"] for r in manifests["calibration"]}
    identity_test = {r["family_id"] for r in manifests["id_test"]}
    assert calibration.isdisjoint(identity_test), \
        "calibration and ID share global family ids"


def test_calibration_is_family_disjoint_from_every_reported_test(manifests):
    calibration = {r["family_id"] for r in manifests["calibration"]}
    reported = {
        r["family_id"]
        for stem in ("id_test", "transfer_test", "orbench_safe_stress",
                     "harmbench_positive_stress")
        for r in manifests[stem]
    }
    assert calibration.isdisjoint(reported), \
        "calibration shares a global family with an ID, transfer, or stress row"


def test_authoritative_transfer_pairs_share_families(manifests):
    grouped = {}
    for row in manifests["transfer_test"]:
        if row["source"] not in {"jailbreakbench", "xstest"}:
            continue
        assert row.get("upstream_family_id"), row["sample_id"]
        grouped.setdefault(row["upstream_family_id"], []).append(row)
    jbb = [rows for fid, rows in grouped.items()
           if fid.startswith("jailbreakbench:")
           and {r["label"] for r in rows} == {"safe", "unsafe"}]
    xstest = [rows for fid, rows in grouped.items()
              if fid.startswith("xstest:direct-contrast:")
              and {r["label"] for r in rows} == {"safe", "unsafe"}]
    assert len(jbb) == 36
    assert len(xstest) == 58
    assert all(len({r["family_id"] for r in rows}) == 1 for rows in jbb + xstest)


# --------------------------------------------------------------------------
# construction sanity (locked 1,200 rows: 400/source, 200/source-label)
# --------------------------------------------------------------------------
def test_train_manifest_is_locked_1200_400_200(train_rows):
    assert len(train_rows) == 1200
    by_src_label = Counter((r["source"], r["label"]) for r in train_rows)
    for src in REPRESENTED_SOURCES:
        assert by_src_label[(src, "safe")] == 200, f"{src} safe != 200"
        assert by_src_label[(src, "unsafe")] == 200, f"{src} unsafe != 200"


def test_stress_sets_are_single_class(manifests):
    assert {r["label"] for r in manifests["orbench_safe_stress"]} == {"safe"}
    assert {r["label"] for r in manifests["harmbench_positive_stress"]} == {"unsafe"}


def test_transfer_rows_only_from_transfer_sources(manifests):
    assert {r["source"] for r in manifests["transfer_test"]}.issubset(TRANSFER_SOURCES)


def test_calibration_and_id_from_represented_sources(manifests):
    for stem in ("calibration", "id_test"):
        assert {r["source"] for r in manifests[stem]}.issubset(REPRESENTED_SOURCES)


def test_manifest_records_pinned_backend_and_resolved_licenses(manifest_meta, all_rows):
    provenance = manifest_meta["provenance"]
    assert provenance["minhash_backend"] == L.MINHASH_BACKEND
    assert provenance["minhash_algorithm_version"] == L.MINHASH_ALGORITHM_VERSION
    assert provenance["minhash_seed"] == L.MINHASH_SEED
    assert all(not str(r["license_id"]).lower().startswith("unknown") for r in all_rows)
    prompt = manifest_meta["sources"]["prompt_injections"]["license_metadata"]
    assert prompt["metadata_values"] == {
        "card_top_level": "Apache-2.0", "dataset_info_nested": "CC-BY-4.0"}


def _assert_text_free_public_index(public_dir):
    index_path = os.path.join(public_dir, "manifest.json")
    assert os.path.exists(index_path), "tracked public manifest index is required"
    index = json.load(open(index_path))
    assert index["source_contract"] in {"legacy_v1_snapshot", "pinned_hf_v2"}
    assert isinstance(index["clean_rerun_compatible"], bool)
    if index["source_contract"] == "legacy_v1_snapshot":
        assert index["clean_rerun_compatible"] is False
        assert index["known_integrity_limitations"]
    banned = {
        "text", "raw_text", "norm_text", "normalized_text",
        "text_or_download_reference", "prompt", "response", "completion", "adversarial",
    }

    def check(value):
        if isinstance(value, dict):
            assert not ({str(k).lower() for k in value} & banned)
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(index)
    for stem in L.MANIFEST_FILES:
        path = os.path.join(public_dir, f"{stem}.jsonl")
        assert os.path.exists(path)
        rows = [json.loads(line) for line in open(path) if line.strip()]
        check(rows)
        assert len(rows) == index["files"][stem]["n_rows"]
        assert L.sha256_of_file(path) == index["files"][stem]["sha256"]
    for name in ("policy_label_crosswalk", "contradictory_label_inventory"):
        rec = index["supplemental_files"][name]
        path = rec["path"] if os.path.isabs(rec["path"]) else os.path.join(_ROOT, rec["path"])
        assert os.path.exists(path)
        assert L.sha256_of_file(path) == rec["sha256"]
        check(json.load(open(path)))


def test_legacy_release_contains_text_free_public_index():
    _assert_text_free_public_index(PUBLIC_MANIFEST_DIR)


def test_built_v2_release_contains_text_free_public_index():
    public_dir = os.path.join(_ROOT, "artifacts", "paper_a_sft_v2", "public_manifests")
    if not os.path.exists(os.path.join(public_dir, "manifest.json")):
        pytest.skip("v2 public release not built; legacy tracked release was checked separately")
    _assert_text_free_public_index(public_dir)
