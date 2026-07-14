"""Manifest-construction helpers for Paper A (builder + audit + tests share this).

Hashing / normalization / MinHash come from ``guard_research.provenance`` (the
single source of truth, plan sec 13.1). Family clustering (LSH banding + union
find), deterministic hash-ranking, and the calibration/ID family split are built
here on top of those primitives so the manifest builder, the split audit, and
the tests are byte-identical by construction.  Import failure is fatal: a
data-defining algorithm must never switch to an implicit fallback.

(This module is intentionally separate from ``experiments/paper_a_common.py``,
which serves the lock/train/eval/analyze pipeline.)
"""

from __future__ import annotations

import hashlib
import os
import sys

# --- make repo-root + experiments/ importable regardless of entrypoint --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- provenance primitives: one required implementation -----------------------
from guard_research.provenance import (  # type: ignore  # noqa: E402
    MINHASH_ALGORITHM_VERSION,
    MINHASH_BACKEND,
    MINHASH_JACCARD_THRESHOLD,
    MINHASH_SEED,
    content_sha256,
    estimated_jaccard,
    minhash_signature,
    normalize_text,
    sha256_of_file,
    sha256_of_obj,
)

PROVENANCE_SOURCE = "guard_research.provenance"


# --- frozen family-construction parameters (plan sec 6.7) ---------------------
NGRAM = 5
NUM_PERM = 256
LSH_BANDS = 32
LSH_ROWS = 8  # LSH_BANDS * LSH_ROWS must equal NUM_PERM
assert LSH_BANDS * LSH_ROWS == NUM_PERM

# Canonical split names, aligned with the plan sec 6.6 manifest file stems.
SPLIT_TRAIN = "train"
SPLIT_CALIBRATION = "calibration"
SPLIT_ID = "id_test"
SPLIT_TRANSFER = "transfer_test"
SPLIT_ORBENCH = "orbench_safe_stress"
SPLIT_HARMBENCH = "harmbench_positive_stress"

MANIFEST_FILES = [
    "train",
    "calibration",
    "id_test",
    "transfer_test",
    "orbench_safe_stress",
    "harmbench_positive_stress",
]
# The five splits a training/validation path must never read (fail-closed set).
EVAL_SPLITS = {SPLIT_CALIBRATION, SPLIT_ID, SPLIT_TRANSFER, SPLIT_ORBENCH, SPLIT_HARMBENCH}

# Canonical row-schema fields (plan sec 6.5).
ROW_SCHEMA_FIELDS = [
    "sample_id",
    "source",
    "source_config",
    "source_revision",
    "source_row_id",
    "upstream_family_id",
    "source_origin",
    "split",
    "label",
    "gold",
    "label_provenance",
    "text_or_download_reference",
    "content_sha256",
    "family_id",
    "license_id",
    "redistribution_class",
    "known_overlap_disposition",
]

# XSTest contains eight 25-row unsafe contrast blocks.  The first five and the
# historical block pair with the immediately preceding safe block; the
# discrimination and privacy contrasts pair with ``real_group_nons_discr`` and
# ``privacy_fictional`` respectively.  The other two safe blocks have no direct
# unsafe counterpart.  Values are stable family namespaces, not labels.
XSTEST_DIRECT_CONTRAST_GROUPS = {
    "homonyms": "homonyms",
    "contrast_homonyms": "homonyms",
    "figurative_language": "figurative_language",
    "contrast_figurative_language": "figurative_language",
    "safe_targets": "safe_targets",
    "contrast_safe_targets": "safe_targets",
    "safe_contexts": "safe_contexts",
    "contrast_safe_contexts": "safe_contexts",
    "definitions": "definitions",
    "contrast_definitions": "definitions",
    "real_group_nons_discr": "discrimination",
    "contrast_discr": "discrimination",
    "historical_events": "historical_events",
    "contrast_historical_events": "historical_events",
    "privacy_fictional": "privacy_fictional",
    "contrast_privacy": "privacy_fictional",
}

# License metadata is frozen at the same source revisions as the data.  These
# three configs previously used ``unknown-verify-before-lock`` even though the
# pinned cards declare licenses.  Prompt-Injections is special: its canonical
# top-level card field says Apache-2.0, while nested generated dataset_info says
# CC-BY-4.0.  Preserve that conflict instead of silently erasing it.
PINNED_LICENSE_METADATA = {
    ("deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd"): {
        "license_id": "Apache-2.0",
        "redistribution_class": "permissive_with_notice",
        "status": "canonical_card_value_with_recorded_metadata_conflict",
        "card_url": (
            "https://huggingface.co/datasets/deepset/prompt-injections/blob/"
            "4f61ecb038e9c3fb77e21034b22511b523772cdd/README.md"
        ),
        "metadata_values": {
            "card_top_level": "Apache-2.0",
            "dataset_info_nested": "CC-BY-4.0",
        },
        "note": "Pinned card has conflicting top-level and nested license metadata.",
    },
    ("jackhhao/jailbreak-classification", "2f2ceeb39658696fd3f462403562b6eea5306287"): {
        "license_id": "Apache-2.0",
        "redistribution_class": "permissive_with_notice",
        "status": "declared_by_pinned_card",
        "card_url": (
            "https://huggingface.co/datasets/jackhhao/jailbreak-classification/blob/"
            "2f2ceeb39658696fd3f462403562b6eea5306287/README.md"
        ),
        "metadata_values": {"card_top_level": "Apache-2.0"},
    },
    ("bench-llm/or-bench", "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"): {
        "license_id": "CC-BY-4.0",
        "redistribution_class": "permissive_with_attribution",
        "status": "declared_by_pinned_card",
        "card_url": (
            "https://huggingface.co/datasets/bench-llm/or-bench/blob/"
            "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272/README.md"
        ),
        "metadata_values": {"card_top_level": "CC-BY-4.0"},
    },
}


def resolved_license_metadata(spec):
    """Return effective and declared license provenance for one source spec."""
    declared = spec.get("license_id", "unknown")
    key = (spec.get("hf_path"), spec.get("revision"))
    pinned = PINNED_LICENSE_METADATA.get(key)
    if pinned:
        out = dict(pinned)
        out["metadata_values"] = dict(pinned.get("metadata_values", {}))
        out["declared_config_license_id"] = declared
        return out
    return {
        "license_id": declared,
        "redistribution_class": spec.get("redistribution_class", "unknown"),
        "status": "declared_by_pinned_config",
        "card_url": (
            f"https://huggingface.co/datasets/{spec.get('hf_path')}/tree/"
            f"{spec.get('revision')}"
        ),
        "metadata_values": {"config": declared},
        "declared_config_license_id": declared,
    }


def to_gold(label) -> int:
    """Map a label to gold 1(unsafe)/0(safe)."""
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return 1 if int(label) == 1 else 0
    s = str(label).strip().lower()
    if s in ("unsafe", "harmful", "1", "true"):
        return 1
    if s in ("safe", "unharmful", "benign", "0", "false"):
        return 0
    raise ValueError(f"unrecognized label: {label!r}")


# --- deterministic hash ranking (plan sec 6.4.1 step 6) -----------------------
def rank_key(data_seed, source_row_id, content_hash) -> str:
    """SHA-256 hex of (data_seed, source_row_id, content_sha256); sort ascending."""
    return hashlib.sha256(
        f"{data_seed}|{source_row_id}|{content_hash}".encode("utf-8")
    ).hexdigest()


def family_sort_key(data_seed, source, family_id) -> str:
    """SHA-256 hex of (data_seed, source, family_id) for the cal/ID family order."""
    return hashlib.sha256(
        f"{data_seed}|{source}|{family_id}".encode("utf-8")
    ).hexdigest()


def global_family_sort_key(data_seed, family_id) -> str:
    """Frozen order for globally assigning whole families to calibration/ID."""
    return hashlib.sha256(f"{data_seed}|global|{family_id}".encode("utf-8")).hexdigest()


# --- union-find ---------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --- LSH banding over MinHash signatures --------------------------------------
def lsh_candidate_pairs(sigs, bands=LSH_BANDS, rows=LSH_ROWS):
    """Set of candidate index pairs (i<j) sharing a band bucket.

    Banding (32 bands x 8 rows) over 256 permutations puts the S-curve midpoint
    near Jaccard 0.65, so all pairs with true Jaccard >= 0.80 are recalled with
    probability ~0.997+ and then exactly re-scored by the caller.
    """
    pairs = set()
    n = len(sigs)
    for band in range(bands):
        s = band * rows
        e = s + rows
        buckets = {}
        for i in range(n):
            key = sigs[i][s:e].tobytes()
            buckets.setdefault(key, []).append(i)
        for idxs in buckets.values():
            if len(idxs) > 1:
                for a in range(len(idxs)):
                    ia = idxs[a]
                    for b in range(a + 1, len(idxs)):
                        ib = idxs[b]
                        pairs.add((ia, ib) if ia < ib else (ib, ia))
    return pairs


def build_minhash_signatures(texts):
    """MinHash signatures for a list of texts (uint64 numpy arrays)."""
    return [minhash_signature(t, num_perm=NUM_PERM, ngram=NGRAM) for t in texts]


def edges_at_threshold(sigs, cand_pairs, threshold):
    """List of (i, j, est_jaccard) with estimated Jaccard >= threshold."""
    out = []
    for (i, j) in cand_pairs:
        est = estimated_jaccard(sigs[i], sigs[j])
        if est >= threshold:
            out.append((i, j, est))
    return out


def connected_components(n, edges, extra_unions=None):
    """Union-find components. edges: iterable of (i, j, ...); extra_unions: (i, j)."""
    uf = UnionFind(n)
    for e in edges:
        uf.union(e[0], e[1])
    if extra_unions:
        for (i, j) in extra_unions:
            uf.union(i, j)
    comp = {}
    for i in range(n):
        comp.setdefault(uf.find(i), []).append(i)
    return list(comp.values())


def family_id_for_component(member_content_hashes) -> str:
    """family_id = sha256 of the lexicographically smallest content hash (plan 6.7 step 7)."""
    smallest = min(member_content_hashes)
    return hashlib.sha256(smallest.encode("utf-8")).hexdigest()


def build_families(texts, content_hashes, upstream_edges=None, sigs=None):
    """Assign a family_id to every row (plan sec 6.7 steps 3-7).

    texts/content_hashes are aligned lists. upstream_edges is an iterable of
    (i, j) authoritative-family edges. Returns
    (family_of, sigs, cand_pairs, edges_085, comps, stats).
    """
    n = len(content_hashes)
    if sigs is None:
        sigs = build_minhash_signatures(texts)
    cand = lsh_candidate_pairs(sigs)
    edges = edges_at_threshold(sigs, cand, MINHASH_JACCARD_THRESHOLD)
    up = list(upstream_edges) if upstream_edges else []
    comps = connected_components(n, edges, extra_unions=up)
    family_of = [None] * n
    multi = 0
    for members in comps:
        fid = family_id_for_component([content_hashes[m] for m in members])
        for m in members:
            family_of[m] = fid
        if len(members) > 1:
            multi += 1
    stats = {
        "n_rows": n,
        "n_candidate_pairs": len(cand),
        "n_minhash_edges_085": len(edges),
        "n_upstream_edges": len(up),
        "n_components": len(comps),
        "n_multi_member_components": multi,
        "num_perm": NUM_PERM,
        "ngram": NGRAM,
        "lsh_bands": LSH_BANDS,
        "lsh_rows": LSH_ROWS,
        "threshold": MINHASH_JACCARD_THRESHOLD,
        "provenance_source": PROVENANCE_SOURCE,
        "minhash_backend": MINHASH_BACKEND,
        "minhash_algorithm_version": MINHASH_ALGORITHM_VERSION,
        "minhash_seed": MINHASH_SEED,
    }
    return family_of, sigs, cand, edges, comps, stats


# --- calibration / ID family-level split (plan sec 6.4.2) ---------------------
def split_calibration_id(source_rows, source, data_seed, cal_frac=0.40):
    """Greedy whole-family 40/60 calibration/ID split for one represented source.

    source_rows: list of dicts each with 'family_id'. Returns
    (calibration_family_ids: set, assignment: dict family_id -> split_label).
    Adds families (in the frozen sha256 order) to calibration while doing so does
    not move the calibration row count strictly farther from the 40% target; the
    first family that would move it farther, and all remaining families, go to ID.
    """
    fam_rows = {}
    for r in source_rows:
        fam_rows.setdefault(r["family_id"], []).append(r)
    total = len(source_rows)
    target = cal_frac * total
    fams = sorted(fam_rows, key=lambda f: family_sort_key(data_seed, source, f))
    cal_ids = set()
    cal_count = 0
    stopped = False
    for f in fams:
        if stopped:
            continue
        sz = len(fam_rows[f])
        if abs((cal_count + sz) - target) <= abs(cal_count - target):
            cal_ids.add(f)
            cal_count += sz
        else:
            stopped = True
    assignment = {f: (SPLIT_CALIBRATION if f in cal_ids else SPLIT_ID) for f in fams}
    return cal_ids, assignment


def split_calibration_id_global(rows, data_seed, cal_frac=0.40):
    """Assign every global family atomically to calibration or ID.

    Start from the frozen per-source family split so unrelated families retain
    their historical assignment.  For a family whose rows cross sources and
    received conflicting local assignments, deterministically choose the single
    global assignment that minimizes summed per-source absolute deviation from
    the requested calibration fraction.  Returns
    ``(calibration_family_ids, assignment)``.
    """
    fam_rows = {}
    totals = {}
    rows_by_source = {}
    for row in rows:
        family_id = row["family_id"]
        source = row["source"]
        fam_rows.setdefault(family_id, []).append(row)
        totals[source] = totals.get(source, 0) + 1
        rows_by_source.setdefault(source, []).append(row)
    targets = {source: cal_frac * total for source, total in totals.items()}
    cal_counts = {source: 0 for source in totals}
    local_assignment = {}
    for source, source_rows in rows_by_source.items():
        _, source_assignment = split_calibration_id(
            source_rows, source, data_seed, cal_frac=cal_frac)
        for family_id, split in source_assignment.items():
            local_assignment[(source, family_id)] = split
        for row in source_rows:
            if source_assignment[row["family_id"]] == SPLIT_CALIBRATION:
                cal_counts[source] += 1

    assignment = {}
    conflicts = []
    for family_id, members in fam_rows.items():
        local_splits = {local_assignment[(row["source"], family_id)] for row in members}
        if len(local_splits) == 1:
            assignment[family_id] = next(iter(local_splits))
        else:
            conflicts.append(family_id)

    for family_id in sorted(conflicts, key=lambda f: global_family_sort_key(data_seed, f)):
        members = fam_rows[family_id]
        additions = {}
        # Remove the inconsistent provisional pieces before comparing the two
        # valid whole-family assignments.
        for row in members:
            source = row["source"]
            additions[source] = additions.get(source, 0) + 1
            if local_assignment[(source, family_id)] == SPLIT_CALIBRATION:
                cal_counts[source] -= 1
        objective_id = sum(abs(cal_counts[s] - targets[s]) for s in totals)
        objective_cal = sum(
            abs(cal_counts[s] + additions.get(s, 0) - targets[s]) for s in totals)
        if objective_cal < objective_id:
            chosen = SPLIT_CALIBRATION
        elif objective_cal > objective_id:
            chosen = SPLIT_ID
        else:
            # Stable tie-break independent of input order.
            chosen = (SPLIT_CALIBRATION
                      if int(global_family_sort_key(data_seed, family_id), 16) % 2 == 0
                      else SPLIT_ID)
        assignment[family_id] = chosen
        if chosen == SPLIT_CALIBRATION:
            for source, count in additions.items():
                cal_counts[source] += count

    cal_ids = {family_id for family_id, split in assignment.items()
               if split == SPLIT_CALIBRATION}
    return cal_ids, assignment


def route_calibration_conflicts_to_id(assignment, reported_test_family_ids):
    """Keep threshold-fitting families off every reported test/stress surface."""
    protected = set(reported_test_family_ids)
    routed = sorted(
        family_id for family_id, split in assignment.items()
        if split == SPLIT_CALIBRATION and family_id in protected)
    corrected = dict(assignment)
    for family_id in routed:
        corrected[family_id] = SPLIT_ID
    return corrected, routed
