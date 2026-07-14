#!/usr/bin/env python
"""Build the immutable, decontaminated Paper A data manifests (plan sec 6).

Pipeline (plan sec 6.4.2 removal precedence; eval-target selection is performed
before family clustering because it is train-independent and fully deterministic):

  1. load pinned upstream rows and apply the native->binary label map;
  2. require nonempty text; derive/preserve source_row_id;
  3. exact-deduplicate within source (keep smallest source_row_id when labels
     agree; QUARANTINE same-text conflicting labels; never unsafe-wins);
  4. construct evaluation candidate pools and select each eval target by frozen
     hash rank (represented-source cal+ID, transfer, single-class stress);
  5. remove EXACT normalized-text overlaps between train candidates and ANY eval
     row (this removes the 4 known jailbreak-classification <-> WildJailbreak rows);
  6. build family clusters over train-eligible U final-eval rows (upstream-id
     edges U char-5gram MinHash>=0.85 connected components);
  7. adjudicate cross-split (train vs eval) near-dup components: remove train-side;
  8. select the final 1,200-row train manifest by frozen hash rank (200 safe +
     200 unsafe per source); STOP if any stratum < 200 after audit;
  9. split represented-source test rows into calibration/ID at the family level.

Usage:
  .venv/bin/python experiments/prepare_paper_a_manifests.py \
    --config configs/paper_a_sft.yaml --out artifacts/paper_a_sft_v2/manifests \
    --public-out artifacts/paper_a_sft_v2/public_manifests
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict

# --- path bootstrap + shared lib ---------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_manifest_lib as L  # noqa: E402

FROZEN_PATH = os.path.join(_ROOT, "data", "frozen_eval_rows.json")


# =============================================================================
# environment / config
# =============================================================================
def load_env():
    def le(p):
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass
    le(os.path.join(_ROOT, ".env"))
    return os.environ.get("HF_TOKEN")


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# label maps
# =============================================================================
def _b(label_bool):
    return "unsafe" if label_bool else "safe"


def label_toxicchat(row):
    return _b(str(row.get("toxicity")).strip() == "1")


def label_promptinj(row):
    return _b(str(row.get("label")).strip() == "1")


def label_jailbreakclf(row):
    return _b(str(row.get("type") or "").strip().lower().startswith("jailbreak"))


def label_xstest(row):
    return _b(str(row.get("type") or "").strip().lower().startswith("contrast"))


# =============================================================================
# raw row loading
# =============================================================================
def _hf(token, path, config, split, revision):
    from datasets import load_dataset
    return load_dataset(path, config, split=split, revision=revision, token=token)


def make_row(source, cfg, rev, split, srid, text, label, prov, spec,
             upstream_family_id=None):
    ct = L.content_sha256(text)
    return {
        "source": source,
        "source_config": cfg,
        "source_revision": rev,
        "source_row_id": srid,
        "upstream_family_id": upstream_family_id,
        "role_split": split,          # working split; final `split` set later
        "text": text,
        "content_sha256": ct,
        "norm_text": L.normalize_text(text),
        "label": label,
        "gold": L.to_gold(label),
        "label_provenance": prov,
        "license_id": spec.get("_effective_license_id", spec.get("license_id", "unknown")),
        "redistribution_class": spec.get(
            "_effective_redistribution_class", spec.get("redistribution_class", "unknown")),
        "source_origin": None,        # set by loader
    }


def load_hf_source(token, name, spec, label_fn, id_field=None):
    """Load an HF source split(s), apply label map, derive source_row_id."""
    rev = spec["revision"]
    cfg = spec["hf_config"]
    field = spec["text_field"]
    prov = f"{spec['hf_path']}@{rev[:8]} :: {spec['label_rule']}"
    splits = spec["split"] if isinstance(spec["split"], list) else [spec["split"]]
    rows = []
    for split in splits:
        ds = _hf(token, spec["hf_path"], cfg, split, rev)
        for pos, r in enumerate(ds):
            text = (r.get(field) or "").strip()
            if not text:
                continue
            if label_fn == "split_harmful_benign":  # JailbreakBench
                label = "unsafe" if split == "harmful" else "safe"
            elif callable(label_fn):
                label = label_fn(r)
            else:
                label = label_fn  # constant "safe"/"unsafe"
            if label is None:
                continue
            up = r.get(id_field) if id_field else None
            if up not in (None, ""):
                srid = f"{cfg}/{split}/{up}"
            else:
                srid = f"{rev[:8]}/{cfg}/{split}/pos{pos}"
            upstream_family_id = None
            if up not in (None, ""):
                upstream_family_id = f"{name}:{id_field}:{up}"
            row = make_row(name, cfg, rev, spec["role"], srid, text, label, prov, spec,
                           upstream_family_id=upstream_family_id)
            row["source_origin"] = f"hf:{spec['hf_path']}@{rev[:8]}"
            rows.append(row)
    return rows


def load_wildguard_labeled(token, name, spec):
    """WildGuardMix wildguardtest: prompt_harm_label harmful->unsafe, unharmful->safe."""
    rev = spec["revision"]
    cfg = spec["hf_config"]
    prov = f"{spec['hf_path']}@{rev[:8]} :: {spec['label_rule']}"
    ds = _hf(token, spec["hf_path"], cfg, spec["split"], rev)
    rows = []
    for pos, r in enumerate(ds):
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        lab = str(r.get("prompt_harm_label") or "").strip().lower()
        if lab == "harmful":
            label = "unsafe"
        elif lab == "unharmful":
            label = "safe"
        else:
            continue  # discard "None"/other
        srid = f"{rev[:8]}/{cfg}/{spec['split']}/pos{pos}"
        row = make_row(name, cfg, rev, spec["role"], srid, text, label, prov, spec)
        row["source_origin"] = f"hf:{spec['hf_path']}@{rev[:8]}"
        rows.append(row)
    return rows


def load_wildjailbreak_labeled(token, name, spec):
    """WildJailbreak eval/train: adversarial field, label 1->unsafe, 0->safe."""
    rev = spec["revision"]
    cfg = spec["hf_config"]
    prov = f"{spec['hf_path']}@{rev[:8]} :: {spec['label_rule']}"
    ds = _hf(token, spec["hf_path"], cfg, spec["split"], rev)
    rows = []
    for pos, r in enumerate(ds):
        text = (r.get("adversarial") or "").strip()
        if not text:
            continue
        label = "unsafe" if str(r.get("label")).strip() == "1" else "safe"
        srid = f"{rev[:8]}/{cfg}/{spec['split']}/pos{pos}"
        row = make_row(name, cfg, rev, spec["role"], srid, text, label, prov, spec)
        row["source_origin"] = f"hf:{spec['hf_path']}@{rev[:8]}"
        rows.append(row)
    return rows


def load_xstest_labeled(token, name, spec):
    """Load XSTest while preserving all eight direct contrast relationships."""
    rev = spec["revision"]
    cfg = spec["hf_config"]
    split = spec["split"]
    prov = f"{spec['hf_path']}@{rev[:8]} :: {spec['label_rule']}"
    ds = _hf(token, spec["hf_path"], cfg, split, rev)
    type_ordinals = Counter()
    rows = []
    for pos, r in enumerate(ds):
        text = (r.get(spec["text_field"]) or "").strip()
        if not text:
            continue
        row_type = str(r.get("type") or "").strip()
        ordinal = type_ordinals[row_type]
        type_ordinals[row_type] += 1
        label = label_xstest(r)
        upstream_id = r.get("id")
        srid = f"{rev[:8]}/{cfg}/{split}/pos{pos}"
        group = L.XSTEST_DIRECT_CONTRAST_GROUPS.get(row_type)
        if group is not None:
            upstream_family_id = f"xstest:direct-contrast:{group}:{ordinal:02d}"
        else:
            # Preserve a stable identifier without inventing a dependency.
            stable = upstream_id if upstream_id not in (None, "") else f"pos{pos}"
            upstream_family_id = f"xstest:unpaired:{stable}"
        row = make_row(name, cfg, rev, spec["role"], srid, text, label, prov, spec,
                       upstream_family_id=upstream_family_id)
        row["source_origin"] = f"hf:{spec['hf_path']}@{rev[:8]}"
        rows.append(row)

    expected_types = set(L.XSTEST_DIRECT_CONTRAST_GROUPS)
    bad = {t: type_ordinals.get(t, 0) for t in expected_types if type_ordinals.get(t, 0) != 25}
    if bad:
        raise RuntimeError(f"XSTest pinned contrast-block shape drift: {bad}")
    grouped = defaultdict(list)
    for row in rows:
        if row["upstream_family_id"].startswith("xstest:direct-contrast:"):
            grouped[row["upstream_family_id"]].append(row)
    invalid = [fid for fid, members in grouped.items()
               if len(members) != 2 or {m["label"] for m in members} != {"safe", "unsafe"}]
    if invalid or len(grouped) != 200:
        raise RuntimeError(
            f"XSTest direct-contrast validation failed: families={len(grouped)} "
            f"invalid={invalid[:5]}")
    return rows


def load_frozen(name, spec, frozen, frozen_sha):
    """Load a transfer/stress eval set from the local frozen cache (gated sets)."""
    key = spec["frozen_key"]
    node = frozen["novel"][key]
    texts, gold = node["texts"], node["gold"]
    rev = spec["revision"]
    cfg = spec["hf_config"]
    prov = f"{spec['hf_path']}@{rev[:8]} :: {spec['label_rule']} (via frozen_eval_rows.json)"
    rows = []
    for pos, (t, g) in enumerate(zip(texts, gold)):
        text = (t or "").strip()
        if not text:
            continue
        label = "unsafe" if int(g) == 1 else "safe"
        srid = f"frozen/{key}/pos{pos}"
        row = make_row(name, cfg, rev, spec["role"], srid, text, label, prov, spec)
        row["source_origin"] = f"frozen_eval_rows.json@{frozen_sha[:8]}:novel.{key}"
        rows.append(row)
    return rows


# =============================================================================
# within-source exact dedup + quarantine (plan sec 6.4.1 steps 4-5)
# =============================================================================
def dedup_within_source(rows):
    """Group by content_sha256 within a source. Returns (kept, quarantined, dup_dropped).

    - all rows in a group share a label -> keep the lexicographically smallest
      source_row_id; drop the rest as exact duplicates;
    - conflicting labels in a group -> QUARANTINE the whole group (never unsafe-wins).
    """
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    kept, quarantined, dup_dropped = [], [], []
    for source, srows in by_source.items():
        groups = defaultdict(list)
        for r in srows:
            groups[r["content_sha256"]].append(r)
        for ch, grp in groups.items():
            labels = {g["label"] for g in grp}
            if len(labels) > 1:
                for g in grp:
                    g["known_overlap_disposition"] = "quarantined_conflicting_label_within_source"
                quarantined.extend(grp)
            else:
                grp_sorted = sorted(grp, key=lambda g: g["source_row_id"])
                kept.append(grp_sorted[0])
                dup_dropped.extend(grp_sorted[1:])
    return kept, quarantined, dup_dropped


# =============================================================================
# eval-target selection by frozen hash rank (plan sec 6.4.1 steps 6-7)
# =============================================================================
def _ranked(rows, data_seed):
    return sorted(rows, key=lambda r: L.rank_key(data_seed, r["source_row_id"], r["content_sha256"]))


def select_target(rows, target, data_seed):
    """Apply a source's deterministic target to its eligible (deduped) rows."""
    by_label = {"safe": [r for r in rows if r["label"] == "safe"],
                "unsafe": [r for r in rows if r["label"] == "unsafe"]}
    out = []
    if target == "at_most_400_per_label":
        for lab in ("safe", "unsafe"):
            out += _ranked(by_label[lab], data_seed)[:400]
    elif target == "balance_to_minority":
        m = min(len(by_label["safe"]), len(by_label["unsafe"]))
        for lab in ("safe", "unsafe"):
            out += _ranked(by_label[lab], data_seed)[:m]
    elif target == "60_per_label":
        for lab in ("safe", "unsafe"):
            out += _ranked(by_label[lab], data_seed)[:60]
    elif target == "120_per_label":
        for lab in ("safe", "unsafe"):
            out += _ranked(by_label[lab], data_seed)[:120]
    elif target == "400_per_label":
        for lab in ("safe", "unsafe"):
            out += _ranked(by_label[lab], data_seed)[:400]
    elif target == "400_rows":
        out += _ranked(rows, data_seed)[:400]
    elif target == "200_rows":
        out += _ranked(rows, data_seed)[:200]
    else:
        raise ValueError(f"unknown target: {target}")
    return out


# =============================================================================
# main build
# =============================================================================
def _redact_text(rec):
    """Copy of an overlap record with raw text fields removed (keep hash+ids)."""
    return {k: v for k, v in rec.items() if k != "normalized_text"}


PUBLIC_BANNED_CONTENT_KEYS = {
    "text",
    "raw_text",
    "norm_text",
    "normalized_text",
    "text_or_download_reference",
    "prompt",
    "response",
    "completion",
    "adversarial",
}

PUBLIC_ROW_FIELDS = (
    "sample_id",
    "source",
    "source_config",
    "source_revision",
    "source_row_id",
    "upstream_family_id",
    "split",
    "label",
    "gold",
    "label_provenance",
    "content_sha256",
    "family_id",
    "license_id",
    "redistribution_class",
    "known_overlap_disposition",
    "source_origin",
)

POLICY_LABEL_CROSSWALK = {
    "toxicchat": {
        "native_field": "toxicity", "native_to_binary": {"1": "unsafe", "else": "safe"},
        "caveat": "Toxicity annotation; not a measured jailbreak or injection success outcome.",
    },
    "prompt_injections": {
        "native_field": "label", "native_to_binary": {"1": "unsafe", "0": "safe"},
        "caveat": "Prompt-injection class label; not a measured attack-success outcome.",
    },
    "jailbreak_classification": {
        "native_field": "type",
        "native_to_binary": {"prefix jailbreak": "unsafe", "otherwise": "safe"},
        "caveat": "Prompt-type label; not a measured jailbreak-success outcome.",
    },
    "jailbreakbench": {
        "native_field": "dataset split",
        "native_to_binary": {"harmful": "unsafe", "benign": "safe"},
        "caveat": "Behavior-policy class; matched harmful/benign indexes form one family.",
    },
    "xstest": {
        "native_field": "type",
        "native_to_binary": {"prefix contrast_": "unsafe", "otherwise": "safe"},
        "caveat": "Functional contrast class, not observed harm; eight direct contrast blocks are paired.",
    },
    "wildguardtest": {
        "native_field": "prompt_harm_label",
        "native_to_binary": {"harmful": "unsafe", "unharmful": "safe"},
        "caveat": "Prompt-harm annotation; response labels are not used.",
    },
    "wildjailbreak": {
        "native_field": "label", "native_to_binary": {"1": "unsafe", "0": "safe"},
        "caveat": "Adversarial prompt class; not a measured jailbreak-success outcome.",
    },
    "orbench": {
        "native_field": "benchmark membership", "native_to_binary": {"all": "safe"},
        "caveat": "Single-class benign over-refusal stress set.",
    },
    "harmbench": {
        "native_field": "benchmark membership", "native_to_binary": {"all": "unsafe"},
        "caveat": "Single-class harmful-behavior recall stress set.",
    },
}


def recursively_redact_public(value):
    """Remove raw-content keys at every nesting level from a public artifact."""
    if isinstance(value, dict):
        return {
            key: recursively_redact_public(item)
            for key, item in value.items()
            if str(key).lower() not in PUBLIC_BANNED_CONTENT_KEYS
        }
    if isinstance(value, list):
        return [recursively_redact_public(item) for item in value]
    return value


def assert_public_artifact_redacted(value, path="$"):
    """Fail closed if a recursively nested raw-content field remains."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in PUBLIC_BANNED_CONTENT_KEYS:
                raise ValueError(f"public artifact contains banned key {path}.{key}")
            assert_public_artifact_redacted(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_public_artifact_redacted(item, f"{path}[{i}]")


def recover_public_upstream_family_id(row):
    """Recover stable public pair IDs for legacy manifests that predate the field."""
    if row.get("upstream_family_id"):
        return row["upstream_family_id"]
    source = row.get("source")
    source_row_id = str(row.get("source_row_id", ""))
    if source == "jailbreakbench" and "/" in source_row_id:
        return f"jailbreakbench:Index:{source_row_id.rsplit('/', 1)[-1]}"
    if source == "xstest" and "/pos" in source_row_id:
        try:
            pos = int(source_row_id.rsplit("/pos", 1)[-1])
        except ValueError:
            return None
        blocks = {
            0: "homonyms", 25: "homonyms",
            50: "figurative_language", 75: "figurative_language",
            100: "safe_targets", 125: "safe_targets",
            150: "safe_contexts", 175: "safe_contexts",
            200: "definitions", 225: "definitions",
            275: "discrimination", 300: "discrimination",
            325: "historical_events", 350: "historical_events",
            400: "privacy_fictional", 425: "privacy_fictional",
        }
        start = (pos // 25) * 25
        if start in blocks:
            return f"xstest:direct-contrast:{blocks[start]}:{pos % 25:02d}"
        return f"xstest:unpaired:pos{pos}"
    return None


def project_public_row(raw, effective_license_by_source):
    """Deterministic text-free projection used by both publisher and auditor."""
    public_row = {field: raw.get(field) for field in PUBLIC_ROW_FIELDS}
    public_row["upstream_family_id"] = recover_public_upstream_family_id(raw)
    license_meta = effective_license_by_source.get(raw.get("source"))
    if license_meta:
        public_row["license_id"] = license_meta["license_id"]
        public_row["redistribution_class"] = license_meta["redistribution_class"]
    return recursively_redact_public(public_row)


def write_public_release(raw_manifest_path, public_out_dir, source_license_overrides=None):
    """Write a text-free public row index and recursively redacted manifest.

    The raw manifests remain local.  Public JSONL files expose only stable IDs,
    hashes, labels, family assignments, revisions, and license/provenance fields.
    """
    raw_manifest = json.load(open(raw_manifest_path, encoding="utf-8"))
    raw_dir = os.path.dirname(raw_manifest_path)
    os.makedirs(public_out_dir, exist_ok=True)
    public_files = {}
    raw_commitments = {}
    all_raw_rows = []
    effective_license_by_source = {}
    source_license_overrides = source_license_overrides or {}
    for source_key, source_meta in raw_manifest.get("sources", {}).items():
        emitted = source_meta.get("emitted_source")
        resolved = L.resolved_license_metadata(
            source_license_overrides.get(source_key, source_meta))
        effective_license_by_source[emitted] = resolved
    for stem in L.MANIFEST_FILES:
        raw_path = os.path.join(raw_dir, f"{stem}.jsonl")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"cannot publish missing raw manifest: {raw_path}")
        public_rows = []
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                all_raw_rows.append(raw)
                public_rows.append(project_public_row(raw, effective_license_by_source))
        assert_public_artifact_redacted(public_rows)
        public_path = os.path.join(public_out_dir, f"{stem}.jsonl")
        with open(public_path, "w", encoding="utf-8") as f:
            for row in public_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        public_files[stem] = {
            "path": os.path.relpath(public_path, _ROOT),
            "n_rows": len(public_rows),
            "sha256": L.sha256_of_file(public_path),
        }
        raw_commitments[stem] = {
            "n_rows": len(public_rows),
            "sha256": L.sha256_of_file(raw_path),
        }

    crosswalk = {}
    for key, source_meta in raw_manifest.get("sources", {}).items():
        emitted = source_meta.get("emitted_source")
        if emitted not in POLICY_LABEL_CROSSWALK:
            continue
        license_meta = L.resolved_license_metadata(
            source_license_overrides.get(key, source_meta))
        record = copy.deepcopy(POLICY_LABEL_CROSSWALK[emitted])
        record.update({
            "source": emitted,
            "source_config_key": key,
            "hf_path": source_meta.get("hf_path"),
            "hf_config": source_meta.get("hf_config"),
            "revision": source_meta.get("revision"),
            "role": source_meta.get("role"),
            "license_id": license_meta["license_id"],
            "license_metadata": license_meta,
        })
        crosswalk[key] = record
    crosswalk_doc = {
        "schema_version": 1,
        "description": "Native dataset annotations mapped to the study's binary prompt-policy labels.",
        "sources": crosswalk,
    }
    assert_public_artifact_redacted(crosswalk_doc)
    crosswalk_path = os.path.join(public_out_dir, "policy_label_crosswalk.json")
    with open(crosswalk_path, "w", encoding="utf-8") as f:
        json.dump(crosswalk_doc, f, indent=2, ensure_ascii=False, sort_keys=True)

    by_content = defaultdict(list)
    for row in all_raw_rows:
        by_content[row["content_sha256"]].append(row)
    final_conflicts = []
    for content_sha, rows in sorted(by_content.items()):
        if len({r["label"] for r in rows}) < 2 or len({r["source"] for r in rows}) < 2:
            continue
        final_conflicts.append({
            "content_sha256": content_sha,
            "rows": [{field: row.get(field) for field in (
                "sample_id", "source", "source_revision", "source_row_id",
                "upstream_family_id", "split", "label", "gold", "family_id")}
                     for row in rows],
        })
    removed_conflicts = []
    for record in raw_manifest.get("removals", {}).get(
            "exact_train_vs_eval", {}).get("records", []):
        if set(record.get("eval_labels", [])) == {record.get("train_label")}:
            continue
        removed_conflicts.append({
            key: record.get(key) for key in (
                "content_sha256", "train_source", "train_source_row_id",
                "train_label", "eval_sources", "eval_labels")
        })
    conflict_doc = {
        "schema_version": 1,
        "matching_rule": "exact SHA-256 of NFKC-lower-collapsed-whitespace content",
        "final_manifest_cross_source_conflicts": final_conflicts,
        "removed_train_eval_conflicts": removed_conflicts,
    }
    assert_public_artifact_redacted(conflict_doc)
    conflict_path = os.path.join(public_out_dir, "contradictory_label_inventory.json")
    with open(conflict_path, "w", encoding="utf-8") as f:
        json.dump(conflict_doc, f, indent=2, ensure_ascii=False, sort_keys=True)

    supplemental_files = {
        "policy_label_crosswalk": {
            "path": os.path.relpath(crosswalk_path, _ROOT),
            "sha256": L.sha256_of_file(crosswalk_path),
        },
        "contradictory_label_inventory": {
            "path": os.path.relpath(conflict_path, _ROOT),
            "sha256": L.sha256_of_file(conflict_path),
        },
    }

    public_manifest = recursively_redact_public(raw_manifest)
    for key, source_meta in public_manifest.get("sources", {}).items():
        resolved = L.resolved_license_metadata(
            source_license_overrides.get(key, raw_manifest["sources"][key]))
        source_meta["license_id"] = resolved["license_id"]
        source_meta["redistribution_class"] = resolved["redistribution_class"]
        source_meta["license_metadata"] = resolved
    raw_source_mode = raw_manifest.get("source_mode")
    family_stats = raw_manifest.get("family_stats", {})
    calibration_families = {
        row["family_id"] for row in all_raw_rows if row.get("split") == L.SPLIT_CALIBRATION}
    id_families = {
        row["family_id"] for row in all_raw_rows if row.get("split") == L.SPLIT_ID}
    limitations = []
    if raw_source_mode != "huggingface_pinned_revisions":
        limitations.append(
            "Historical v1 snapshot uses the previously inspected legacy frozen transfer cohorts.")
    if int(family_stats.get("n_upstream_edges", 0)) < 94:
        limitations.append(
            "Historical family IDs omit JailbreakBench and XSTest authoritative pair edges.")
    shared_cal_id = calibration_families & id_families
    if shared_cal_id:
        limitations.append(
            f"Historical calibration and ID assignments share {len(shared_cal_id)} global families.")
    observed_backend = raw_manifest.get("provenance", {}).get("minhash_backend")
    if observed_backend != L.MINHASH_BACKEND:
        limitations.append(
            f"Historical manifest records MinHash backend {observed_backend!r}, not the pinned v2 name.")
    if limitations:
        limitations.append(
            "This public index is an auditable commitment to historical rows, not corrected rerun evidence.")

    public_manifest["public_schema_version"] = 2
    public_manifest["source_contract"] = (
        "pinned_hf_v2" if not limitations else "legacy_v1_snapshot")
    public_manifest["clean_rerun_compatible"] = not limitations
    public_manifest["known_integrity_limitations"] = limitations
    public_manifest["release_kind"] = "text_free_identifier_hash_index"
    public_manifest["row_schema_fields"] = list(PUBLIC_ROW_FIELDS)
    public_manifest["files"] = public_files
    public_manifest["supplemental_files"] = supplemental_files
    public_manifest["raw_artifact_commitment"] = {
        "manifest_sha256": L.sha256_of_file(raw_manifest_path),
        "splits": raw_commitments,
    }
    assert_public_artifact_redacted(public_manifest)
    public_manifest_path = os.path.join(public_out_dir, "manifest.json")
    with open(public_manifest_path, "w", encoding="utf-8") as f:
        json.dump(public_manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    return {
        "manifest_path": public_manifest_path,
        "manifest_sha256": L.sha256_of_file(public_manifest_path),
        "files": public_files,
        "supplemental_files": supplemental_files,
    }


def _prepare_sources(config_sources):
    """Validate immutable revisions and attach effective license metadata."""
    sources = copy.deepcopy(config_sources)
    for key, spec in sources.items():
        revision = str(spec.get("revision", ""))
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"{key}: source revision must be a full 40-hex commit, got {revision!r}")
        license_meta = L.resolved_license_metadata(spec)
        spec["_license_metadata"] = license_meta
        spec["_effective_license_id"] = license_meta["license_id"]
        spec["_effective_redistribution_class"] = license_meta["redistribution_class"]
        if str(license_meta["license_id"]).lower().startswith("unknown"):
            raise ValueError(f"{key}: unresolved license at pinned revision {revision}")
    return sources


def build(config_path, out_dir, *, allow_legacy_frozen_cohorts=False,
          public_out_dir=None):
    token = load_env()
    cfg = load_config(config_path)
    data_seed = cfg["data_seed"]
    sources = _prepare_sources(cfg["sources"])
    os.makedirs(out_dir, exist_ok=True)

    # Pinned Hugging Face revisions are the only default.  The legacy cache is
    # a previously inspected seed-7 cohort and is available solely for explicit
    # retrospective reproduction.
    frozen = None
    frozen_sha = None
    if allow_legacy_frozen_cohorts:
        if not os.path.exists(FROZEN_PATH):
            raise FileNotFoundError(
                f"legacy frozen cohort requested but missing: {FROZEN_PATH}")
        frozen = json.load(open(FROZEN_PATH))
        frozen_sha = L.sha256_of_file(FROZEN_PATH)
        print("!! LEGACY RETROSPECTIVE MODE: using previously inspected frozen cohorts !!",
              flush=True)

    LABEL_FN = {
        "toxicchat": label_toxicchat, "toxicchat_test": label_toxicchat,
        "prompt_injections": label_promptinj, "prompt_injections_test": label_promptinj,
        "jailbreak_classification": label_jailbreakclf,
        "jailbreak_classification_test": label_jailbreakclf,
        "xstest": label_xstest,
    }
    ID_FIELD = {"toxicchat": "conv_id", "toxicchat_test": "conv_id",
                "jailbreakbench": "Index"}
    # emitted `source` value per config key (aligns represented train/test)
    SRC_NAME = {
        "toxicchat": "toxicchat", "toxicchat_test": "toxicchat",
        "prompt_injections": "prompt_injections", "prompt_injections_test": "prompt_injections",
        "jailbreak_classification": "jailbreak_classification",
        "jailbreak_classification_test": "jailbreak_classification",
        "jailbreakbench": "jailbreakbench", "xstest": "xstest",
        "wildguardtest": "wildguardtest", "wildjailbreak": "wildjailbreak",
        "orbench_hard": "orbench", "harmbench": "harmbench",
    }

    print("== loading sources ==", flush=True)
    raw = {}
    for key, spec in sources.items():
        role = spec["role"]
        use_legacy_frozen = allow_legacy_frozen_cohorts and key in {
            "wildguardtest", "wildjailbreak"}
        if key in ("wildguardtest",):
            if use_legacy_frozen:
                rows = load_frozen(SRC_NAME[key], spec, frozen, frozen_sha)
            else:
                rows = load_wildguard_labeled(token, SRC_NAME[key], spec)
        elif key in ("wildjailbreak",):
            if use_legacy_frozen:
                rows = load_frozen(SRC_NAME[key], spec, frozen, frozen_sha)
            else:
                rows = load_wildjailbreak_labeled(token, SRC_NAME[key], spec)
        elif key == "jailbreakbench":
            rows = load_hf_source(token, SRC_NAME[key], spec, "split_harmful_benign",
                                  id_field=ID_FIELD.get(key))
        elif key == "xstest":
            rows = load_xstest_labeled(token, SRC_NAME[key], spec)
        elif key == "orbench_hard":
            rows = load_hf_source(token, SRC_NAME[key], spec, "safe")
        elif key == "harmbench":
            rows = load_hf_source(token, SRC_NAME[key], spec, "unsafe")
        else:
            rows = load_hf_source(token, SRC_NAME[key], spec, LABEL_FN[key],
                                  id_field=ID_FIELD.get(key))
        raw[key] = rows
        print(f"  {key:32s} role={role:16s} rows={len(rows)} "
              f"({Counter(r['label'] for r in rows)})", flush=True)

    # ---- within-source exact dedup + quarantine (all sources) ---------------
    print("== exact dedup within source ==", flush=True)
    eligible = {}
    quarantine_records = []
    dedup_stats = {}
    for key, rows in raw.items():
        kept, quar, dup = dedup_within_source(rows)
        eligible[key] = kept
        dedup_stats[key] = {"raw": len(rows), "kept": len(kept),
                            "quarantined": len(quar), "exact_dup_dropped": len(dup)}
        for q in quar:
            quarantine_records.append({
                "source": q["source"], "source_row_id": q["source_row_id"],
                "content_sha256": q["content_sha256"],
                "disposition": q["known_overlap_disposition"]})
        if quar or dup:
            print(f"  {key}: kept={len(kept)} quarantined={len(quar)} dup_dropped={len(dup)}",
                  flush=True)

    # ---- construct + select eval final sets --------------------------------
    print("== select eval targets (frozen hash rank) ==", flush=True)
    eval_final = {}   # key -> selected rows
    for key, spec in sources.items():
        if spec["role"] == "train":
            continue
        sel = select_target(eligible[key], spec["target"], data_seed)
        eval_final[key] = sel
        print(f"  {key:32s} target={spec['target']:22s} -> {len(sel)} "
              f"({Counter(r['label'] for r in sel)})", flush=True)

    # flatten all eval rows (every row a train candidate must be disjoint from)
    eval_rows_all = [r for k, rows in eval_final.items() for r in rows]
    eval_hash_to_rows = defaultdict(list)
    for r in eval_rows_all:
        eval_hash_to_rows[r["content_sha256"]].append(r)

    # ---- remove EXACT train<->eval overlaps (train side) -------------------
    print("== remove exact train<->eval overlaps ==", flush=True)
    train_keys = [k for k, s in sources.items() if s["role"] == "train"]
    exact_overlap_records = []
    train_after_exact = {}
    for key in train_keys:
        keep, removed = [], []
        for r in eligible[key]:
            hit = eval_hash_to_rows.get(r["content_sha256"])
            if hit:
                eval_srcs = sorted({h["source"] for h in hit})
                r["known_overlap_disposition"] = f"removed_exact_overlap_with:{','.join(eval_srcs)}"
                removed.append(r)
                exact_overlap_records.append({
                    "train_source": r["source"], "train_source_row_id": r["source_row_id"],
                    "train_label": r["label"], "content_sha256": r["content_sha256"],
                    "eval_sources": eval_srcs,
                    "eval_labels": sorted({h["label"] for h in hit}),
                    "normalized_text": r["norm_text"]})
            else:
                keep.append(r)
        train_after_exact[key] = keep
        if removed:
            print(f"  {key}: removed {len(removed)} exact-overlap rows", flush=True)

    # the 4 known jailbreak-classification <-> wildjailbreak rows
    jc_wjb = [e for e in exact_overlap_records
              if e["train_source"] == "jailbreak_classification"
              and "wildjailbreak" in e["eval_sources"]]

    # ---- family clustering over train-eligible U final-eval ----------------
    print("== family clustering (MinHash 5-gram, 256 perm, J>=0.85) ==", flush=True)
    graph_rows = []
    for key in train_keys:
        for r in train_after_exact[key]:
            r["_graph_side"] = "train"
            graph_rows.append(r)
    for r in eval_rows_all:
        r["_graph_side"] = "eval"
        graph_rows.append(r)
    texts = [r["norm_text"] for r in graph_rows]
    chashes = [r["content_sha256"] for r in graph_rows]
    # Authoritative family edges are independent of split and row serialization.
    # In particular this joins JailbreakBench harmful/benign matched indexes and
    # all eight XSTest direct-contrast blocks when both members are selected.
    up_index = defaultdict(list)
    for i, r in enumerate(graph_rows):
        upstream_family_id = r.get("upstream_family_id")
        if upstream_family_id:
            up_index[upstream_family_id].append(i)
    upstream_edges = []
    for members in up_index.values():
        for j in members[1:]:
            upstream_edges.append((members[0], j))
    family_of, sigs, cand_pairs, edges085, comps, fam_stats = L.build_families(
        texts, chashes, upstream_edges=upstream_edges)
    for i, r in enumerate(graph_rows):
        r["family_id"] = family_of[i]
    print(f"  rows={fam_stats['n_rows']} components={fam_stats['n_components']} "
          f"minhash_edges={fam_stats['n_minhash_edges_085']} "
          f"multi_family={fam_stats['n_multi_member_components']} "
          f"backend={fam_stats['provenance_source']}", flush=True)

    # ---- cross-split near-dup adjudication: remove train-side ---------------
    print("== cross-split near-dup adjudication ==", flush=True)
    comp_sides = []
    cross_split_records = []
    train_remove_ids = set()
    for members in comps:
        sides = {graph_rows[m]["_graph_side"] for m in members}
        if sides == {"train", "eval"} or (len(members) > 1 and "train" in sides and "eval" in sides):
            fam = graph_rows[members[0]]["family_id"]
            train_members = [graph_rows[m] for m in members if graph_rows[m]["_graph_side"] == "train"]
            eval_members = [graph_rows[m] for m in members if graph_rows[m]["_graph_side"] == "eval"]
            for tm in train_members:
                tm["known_overlap_disposition"] = "removed_cross_split_near_dup"
                train_remove_ids.add((tm["source"], tm["source_row_id"]))
            cross_split_records.append({
                "family_id": fam,
                "train_members": [{"source": t["source"], "source_row_id": t["source_row_id"],
                                   "label": t["label"]} for t in train_members],
                "eval_members": [{"source": e["source"], "source_row_id": e["source_row_id"],
                                  "label": e["label"]} for e in eval_members],
                "disposition": "removed_train_side"})
    print(f"  cross-split near-dup components: {len(cross_split_records)} "
          f"(train rows removed: {len(train_remove_ids)})", flush=True)

    train_survivors = {}
    for key in train_keys:
        survivors = [r for r in train_after_exact[key]
                     if (r["source"], r["source_row_id"]) not in train_remove_ids]
        train_survivors[key] = survivors

    # ---- select final 1,200-row train by frozen hash rank ------------------
    print("== select final train (200 safe + 200 unsafe per source) ==", flush=True)
    per_label = cfg["rows_per_source_label"]
    shortfalls = []
    train_selected = []
    train_breakdown = {}
    for key in train_keys:
        src = SRC_NAME[key]
        picked = {}
        for lab in ("safe", "unsafe"):
            pool = _ranked([r for r in train_survivors[key] if r["label"] == lab], data_seed)
            if len(pool) < per_label:
                shortfalls.append({"source": src, "label": lab,
                                   "available": len(pool), "target": per_label})
            picked[lab] = pool[:per_label]
        train_breakdown[src] = {"safe": len(picked["safe"]), "unsafe": len(picked["unsafe"])}
        for lab in ("safe", "unsafe"):
            for r in picked[lab]:
                r["known_overlap_disposition"] = "none"
                train_selected.append(r)
        print(f"  {src:28s} safe={len(picked['safe'])} unsafe={len(picked['unsafe'])} "
              f"(survivors safe={sum(1 for r in train_survivors[key] if r['label']=='safe')} "
              f"unsafe={sum(1 for r in train_survivors[key] if r['label']=='unsafe')})", flush=True)

    if shortfalls:
        print("\n!! STOP: stratum below target after audit (plan sec 6.1). "
              "Do NOT resample with replacement. Revise the manifest rule.", flush=True)
        for s in shortfalls:
            print(f"   shortfall: {s}", flush=True)
        raise SystemExit(2)

    # ---- calibration / ID family-level split (plan sec 6.4.2 step 10) ------
    print("== calibration/ID family split (40/60) ==", flush=True)
    represented_test_keys = [k for k, s in sources.items() if s["role"] == "represented_test"]
    represented_rows = [r for key in represented_test_keys for r in eval_final[key]]
    _, global_assignment = L.split_calibration_id_global(
        represented_rows, data_seed, cal_frac=0.40)
    # Calibration is used to choose deployment thresholds, so it must be
    # family-disjoint from *every* reported test/stress surface, not only ID.
    noncal_eval_rows = [
        row for key, spec in sources.items()
        if spec["role"] != "represented_test"
        for row in eval_final.get(key, [])
    ]
    noncal_eval_families = {row["family_id"] for row in noncal_eval_rows}
    global_assignment, routed_from_calibration = L.route_calibration_conflicts_to_id(
        global_assignment, noncal_eval_families)
    cal_rows, id_rows = [], []
    calid_summary = {}
    for key in represented_test_keys:
        src = SRC_NAME[key]
        srows = eval_final[key]
        c_safe = c_unsafe = i_safe = i_unsafe = 0
        for r in srows:
            split = global_assignment[r["family_id"]]
            r["known_overlap_disposition"] = "none"
            if split == L.SPLIT_CALIBRATION:
                r["_final_split"] = L.SPLIT_CALIBRATION
                cal_rows.append(r)
                c_safe += r["label"] == "safe"; c_unsafe += r["label"] == "unsafe"
            else:
                r["_final_split"] = L.SPLIT_ID
                id_rows.append(r)
                i_safe += r["label"] == "safe"; i_unsafe += r["label"] == "unsafe"
        calid_summary[src] = {
            "total": len(srows), "cal": c_safe + c_unsafe, "id": i_safe + i_unsafe,
            "cal_safe": c_safe, "cal_unsafe": c_unsafe, "id_safe": i_safe, "id_unsafe": i_unsafe,
            "cal_frac_realized": round((c_safe + c_unsafe) / max(1, len(srows)), 4)}
        print(f"  {src:28s} cal={c_safe+c_unsafe} (s{c_safe}/u{c_unsafe}) "
              f"id={i_safe+i_unsafe} (s{i_safe}/u{i_unsafe})", flush=True)

    # calibration assertions (plan sec 6.4.2)
    calib_problems = []
    pooled_cal_neg = sum(v["cal_safe"] for v in calid_summary.values())
    for src, v in calid_summary.items():
        if v["cal_safe"] == 0 or v["cal_unsafe"] == 0 or v["id_safe"] == 0 or v["id_unsafe"] == 0:
            calib_problems.append(f"{src}: both labels required in cal and ID "
                                  f"(cal s{v['cal_safe']}/u{v['cal_unsafe']}, id s{v['id_safe']}/u{v['id_unsafe']})")
        if v["cal_safe"] < 10:
            calib_problems.append(f"{src}: <10 calibration negatives (cal_safe={v['cal_safe']})")
    if pooled_cal_neg < 60:
        calib_problems.append(f"pooled calibration negatives {pooled_cal_neg} < 60")
    if calib_problems:
        print("\n!! STOP: calibration/ID split assertion failed (plan sec 6.4.2).", flush=True)
        for p in calib_problems:
            print("   " + p, flush=True)
        raise SystemExit(3)

    cal_families = {r["family_id"] for r in cal_rows}
    id_families = {r["family_id"] for r in id_rows}
    shared_cal_id = cal_families & id_families
    if shared_cal_id:
        raise RuntimeError(
            f"global calibration/ID split leaked {len(shared_cal_id)} family ids")
    shared_cal_reported = cal_families & noncal_eval_families
    if shared_cal_reported:
        raise RuntimeError(
            "global calibration split leaked families into transfer/stress surfaces: "
            f"{len(shared_cal_reported)}")

    # ---- assemble + write manifests ----------------------------------------
    print("== writing manifests ==", flush=True)

    def finalize(r, split):
        return {
            "sample_id": f"{r['source']}::{r['source_row_id']}",
            "source": r["source"],
            "source_config": r["source_config"],
            "source_revision": r["source_revision"],
            "source_row_id": r["source_row_id"],
            "upstream_family_id": r.get("upstream_family_id"),
            "split": split,
            "label": r["label"],
            "gold": r["gold"],
            "label_provenance": r["label_provenance"],
            "text_or_download_reference": r["text"],
            "content_sha256": r["content_sha256"],
            "family_id": r["family_id"],
            "license_id": r["license_id"],
            "redistribution_class": r["redistribution_class"],
            "known_overlap_disposition": r.get("known_overlap_disposition", "none"),
            "source_origin": r["source_origin"],
        }

    SRC_ORDER = {"toxicchat": 0, "prompt_injections": 1, "jailbreak_classification": 2,
                 "jailbreakbench": 3, "xstest": 4, "wildguardtest": 5, "wildjailbreak": 6,
                 "orbench": 7, "harmbench": 8}

    def order(rows):
        return sorted(rows, key=lambda r: (
            SRC_ORDER.get(r["source"], 99), 0 if r["label"] == "safe" else 1,
            L.rank_key(data_seed, r["source_row_id"], r["content_sha256"])))

    transfer_keys = [k for k, s in sources.items() if s["role"] == "transfer"]
    transfer_rows = [r for k in transfer_keys for r in eval_final[k]]
    orbench_rows = eval_final["orbench_hard"]
    harmbench_rows = eval_final["harmbench"]

    manifests = {
        "train": [finalize(r, L.SPLIT_TRAIN) for r in order(train_selected)],
        "calibration": [finalize(r, L.SPLIT_CALIBRATION) for r in order(cal_rows)],
        "id_test": [finalize(r, L.SPLIT_ID) for r in order(id_rows)],
        "transfer_test": [finalize(r, L.SPLIT_TRANSFER) for r in order(transfer_rows)],
        "orbench_safe_stress": [finalize(r, L.SPLIT_ORBENCH) for r in order(orbench_rows)],
        "harmbench_positive_stress": [finalize(r, L.SPLIT_HARMBENCH) for r in order(harmbench_rows)],
    }

    file_meta = {}
    for stem, rows in manifests.items():
        path = os.path.join(out_dir, f"{stem}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file_meta[stem] = {
            "path": os.path.relpath(path, _ROOT),
            "n_rows": len(rows),
            "sha256": L.sha256_of_file(path),
            "class_counts": dict(Counter(r["label"] for r in rows)),
            "per_source": {s: dict(Counter(r["label"] for r in rows if r["source"] == s))
                           for s in sorted({r["source"] for r in rows})},
        }
        print(f"  wrote {stem}.jsonl  n={len(rows)}  sha={file_meta[stem]['sha256'][:12]}",
              flush=True)

    # ---- manifest.json ------------------------------------------------------
    source_report = {}
    for key, spec in sources.items():
        effective_origin = (
            "legacy_frozen_seed7_opt_in"
            if allow_legacy_frozen_cohorts and key in {"wildguardtest", "wildjailbreak"}
            else "huggingface_pinned_revision"
        )
        source_report[key] = {
            "emitted_source": SRC_NAME[key], "role": spec["role"],
            "hf_path": spec["hf_path"], "hf_config": spec["hf_config"],
            "split": spec["split"], "revision": spec["revision"],
            "text_field": spec.get("text_field"), "label_rule": spec["label_rule"],
            "license_id": spec.get("_effective_license_id"),
            "redistribution_class": spec.get("_effective_redistribution_class"),
            "license_metadata": spec.get("_license_metadata"),
            "configured_origin": spec.get("origin", "hf"),
            "effective_origin": effective_origin,
            "target": spec.get("target"),
            "raw_rows": dedup_stats.get(key, {}).get("raw"),
            "eligible_after_dedup": dedup_stats.get(key, {}).get("kept"),
        }

    manifest = {
        "schema_version": cfg["schema_version"],
        "study_id": cfg["study_id"],
        "data_branch": cfg["data_branch"],
        "data_seed": data_seed,
        "data_order_seed": cfg["data_order_seed"],
        "row_schema_fields": L.ROW_SCHEMA_FIELDS,
        "build_order_note": (
            "plan sec 6.4.2; eval-target selection performed before family "
            "clustering (train-independent + deterministic); family graph and "
            "exact/near-dup checks scoped to train-eligible U final-eval rows."
        ),
        "config_path": os.path.relpath(config_path, _ROOT),
        "config_sha256": L.sha256_of_obj(cfg),
        "frozen_eval_rows_sha256": frozen_sha,
        "source_mode": (
            "legacy_frozen_seed7_retrospective_opt_in"
            if allow_legacy_frozen_cohorts else "huggingface_pinned_revisions"
        ),
        "provenance": {
            "provenance_source": L.PROVENANCE_SOURCE,
            "minhash_backend": L.MINHASH_BACKEND,
            "minhash_algorithm_version": L.MINHASH_ALGORITHM_VERSION,
            "minhash_seed": L.MINHASH_SEED,
            "normalization": "NFKC+lowercase+collapse_whitespace",
            "minhash_num_perm": L.NUM_PERM, "minhash_ngram": L.NGRAM,
            "minhash_jaccard_threshold": L.MINHASH_JACCARD_THRESHOLD,
            "lsh_bands": L.LSH_BANDS, "lsh_rows": L.LSH_ROWS,
            "rank_rule": "sha256(data_seed|source_row_id|content_sha256)",
        },
        "sources": source_report,
        "files": file_meta,
        "train_breakdown": train_breakdown,
        "train_total": sum(v["safe"] + v["unsafe"] for v in train_breakdown.values()),
        "calibration_id_split": calid_summary,
        "calibration_family_routing": {
            "policy": "families present on transfer or stress surfaces route to id_test",
            "n_families_routed_from_calibration": len(routed_from_calibration),
            "routed_family_ids": routed_from_calibration,
            "calibration_reported_test_shared_families": 0,
        },
        "pooled_calibration_negatives": pooled_cal_neg,
        "family_stats": fam_stats,
        "dedup_stats": dedup_stats,
        "removals": {
            # normalized_text is redacted from this tracked artifact: raw
            # third-party text is not redistributed (plan sec 6.2 Branch A / 6.5).
            # The (source, source_revision, source_row_id, content_sha256) tuple is
            # the reconstruction key; full text is printed to stdout at build time.
            "exact_train_vs_eval": {
                "count": len(exact_overlap_records),
                "records": [_redact_text(e) for e in exact_overlap_records]},
            "cross_split_near_dup": {
                "count": len(cross_split_records), "components": cross_split_records},
            "quarantined_conflicting_within_source": {
                "count": len(quarantine_records), "records": quarantine_records},
        },
        "known_wildjailbreak_overlaps": {
            "count": len(jc_wjb),
            "expected": 4,
            "note": "raw text redacted; reconstruct from source_revision+source_row_id",
            "records": [_redact_text(e) for e in jc_wjb],
        },
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"  wrote manifest.json  sha={L.sha256_of_file(mpath)[:12]}", flush=True)
    if public_out_dir is None:
        public_out_dir = os.path.join(os.path.dirname(os.path.abspath(out_dir)),
                                      "public_manifests")
    public_release = write_public_release(
        mpath, public_out_dir, source_license_overrides=cfg.get("sources"))
    print(f"  wrote public text-free index  sha={public_release['manifest_sha256'][:12]}",
          flush=True)

    # ---- summary ------------------------------------------------------------
    print("\n== SUMMARY ==", flush=True)
    print(f"train total: {manifest['train_total']}  breakdown: {train_breakdown}", flush=True)
    print(f"calibration: {file_meta['calibration']['n_rows']}  id_test: {file_meta['id_test']['n_rows']}  "
          f"transfer: {file_meta['transfer_test']['n_rows']}  "
          f"orbench: {file_meta['orbench_safe_stress']['n_rows']}  "
          f"harmbench: {file_meta['harmbench_positive_stress']['n_rows']}", flush=True)
    print(f"exact train<->eval overlaps removed: {len(exact_overlap_records)} "
          f"(jailbreak-classification<->wildjailbreak: {len(jc_wjb)})", flush=True)
    print(f"cross-split near-dup train rows removed: {len(train_remove_ids)}", flush=True)
    print("\n-- the 4 known jailbreak-classification <-> WildJailbreak overlaps (removed from train) --",
          flush=True)
    for i, e in enumerate(jc_wjb):
        print(f"  [{i}] train_id={e['train_source_row_id']} label={e['train_label']} "
              f"sha={e['content_sha256'][:12]}\n      text: {e['normalized_text'][:110]}", flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--public-out", default=None,
        help="text-free public index directory (default: sibling public_manifests)")
    ap.add_argument(
        "--allow-legacy-frozen-cohorts", action="store_true",
        help=("RETROSPECTIVE ONLY: use the previously inspected seed-7 WildGuard/"
              "WildJailbreak cache instead of pinned Hugging Face revisions"))
    args = ap.parse_args()
    build(args.config, args.out,
          allow_legacy_frozen_cohorts=args.allow_legacy_frozen_cohorts,
          public_out_dir=args.public_out)


if __name__ == "__main__":
    main()
