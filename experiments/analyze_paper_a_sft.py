#!/usr/bin/env python
"""Paper A analysis (plan sec 12, 15.5, 15.6, 16).

From the keyed score table it computes, using the canonical metric module
(guard_research.metrics -- never a custom AP):

  * per (model_key, condition, seed, benchmark) tie-aware AP + AUROC;
  * macro-AP over represented-source and over transfer benchmarks;
  * base->SFT deltas per checkpoint (mean over seeds) and the fixed-panel
    aggregate (mean over 4 checkpoints);
  * the hierarchical PAIRED bootstrap of plan sec 12.4 (10000 reps, seed
    20260712: 4 checkpoints fixed; resample 5 seed indices within each
    checkpoint; one Poisson(1) weight per GLOBAL family_id applied to all its
    rows across datasets; weighted tie-aware AP per benchmark; macro-average;
    one-sided 95% LCB/UCB and two-sided 95%);
  * leave-one-benchmark-out and leave-one-base-out sensitivity;
  * secondary 5% FPR operating point (TPR + realized FPR, represented/transfer);
  * OR-Bench benign FPR + HarmBench recall (one-class, NO AP);
  * descriptive joint criteria with precision-focused estimation language;
    powered-confirmatory claims are intentionally disabled.

Emits deterministic results.json, runtime-bound analysis_metadata.json,
seed_values.csv, per_benchmark.csv, sensitivity.json, claim_checks.json,
LaTeX Table 3/4 plus appendix seed-table fragments, result
macros, and the specialization-plane figure -- all generated from artifacts
(no hand-entered numbers).

Usage:
  python experiments/analyze_paper_a_sft.py --lock artifacts/paper_a_sft_v2/LOCK.json \
    --scores artifacts/paper_a_sft_v2/scores/scores.parquet \
    --out artifacts/paper_a_sft_v2/analysis
  python experiments/analyze_paper_a_sft.py --self-test   # synthetic end-to-end check
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402

ANALYSIS_CODE_VERSION = "paper_a_sft_analysis_v2"
AP_SPLITS = {"represented": "id_test", "transfer": "transfer_test"}
CURRENT_ANALYSIS_SOURCE_FILES = (
    "experiments/analyze_paper_a_sft.py",
    "experiments/paper_a_common.py",
    "guard_research/__init__.py",
    "guard_research/metrics.py",
    "guard_research/thresholds.py",
)
RELEASE_ANALYSIS_SOFTWARE_KEYS = (
    "python", "numpy", "pandas", "pyarrow", "sklearn", "scipy", "matplotlib",
)

# This is deliberately duplicated from the scorer rather than imported from it: analysis is a
# final, independent integrity boundary and must reject a score table whose schema merely happens
# to agree with self-reported metadata.
EXPECTED_SCORE_COLUMNS = [
    "sample_id", "content_sha256", "source", "split", "gold", "family_id",
    "model_key", "model_revision", "condition", "seed", "adapter_sha256",
    "prompt_sha256", "safe_token_id", "unsafe_token_id", "safe_logit",
    "unsafe_logit", "score_raw", "probability_raw", "probability_calibrated",
    "threshold_id", "prediction", "original_token_count", "scored_token_count",
    "truncated", "latency_ms",
]
SCORE_COLUMNS_V2 = EXPECTED_SCORE_COLUMNS[:-1] + ["truncation_strategy", "latency_ms"]
SCORE_SCHEMAS = {
    "paper_a_sft_scorer_v1": EXPECTED_SCORE_COLUMNS,
    "paper_a_sft_scorer_v2": SCORE_COLUMNS_V2,
}
MODEL_LABELS = {
    "qwen25_15b": "Qwen2.5-1.5B",
    "smollm2_17b": "SmolLM2-1.7B",
    "smollm3_3b": "SmolLM3-3B",
    "qwen3_4b": "Qwen3-4B",
}
SCORING_SPLIT_FILES = {
    "calibration": "calibration.jsonl",
    "id_test": "id_test.jsonl",
    "transfer_test": "transfer_test.jsonl",
    "stress_orbench": "orbench_safe_stress.jsonl",
    "stress_harmbench": "harmbench_positive_stress.jsonl",
}


class ScoreValidationError(ValueError):
    """The score table is not the exact locked Paper A evidence matrix."""


def _require(ok, message):
    if not ok:
        raise ScoreValidationError(f"score validation failed: {message}")


def _expected_bundle_keys(model_keys, seeds):
    keys = {f"{mk}:base" for mk in model_keys}
    keys.update(f"{mk}:sft:seed_{s}" for mk in model_keys for s in seeds)
    return keys


def load_locked_scoring_manifest_rows(lock):
    """Load the five lock-verified scoring manifests for an independent join."""
    manifest_dir = C.resolved_path(C.artifact_paths(lock)["manifests"])
    rows = []
    for expected_split, filename in SCORING_SPLIT_FILES.items():
        path = manifest_dir / filename
        for row in C.read_jsonl(path):
            normalized = dict(row)
            normalized["split"] = expected_split
            normalized["gold"] = C.row_gold(row)
            rows.append(normalized)
    return rows


def load_public_scoring_manifest_rows(lock):
    """Load score identities from the lock-verified text-free public release."""
    public_dir = C.resolved_path(C.artifact_paths(lock)["public_manifests"])
    rows = []
    required = {"sample_id", "content_sha256", "source", "split", "gold", "family_id"}
    for expected_split, filename in SCORING_SPLIT_FILES.items():
        path = public_dir / filename
        manifest_split = pathlib.Path(filename).stem
        for row in C.read_jsonl(path):
            _require(required.issubset(row),
                     f"public manifest row lacks score identity fields: {path}")
            _require(row.get("split") == manifest_split,
                     f"public manifest row has wrong split in {path}")
            normalized = dict(row)
            # Stress manifests retain their file-role names in the public projection,
            # while score rows use the canonical evaluation split names. Raw-manifest
            # analysis performs the same normalization before its identity join.
            normalized["split"] = expected_split
            normalized["gold"] = C.row_gold(row)
            rows.append(normalized)
    return rows


def validate_reference_against_manifests(reference, manifest_rows):
    """Require score-row identities to equal the raw locked manifests exactly."""
    _require(manifest_rows is not None, "strict analysis requires locked manifest rows")
    expected = {}
    for row in manifest_rows:
        sample_id = str(row.get("sample_id", ""))
        _require(sample_id and sample_id not in expected,
                 f"locked manifests contain duplicate/empty sample_id {sample_id!r}")
        expected[sample_id] = (
            str(row.get("content_sha256")), str(row.get("source")), str(row.get("split")),
            int(C.row_gold(row)), str(row.get("family_id")),
        )
    observed = {
        str(row.sample_id): (
            str(row.content_sha256), str(row.source), str(row.split), int(row.gold),
            str(row.family_id),
        )
        for row in reference[
            ["sample_id", "content_sha256", "source", "split", "gold", "family_id"]
        ].itertuples(index=False)
    }
    _require(observed == expected,
             "score identities do not join one-to-one to the locked scoring manifests")
    return {"n_manifest_rows_joined": len(expected)}


def validate_locked_threshold_selection(cell, threshold, target_fpr, key="bundle"):
    """Independently rerun the canonical calibration threshold selector."""
    calibration_rows = cell[cell["split"] == "calibration"]
    selected = C.normalize_threshold_result(C.require_select_threshold()(
        calibration_rows["probability_calibrated"].astype(float).tolist(),
        calibration_rows["gold"].astype(int).tolist(),
        float(target_fpr),
    ))
    _require(threshold.get("status") == selected["status"],
             f"threshold status for {key} is not canonical at locked target FPR")
    selected_value = selected["threshold"]
    stored_value = threshold.get("threshold_value")
    if selected["status"] in ("NO_FEASIBLE_THRESHOLD", "PREDICT_NONE"):
        _require(stored_value is None, f"threshold value for {key} must be null")
    else:
        try:
            stored_value = float(stored_value)
        except (TypeError, ValueError):
            stored_value = float("nan")
        _require(np.isfinite(stored_value)
                 and np.isclose(stored_value, float(selected_value), rtol=0, atol=1e-12),
                 f"threshold value for {key} is not canonical at locked target FPR")
    return selected


def validate_analysis_paths(lock, out_dir, scores_path, *, nonfinal=False):
    """Keep final and diagnostic analysis writes in disjoint namespaces."""
    apaths = C.artifact_paths(lock)
    if nonfinal:
        protected_roots = {
            apaths["root"], C.DEFAULT_ARTIFACTS["root"], C.DEFAULT_ARTIFACTS_V2["root"],
        }
        _require(not any(C.path_is_within(out_dir, root) for root in protected_roots),
                 "nonfinal analysis output must be outside canonical v1/v2 artifact roots")
    else:
        _require(C.resolved_path(out_dir) == C.resolved_path(apaths["analysis"]),
                 "final analysis output is lock-authoritative; use --nonfinal for diagnostics")
        expected_scores = os.path.join(apaths["scores"], "scores.parquet")
        _require(C.resolved_path(scores_path) == C.resolved_path(expected_scores),
                 "final score input is lock-authoritative; use --nonfinal for diagnostics")
    return {"out_dir": str(C.resolved_path(out_dir)),
            "scores_path": str(C.resolved_path(scores_path))}


def _python_major_minor(version):
    """Return a strict ``(major, minor)`` pair for a dotted Python version."""
    parts = str(version).split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return tuple(int(part) for part in parts[:2])


def validate_analysis_runtime(lock, *, nonfinal=False, release_cache=False):
    """Require the locked scientific runtime for canonical v2 analysis.

    A score-only release may be reproduced on a later patch of the same Python
    major/minor line.  Its numerical libraries remain exact-version locked and
    the complete current analysis runtime is attested in ``analysis_metadata``.
    Full raw-artifact analysis retains the stricter patch-exact contract.
    """
    strict = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    if strict and not nonfinal:
        actual = C.software_versions()
        expected = lock.get("software_versions") or {}
        if release_cache:
            issues = []
            for key in RELEASE_ANALYSIS_SOFTWARE_KEYS:
                if expected.get(key) is None:
                    issues.append(f"locked_{key}_missing")
                elif actual.get(key) != expected.get(key):
                    issues.append(f"{key}_mismatch")
        else:
            issues = C.protocol_software_issues(actual, expected)
        if release_cache and "python_mismatch" in issues:
            if (_python_major_minor(actual.get("python"))
                    == _python_major_minor(expected.get("python"))
                    and _python_major_minor(actual.get("python")) is not None):
                issues.remove("python_mismatch")
        _require(not issues, f"analysis runtime software differs from LOCK.json: {issues}")


def validate_release_cache_request(lock, *, release_cache=False, nonfinal=False):
    """Keep the reduced-artifact path explicit, strict-v2-only, and canonical."""
    if not release_cache:
        return
    strict = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    _require(strict and lock.get("finalization_status") == "final",
             "--release-cache requires a strict final v2 lock")
    _require(not nonfinal, "--release-cache cannot be combined with --nonfinal")


def validate_score_artifacts(df, lock, metadata, *, allow_synthetic=False,
                             manifest_rows=None, release_cache=False):
    """Fail closed unless *df* is the exact matrix bound by LOCK and metadata.json.

    Validation intentionally happens before benchmark discovery.  Otherwise a missing model,
    seed, benchmark, or row could silently redefine the estimand and even make a vacuous
    leave-one-out check appear successful.
    """
    import pandas as pd

    strict = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    _require(not release_cache or strict,
             "release-cache score validation requires a strict v2 lock")
    _require(isinstance(metadata, dict), "sibling scores/metadata.json is required")
    _require(allow_synthetic or not metadata.get("synthetic", False),
             "synthetic score metadata is not admissible for final analysis")
    score_version = metadata.get("score_code_version")
    expected_columns = SCORE_SCHEMAS.get(score_version)
    _require(expected_columns is not None, f"unsupported score schema version {score_version!r}")
    _require(list(df.columns) == expected_columns,
             f"schema mismatch; expected columns in canonical order {expected_columns}")
    _require(metadata.get("columns") == expected_columns,
             "metadata column schema does not match the canonical score schema")
    _require(metadata.get("lock_sha256") == lock.get("lock_sha256"),
             "metadata lock_sha256 does not match LOCK.json")
    _require(metadata.get("score_code_version") == lock.get("score_code_version"),
             "metadata score_code_version does not match LOCK.json")
    if strict:
        _require(lock.get("analysis_code_version") == ANALYSIS_CODE_VERSION,
                 "strict LOCK.json does not bind this analysis-code version")
    _require(int(metadata.get("n_rows_total", -1)) == len(df),
             "metadata row count does not match scores.parquet")

    model_keys = list(C.MODEL_KEYS)
    seeds = [int(s) for s in C.lock_seeds(lock)]
    locked_models = C.lock_model_panel(lock)
    _require(set(locked_models) == set(model_keys),
             f"LOCK model panel must contain exactly {model_keys}")
    _require(len(seeds) >= 2 and len(set(seeds)) == len(seeds),
             "LOCK must contain at least two unique SFT seeds")
    _require(set(metadata.get("models", {})) == set(model_keys),
             "metadata model panel is incomplete or contains extras")
    _require([int(s) for s in metadata.get("seeds", [])] == seeds,
             "metadata seeds do not match LOCK.json")
    locked_target_fpr = float((lock.get("operating_point") or {}).get(
        "target_fpr", C.DEFAULT_TARGET_FPR))
    if strict:
        try:
            metadata_target_fpr = float(metadata["target_fpr"])
        except (KeyError, TypeError, ValueError):
            metadata_target_fpr = float("nan")
        _require(np.isfinite(metadata_target_fpr)
                 and np.isclose(metadata_target_fpr, locked_target_fpr, rtol=0, atol=1e-15),
                 "score metadata target_fpr differs from LOCK.json")
        _require(isinstance(metadata.get("producer_runtime"), dict)
                 and isinstance(metadata["producer_runtime"].get("sha256"), str),
                 "strict score metadata lacks producer runtime provenance")
        _require(metadata["producer_runtime"]["sha256"]
                 == C.canonical_obj_sha256(metadata["producer_runtime"].get("details")),
                 "score producer runtime fingerprint is internally inconsistent")
        _require(int(metadata.get("batch_size", 0)) > 0,
                 "strict score metadata lacks a positive batch size")

    regimes = lock.get("regime_benchmarks") or {}
    represented = list(regimes.get("represented") or [])
    transfer = list(regimes.get("transfer") or [])
    stress = list(regimes.get("stress") or [])
    _require(len(represented) >= 2 and len(transfer) >= 2,
             "each AP regime needs at least two locked benchmarks for finite LOO checks")
    _require(len(set(represented)) == len(represented)
             and len(set(transfer)) == len(transfer),
             "LOCK benchmark lists contain duplicates")
    _require(set(stress) == {"orbench", "harmbench"},
             "LOCK stress regime must contain exactly orbench and harmbench")
    expected_split_sources = {
        "calibration": set(represented),
        "id_test": set(represented),
        "transfer_test": set(transfer),
        "stress_orbench": {"orbench"},
        "stress_harmbench": {"harmbench"},
    }

    _require(len(df) > 0, "score table is empty")
    for col in ("sample_id", "content_sha256", "source", "split", "family_id",
                "model_key", "model_revision", "condition", "prompt_sha256",
                "threshold_id"):
        values = df[col]
        _require(not values.isna().any(), f"{col} contains null values")
        _require(not values.astype(str).str.strip().eq("").any(), f"{col} contains empty values")

    numeric = [
        "gold", "seed", "safe_token_id", "unsafe_token_id", "safe_logit", "unsafe_logit",
        "score_raw", "probability_raw", "probability_calibrated", "prediction",
        "original_token_count", "scored_token_count", "latency_ms",
    ]
    for col in numeric:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        _require(np.isfinite(vals).all(), f"{col} contains non-numeric or non-finite values")
    for col in ("gold", "seed", "safe_token_id", "unsafe_token_id", "prediction",
                "original_token_count", "scored_token_count"):
        vals = pd.to_numeric(df[col]).to_numpy(float)
        _require(np.equal(vals, np.floor(vals)).all(), f"{col} must contain integer values")
    truncated = df["truncated"]
    _require(set(truncated.unique()) <= {False, True, 0, 1}, "truncated must be boolean")
    if "truncation_strategy" in df:
        strategy = df["truncation_strategy"]
        _require(not strategy.isna().any()
                 and strategy.astype(str).str.strip().ne("").all(),
                 "truncation_strategy must be present for every v2 score row")
    _require(set(pd.to_numeric(df["gold"]).astype(int).unique()) <= {0, 1},
             "gold must be binary")
    _require(set(pd.to_numeric(df["prediction"]).astype(int).unique()) <= {0, 1},
             "prediction must be binary")
    _require(((df["probability_raw"] >= 0) & (df["probability_raw"] <= 1)).all(),
             "probability_raw must lie in [0,1]")
    _require(((df["probability_calibrated"] >= 0)
              & (df["probability_calibrated"] <= 1)).all(),
             "probability_calibrated must lie in [0,1]")
    _require((pd.to_numeric(df["safe_token_id"]) != pd.to_numeric(df["unsafe_token_id"])).all(),
             "safe and unsafe decision-token IDs must differ")
    _require(np.allclose(df["score_raw"].to_numpy(float),
                         df["unsafe_logit"].to_numpy(float) - df["safe_logit"].to_numpy(float),
                         rtol=0, atol=1e-10),
             "score_raw is not unsafe_logit - safe_logit")
    expected_prob = 1.0 / (1.0 + np.exp(-np.clip(df["score_raw"].to_numpy(float), -700, 700)))
    _require(np.allclose(df["probability_raw"].to_numpy(float), expected_prob,
                         rtol=1e-10, atol=1e-12),
             "probability_raw is inconsistent with score_raw")

    _require(set(df["model_key"].unique()) == set(model_keys),
             "score table model panel is incomplete or contains extras")
    _require(set(df["condition"].unique()) == {"base", "sft"},
             "conditions must be exactly base and sft")
    base_seeds = set(pd.to_numeric(df.loc[df.condition == "base", "seed"]).astype(int))
    sft_seeds = set(pd.to_numeric(df.loc[df.condition == "sft", "seed"]).astype(int))
    _require(base_seeds == {-1}, "base rows must use seed=-1")
    _require(sft_seeds == set(seeds), "SFT seed cells are incomplete or contain extras")

    actual_split_sources = {
        split: set(df.loc[df.split == split, "source"].unique())
        for split in df["split"].unique()
    }
    _require(set(actual_split_sources) == set(expected_split_sources),
             "score table splits are incomplete or contain extras")
    for split, sources in expected_split_sources.items():
        _require(actual_split_sources.get(split) == sources,
                 f"{split} sources do not match LOCK.json")
    for split in ("calibration", "id_test", "transfer_test"):
        for source in expected_split_sources[split]:
            labels = set(pd.to_numeric(df.loc[(df.split == split) & (df.source == source),
                                               "gold"]).astype(int))
            _require(labels == {0, 1}, f"{split}/{source} is not a nonempty two-class cell")
    _require(set(pd.to_numeric(df.loc[df.split == "stress_orbench", "gold"]).astype(int)) == {0},
             "OR-Bench stress rows must be nonempty and all benign")
    _require(set(pd.to_numeric(df.loc[df.split == "stress_harmbench", "gold"]).astype(int)) == {1},
             "HarmBench stress rows must be nonempty and all harmful")

    composite = ["model_key", "condition", "seed", "sample_id"]
    _require(not df.duplicated(composite).any(), "duplicate bundle/sample composite keys")
    expected_bundles = {(mk, "base", -1) for mk in model_keys}
    expected_bundles.update((mk, "sft", s) for mk in model_keys for s in seeds)
    actual_bundles = {
        (str(mk), str(cond), int(seed))
        for mk, cond, seed in df[["model_key", "condition", "seed"]].drop_duplicates().itertuples(index=False)
    }
    _require(actual_bundles == expected_bundles,
             "bundle cross-product is incomplete or contains unexpected cells")

    reference = df[(df.model_key == model_keys[0]) & (df.condition == "base")].copy()
    _require(not reference["sample_id"].duplicated().any(),
             "reference bundle contains duplicate sample IDs")
    manifest_join = (validate_reference_against_manifests(reference, manifest_rows)
                     if strict else {"n_manifest_rows_joined": None})
    identity_cols = ["content_sha256", "source", "split", "gold", "family_id"]
    reference_identity = {
        str(row.sample_id): tuple(getattr(row, c) for c in identity_cols)
        for row in reference[["sample_id", *identity_cols]].itertuples(index=False)
    }
    for mk, cond, seed in sorted(expected_bundles):
        cell = df[(df.model_key == mk) & (df.condition == cond) & (df.seed == seed)]
        _require(len(cell) == len(reference), f"{mk}/{cond}/seed_{seed} row count is incomplete")
        identity = {
            str(row.sample_id): tuple(getattr(row, c) for c in identity_cols)
            for row in cell[["sample_id", *identity_cols]].itertuples(index=False)
        }
        _require(identity == reference_identity,
                 f"{mk}/{cond}/seed_{seed} sample identity differs from the reference bundle")
        adapters = cell["adapter_sha256"]
        if cond == "base":
            _require(adapters.isna().all(), f"base bundle {mk} unexpectedly has an adapter hash")
        else:
            _require(not adapters.isna().any()
                     and adapters.astype(str).str.strip().ne("").all()
                     and adapters.astype(str).nunique() == 1,
                     f"SFT bundle {mk}/seed_{seed} lacks one immutable adapter hash")
        _require(cell["threshold_id"].nunique() == 1,
                 f"{mk}/{cond}/seed_{seed} has multiple threshold IDs")

    for mk in model_keys:
        locked = locked_models[mk]
        _require(set(df.loc[df.model_key == mk, "model_revision"]) == {locked["model_revision"]},
                 f"{mk} model revision differs from LOCK.json")
        locked_prompt = (lock.get("prompt", {}).get("per_model_template_sha256", {}).get(mk))
        _require(bool(locked_prompt), f"LOCK lacks a prompt-template hash for {mk}")
        _require(set(df.loc[df.model_key == mk, "prompt_sha256"]) == {locked_prompt},
                 f"{mk} prompt hash differs from LOCK.json")
        probe = (lock.get("tokenizer_probe") or {}).get(mk) or {}
        if probe:
            _require(set(pd.to_numeric(df.loc[df.model_key == mk, "safe_token_id"]).astype(int))
                     == {int(probe["safe_token_id"])}, f"{mk} safe token ID differs from LOCK")
            _require(set(pd.to_numeric(df.loc[df.model_key == mk, "unsafe_token_id"]).astype(int))
                     == {int(probe["unsafe_token_id"])}, f"{mk} unsafe token ID differs from LOCK")
        meta_model = metadata["models"][mk]
        runtime_fields = ("model_id", "model_revision", "tokenizer_revision", "dtype",
                          "attn_implementation", "trust_remote_code")
        if strict:
            _require({field: meta_model.get(field) for field in runtime_fields}
                     == {field: locked.get(field) for field in runtime_fields},
                     f"metadata model runtime identity for {mk} differs from LOCK.json")
            _require((metadata.get("dtype_by_model") or {}).get(mk) == locked.get("dtype"),
                     f"metadata dtype_by_model for {mk} differs from LOCK.json")
        else:
            _require(meta_model.get("model_revision") == locked["model_revision"]
                     and meta_model.get("tokenizer_revision") == locked["tokenizer_revision"],
                     f"legacy metadata model identity for {mk} differs from LOCK.json")

    manifest_meta = metadata.get("manifest_fingerprints") or {}
    _require(int(manifest_meta.get("n_rows", -1)) == len(reference),
             "metadata per-bundle manifest row count is wrong")
    _require(manifest_meta.get("sample_ids_fingerprint")
             == C.sha256_ordered(reference["sample_id"].astype(str).tolist()),
             "sample-ID fingerprint does not match scores.parquet")
    _require(manifest_meta.get("content_fingerprint")
             == C.sha256_ordered(reference["content_sha256"].astype(str).tolist()),
             "content fingerprint does not match scores.parquet")
    per_split_meta = manifest_meta.get("per_split_manifest_sha256") or {}
    locked_splits = (lock.get("manifests") or {}).get("splits") or {}
    for split, filename in SCORING_SPLIT_FILES.items():
        locked_split = locked_splits.get(filename)
        _require(isinstance(locked_split, dict), f"LOCK lacks {filename}")
        count = int((reference.split == split).sum())
        _require(int(locked_split.get("rows", -1)) == count,
                 f"{split} row count differs from LOCK.json")
        _require(per_split_meta.get(split) == locked_split.get("sha256"),
                 f"metadata hash for {split} differs from LOCK.json")
    recomputed_manifest_sha = C.sha256_ordered(
        [f"{k}={v}" for k, v in sorted(per_split_meta.items())])
    _require(manifest_meta.get("manifest_sha256") == recomputed_manifest_sha,
             "combined manifest fingerprint is internally inconsistent")

    bundle_meta = metadata.get("bundles") or {}
    _require(set(bundle_meta) == _expected_bundle_keys(model_keys, seeds),
             "metadata bundle inventory is incomplete or contains extras")
    _require(int(metadata.get("n_bundles", -1)) == len(expected_bundles),
             "metadata bundle count is wrong")
    expected_sft_keys = {
        f"{mk}:sft:seed_{seed}" for mk in model_keys for seed in seeds}
    adapter_inventory = metadata.get("adapter_inventory") or {}
    if strict:
        _require(set(adapter_inventory) == expected_sft_keys,
                 "strict score metadata adapter inventory is incomplete or contains extras")
    observed_adapter_hashes = []
    for mk, cond, seed in expected_bundles:
        key = f"{mk}:base" if cond == "base" else f"{mk}:sft:seed_{seed}"
        bundle_record = bundle_meta.get(key) or {}
        cell = df[(df.model_key == mk) & (df.condition == cond) & (df.seed == seed)]
        row_adapter_hashes = set(cell["adapter_sha256"].dropna().astype(str))
        if cond == "base":
            _require(not row_adapter_hashes and bundle_record.get("adapter_sha256") is None,
                     f"base bundle {key} unexpectedly binds an adapter")
        elif strict:
            inventory = adapter_inventory.get(key) or {}
            _require(len(row_adapter_hashes) == 1,
                     f"strict SFT bundle {key} lacks one adapter hash")
            adapter_hash = next(iter(row_adapter_hashes))
            _require(bundle_record.get("adapter_sha256") == adapter_hash
                     and inventory.get("adapter_sha256") == adapter_hash,
                     f"adapter inventory for {key} differs from scores.parquet")
            _require(bundle_record.get("run_meta_sha256") == inventory.get("run_meta_sha256")
                     and bundle_record.get("run_meta_path") == inventory.get("run_meta_path"),
                     f"run-metadata inventory for {key} is inconsistent")
            run_meta_path = inventory.get("run_meta_path")
            run_meta_sha = inventory.get("run_meta_sha256")
            _require(isinstance(run_meta_path, str) and isinstance(run_meta_sha, str)
                     and len(run_meta_sha) == 64,
                     f"run metadata binding for {key} is missing")
            _require(C.path_is_within(run_meta_path, C.artifact_paths(lock)["runs"]),
                     f"run metadata for {key} resolves outside the locked runs root")
            if not release_cache:
                resolved_run_meta = C.resolved_path(run_meta_path)
                _require(resolved_run_meta.is_file()
                         and C.sha256_file(resolved_run_meta) == run_meta_sha,
                         f"run metadata hash for {key} does not verify")
                run_meta = C.read_json(resolved_run_meta)
                _require(run_meta.get("model_key") == mk
                         and int(run_meta.get("seed", -1)) == seed
                         and run_meta.get("adapter_sha256") == adapter_hash
                         and run_meta.get("lock_sha256") == lock.get("lock_sha256"),
                         f"run metadata identity for {key} differs from score metadata")
                adapter_dir = resolved_run_meta.parent / "adapter"
                _require(C.adapter_is_present(str(adapter_dir))
                         and C.sha256_dir(adapter_dir) == adapter_hash,
                         f"adapter bytes for {key} differ from the score/run binding")
                _require(not C.adapter_config_issues(adapter_dir, lock.get("recipe") or {}),
                         f"serialized adapter config for {key} differs from LOCK.json")
            observed_adapter_hashes.append(adapter_hash)
        if strict:
            _require(bundle_record.get("batch_size") == metadata.get("batch_size")
                     and bundle_record.get("producer_runtime_sha256")
                     == metadata["producer_runtime"]["sha256"],
                     f"cache producer identity for {key} differs from combined metadata")
        calibration = bundle_record.get("calibration") or {}
        _require(calibration.get("status") == "ok",
                 f"metadata calibration for {key} is not an accepted two-class fit")
        if strict:
            _require(calibration.get("optim_success") is True,
                     f"calibration optimizer did not succeed for {key}")
        try:
            temperature = float(calibration["temperature"])
        except (KeyError, TypeError, ValueError):
            temperature = float("nan")
        _require(np.isfinite(temperature) and temperature > 0,
                 f"metadata calibration temperature for {key} is invalid")
        threshold = bundle_record.get("threshold") or {}
        actual_threshold = df.loc[(df.model_key == mk) & (df.condition == cond)
                                  & (df.seed == seed), "threshold_id"].iloc[0]
        _require(threshold.get("threshold_id") == actual_threshold,
                 f"metadata threshold identity for {key} differs from scores.parquet")
        score = cell["score_raw"].to_numpy(float)
        expected_calibrated = 1.0 / (
            1.0 + np.exp(-np.clip(score / temperature, -700, 700)))
        _require(np.allclose(cell["probability_calibrated"].to_numpy(float),
                             expected_calibrated, rtol=1e-10, atol=1e-12),
                 f"calibrated probabilities for {key} do not match score/temperature")
        threshold_status = threshold.get("status")
        threshold_value = threshold.get("threshold_value")
        if strict:
            validate_locked_threshold_selection(cell, threshold, locked_target_fpr, key)
        if threshold_status == "ok":
            try:
                threshold_value = float(threshold_value)
            except (TypeError, ValueError):
                threshold_value = float("nan")
            _require(np.isfinite(threshold_value) and 0 <= threshold_value <= 1,
                     f"metadata threshold value for {key} is invalid")
            expected_prediction = (expected_calibrated >= threshold_value).astype(int)
        elif threshold_status in ("NO_FEASIBLE_THRESHOLD", "PREDICT_NONE"):
            _require(threshold_value is None,
                     f"{key} predict-none threshold must store threshold_value=null")
            expected_prediction = np.zeros(len(cell), dtype=int)
        else:
            raise ScoreValidationError(
                f"score validation failed: unsupported threshold status for {key}: "
                f"{threshold_status!r}")
        _require(np.array_equal(cell["prediction"].to_numpy(int), expected_prediction),
                 f"predictions for {key} do not match calibrated probability/threshold semantics")

    if strict:
        _require(len(observed_adapter_hashes) == len(expected_sft_keys)
                 and len(set(observed_adapter_hashes)) == len(expected_sft_keys),
                 "strict score artifact reuses an adapter hash across SFT cells")

    return {
        "n_rows": int(len(df)), "n_samples_per_bundle": int(len(reference)),
        "n_bundles": int(len(expected_bundles)), "model_keys": model_keys, "seeds": seeds,
        **manifest_join,
    }


# --------------------------------------------------------------------------------------
# build per-benchmark aligned arrays from the score table
# --------------------------------------------------------------------------------------
def build_bench_data(df, regimes, model_keys, seeds):
    """Return (data, families) where
       data[mk][bench] = {regime, split, gold, fam_idx, base(scores), sft{seed:scores},
                          base_pred, sft_pred{seed}}
       families = ordered list of global family_ids over AP rows."""
    ap_rows = df[df["split"].isin(list(AP_SPLITS.values()))]
    families = sorted(map(str, ap_rows["family_id"].dropna().unique()))
    fam_index = {f: i for i, f in enumerate(families)}

    # which benchmarks are present in each regime
    present = {}
    for regime, split in AP_SPLITS.items():
        srcs = set(df[df["split"] == split]["source"].unique())
        present[regime] = [b for b in regimes[regime] if b in srcs]

    data = {}
    for mk in model_keys:
        data[mk] = {}
        for regime, benches in present.items():
            split = AP_SPLITS[regime]
            for bench in benches:
                base = df[(df.model_key == mk) & (df.condition == "base")
                          & (df.split == split) & (df.source == bench)].sort_values("sample_id")
                if base.empty:
                    continue
                order = base["sample_id"].tolist()
                gold = base["gold"].to_numpy(int)
                fam_idx = np.array([fam_index[str(f)] for f in base["family_id"]], int)
                entry = {
                    "regime": regime, "split": split, "n": len(order),
                    "gold": gold, "fam_idx": fam_idx,
                    "base": base["score_raw"].to_numpy(float),
                    "base_pred": base["prediction"].to_numpy(int),
                    "sft": {}, "sft_pred": {},
                }
                for s in seeds:
                    sf = df[(df.model_key == mk) & (df.condition == "sft") & (df.seed == s)
                            & (df.split == split) & (df.source == bench)]
                    sf = sf.set_index("sample_id").reindex(order)
                    entry["sft"][s] = sf["score_raw"].to_numpy(float)
                    entry["sft_pred"][s] = sf["prediction"].to_numpy(float)
                data[mk][bench] = entry
    return data, families, present


# --------------------------------------------------------------------------------------
# point estimates (plan sec 12.1-12.3)
# --------------------------------------------------------------------------------------
def macro_ap(data, mk, benches, ap_fn, condition, seed=None, weights=None):
    vals = []
    for b in benches:
        e = data[mk].get(b)
        if e is None:
            continue
        scores = e["base"] if condition == "base" else e["sft"][seed]
        w = None if weights is None else weights[e["fam_idx"]]
        vals.append(C.weighted_metric(ap_fn, scores, e["gold"], w))
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def point_estimates(data, present, model_keys, seeds, ap_fn):
    out = {"per_checkpoint": {}, "aggregate": {}, "seed_values": []}
    for regime, benches in present.items():
        ck = {}
        for mk in model_keys:
            base = macro_ap(data, mk, benches, ap_fn, "base")
            sft_seed = {s: macro_ap(data, mk, benches, ap_fn, "sft", s) for s in seeds}
            sft_mean = float(np.mean(list(sft_seed.values())))
            ck[mk] = {"base": base, "sft_mean": sft_mean,
                      "sft_by_seed": sft_seed, "delta": sft_mean - base,
                      "seed_deltas": {s: sft_seed[s] - base for s in seeds}}
        out["per_checkpoint"][regime] = ck
        out["aggregate"][regime] = float(np.mean([ck[mk]["delta"] for mk in model_keys]))
    # tidy seed_values rows
    for mk in model_keys:
        for s in seeds:
            row = {"model_key": mk, "seed": s}
            for regime in present:
                ck = out["per_checkpoint"][regime][mk]
                row[f"{regime}_base"] = ck["base"]
                row[f"{regime}_sft"] = ck["sft_by_seed"][s]
                row[f"{regime}_delta"] = ck["seed_deltas"][s]
            out["seed_values"].append(row)
    return out


# --------------------------------------------------------------------------------------
# hierarchical paired bootstrap (plan sec 12.4)
# --------------------------------------------------------------------------------------
def hierarchical_bootstrap(data, present, model_keys, seeds, ap_fn, reps, rng_seed,
                           max_redraw=2000):
    rng = np.random.default_rng(rng_seed)
    n_fam = 1 + max((e["fam_idx"].max() for mk in data for e in data[mk].values()
                     if e["fam_idx"].size), default=-1)
    n_seeds = len(seeds)
    regimes = list(present.keys())

    # precompute per (regime,mk) bench list; per bench pos/neg fam-membership for validity
    bench_of = {r: present[r] for r in regimes}
    # collect the set of (mk,bench) entries and, per bench, index arrays for validity test
    entries = [(mk, b, data[mk][b]) for mk in model_keys for r in regimes
               for b in bench_of[r] if b in data[mk]]

    agg_samples = {r: np.empty(reps) for r in regimes}
    ck_samples = {r: {mk: np.empty(reps) for mk in model_keys} for r in regimes}
    total_redraw = 0
    print(f"[bootstrap] starting {reps} paired replicates across {n_fam} families", flush=True)
    progress_every = max(1, reps // 10)

    def weights_valid(w):
        for _mk, _b, e in entries:
            g = e["gold"]; fi = e["fam_idx"]
            if w[fi[g == 1]].sum() <= 0 or w[fi[g == 0]].sum() <= 0:
                return False
        return True

    for rep in range(reps):
        # 1. valid Poisson(1) family weights (redraw whole vector on zero-class)
        redraws = 0
        while True:
            w = rng.poisson(1.0, size=n_fam).astype(float)
            if weights_valid(w):
                break
            redraws += 1; total_redraw += 1
            if redraws > max_redraw:
                raise RuntimeError("bootstrap: exceeded redraw cap (data too sparse?)")
        # 2. seed resample indices per checkpoint (with replacement)
        seed_pick = {mk: rng.integers(0, n_seeds, size=n_seeds) for mk in model_keys}
        # 3-9. weighted macro AP -> per-ckpt delta -> aggregate
        for regime in regimes:
            benches = bench_of[regime]
            per_ck = []
            for mk in model_keys:
                base_M = macro_ap(data, mk, benches, ap_fn, "base", weights=w)
                seed_M = {s: macro_ap(data, mk, benches, ap_fn, "sft", s, weights=w) for s in seeds}
                picked = [seed_M[seeds[j]] for j in seed_pick[mk]]
                delta = float(np.mean(picked)) - base_M
                per_ck.append(delta)
                ck_samples[regime][mk][rep] = delta
            agg_samples[regime][rep] = float(np.mean(per_ck))
        if reps >= 1000 and ((rep + 1) % progress_every == 0 or rep + 1 == reps):
            print(f"[bootstrap] {rep + 1}/{reps} replicates complete", flush=True)

    def summarize(arr):
        return {
            "std": float(np.std(arr, ddof=1)),
            "lcb95_one_sided": float(np.percentile(arr, 5)),
            "ucb95_one_sided": float(np.percentile(arr, 95)),
            "ci95_two_sided": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        }

    result = {"aggregate": {r: summarize(agg_samples[r]) for r in regimes},
              "per_checkpoint": {r: {mk: summarize(ck_samples[r][mk]) for mk in model_keys}
                                 for r in regimes},
              "reps": reps, "rng_seed": rng_seed, "n_families": int(n_fam),
              "redraws": int(total_redraw),
              "rejected_fraction": float(total_redraw / (reps + total_redraw)) if reps else 0.0}
    return result


# --------------------------------------------------------------------------------------
# sensitivity: leave-one-benchmark-out, leave-one-base-out, heterogeneity (plan sec 12.4)
# --------------------------------------------------------------------------------------
def sensitivity(data, present, model_keys, seeds, ap_fn, point):
    out = {"leave_one_benchmark_out": {}, "leave_one_base_out": {},
           "per_base_delta": {}, "per_benchmark_delta": {}, "sign_table": {},
           "range_across_bases": {}, "range_across_benchmarks": {}}

    def agg_delta(regime, benches, mks):
        vals = []
        for mk in mks:
            base = macro_ap(data, mk, benches, ap_fn, "base")
            sft = float(np.mean([macro_ap(data, mk, benches, ap_fn, "sft", s) for s in seeds]))
            vals.append(sft - base)
        return float(np.mean(vals)) if vals else float("nan")

    for regime, benches in present.items():
        full = point["aggregate"][regime]
        # leave-one-benchmark-out
        loo_b = {}
        for b in benches:
            rem = [x for x in benches if x != b]
            loo_b[b] = agg_delta(regime, rem, model_keys) if rem else float("nan")
        out["leave_one_benchmark_out"][regime] = {
            "full": full, "loo": loo_b,
            "sign_stable": _sign_stable(full, list(loo_b.values()))}
        # leave-one-base-out
        loo_base = {}
        for mk in model_keys:
            rem = [x for x in model_keys if x != mk]
            loo_base[mk] = agg_delta(regime, benches, rem)
        out["leave_one_base_out"][regime] = {
            "full": full, "loo": loo_base,
            "sign_stable": _sign_stable(full, list(loo_base.values()))}
        # per-base delta + range/std
        pbd = {mk: point["per_checkpoint"][regime][mk]["delta"] for mk in model_keys}
        out["per_base_delta"][regime] = pbd
        out["range_across_bases"][regime] = {
            "min": float(min(pbd.values())), "max": float(max(pbd.values())),
            "range": float(max(pbd.values()) - min(pbd.values())),
            "std": float(np.std(list(pbd.values()), ddof=1)) if len(pbd) > 1 else 0.0}
        # per-benchmark delta (fixed-panel mean over bases) + range/std + sign row
        pbench = {}
        sign_row = {}
        for b in benches:
            deltas = []
            for mk in model_keys:
                e = data[mk].get(b)
                if e is None:
                    continue
                base = C.weighted_metric(ap_fn, e["base"], e["gold"])
                sft = float(np.mean([C.weighted_metric(ap_fn, e["sft"][s], e["gold"]) for s in seeds]))
                deltas.append(sft - base)
                sign_row[f"{mk}:{b}"] = int(np.sign(sft - base))
            pbench[b] = float(np.mean(deltas)) if deltas else float("nan")
        out["per_benchmark_delta"][regime] = pbench
        vv = [v for v in pbench.values() if not math.isnan(v)]
        out["range_across_benchmarks"][regime] = {
            "min": float(min(vv)) if vv else float("nan"),
            "max": float(max(vv)) if vv else float("nan"),
            "range": float(max(vv) - min(vv)) if vv else float("nan"),
            "std": float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0}
        out["sign_table"][regime] = sign_row
    return out


def _sign_stable(full, loos):
    if not np.isfinite(full):
        raise ScoreValidationError("leave-one-out full estimate is not finite")
    if not loos:
        raise ScoreValidationError("leave-one-out complement set is empty")
    if not all(np.isfinite(v) for v in loos):
        raise ScoreValidationError("leave-one-out estimates must all be finite")
    fs = np.sign(full)
    if fs == 0:
        return False
    return all(np.sign(v) == fs for v in loos)


# --------------------------------------------------------------------------------------
# secondary 5% FPR operating point (RQ4) + one-class stress (plan sec 10.5, 11.2)
# --------------------------------------------------------------------------------------
def operating_point(df, present, model_keys, seeds):
    def tpr_fpr(sub):
        g = sub["gold"].to_numpy(int); p = sub["prediction"].to_numpy(int)
        pos = g == 1; neg = g == 0
        tpr = float(p[pos].mean()) if pos.any() else float("nan")
        fpr = float(p[neg].mean()) if neg.any() else float("nan")
        return tpr, fpr

    def macro(mk, condition, seed, benches, split):
        tprs, fprs = [], []
        for b in benches:
            sub = df[(df.model_key == mk) & (df.condition == condition) & (df.split == split)
                     & (df.source == b) & ((df.seed == seed) if condition == "sft" else True)]
            if sub.empty:
                continue
            t, f = tpr_fpr(sub)
            tprs.append(t); fprs.append(f)
        return (float(np.nanmean(tprs)) if tprs else float("nan"),
                float(np.nanmean(fprs)) if fprs else float("nan"))

    def pooled_fpr(mk, condition, seed, benches, split):
        sub = df[(df.model_key == mk) & (df.condition == condition)
                 & (df.split == split) & (df.source.isin(benches))
                 & ((df.seed == seed) if condition == "sft" else True)]
        negatives = sub[sub["gold"] == 0]
        return float(negatives["prediction"].mean()) if len(negatives) else float("nan")

    out = {}
    for regime, benches in present.items():
        split = AP_SPLITS[regime]
        reference = df[(df.model_key == model_keys[0]) & (df.condition == "base")
                       & (df.split == split) & (df.source.isin(benches))]
        base_t, base_f, sft_t, sft_f = [], [], [], []
        base_pooled_f, sft_pooled_f = [], []
        for mk in model_keys:
            bt, bf = macro(mk, "base", None, benches, split)
            base_t.append(bt); base_f.append(bf)
            base_pooled_f.append(pooled_fpr(mk, "base", None, benches, split))
            st = [macro(mk, "sft", s, benches, split) for s in seeds]
            sft_t.append(float(np.nanmean([x[0] for x in st])))
            sft_f.append(float(np.nanmean([x[1] for x in st])))
            sft_pooled_f.append(float(np.nanmean(
                [pooled_fpr(mk, "sft", s, benches, split) for s in seeds])))
        out[regime] = {
            "base_macro_tpr": float(np.nanmean(base_t)), "base_macro_fpr": float(np.nanmean(base_f)),
            "sft_macro_tpr": float(np.nanmean(sft_t)), "sft_macro_fpr": float(np.nanmean(sft_f)),
            "delta_tpr": float(np.nanmean(sft_t) - np.nanmean(base_t)),
            "delta_fpr": float(np.nanmean(sft_f) - np.nanmean(base_f)),
            "base_pooled_fpr": float(np.nanmean(base_pooled_f)),
            "sft_pooled_fpr": float(np.nanmean(sft_pooled_f)),
            "delta_pooled_fpr": float(
                np.nanmean(sft_pooled_f) - np.nanmean(base_pooled_f)),
            "n_rows": int(len(reference)),
            "n_positive_rows": int((reference["gold"] == 1).sum()),
            "n_negative_rows": int((reference["gold"] == 0).sum()),
            "aggregation": "benchmark macro, then fixed-model-panel mean; SFT also mean over seeds",
            "pooled_fpr_aggregation": (
                "pool negative rows across locked benchmarks, then fixed-model-panel mean; "
                "SFT also mean over seeds"),
            "realized_fpr_note": "calibration-targeted; realized test FPR is reported, not assumed",
        }
    return out


def stress_metrics(df, model_keys, seeds):
    def rate(split, mk, condition, seed):
        sub = df[(df.model_key == mk) & (df.condition == condition) & (df.split == split)
                 & ((df.seed == seed) if condition == "sft" else True)]
        return float(sub["prediction"].mean()) if len(sub) else float("nan")
    out = {}
    for name, split in (("orbench_benign_fpr", "stress_orbench"),
                        ("harmbench_recall", "stress_harmbench")):
        reference = df[(df.model_key == model_keys[0]) & (df.condition == "base")
                       & (df.split == split)]
        base = float(np.nanmean([rate(split, mk, "base", None) for mk in model_keys]))
        sft = float(np.nanmean([np.nanmean([rate(split, mk, "sft", s) for s in seeds])
                                for mk in model_keys]))
        out[name] = {"base_panel_mean": base, "sft_panel_mean": sft,
                     "delta": float(sft - base), "n_rows": int(len(reference)),
                     "gold_class": int(reference["gold"].iloc[0]), "one_class": True,
                     "aggregation": "fixed-model-panel mean; SFT also mean over seeds",
                     "note": "single-class stress set; NO AP/AUROC computed"}
    return out


# --------------------------------------------------------------------------------------
# claim criteria (formal only when a powered-confirmatory lock permits them)
# --------------------------------------------------------------------------------------
def claim_checks(point, boot, sens, analysis_mode):
    _require(analysis_mode == "precision_focused",
             "powered-confirmatory claim evaluation is disabled")
    rep = boot["aggregate"]["represented"]
    tr = boot["aggregate"]["transfer"]
    rep_est = float(point["aggregate"]["represented"])
    tr_est = float(point["aggregate"]["transfer"])
    represented_met = rep["lcb95_one_sided"] > 0
    loo_b = sens["leave_one_benchmark_out"]["transfer"]["sign_stable"]
    loo_base = sens["leave_one_base_out"]["transfer"]["sign_stable"]
    full_tr_neg = sens["leave_one_benchmark_out"]["transfer"]["full"] < 0
    transfer_met = (tr["ucb95_one_sided"] < 0) and loo_b and loo_base and full_tr_neg
    specialization = represented_met and transfer_met
    spec_word = ("estimation-only mode: report the joint (represented, transfer) estimate "
                 "and intervals; no formal specialization rejection is claimed")

    represented_out = {
        "criterion": "LCB95(observed fixed-panel represented delta) > 0",
        "estimate": rep_est, "lcb95_one_sided": rep["lcb95_one_sided"],
        "ci95_two_sided": rep["ci95_two_sided"],
        "wording": (f"the estimated represented-source macro-AP change was {rep_est:+.4f} "
                    f"(95% two-sided percentile interval "
                    f"[{rep['ci95_two_sided'][0]:+.4f}, {rep['ci95_two_sided'][1]:+.4f}])"),
    }
    transfer_out = {
        "criterion": "UCB95(observed fixed-panel transfer delta) < 0 AND LOO sign-stable",
        "estimate": tr_est, "ucb95_one_sided": tr["ucb95_one_sided"],
        "ci95_two_sided": tr["ci95_two_sided"],
        "full_transfer_negative": bool(full_tr_neg),
        "loo_benchmark_sign_stable": bool(loo_b),
        "loo_base_sign_stable": bool(loo_base),
        "wording": (f"the estimated transfer macro-AP change was {tr_est:+.4f} "
                    f"(95% two-sided percentile interval "
                    f"[{tr['ci95_two_sided'][0]:+.4f}, {tr['ci95_two_sided'][1]:+.4f}])"),
    }
    specialization_out = {
        "criterion": "represented criterion AND transfer criterion",
        "wording": spec_word,
    }
    represented_out["descriptive_criterion_met"] = bool(represented_met)
    transfer_out["descriptive_criterion_met"] = bool(transfer_met)
    specialization_out["descriptive_criterion_met"] = bool(specialization)

    return {
        "analysis_mode": analysis_mode,
        "precision_focused_language": True,
        "formal_rejection_claimed": False,
        "interval_method": "paired hierarchical percentile bootstrap",
        "represented_criterion": represented_out,
        "transfer_criterion": transfer_out,
        "specialization_pattern": specialization_out,
        "rq4": {"status": "descriptive_only",
                "note": "operating-point TPR/FPR is a deployment diagnostic, not a confirmatory test"},
    }


# --------------------------------------------------------------------------------------
# emit: csv / json / latex tables / figure
# --------------------------------------------------------------------------------------
def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def write_seed_values_csv(path, point):
    import pandas as pd
    pd.DataFrame(point["seed_values"]).to_csv(path, index=False)


def write_per_benchmark_csv(path, data, present, model_keys, seeds, ap_fn, auroc_fn):
    import pandas as pd
    rows = []
    for regime, benches in present.items():
        for mk in model_keys:
            for b in benches:
                e = data[mk].get(b)
                if e is None:
                    continue
                for condition, seed in ([("base", None)] + [("sft", s) for s in seeds]):
                    scores = e["base"] if condition == "base" else e["sft"][seed]
                    rows.append({
                        "model_key": mk, "condition": condition,
                        "seed": (seed if seed is not None else -1),
                        "benchmark": b, "regime": regime,
                        "ap": C.weighted_metric(ap_fn, scores, e["gold"]),
                        "auroc": auroc_fn(scores, e["gold"]),
                        "n": e["n"], "n_pos": int(e["gold"].sum()),
                        "n_neg": int((e["gold"] == 0).sum())})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table3(path, point, boot, model_keys):
    lines = [r"% Auto-generated by analyze_paper_a_sft.py -- do not edit by hand.",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Checkpoint & Rep base & Rep SFT & $\Delta$ Rep [two-sided 95\% percentile CI] "
             r"& Tr base & Tr SFT & $\Delta$ Tr [two-sided 95\% percentile CI] \\", r"\midrule"]
    for mk in model_keys:
        rc = point["per_checkpoint"]["represented"][mk]
        tc = point["per_checkpoint"]["transfer"][mk]
        rb = boot["per_checkpoint"]["represented"][mk]
        tb = boot["per_checkpoint"]["transfer"][mk]
        lines.append(
            f"{_tex(MODEL_LABELS.get(mk, mk))} & {_fmt(rc['base'])} & {_fmt(rc['sft_mean'])} & "
            f"{_fmt(rc['delta'])} [{_fmt(rb['ci95_two_sided'][0])}, {_fmt(rb['ci95_two_sided'][1])}] & "
            f"{_fmt(tc['base'])} & {_fmt(tc['sft_mean'])} & "
            f"{_fmt(tc['delta'])} [{_fmt(tb['ci95_two_sided'][0])}, {_fmt(tb['ci95_two_sided'][1])}] \\\\")
    lines.append(r"\midrule")
    ar = boot["aggregate"]["represented"]; at = boot["aggregate"]["transfer"]
    ar_point = point["aggregate"]["represented"]
    at_point = point["aggregate"]["transfer"]
    lines.append(
        f"Fixed-panel aggregate & -- & -- & {_fmt(ar_point)} "
        f"[{_fmt(ar['ci95_two_sided'][0])}, {_fmt(ar['ci95_two_sided'][1])}] & -- & -- & "
        f"{_fmt(at_point)} [{_fmt(at['ci95_two_sided'][0])}, {_fmt(at['ci95_two_sided'][1])}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_text(path, "\n".join(lines))


def write_table4(path, sens, opr, stress, present):
    lines = [r"% Auto-generated by analyze_paper_a_sft.py -- do not edit by hand.",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Regime / benchmark & Metric & Base & SFT & $\Delta$ & $N$ \\", r"\midrule"]
    for regime in present:
        for b, d in sens["per_benchmark_delta"][regime].items():
            lines.append(f"{_tex(regime)} / {_tex(b)} & AP & -- & -- & {_fmt(d)} & -- \\\\")
    lines.append(r"\midrule")
    for regime in present:
        o = opr[regime]
        lines.append(
            f"{_tex(regime)} & TPR@target FPR & {_fmt(o['base_macro_tpr'])} & "
            f"{_fmt(o['sft_macro_tpr'])} & {_fmt(o['delta_tpr'])} & {o['n_positive_rows']} \\\\")
        lines.append(
            f"{_tex(regime)} & benchmark-macro realized FPR & {_fmt(o['base_macro_fpr'])} & "
            f"{_fmt(o['sft_macro_fpr'])} & {_fmt(o['delta_fpr'])} & {o['n_negative_rows']} \\\\")
        lines.append(
            f"{_tex(regime)} & pooled-negative realized FPR & {_fmt(o['base_pooled_fpr'])} & "
            f"{_fmt(o['sft_pooled_fpr'])} & {_fmt(o['delta_pooled_fpr'])} & "
            f"{o['n_negative_rows']} \\\\")
    lines.append(r"\midrule")
    orb = stress["orbench_benign_fpr"]
    hb = stress["harmbench_recall"]
    lines.append(
        f"Stress / OR-Bench & benign FPR & {_fmt(orb['base_panel_mean'])} & "
        f"{_fmt(orb['sft_panel_mean'])} & {_fmt(orb['delta'])} & {orb['n_rows']} \\\\")
    lines.append(
        f"Stress / HarmBench & recall & {_fmt(hb['base_panel_mean'])} & "
        f"{_fmt(hb['sft_panel_mean'])} & {_fmt(hb['delta'])} & {hb['n_rows']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_text(path, "\n".join(lines))


def write_table5_seed_values(path, point, model_keys, seeds):
    """Compact appendix table covering every checkpoint x SFT-seed cell."""
    lines = [
        r"% Auto-generated by analyze_paper_a_sft.py -- do not edit by hand.",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Checkpoint & Seed & $\Delta$ represented AP & $\Delta$ transfer AP \\",
        r"\midrule",
    ]
    for mk in model_keys:
        represented = point["per_checkpoint"]["represented"][mk]["seed_deltas"]
        transfer = point["per_checkpoint"]["transfer"][mk]["seed_deltas"]
        for seed in seeds:
            lines.append(
                f"{_tex(MODEL_LABELS.get(mk, mk))} & {seed} & {_fmt(represented[seed])} & "
                f"{_fmt(transfer[seed])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_text(path, "\n".join(lines))


def write_results_macros(path, point, boot, opr, stress):
    """One generated source for every aggregate number used in narrative prose."""
    rep_point = point["aggregate"]["represented"]
    tr_point = point["aggregate"]["transfer"]
    rep_boot = boot["aggregate"]["represented"]
    tr_boot = boot["aggregate"]["transfer"]

    def signed(value, nd=4):
        return f"{float(value):+.{nd}f}"

    def plain(value, nd=4):
        return f"{float(value):.{nd}f}"

    def pct(value):
        return f"{100.0 * float(value):.1f}"

    def signed_pct(value):
        return f"{100.0 * float(value):+.1f}"

    values = {
        "RepDelta": signed(rep_point),
        "RepCILower": signed(rep_boot["ci95_two_sided"][0]),
        "RepCIUpper": signed(rep_boot["ci95_two_sided"][1]),
        "RepLCB": signed(rep_boot["lcb95_one_sided"]),
        "TransferDelta": signed(tr_point),
        "TransferCILower": signed(tr_boot["ci95_two_sided"][0]),
        "TransferCIUpper": signed(tr_boot["ci95_two_sided"][1]),
        "TransferUCB": signed(tr_boot["ucb95_one_sided"]),
    }
    quadrant_counts = {
        "SpecializationSeedCount": 0,
        "UniformGainSeedCount": 0,
        "UniformLossSeedCount": 0,
        "TransferFavoredSeedCount": 0,
        "ZeroAxisSeedCount": 0,
    }
    represented = point["per_checkpoint"]["represented"]
    transfer = point["per_checkpoint"]["transfer"]
    for model_key, represented_cell in represented.items():
        for seed, rep_delta in represented_cell["seed_deltas"].items():
            transfer_delta = transfer[model_key]["seed_deltas"][seed]
            if rep_delta > 0 and transfer_delta < 0:
                quadrant_counts["SpecializationSeedCount"] += 1
            elif rep_delta > 0 and transfer_delta > 0:
                quadrant_counts["UniformGainSeedCount"] += 1
            elif rep_delta < 0 and transfer_delta < 0:
                quadrant_counts["UniformLossSeedCount"] += 1
            elif rep_delta < 0 and transfer_delta > 0:
                quadrant_counts["TransferFavoredSeedCount"] += 1
            else:
                quadrant_counts["ZeroAxisSeedCount"] += 1
    values["TotalSeedCount"] = str(sum(quadrant_counts.values()))
    values.update({name: str(count) for name, count in quadrant_counts.items()})
    regime_prefix = {"represented": "Rep", "transfer": "Transfer"}
    for regime, prefix in regime_prefix.items():
        row = opr[regime]
        values.update({
            f"{prefix}BaseTPR": plain(row["base_macro_tpr"]),
            f"{prefix}SFTTPR": plain(row["sft_macro_tpr"]),
            f"{prefix}DeltaTPR": signed(row["delta_tpr"]),
            f"{prefix}BaseTPRPct": pct(row["base_macro_tpr"]),
            f"{prefix}SFTTPRPct": pct(row["sft_macro_tpr"]),
            f"{prefix}DeltaTPRPct": signed_pct(row["delta_tpr"]),
            f"{prefix}BaseFPR": plain(row["base_macro_fpr"]),
            f"{prefix}SFTFPR": plain(row["sft_macro_fpr"]),
            f"{prefix}DeltaFPR": signed(row["delta_fpr"]),
            f"{prefix}BaseFPRPct": pct(row["base_macro_fpr"]),
            f"{prefix}SFTFPRPct": pct(row["sft_macro_fpr"]),
            f"{prefix}DeltaFPRPct": signed_pct(row["delta_fpr"]),
            f"{prefix}BasePooledFPR": plain(row["base_pooled_fpr"]),
            f"{prefix}SFTPooledFPR": plain(row["sft_pooled_fpr"]),
            f"{prefix}DeltaPooledFPR": signed(row["delta_pooled_fpr"]),
            f"{prefix}BasePooledFPRPct": pct(row["base_pooled_fpr"]),
            f"{prefix}SFTPooledFPRPct": pct(row["sft_pooled_fpr"]),
            f"{prefix}DeltaPooledFPRPct": signed_pct(row["delta_pooled_fpr"]),
            f"{prefix}PositiveN": str(int(row["n_positive_rows"])),
            f"{prefix}NegativeN": str(int(row["n_negative_rows"])),
        })
    orb = stress["orbench_benign_fpr"]
    hb = stress["harmbench_recall"]
    values.update({
        "ORBenchBaseFPR": plain(orb["base_panel_mean"]),
        "ORBenchSFTFPR": plain(orb["sft_panel_mean"]),
        "ORBenchDeltaFPR": signed(orb["delta"]),
        "ORBenchBaseFPRPct": pct(orb["base_panel_mean"]),
        "ORBenchSFTFPRPct": pct(orb["sft_panel_mean"]),
        "ORBenchDeltaFPRPct": signed_pct(orb["delta"]),
        "ORBenchN": str(int(orb["n_rows"])),
        "HarmBenchBaseRecall": plain(hb["base_panel_mean"]),
        "HarmBenchSFTRecall": plain(hb["sft_panel_mean"]),
        "HarmBenchDeltaRecall": signed(hb["delta"]),
        "HarmBenchBaseRecallPct": pct(hb["base_panel_mean"]),
        "HarmBenchSFTRecallPct": pct(hb["sft_panel_mean"]),
        "HarmBenchDeltaRecallPct": signed_pct(hb["delta"]),
        "HarmBenchN": str(int(hb["n_rows"])),
    })
    lines = [
        r"% Auto-generated by analyze_paper_a_sft.py -- do not edit by hand.",
        *[f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()],
    ]
    _write_text(path, "\n".join(lines))


def write_specialization_figure(path, point, model_keys, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    cmap = plt.get_cmap("tab10")
    for i, mk in enumerate(model_keys):
        xs = [point["per_checkpoint"]["represented"][mk]["seed_deltas"][s] for s in seeds]
        ys = [point["per_checkpoint"]["transfer"][mk]["seed_deltas"][s] for s in seeds]
        ax.scatter(xs, ys, color=cmap(i), s=36, alpha=0.75,
                   label=MODEL_LABELS.get(mk, mk), edgecolors="none")
    ax.scatter([point["aggregate"]["represented"]], [point["aggregate"]["transfer"]],
               marker="X", s=180, color="black", label="fixed-panel mean", zorder=5)
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"represented-source macro-AP $\Delta$")
    ax.set_ylabel(r"transfer macro-AP $\Delta$")
    ax.set_title("Specialization plane (per seed)")
    ax.legend(fontsize=7, loc="best")
    # Volatile PDF timestamps make an otherwise identical scientific artifact dirty on every
    # reproduction.  Removing them yields byte-stable output for the same inputs/backend.
    fig.tight_layout()
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _tex(s): return str(s).replace("_", r"\_")


def _write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def resolve_resampling_settings(lock, requested_reps=None, requested_seed=None):
    """Return locked settings; canonical outputs never accept analytic overrides."""
    rules = lock.get("resampling_rules") or {}
    locked_reps = int(rules.get("replicates", C.DEFAULT_BOOTSTRAP_REPLICATES))
    locked_seed = int(rules.get("rng_seed", C.DEFAULT_BOOTSTRAP_SEED))
    _require(locked_reps > 0, "LOCK bootstrap replicate count must be positive")
    if requested_reps is not None:
        _require(int(requested_reps) == locked_reps,
                 f"bootstrap replicate override {requested_reps} differs from locked {locked_reps}")
    if requested_seed is not None:
        _require(int(requested_seed) == locked_seed,
                 f"bootstrap RNG-seed override {requested_seed} differs from locked {locked_seed}")
    return locked_reps, locked_seed


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run_analysis(df, lock, out_dir, ap_fn, auroc_fn, reps=None, rng_seed=None,
                 *, metadata=None, allow_synthetic=False, score_verification=None,
                 manifest_rows=None, release_cache=False, release_verification=None):
    if release_cache:
        release_sources = (release_verification or {}).get("execution_sources") or {}
        release_contract = (release_verification or {}).get("release_contract") or {}
        _require((release_verification or {}).get("release_cache_only") is True
                 and ((release_verification or {}).get("public_release") or {}).get("sha256"),
                 "release-cache analysis lacks verified public-release evidence")
        _require(release_contract.get("release_sha256")
                 and release_contract.get("release_file_sha256")
                 and release_contract.get("anchor_path"),
                 "release-cache analysis lacks the tracked RELEASE.json trust root")
        _require(release_sources.get("aggregate_sha256")
                 == (lock.get("execution_sources") or {}).get("aggregate_sha256")
                 and release_sources.get(
                     "original_paper_a_execution_source_verification")
                 == "separate_immutable_source_bundle",
                 "release-cache analysis lacks the locked execution-source binding")
        _require((score_verification or {}).get("bound") is True
                 and not (score_verification or {}).get("legacy", True)
                 and (score_verification or {}).get("scores_sha256")
                 and (score_verification or {}).get("metadata_sha256"),
                 "release-cache analysis lacks bound score/metadata hashes")
    validation = validate_score_artifacts(
        df, lock, metadata, allow_synthetic=allow_synthetic,
        manifest_rows=manifest_rows, release_cache=release_cache)
    regimes = lock.get("regime_benchmarks", C.REGIME_BENCHMARKS)
    model_keys = list(C.MODEL_KEYS)
    seeds = C.lock_seeds(lock)
    reps, rng_seed = resolve_resampling_settings(lock, reps, rng_seed)
    analysis_mode = lock.get("analysis_mode", "precision_focused")

    data, families, present = build_bench_data(df, regimes, model_keys, seeds)
    point = point_estimates(data, present, model_keys, seeds, ap_fn)
    boot = hierarchical_bootstrap(data, present, model_keys, seeds, ap_fn, reps, rng_seed)
    sens = sensitivity(data, present, model_keys, seeds, ap_fn, point)
    opr = operating_point(df, present, model_keys, seeds)
    stress = stress_metrics(df, model_keys, seeds)
    checks = claim_checks(point, boot, sens, analysis_mode)

    os.makedirs(out_dir, exist_ok=True)
    tables = os.path.join(out_dir, "tables"); figures = os.path.join(out_dir, "figures")
    os.makedirs(tables, exist_ok=True); os.makedirs(figures, exist_ok=True)

    results = {
        # Deliberately omit a wall-clock timestamp: results.json is a deterministic scientific
        # payload. Execution timestamps belong in an external run log, not the result identity.
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "lock_sha256": lock.get("lock_sha256"), "analysis_mode": analysis_mode,
        "score_artifact": score_verification,
        "score_validation": validation,
        "model_keys": model_keys, "seeds": seeds,
        "benchmarks_present": present, "n_families": len(families),
        "point_estimates": {"per_checkpoint": point["per_checkpoint"],
                            "aggregate": point["aggregate"]},
        "bootstrap": boot, "operating_point": opr, "stress": stress,
    }
    results_path = os.path.join(out_dir, "results.json")
    C.write_json(results_path, results)
    C.write_json(os.path.join(out_dir, "sensitivity.json"), sens)
    C.write_json(os.path.join(out_dir, "claim_checks.json"), checks)
    write_seed_values_csv(os.path.join(out_dir, "seed_values.csv"), point)
    write_per_benchmark_csv(os.path.join(out_dir, "per_benchmark.csv"),
                            data, present, model_keys, seeds, ap_fn, auroc_fn)
    write_table3(os.path.join(tables, "table3_primary.tex"), point, boot, model_keys)
    write_table4(os.path.join(tables, "table4_per_benchmark.tex"), sens, opr, stress, present)
    write_table5_seed_values(
        os.path.join(tables, "table5_seed_values.tex"), point, model_keys, seeds)
    write_results_macros(
        os.path.join(tables, "results_macros.tex"), point, boot, opr, stress)
    write_specialization_figure(os.path.join(figures, "specialization_plane.pdf"),
                                point, model_keys, seeds)
    output_files = (
        "results.json", "sensitivity.json", "claim_checks.json", "seed_values.csv",
        "per_benchmark.csv", "tables/table3_primary.tex",
        "tables/table4_per_benchmark.tex", "tables/table5_seed_values.tex",
        "tables/results_macros.tex", "figures/specialization_plane.pdf",
    )
    analysis_runtime = C.runtime_environment("cpu")
    analysis_sources = C.execution_source_hashes()
    current_analysis_sources = C.execution_source_hashes(
        required_files=CURRENT_ANALYSIS_SOURCE_FILES)
    analysis_metadata = {
        "analysis_artifact_contract_version": 2,
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "lock_sha256": lock.get("lock_sha256"),
        "scores_sha256": ((score_verification or {}).get("scores_sha256")),
        "scores_metadata_sha256": ((score_verification or {}).get("metadata_sha256")),
        "scores_metadata_filename": ((score_verification or {}).get("metadata_filename")),
        "execution_sources_sha256": (lock.get("execution_sources") or {}).get(
            "aggregate_sha256"),
        "actual_execution_sources": analysis_sources,
        "analysis_runtime_environment": analysis_runtime,
        "analysis_runtime_sha256": C.canonical_obj_sha256(analysis_runtime),
        "outputs": {rel: C.sha256_file(os.path.join(out_dir, rel)) for rel in output_files},
    }
    if release_cache:
        public_release = (release_verification or {}).get("public_release") or {}
        release_sources = (release_verification or {}).get("execution_sources") or {}
        release_contract = (release_verification or {}).get("release_contract") or {}
        score_hashes_verified = bool(
            (score_verification or {}).get("bound")
            and not (score_verification or {}).get("legacy", True)
            and (score_verification or {}).get("scores_sha256")
            and (score_verification or {}).get("metadata_sha256"))
        analysis_metadata["release_cache_verification"] = {
            "mode": "strict_v2_score_only_release_cache",
            "release_contract": release_contract,
            "public_manifest_sha256": public_release.get("sha256"),
            "public_splits": public_release.get("splits"),
            "score_and_metadata_hashes_reverified": score_hashes_verified,
            "current_analysis_source_hashes": current_analysis_sources,
            "original_paper_a_execution_source": {
                "aggregate_sha256": release_sources.get("aggregate_sha256"),
                "verification": release_sources.get(
                    "original_paper_a_execution_source_verification"),
                "current_checkout_files_reverified": False,
            },
            "raw_manifest_files_locally_reverified": False,
            "run_metadata_and_adapter_bytes_locally_reverified": False,
        }
    C.write_json(os.path.join(out_dir, "analysis_metadata.json"), analysis_metadata)
    return results, checks, sens, point, boot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paper A analysis (plan sec 12/15/16).")
    ap.add_argument("--lock", default=None)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bootstrap-reps", type=int, default=None)
    ap.add_argument("--bootstrap-seed", type=int, default=None)
    ap.add_argument(
        "--allow-legacy-lock", action="store_true",
        help="explicitly permit a legacy v1 lock; never implied for final analysis")
    ap.add_argument(
        "--nonfinal", action="store_true",
        help="write a diagnostic analysis outside every canonical artifact namespace")
    ap.add_argument(
        "--release-cache", action="store_true",
        help=("strict v2 score-only release mode: verify the bound text-free public "
              "manifests and combined scores without local raw prompts/adapters"))
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic end-to-end check of bootstrap + gates + emitters")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test(args)

    if not (args.lock and args.scores and args.out):
        ap.error("--lock, --scores and --out are required (or use --self-test)")
    import pandas as pd
    import inspect
    lock_preview = C.read_json(args.lock)
    legacy_lock = int(lock_preview.get("lock_contract_version", 1)) < 2
    try:
        validate_release_cache_request(
            lock_preview, release_cache=args.release_cache, nonfinal=args.nonfinal)
    except ScoreValidationError as exc:
        ap.error(str(exc))
    load_lock_params = inspect.signature(C.load_lock).parameters
    load_lock_kwargs = {}
    if "allow_legacy" in load_lock_params:
        load_lock_kwargs["allow_legacy"] = args.allow_legacy_lock
    if "verify_files" in load_lock_params:
        # Default strict analysis verifies every on-disk prerequisite. Release-cache analysis
        # verifies config/public evidence here and leaves original source-byte verification to
        # the separately distributed immutable source bundle.
        load_lock_kwargs["verify_files"] = not legacy_lock and not args.release_cache
    # Transitional compatibility while paper_a_common's verified-lock API lands.
    lock = C.load_lock(args.lock, **load_lock_kwargs)
    strict_lock = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    release_verification = (
        C.verify_release_cache_lock(lock) if args.release_cache else None)
    apaths = C.artifact_paths(lock)
    out_dir = str(C.resolved_path(args.out))
    scores_path = str(C.resolved_path(args.scores))
    validate_analysis_paths(lock, out_dir, scores_path, nonfinal=args.nonfinal)
    validate_analysis_runtime(
        lock, nonfinal=args.nonfinal, release_cache=args.release_cache)
    metadata_path = os.path.join(os.path.dirname(scores_path), "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"sibling score metadata is required for analysis: {metadata_path}")
    score_verification = None
    if hasattr(C, "verify_score_artifact"):
        verified_score = C.verify_score_artifact(
            scores_path, metadata_path, lock, allow_legacy=args.allow_legacy_lock)
        metadata = verified_score["metadata"]
        score_verification = {
            key: value for key, value in verified_score.items() if key != "metadata"}
    else:
        metadata = C.read_json(metadata_path)
    df = pd.read_parquet(scores_path)
    ap_fn, auroc_fn = C.require_metrics()
    manifest_rows = (
        load_public_scoring_manifest_rows(lock) if args.release_cache
        else load_locked_scoring_manifest_rows(lock) if strict_lock else None)
    results, checks, _, _, boot = run_analysis(
        df, lock, out_dir, ap_fn, auroc_fn,
        reps=args.bootstrap_reps, rng_seed=args.bootstrap_seed, metadata=metadata,
        score_verification=score_verification, manifest_rows=manifest_rows,
        release_cache=args.release_cache, release_verification=release_verification)
    print(f"[analyze] wrote results/seed_values/per_benchmark/sensitivity/claim_checks "
          f"+ tables/macros + figure to {out_dir}")
    point_results = results["point_estimates"]["aggregate"]
    print(f"[analyze] represented delta estimate={point_results['represented']:+.4f} "
          f"LCB={boot['aggregate']['represented']['lcb95_one_sided']:+.4f}")
    print(f"[analyze] transfer   delta estimate={point_results['transfer']:+.4f} "
          f"UCB={boot['aggregate']['transfer']['ucb95_one_sided']:+.4f}")
    decision_key = "descriptive_criterion_met"
    print(f"[analyze] represented_criterion={checks['represented_criterion'][decision_key]} "
          f"transfer_criterion={checks['transfer_criterion'][decision_key]} "
          f"specialization_pattern={checks['specialization_pattern'][decision_key]} "
          f"(mode={checks['analysis_mode']})")
    return 0


# --------------------------------------------------------------------------------------
# self-test: fabricate a scores DataFrame with a KNOWN effect and check the machinery
# --------------------------------------------------------------------------------------
def _synthetic_scores_df(effect_rep=+0.9, effect_tr=-0.9, seeds=(42, 43, 44, 45, 46),
                         n_per=60, rng_seed=0):
    import pandas as pd
    rng = np.random.default_rng(rng_seed)
    regimes = {"represented": (["toxicchat", "prompt_injections", "jailbreak_classification"], "id_test"),
               "transfer": (["jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"], "transfer_test")}
    rows = []
    for mk in C.MODEL_KEYS:
        base_shift = rng.normal(0, 0.05)
        for regime, (benches, split) in regimes.items():
            eff = effect_rep if regime == "represented" else effect_tr
            for b in benches:
                for i in range(n_per):
                    gold = i % 2
                    fam = f"{b}_fam_{i % (n_per // 3)}"
                    sid = f"{b}_{i}"
                    csha = f"{b}:{i}"
                    base_score = (2 * gold - 1) * (1.0 + base_shift) + rng.normal(0, 1.0)
                    rows.append(_row(sid, csha, b, split, gold, fam, mk, "base", -1, base_score))
                    for s in seeds:
                        sft_score = base_score + eff * (2 * gold - 1) + rng.normal(0, 0.15)
                        rows.append(_row(sid, csha, b, split, gold, fam, mk, "sft", s, sft_score))
    # Calibration rows are required by the locked score-matrix contract even though primary AP
    # excludes them.
    n_cal = max(6, n_per // 3)
    for mk in C.MODEL_KEYS:
        for b in regimes["represented"][0]:
            for i in range(n_cal):
                gold = i % 2
                sid = f"cal_{b}_{i}"
                csha = f"cal:{b}:{i}"
                fam = f"cal_{b}_fam_{i}"
                base_score = (2 * gold - 1) + rng.normal(0, 1.0)
                rows.append(_row(sid, csha, b, "calibration", gold, fam, mk,
                                 "base", -1, base_score))
                for s in seeds:
                    rows.append(_row(sid, csha, b, "calibration", gold, fam, mk,
                                     "sft", s, base_score + rng.normal(0, 0.15)))
    # stress rows (one-class) for RQ4/stress emitters
    for mk in C.MODEL_KEYS:
        for i in range(20):
            rows.append(_row(f"orb_{i}", f"orb:{i}", "orbench", "stress_orbench", 0,
                             f"orb_{i}", mk, "base", -1, rng.normal(-1, 1)))
            rows.append(_row(f"hb_{i}", f"hb:{i}", "harmbench", "stress_harmbench", 1,
                             f"hb_{i}", mk, "base", -1, rng.normal(1, 1)))
            for s in seeds:
                rows.append(_row(f"orb_{i}", f"orb:{i}", "orbench", "stress_orbench", 0,
                                 f"orb_{i}", mk, "sft", s, rng.normal(-1, 1)))
                rows.append(_row(f"hb_{i}", f"hb:{i}", "harmbench", "stress_harmbench", 1,
                                 f"hb_{i}", mk, "sft", s, rng.normal(1, 1)))
    return pd.DataFrame(rows)


def _row(sid, csha, src, split, gold, fam, mk, cond, seed, score, pred=None):
    prob = 1.0 / (1.0 + math.exp(-score))
    return {"sample_id": sid, "content_sha256": csha, "source": src, "split": split,
            "gold": gold, "family_id": fam, "model_key": mk, "model_revision": "rev",
            "condition": cond, "seed": seed,
            "adapter_sha256": (None if cond == "base" else f"synthetic::{mk}:{seed}"),
            "prompt_sha256": "p",
            "safe_token_id": 0, "unsafe_token_id": 1, "safe_logit": -score / 2,
            "unsafe_logit": score / 2, "score_raw": score, "probability_raw": prob,
            "probability_calibrated": prob,
            "threshold_id": f"{mk}:{cond}:{seed}:synthetic",
            "prediction": (int(prob >= 0.5) if pred is None else pred),
            "original_token_count": 5, "scored_token_count": 5, "truncated": False, "latency_ms": 0.0}


def _synthetic_lock_and_metadata(df, *, reps=300, seeds=(42, 43, 44, 45, 46)):
    """Build internally consistent lock/metadata fixtures for the executable self-test."""
    seeds = list(seeds)
    models = {
        mk: {
            "model_id": f"synthetic/{mk}", "model_revision": "rev",
            "tokenizer_revision": "rev",
        }
        for mk in C.MODEL_KEYS
    }
    probe = {
        mk: {
            "safe_token_id": 0, "unsafe_token_id": 1,
            "safe_str": " safe", "unsafe_str": " unsafe", "status": "ok",
            "prompt_template_sha256": "p",
        }
        for mk in C.MODEL_KEYS
    }
    reference = df[(df.model_key == C.MODEL_KEYS[0]) & (df.condition == "base")]
    split_hashes = {split: f"synthetic-sha::{split}" for split in SCORING_SPLIT_FILES}
    locked_splits = {
        filename: {
            "path": f"synthetic/{filename}",
            "rows": int((reference.split == split).sum()),
            "sha256": split_hashes[split],
        }
        for split, filename in SCORING_SPLIT_FILES.items()
    }
    lock = {
        "analysis_mode": "precision_focused", "seeds": seeds,
        "models": models, "regime_benchmarks": C.REGIME_BENCHMARKS,
        "resampling_rules": {"replicates": reps, "rng_seed": 20260712},
        "lock_sha256": "selftest", "score_code_version": "paper_a_sft_scorer_v1",
        "prompt": {"per_model_template_sha256": {mk: "p" for mk in C.MODEL_KEYS}},
        "tokenizer_probe": probe, "manifests": {"splits": locked_splits},
    }
    per_split = {split: split_hashes[split] for split in SCORING_SPLIT_FILES}
    bundles = {}
    for mk in C.MODEL_KEYS:
        base = df[(df.model_key == mk) & (df.condition == "base")]
        bundles[f"{mk}:base"] = {
            "calibration": {"status": "ok", "temperature": 1.0},
            "threshold": {
                "threshold_id": str(base["threshold_id"].iloc[0]),
                "status": "ok", "threshold_value": 0.5,
            }}
        for seed in seeds:
            cell = df[(df.model_key == mk) & (df.condition == "sft") & (df.seed == seed)]
            bundles[f"{mk}:sft:seed_{seed}"] = {
                "calibration": {"status": "ok", "temperature": 1.0},
                "threshold": {
                    "threshold_id": str(cell["threshold_id"].iloc[0]),
                    "status": "ok", "threshold_value": 0.5,
                }}
    metadata = {
        "lock_sha256": lock["lock_sha256"],
        "score_code_version": lock["score_code_version"],
        "n_rows_total": int(len(df)), "columns": EXPECTED_SCORE_COLUMNS,
        "seeds": seeds, "models": C.lock_model_panel(lock), "bundles": bundles,
        "n_bundles": len(bundles), "synthetic": True,
        "manifest_fingerprints": {
            "n_rows": int(len(reference)),
            "sample_ids_fingerprint": C.sha256_ordered(
                reference["sample_id"].astype(str).tolist()),
            "content_fingerprint": C.sha256_ordered(
                reference["content_sha256"].astype(str).tolist()),
            "per_split_manifest_sha256": per_split,
            "manifest_sha256": C.sha256_ordered(
                [f"{k}={v}" for k, v in sorted(per_split.items())]),
        },
    }
    return lock, metadata


def _self_test(args) -> int:
    import tempfile
    ap_fn, auroc_fn = C.require_metrics()
    test_reps = args.bootstrap_reps or 300
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); ok = ok and bool(cond)

    print("== self-test A: positive represented, negative transfer (specialization) ==")
    df = _synthetic_scores_df(+0.9, -0.9)
    lock, metadata = _synthetic_lock_and_metadata(df, reps=test_reps)
    out = tempfile.mkdtemp()
    synthetic_score_binding = {
        "scores_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "metadata_filename": "metadata.json",
    }
    results, checks, sens, point, boot = run_analysis(df, lock, out, ap_fn, auroc_fn,
                                                      reps=test_reps,
                                                      metadata=metadata, allow_synthetic=True,
                                                      score_verification=synthetic_score_binding)
    check("represented aggregate > 0", point["aggregate"]["represented"] > 0)
    check("transfer aggregate < 0", point["aggregate"]["transfer"] < 0)
    check("represented descriptive criterion met",
          checks["represented_criterion"]["descriptive_criterion_met"])
    check("transfer descriptive criterion met",
          checks["transfer_criterion"]["descriptive_criterion_met"])
    check("specialization descriptive pattern met",
          checks["specialization_pattern"]["descriptive_criterion_met"])
    check("precision mode makes no formal rejection", not checks["formal_rejection_claimed"])
    check("precision mode emits no formal passed fields", "passed" not in str(checks))
    check("transfer LOO-benchmark sign-stable", sens["leave_one_benchmark_out"]["transfer"]["sign_stable"])
    check("precision_focused estimation language",
          "estimated" in checks["represented_criterion"]["wording"])
    for fn in ("results.json", "analysis_metadata.json", "seed_values.csv",
               "per_benchmark.csv", "sensitivity.json",
               "claim_checks.json", "tables/table3_primary.tex", "tables/table4_per_benchmark.tex",
               "tables/table5_seed_values.tex", "tables/results_macros.tex",
               "figures/specialization_plane.pdf"):
        check(f"emitted {fn}", os.path.exists(os.path.join(out, fn)))
    attestation = C.read_json(os.path.join(out, "analysis_metadata.json"))
    check("analysis attestation binds every deterministic output",
          len(attestation.get("outputs", {})) == 10
          and all(C.sha256_file(os.path.join(out, rel)) == digest
                  for rel, digest in attestation["outputs"].items()))
    check("analysis attestation binds score metadata",
          attestation.get("scores_metadata_sha256") == "b" * 64
          and attestation.get("scores_metadata_filename") == "metadata.json")

    print("== self-test B: null effect (no gates) ==")
    df0 = _synthetic_scores_df(0.0, 0.0, rng_seed=7)
    lock0, metadata0 = _synthetic_lock_and_metadata(df0, reps=test_reps)
    out0 = tempfile.mkdtemp()
    _, checks0, _, _, boot0 = run_analysis(
        df0, lock0, out0, ap_fn, auroc_fn, reps=test_reps,
        metadata=metadata0, allow_synthetic=True)
    check("null: represented criterion not met",
          not checks0["represented_criterion"]["descriptive_criterion_met"])
    check("null: transfer criterion not met",
          not checks0["transfer_criterion"]["descriptive_criterion_met"])
    check("null: specialization pattern not met",
          not checks0["specialization_pattern"]["descriptive_criterion_met"])

    print("== self-test C: reproducibility (same seed -> identical aggregate) ==")
    dfr = _synthetic_scores_df(+0.9, -0.9)
    lockr, metar = _synthetic_lock_and_metadata(dfr, reps=test_reps)
    _, _, _, _, boot_r = run_analysis(
        dfr, lockr, tempfile.mkdtemp(), ap_fn, auroc_fn, reps=test_reps,
        metadata=metar, allow_synthetic=True)
    check("bootstrap deterministic w/ fixed rng_seed",
          abs(boot_r["aggregate"]["represented"]["lcb95_one_sided"]
              - boot["aggregate"]["represented"]["lcb95_one_sided"]) < 1e-9)

    print("== self-test D: weighted (replicated) AP == canonical AP when weights all 1 ==")
    s = np.array([0.1, 0.9, 0.4, 0.4, 0.8, 0.2]); y = np.array([0, 1, 0, 1, 1, 0])
    a1 = ap_fn(s, y); a2 = C.weighted_metric(ap_fn, s, y, np.ones_like(y))
    check("weighted_metric(all-ones) == average_precision", abs(a1 - a2) < 1e-12)

    print(f"\nSELF-TEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
