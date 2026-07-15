"""Regression tests for Paper A's fail-closed analysis and reporting estimands."""

import json
import os
import pathlib
import shutil
import sys
import copy

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
for _path in (_ROOT, _EXP):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze_paper_a_sft as A  # noqa: E402


@pytest.fixture()
def synthetic_artifacts():
    df = A._synthetic_scores_df(n_per=12, rng_seed=3)
    lock, metadata = A._synthetic_lock_and_metadata(df, reps=12)
    return df, lock, metadata


def _strict_release_artifacts(tmp_path, synthetic_artifacts):
    """Create one complete strict matrix with locally verifiable fake adapters."""
    df, lock, metadata = synthetic_artifacts
    df = df.copy(deep=True)
    df.insert(len(df.columns) - 1, "truncation_strategy", "none")
    lock = copy.deepcopy(lock)
    metadata = copy.deepcopy(metadata)
    root = tmp_path / "v2"
    lock.update({
        "lock_contract_version": A.C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
        "analysis_code_version": A.ANALYSIS_CODE_VERSION,
        "score_code_version": "paper_a_sft_scorer_v2",
        "lock_sha256": "1" * 64,
        "artifact_paths": A.C.artifact_paths_for_root(root),
        "recipe": copy.deepcopy(A.C.DEFAULT_RECIPE),
        "operating_point": copy.deepcopy(A.C.DEFAULT_OPERATING_POINT),
        "execution_sources": {"aggregate_sha256": "2" * 64},
        "software_versions": {
            key: f"fixture-{key}" for key in A.C.PROTOCOL_SOFTWARE_KEYS},
    })
    for model in lock["models"].values():
        model.update({
            "dtype": "bfloat16", "attn_implementation": None,
            "trust_remote_code": False,
        })

    runtime_details = {
        "software_versions": dict(lock["software_versions"]),
        "requested_device": "fixture",
    }
    runtime_sha = A.C.canonical_obj_sha256(runtime_details)
    metadata.update({
        "score_artifact_contract_version": 2,
        "finalization_status": "final",
        "lock_sha256": lock["lock_sha256"],
        "score_code_version": lock["score_code_version"],
        "execution_sources_sha256": lock["execution_sources"]["aggregate_sha256"],
        "columns": A.SCORE_COLUMNS_V2,
        "synthetic": False,
        "target_fpr": A.C.DEFAULT_TARGET_FPR,
        "producer_runtime": {"sha256": runtime_sha, "details": runtime_details},
        "software_versions": dict(lock["software_versions"]),
        "batch_size": 4,
        "models": A.C.lock_model_panel(lock),
        "dtype_by_model": {mk: "bfloat16" for mk in A.C.MODEL_KEYS},
    })

    adapter_inventory = {}
    for mk in A.C.MODEL_KEYS:
        base_key = f"{mk}:base"
        metadata["bundles"][base_key].update({
            "adapter_sha256": None, "run_meta_path": None, "run_meta_sha256": None,
            "batch_size": 4, "producer_runtime_sha256": runtime_sha,
        })
        for seed in lock["seeds"]:
            key = f"{mk}:sft:seed_{seed}"
            run_dir = root / "runs" / mk / "sft" / f"seed_{seed}"
            adapter = run_dir / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(json.dumps({
                "r": lock["recipe"]["lora"]["r"],
                "lora_alpha": lock["recipe"]["lora"]["alpha"],
                "lora_dropout": lock["recipe"]["lora"]["dropout"],
                "target_modules": lock["recipe"]["lora"]["target_modules"],
                "task_type": "CAUSAL_LM",
            }), encoding="utf-8")
            (adapter / "adapter_model.bin").write_text(
                f"fixture weights {mk} {seed}", encoding="utf-8")
            adapter_sha = A.C.sha256_dir(adapter)
            mask = ((df.model_key == mk) & (df.condition == "sft") & (df.seed == seed))
            df.loc[mask, "adapter_sha256"] = adapter_sha
            run_meta_path = run_dir / "run_meta.json"
            A.C.write_json(run_meta_path, {
                "model_key": mk, "seed": seed, "adapter_sha256": adapter_sha,
                "lock_sha256": lock["lock_sha256"],
            })
            run_meta_sha = A.C.sha256_file(run_meta_path)
            metadata["bundles"][key].update({
                "adapter_sha256": adapter_sha,
                "run_meta_path": str(run_meta_path),
                "run_meta_sha256": run_meta_sha,
                "batch_size": 4,
                "producer_runtime_sha256": runtime_sha,
            })
            adapter_inventory[key] = {
                "adapter_sha256": adapter_sha,
                "run_meta_path": str(run_meta_path),
                "run_meta_sha256": run_meta_sha,
            }

    metadata["adapter_inventory"] = adapter_inventory
    for mk in A.C.MODEL_KEYS:
        for condition, seed in [("base", -1), *[("sft", s) for s in lock["seeds"]]]:
            mask = ((df.model_key == mk) & (df.condition == condition) & (df.seed == seed))
            cell = df.loc[mask]
            calibration = cell[cell.split == "calibration"]
            selected = A.C.normalize_threshold_result(A.C.require_select_threshold()(
                calibration["probability_calibrated"].tolist(),
                calibration["gold"].tolist(), A.C.DEFAULT_TARGET_FPR))
            threshold_value = (None if selected["status"] in {
                "NO_FEASIBLE_THRESHOLD", "PREDICT_NONE"} else selected["threshold"])
            cutoff = float("inf") if threshold_value is None else float(threshold_value)
            df.loc[mask, "prediction"] = (
                df.loc[mask, "probability_calibrated"].to_numpy(float) >= cutoff).astype(int)
            key = f"{mk}:base" if condition == "base" else f"{mk}:sft:seed_{seed}"
            metadata["bundles"][key]["calibration"].update({
                "status": "ok", "optim_success": True, "temperature": 1.0})
            metadata["bundles"][key]["threshold"].update({
                "status": selected["status"], "threshold_value": threshold_value})

    reference = df[(df.model_key == A.C.MODEL_KEYS[0]) & (df.condition == "base")]
    manifest_rows = reference[
        ["sample_id", "content_sha256", "source", "split", "gold", "family_id"]
    ].to_dict("records")
    return df, lock, metadata, manifest_rows, root


def test_exact_synthetic_score_matrix_validates(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    result = A.validate_score_artifacts(
        df, lock, metadata, allow_synthetic=True)
    assert result["n_bundles"] == 24
    assert result["n_samples_per_bundle"] > 0


def test_exact_v2_score_schema_validates(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    df = df.copy()
    df.insert(len(df.columns) - 1, "truncation_strategy", "none")
    lock = dict(lock)
    lock["score_code_version"] = "paper_a_sft_scorer_v2"
    metadata = dict(metadata)
    metadata["score_code_version"] = "paper_a_sft_scorer_v2"
    metadata["columns"] = A.SCORE_COLUMNS_V2
    result = A.validate_score_artifacts(
        df, lock, metadata, allow_synthetic=True)
    assert result["n_bundles"] == 24


def test_reference_identity_must_match_locked_manifest_gold_and_family():
    import pandas as pd
    reference = pd.DataFrame([{
        "sample_id": "s1", "content_sha256": "a" * 64, "source": "fixture",
        "split": "id_test", "gold": 1, "family_id": "b" * 64,
    }])
    manifest = [{
        "sample_id": "s1", "content_sha256": "a" * 64, "source": "fixture",
        "split": "id_test", "gold": 1, "family_id": "b" * 64,
    }]
    assert A.validate_reference_against_manifests(reference, manifest) == {
        "n_manifest_rows_joined": 1}
    for field, value in (("gold", 0), ("family_id", "c" * 64)):
        bad = reference.copy()
        bad.loc[0, field] = value
        with pytest.raises(A.ScoreValidationError, match="locked scoring manifests"):
            A.validate_reference_against_manifests(bad, manifest)


def test_locked_target_fpr_threshold_is_recomputed_not_trusted():
    import pandas as pd
    cell = pd.DataFrame({
        "split": ["calibration"] * 120,
        "gold": [0] * 100 + [1] * 20,
        "probability_calibrated": [0.1] * 100 + [0.9] * 20,
    })
    selected = A.C.normalize_threshold_result(A.C.require_select_threshold()(
        cell["probability_calibrated"].tolist(), cell["gold"].tolist(), 0.05))
    threshold = {
        "status": selected["status"], "threshold_value": selected["threshold"]}
    A.validate_locked_threshold_selection(cell, threshold, 0.05, "fixture")
    bad = dict(threshold)
    bad["threshold_value"] = 0.2
    with pytest.raises(A.ScoreValidationError, match="not canonical"):
        A.validate_locked_threshold_selection(cell, bad, 0.05, "fixture")


def test_positive_infinity_threshold_round_trips_as_explicit_predict_none():
    import pandas as pd
    cell = pd.DataFrame({
        "split": ["calibration"] * 120,
        "gold": [0] * 100 + [1] * 20,
        "probability_calibrated": [0.9] * 100 + [0.1] * 20,
    })
    selected = A.C.normalize_threshold_result(A.C.require_select_threshold()(
        cell["probability_calibrated"].tolist(), cell["gold"].tolist(), 0.05))
    assert selected["status"] == "PREDICT_NONE"
    assert np.isposinf(selected["threshold"])
    stored = {"status": "PREDICT_NONE", "threshold_value": None}
    A.validate_locked_threshold_selection(cell, stored, 0.05, "fixture")
    assert json.dumps(stored, allow_nan=False)


def test_nonfinal_analysis_cannot_write_canonical_or_legacy_namespace(tmp_path):
    root = tmp_path / "v2"
    lock = {
        "lock_contract_version": A.C.LOCK_CONTRACT_VERSION,
        "artifact_paths": A.C.artifact_paths_for_root(root),
    }
    scores = root / "scores" / "scores.parquet"
    with pytest.raises(A.ScoreValidationError, match="outside canonical"):
        A.validate_analysis_paths(
            lock, root / "analysis", scores, nonfinal=True)
    with pytest.raises(A.ScoreValidationError, match="outside canonical"):
        A.validate_analysis_paths(
            lock, A.C.DEFAULT_ARTIFACTS["analysis"], scores, nonfinal=True)
    diagnostic = tmp_path / "diagnostic-analysis"
    result = A.validate_analysis_paths(lock, diagnostic, scores, nonfinal=True)
    assert result["out_dir"] == str(diagnostic.resolve())


def test_canonical_analysis_preflights_locked_software(monkeypatch):
    locked = {key: f"locked-{key}" for key in A.C.PROTOCOL_SOFTWARE_KEYS}
    runtime = dict(locked)
    runtime["numpy"] = "drifted"
    lock = {
        "lock_contract_version": A.C.LOCK_CONTRACT_VERSION,
        "software_versions": locked,
    }
    monkeypatch.setattr(A.C, "software_versions", lambda: dict(runtime))
    with pytest.raises(A.ScoreValidationError, match="runtime software differs"):
        A.validate_analysis_runtime(lock)
    A.validate_analysis_runtime(lock, nonfinal=True)


def test_release_cache_allows_only_python_patch_drift(monkeypatch):
    locked = {key: f"locked-{key}" for key in A.C.PROTOCOL_SOFTWARE_KEYS}
    locked["python"] = "3.12.3"
    runtime = dict(locked)
    runtime["python"] = "3.12.10"
    for key in set(A.C.PROTOCOL_SOFTWARE_KEYS) - set(A.RELEASE_ANALYSIS_SOFTWARE_KEYS):
        runtime[key] = None
    lock = {
        "lock_contract_version": A.C.LOCK_CONTRACT_VERSION,
        "software_versions": locked,
    }

    monkeypatch.setattr(A.C, "software_versions", lambda: dict(runtime))
    A.validate_analysis_runtime(lock, release_cache=True)
    with pytest.raises(A.ScoreValidationError, match="python_mismatch"):
        A.validate_analysis_runtime(lock)

    runtime["python"] = "3.13.0"
    with pytest.raises(A.ScoreValidationError, match="python_mismatch"):
        A.validate_analysis_runtime(lock, release_cache=True)

    runtime["python"] = "3.12.10"
    runtime["numpy"] = "drifted"
    with pytest.raises(A.ScoreValidationError, match="numpy_mismatch"):
        A.validate_analysis_runtime(lock, release_cache=True)


def test_release_cache_is_strict_final_and_not_nonfinal():
    with pytest.raises(A.ScoreValidationError, match="strict final v2"):
        A.validate_release_cache_request(
            {"lock_contract_version": 1}, release_cache=True)
    strict = {
        "lock_contract_version": A.C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
    }
    with pytest.raises(A.ScoreValidationError, match="cannot be combined"):
        A.validate_release_cache_request(
            strict, release_cache=True, nonfinal=True)
    A.validate_release_cache_request(strict, release_cache=True)


def test_release_cache_matches_full_analysis_without_local_adapters(
        tmp_path, synthetic_artifacts):
    df, lock, metadata, manifest_rows, root = _strict_release_artifacts(
        tmp_path, synthetic_artifacts)
    public_dir = root / "public_manifests"
    public_dir.mkdir(parents=True)
    for split, filename in A.SCORING_SPLIT_FILES.items():
        rows = [row for row in manifest_rows if row["split"] == split]
        public_rows = []
        for row in rows:
            public_row = dict(row)
            public_row["split"] = pathlib.Path(filename).stem
            public_rows.append(public_row)
        (public_dir / filename).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in public_rows),
            encoding="utf-8")
    public_rows = A.load_public_scoring_manifest_rows(lock)
    expected_identity = {row["sample_id"]: row for row in manifest_rows}
    observed_identity = {row["sample_id"]: row for row in public_rows}
    assert observed_identity == expected_identity
    ap_fn, auroc_fn = A.C.require_metrics()
    score_binding = {
        "scores_sha256": "3" * 64,
        "metadata_sha256": "4" * 64,
        "metadata_filename": "metadata.json",
        "bound": True,
        "legacy": False,
    }
    full_out = tmp_path / "full"
    release_out = tmp_path / "release"
    A.run_analysis(
        df, lock, str(full_out), ap_fn, auroc_fn, metadata=metadata,
        score_verification=score_binding, manifest_rows=manifest_rows)

    shutil.rmtree(root / "runs")
    with pytest.raises(A.ScoreValidationError, match="run metadata hash"):
        A.validate_score_artifacts(
            df, lock, metadata, manifest_rows=manifest_rows)

    release_verification = {
        "release_cache_only": True,
        "release_contract": {
            "release_sha256": "6" * 64,
            "release_file_sha256": "7" * 64,
            "anchor_path": "configs/paper_a_sft_v2_release_anchor.json",
        },
        "public_release": {"sha256": "5" * 64, "splits": {}},
        "execution_sources": {
            "aggregate_sha256": lock["execution_sources"]["aggregate_sha256"],
            "local_files_verified": False,
            "original_paper_a_execution_source_verification":
                "separate_immutable_source_bundle",
        },
    }
    A.run_analysis(
        df, lock, str(release_out), ap_fn, auroc_fn, metadata=metadata,
        score_verification=score_binding, manifest_rows=manifest_rows,
        release_cache=True, release_verification=release_verification)

    deterministic_outputs = (
        "results.json", "sensitivity.json", "claim_checks.json", "seed_values.csv",
        "per_benchmark.csv", "tables/table3_primary.tex",
        "tables/table4_per_benchmark.tex", "tables/table5_seed_values.tex",
        "tables/results_macros.tex", "figures/specialization_plane.pdf",
    )
    for rel in deterministic_outputs:
        assert (full_out / rel).read_bytes() == (release_out / rel).read_bytes(), rel
    attestation = A.C.read_json(release_out / "analysis_metadata.json")
    release = attestation["release_cache_verification"]
    assert release["score_and_metadata_hashes_reverified"] is True
    assert release["release_contract"] == release_verification["release_contract"]
    assert set(release["current_analysis_source_hashes"]["files"]) == set(
        A.CURRENT_ANALYSIS_SOURCE_FILES)
    assert release["original_paper_a_execution_source"] == {
        "aggregate_sha256": lock["execution_sources"]["aggregate_sha256"],
        "verification": "separate_immutable_source_bundle",
        "current_checkout_files_reverified": False,
    }
    assert release["raw_manifest_files_locally_reverified"] is False
    assert release["run_metadata_and_adapter_bytes_locally_reverified"] is False


def test_incomplete_one_model_matrix_is_rejected(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    one_model = df[df["model_key"] == A.C.MODEL_KEYS[0]].copy()
    with pytest.raises(A.ScoreValidationError, match="row count|model panel"):
        A.validate_score_artifacts(
            one_model, lock, metadata, allow_synthetic=True)


def test_incomplete_one_benchmark_matrix_is_rejected(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    one_benchmark = df[
        (df["source"] == "toxicchat") | df["split"].str.startswith("stress_")
    ].copy()
    with pytest.raises(A.ScoreValidationError, match="row count|sources"):
        A.validate_score_artifacts(
            one_benchmark, lock, metadata, allow_synthetic=True)


def test_identity_drift_between_bundles_is_rejected(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    bad = df.copy()
    mask = (
        (bad["model_key"] == A.C.MODEL_KEYS[1])
        & (bad["condition"] == "sft")
        & (bad["seed"] == lock["seeds"][0])
    )
    idx = bad.index[mask][0]
    bad.loc[idx, "content_sha256"] = "drifted"
    with pytest.raises(A.ScoreValidationError, match="sample identity"):
        A.validate_score_artifacts(
            bad, lock, metadata, allow_synthetic=True)


def test_mutated_calibrated_probability_is_rejected(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    bad = df.copy()
    idx = bad.index[bad["split"] == "transfer_test"][0]
    bad.loc[idx, "probability_calibrated"] += 0.01
    with pytest.raises(A.ScoreValidationError, match="calibrated probabilities"):
        A.validate_score_artifacts(
            bad, lock, metadata, allow_synthetic=True)


def test_mutated_stress_prediction_is_rejected(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    bad = df.copy()
    idx = bad.index[bad["split"] == "stress_harmbench"][0]
    bad.loc[idx, "prediction"] = 1 - int(bad.loc[idx, "prediction"])
    with pytest.raises(A.ScoreValidationError, match="predictions"):
        A.validate_score_artifacts(
            bad, lock, metadata, allow_synthetic=True)


def test_no_feasible_threshold_requires_all_zero_predictions(synthetic_artifacts):
    df, lock, metadata = synthetic_artifacts
    metadata = copy.deepcopy(metadata)
    bundle = f"{A.C.MODEL_KEYS[0]}:base"
    metadata["bundles"][bundle]["threshold"].update({
        "status": "NO_FEASIBLE_THRESHOLD", "threshold_value": None,
    })
    adjusted = df.copy()
    mask = ((adjusted["model_key"] == A.C.MODEL_KEYS[0])
            & (adjusted["condition"] == "base"))
    adjusted.loc[mask, "prediction"] = 0
    A.validate_score_artifacts(adjusted, lock, metadata, allow_synthetic=True)
    idx = adjusted.index[mask][0]
    adjusted.loc[idx, "prediction"] = 1
    with pytest.raises(A.ScoreValidationError, match="predictions"):
        A.validate_score_artifacts(
            adjusted, lock, metadata, allow_synthetic=True)


def test_canonical_resampling_rejects_overrides():
    lock = {"resampling_rules": {"replicates": 10000, "rng_seed": 20260712}}
    assert A.resolve_resampling_settings(lock) == (10000, 20260712)
    assert A.resolve_resampling_settings(lock, 10000, 20260712) == (10000, 20260712)
    with pytest.raises(A.ScoreValidationError, match="replicate override"):
        A.resolve_resampling_settings(lock, 9999, None)
    with pytest.raises(A.ScoreValidationError, match="RNG-seed override"):
        A.resolve_resampling_settings(lock, None, 7)


def _point_for_table():
    cell = {
        "base": 0.2, "sft_mean": 0.5, "delta": 0.3,
        "sft_by_seed": {}, "seed_deltas": {},
    }
    return {
        "per_checkpoint": {
            "represented": {"model": dict(cell)},
            "transfer": {"model": {**cell, "delta": -0.04}},
        },
        "aggregate": {"represented": 0.3333, "transfer": -0.0503},
    }


def _boot_for_table():
    represented = {
        "bootstrap_mean": 0.1111,
        "ci95_two_sided": [0.27, 0.38],
        "lcb95_one_sided": 0.28,
        "ucb95_one_sided": 0.37,
    }
    transfer = {
        "bootstrap_mean": -0.2222,
        "ci95_two_sided": [-0.08, -0.02],
        "lcb95_one_sided": -0.07,
        "ucb95_one_sided": -0.03,
    }
    return {
        "aggregate": {"represented": represented, "transfer": transfer},
        "per_checkpoint": {
            "represented": {"model": represented},
            "transfer": {"model": transfer},
        },
    }


def test_primary_table_uses_observed_point_not_bootstrap_mean(tmp_path):
    output = tmp_path / "table3.tex"
    A.write_table3(output, _point_for_table(), _boot_for_table(), ["model"])
    text = output.read_text()
    aggregate = next(line for line in text.splitlines()
                     if line.startswith("Fixed-panel aggregate"))
    assert "0.3333" in aggregate
    assert "-0.0503" in aggregate
    assert "0.1111" not in aggregate
    assert "-0.2222" not in aggregate
    assert "two-sided 95\\% percentile CI" in text


def _sensitivity_for_claims():
    return {
        "leave_one_benchmark_out": {
            "transfer": {"full": -0.0503, "loo": {"a": -0.04, "b": -0.06},
                         "sign_stable": True},
        },
        "leave_one_base_out": {
            "transfer": {"full": -0.0503, "loo": {"a": -0.03, "b": -0.07},
                         "sign_stable": True},
        },
    }


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_precision_claims_are_descriptive_and_use_observed_point():
    point = {"aggregate": {"represented": 0.3333, "transfer": -0.0503}}
    checks = A.claim_checks(
        point, _boot_for_table(), _sensitivity_for_claims(), "precision_focused")
    keys = set(_all_keys(checks))
    assert checks["formal_rejection_claimed"] is False
    assert checks["represented_criterion"]["estimate"] == pytest.approx(0.3333)
    assert checks["transfer_criterion"]["estimate"] == pytest.approx(-0.0503)
    assert checks["represented_criterion"]["descriptive_criterion_met"] is True
    assert checks["transfer_criterion"]["descriptive_criterion_met"] is True
    assert "passed" not in keys
    assert "p" not in keys
    assert not any("holm" in key.lower() for key in keys)
    assert "0.1111" not in json.dumps(checks)


def test_nonfinite_or_empty_leave_one_out_is_rejected():
    with pytest.raises(A.ScoreValidationError, match="empty"):
        A._sign_stable(-0.1, [])
    with pytest.raises(A.ScoreValidationError, match="finite"):
        A._sign_stable(-0.1, [-0.2, np.nan])


def test_secondary_table_reports_complete_point_diagnostics(tmp_path):
    sens = {
        "per_benchmark_delta": {
            "represented": {"toxicchat": 0.2},
            "transfer": {"xstest": -0.1},
        }
    }
    opr = {
        regime: {
            "base_macro_tpr": 0.1, "sft_macro_tpr": 0.7, "delta_tpr": 0.6,
            "base_macro_fpr": 0.02, "sft_macro_fpr": 0.08, "delta_fpr": 0.06,
            "base_pooled_fpr": 0.01, "sft_pooled_fpr": 0.07,
            "delta_pooled_fpr": 0.06,
            "n_positive_rows": 10, "n_negative_rows": 11,
        }
        for regime in ("represented", "transfer")
    }
    stress = {
        "orbench_benign_fpr": {
            "base_panel_mean": 0.12, "sft_panel_mean": 0.10,
            "delta": -0.02, "n_rows": 400,
        },
        "harmbench_recall": {
            "base_panel_mean": 0.78, "sft_panel_mean": 0.57,
            "delta": -0.21, "n_rows": 200,
        },
    }
    output = tmp_path / "table4.tex"
    A.write_table4(output, sens, opr, stress,
                   {"represented": ["toxicchat"], "transfer": ["xstest"]})
    text = output.read_text()
    assert "Metric & Base & SFT & $\\Delta$ & $N$" in text
    assert "TPR@target FPR & 0.1000 & 0.7000 & 0.6000 & 10" in text
    assert "benchmark-macro realized FPR & 0.0200 & 0.0800 & 0.0600 & 11" in text
    assert "pooled-negative realized FPR & 0.0100 & 0.0700 & 0.0600 & 11" in text
    assert "OR-Bench & benign FPR & 0.1200 & 0.1000 & -0.0200 & 400" in text
    assert "HarmBench & recall & 0.7800 & 0.5700 & -0.2100 & 200" in text


def test_seed_appendix_covers_each_checkpoint_seed_cell(tmp_path):
    point = {
        "per_checkpoint": {
            "represented": {
                "m1": {"seed_deltas": {42: 0.1, 43: 0.2}},
                "m2": {"seed_deltas": {42: 0.3, 43: 0.4}},
            },
            "transfer": {
                "m1": {"seed_deltas": {42: -0.1, 43: -0.2}},
                "m2": {"seed_deltas": {42: -0.3, 43: -0.4}},
            },
        }
    }
    output = tmp_path / "table5.tex"
    A.write_table5_seed_values(output, point, ["m1", "m2"], [42, 43])
    rows = [line for line in output.read_text().splitlines()
            if line.startswith(("m1 ", "m2 "))]
    assert len(rows) == 4
    assert "m1 & 42 & 0.1000 & -0.1000" in rows[0]
    assert "m2 & 43 & 0.4000 & -0.4000" in rows[-1]


def test_result_macros_use_observed_aggregate_and_complete_secondary_values(tmp_path):
    opr = {
        regime: {
            "base_macro_tpr": 0.1, "sft_macro_tpr": 0.7, "delta_tpr": 0.6,
            "base_macro_fpr": 0.02, "sft_macro_fpr": 0.08, "delta_fpr": 0.06,
            "base_pooled_fpr": 0.01, "sft_pooled_fpr": 0.07,
            "delta_pooled_fpr": 0.06,
            "n_positive_rows": 10, "n_negative_rows": 11,
        }
        for regime in ("represented", "transfer")
    }
    stress = {
        "orbench_benign_fpr": {
            "base_panel_mean": 0.12, "sft_panel_mean": 0.10,
            "delta": -0.02, "n_rows": 400,
        },
        "harmbench_recall": {
            "base_panel_mean": 0.78, "sft_panel_mean": 0.57,
            "delta": -0.21, "n_rows": 200,
        },
    }
    output = tmp_path / "results_macros.tex"
    point = _point_for_table()
    point["per_checkpoint"]["represented"]["model"]["seed_deltas"] = {
        42: 0.10, 43: 0.20, 44: 0.30, 45: 0.40,
    }
    point["per_checkpoint"]["transfer"]["model"]["seed_deltas"] = {
        42: -0.10, 43: -0.20, 44: 0.30, 45: 0.40,
    }
    A.write_results_macros(
        output, point, _boot_for_table(), opr, stress)
    text = output.read_text()
    assert "\\newcommand{\\RepDelta}{+0.3333}" in text
    assert "\\newcommand{\\TransferDelta}{-0.0503}" in text
    assert "0.1111" not in text
    assert "\\newcommand{\\TransferDeltaFPR}{+0.0600}" in text
    assert "\\newcommand{\\TransferBasePooledFPR}{0.0100}" in text
    assert "\\newcommand{\\HarmBenchDeltaRecall}{-0.2100}" in text
    assert "\\newcommand{\\TransferBaseFPRPct}{2.0}" in text
    assert "\\newcommand{\\ORBenchBaseFPRPct}{12.0}" in text
    assert "\\newcommand{\\HarmBenchBaseRecallPct}{78.0}" in text
    assert "\\newcommand{\\TotalSeedCount}{4}" in text
    assert "\\newcommand{\\SpecializationSeedCount}{2}" in text
    assert "\\newcommand{\\UniformGainSeedCount}{2}" in text
    assert "\\newcommand{\\ZeroAxisSeedCount}{0}" in text


def test_specialization_figure_is_byte_idempotent(tmp_path):
    point = {
        "aggregate": {"represented": 0.3, "transfer": -0.05},
        "per_checkpoint": {
            "represented": {"model": {"seed_deltas": {42: 0.3, 43: 0.4}}},
            "transfer": {"model": {"seed_deltas": {42: -0.1, 43: 0.02}}},
        },
    }
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    A.write_specialization_figure(first, point, ["model"], [42, 43])
    A.write_specialization_figure(second, point, ["model"], [42, 43])
    assert first.read_bytes() == second.read_bytes()
