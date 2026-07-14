"""Regression tests for Paper A's fail-closed analysis and reporting estimands."""

import json
import os
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
    A.write_results_macros(
        output, _point_for_table(), _boot_for_table(), opr, stress)
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
