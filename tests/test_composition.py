"""Unit + self-tests for the composition ("compose, don't tune") analyzer.

Covers the parts that must be correct for the paper's numbers: the weighted tie-aware
AP wrapper, the combiner math, seed-averaged benchmark-macro AP, and end-to-end
determinism of the paired hierarchical bootstrap (same rng_seed -> identical CIs).
"""

import os
import sys
import json
import subprocess

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze_composition as AC  # noqa: E402


# --------------------------------------------------------------------------- wap
def test_wap_matches_sklearn_and_arg_order():
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 0, 1, 1, 0])
    s = rng.random(6)
    # AC.wap(scores, labels): scores first (the guard_research convention)
    assert AC.wap(s, y) == pytest.approx(average_precision_score(y, s))


def test_wap_single_class_is_nan():
    assert np.isnan(AC.wap([0.1, 0.2, 0.3], [0, 0, 0]))
    assert np.isnan(AC.wap([], []))


def test_wap_weighted_matches_sklearn_sample_weight():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.2, 0.9, 0.4, 0.6])
    w = np.array([1.0, 2.0, 1.0, 3.0])
    assert AC.wap(s, y, weights=w) == pytest.approx(
        average_precision_score(y, s, sample_weight=w))


def test_wap_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])  # all positives ranked above negatives
    assert AC.wap(s, y) == pytest.approx(1.0)


def test_strict_load_does_not_normalize_score_evidence(monkeypatch):
    frame = AC.pd.DataFrame({"gold": [0, 1], "seed": [42.5, 43.0]})
    monkeypatch.setattr(AC.pd, "read_parquet", lambda _path: frame.copy())
    strict = AC.load("scores.parquet", strict=True)
    assert strict["seed"].tolist() == [42.5, 43.0]
    legacy = AC.load("scores.parquet", strict=False)
    assert legacy["seed"].tolist() == [42, 43]


def test_composition_source_aggregate_covers_dependencies_and_changes_on_drift(tmp_path):
    for rel in AC.COMPOSITION_ANALYSIS_SOURCE_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture {rel}\n", encoding="utf-8")
    first = AC.composition_analysis_source_hashes(repo_root=tmp_path)
    assert set(first["files"]) == set(AC.COMPOSITION_ANALYSIS_SOURCE_FILES)

    drifted = tmp_path / "guard_research/metrics.py"
    drifted.write_text(drifted.read_text(encoding="utf-8") + "# drift\n",
                       encoding="utf-8")
    second = AC.composition_analysis_source_hashes(repo_root=tmp_path)
    assert second["files"]["guard_research/metrics.py"] != (
        first["files"]["guard_research/metrics.py"])
    assert second["aggregate_sha256"] != first["aggregate_sha256"]


def test_composition_attestation_records_runtime_and_all_analysis_parameters(monkeypatch):
    runtime = {"software_versions": {"python": "3.12.10"}, "requested_device": "cpu"}
    sources = {"files": {"analysis.py": "a" * 64}, "aggregate_sha256": "b" * 64}
    monkeypatch.setattr(AC.C, "runtime_environment", lambda _device: runtime)
    monkeypatch.setattr(AC, "composition_analysis_source_hashes", lambda: sources)
    parameters = AC.composition_parameter_record(4000, 20260712, 0.05)
    got = AC.composition_analysis_attestation(parameters)
    assert got["analysis_runtime_environment"] == runtime
    assert got["analysis_runtime_sha256"] == AC.C.canonical_obj_sha256(runtime)
    assert got["analysis_source_hashes"] == sources
    assert got["analysis_parameters"] == {
        "reps": 4000,
        "rng_seed": 20260712,
        "shuffle_rng_seed": 20260714,
        "target_fpr": 0.05,
        "primary_combiner": "calibrated_avg",
        "status": "fixed_prototype_constants_not_paper_a_lock",
    }


def test_canonical_prototype_parameters_are_fixed_and_overrides_are_nonfinal_only():
    canonical = AC.composition_parameter_record(
        AC.PROTOTYPE_REPS, AC.PROTOTYPE_RNG_SEED, AC.PROTOTYPE_TARGET_FPR)
    assert canonical["primary_combiner"] == AC.PROTOTYPE_PRIMARY_COMBINER
    assert canonical["status"] == "fixed_prototype_constants_not_paper_a_lock"
    with pytest.raises(AC.C.ArtifactContractError, match="fixed prototype constants"):
        AC.composition_parameter_record(
            AC.PROTOTYPE_REPS + 1, AC.PROTOTYPE_RNG_SEED,
            AC.PROTOTYPE_TARGET_FPR)
    diagnostic = AC.composition_parameter_record(
        10, 7, 0.1, nonfinal=True)
    assert diagnostic["status"] == "nonfinal_override_not_lock_bound"


def test_full_and_release_scientific_outputs_are_byte_identical_with_separate_metadata(
        monkeypatch, tmp_path):
    parameters = AC.composition_parameter_record(
        AC.PROTOTYPE_REPS, AC.PROTOTYPE_RNG_SEED, AC.PROTOTYPE_TARGET_FPR)
    result = {
        "analysis": "composition_v2",
        "scores_sha256": "a" * 64,
        "analysis_parameters": parameters,
        "point_estimates": {"same": True},
    }
    lock = {"lock_sha256": "b" * 64}
    full_verification = {"legacy": False, "mode_specific": "full"}
    release_verification = {"legacy": False, "mode_specific": "release"}
    monkeypatch.setattr(
        AC, "render_md", lambda value: json.dumps(value, sort_keys=True))
    monkeypatch.setattr(
        AC, "composition_analysis_attestation",
        lambda record: {"analysis_parameters": dict(record), "runtime": "fixture"})

    full = tmp_path / "full"
    release = tmp_path / "release"
    _, full_meta = AC.write_composition_artifacts(
        result, full, lock=lock, input_verification=full_verification,
        parameter_record=parameters)
    _, release_meta = AC.write_composition_artifacts(
        result, release, lock=lock, input_verification=release_verification,
        parameter_record=parameters, release_cache=True)

    assert (full / "composition.json").read_bytes() == (
        release / "composition.json").read_bytes()
    assert (full / "composition.md").read_bytes() == (
        release / "composition.md").read_bytes()
    assert full_meta["execution_mode"] == "full_artifact"
    assert release_meta["execution_mode"] == "release_cache"
    assert full_meta["input_verification"] != release_meta["input_verification"]
    assert full_meta["outputs"] == release_meta["outputs"]
    assert "input_verification" not in result


# --------------------------------------------------------------------------- combiners
def _entry(seed_ids=(42, 43)):
    base = {"cal": np.array([0.2, 0.8, 0.4, 0.6]),
            "raw": np.array([0.1, 0.7, 0.3, 0.5]),
            "logit": np.array([-1.0, 2.0, 0.0, 1.0])}
    sft = {s: {"cal": np.array([0.9, 0.9, 0.1, 0.1]),
               "raw": np.array([0.8, 0.85, 0.2, 0.15]),
               "logit": np.array([3.0, 3.0, -3.0, -3.0])} for s in seed_ids}
    return {"gold": np.array([0, 1, 0, 1]), "fam": ["a", "b", "c", "d"],
            "base": base, "sft": sft}


def test_combiner_math():
    e = _entry()
    b, s = e["base"], e["sft"][42]
    assert np.allclose(AC.combiner_score(e, 42, "base"), b["cal"])
    assert np.allclose(AC.combiner_score(e, 42, "sft"), s["cal"])
    assert np.allclose(AC.combiner_score(e, 42, "calibrated_avg"), 0.5 * (b["cal"] + s["cal"]))
    assert np.allclose(AC.combiner_score(e, 42, "logit_avg"), 0.5 * (b["logit"] + s["logit"]))
    assert np.allclose(AC.combiner_score(e, 42, "max_cal"), np.maximum(b["cal"], s["cal"]))
    # convex:w = w*sft + (1-w)*base
    assert np.allclose(AC.combiner_score(e, 42, "convex:0.25"), 0.25 * s["cal"] + 0.75 * b["cal"])


# --------------------------------------------------------------------------- macro
def test_macro_is_mean_of_per_seed_ap():
    e = _entry()
    data = {"m": {"id_test": {"toxicchat": e}}}
    # both seeds identical here -> macro == single-seed AP of the calibrated_avg score
    got = AC.macro(data, "m", "id_test", ["toxicchat"], [42, 43], "calibrated_avg")
    exp = AC.wap(0.5 * (e["base"]["cal"] + e["sft"][42]["cal"]), e["gold"])
    assert got == pytest.approx(exp)


def test_macro_is_not_ap_of_seed_mean_scores():
    e = _entry()
    e["base"]["cal"] = np.full(4, 0.5)
    e["sft"][42]["cal"] = np.array([0.1, 0.9, 0.2, 0.8])
    e["sft"][43]["cal"] = np.array([0.9, 0.8, 0.7, 0.6])
    data = {"m": {"id_test": {"toxicchat": e}}}

    got = AC.macro(
        data, "m", "id_test", ["toxicchat"], [42, 43], "calibrated_avg")
    per_seed = [
        AC.wap(AC.combiner_score(e, seed, "calibrated_avg"), e["gold"])
        for seed in (42, 43)
    ]
    ap_of_mean_scores = AC.wap(
        np.mean([
            AC.combiner_score(e, seed, "calibrated_avg")
            for seed in (42, 43)
        ], axis=0),
        e["gold"],
    )
    assert got == pytest.approx(float(np.mean(per_seed)))
    assert got != pytest.approx(ap_of_mean_scores)


def test_macro_skips_single_class_source():
    e = _entry()
    bad = _entry(); bad["gold"] = np.array([1, 1, 1, 1])  # single-class -> nan, skipped
    data = {"m": {"id_test": {"toxicchat": e, "prompt_injections": bad}}}
    got = AC.macro(data, "m", "id_test", ["toxicchat", "prompt_injections"], [42], "sft")
    exp = AC.wap(e["sft"][42]["cal"], e["gold"])  # only the valid source counts
    assert got == pytest.approx(exp)


def test_convex_weight_selection_uses_calibration_only(monkeypatch):
    calls = []

    def fake_panel(_data, split, _sources, _seeds, name, pit=None):
        del pit
        calls.append(split)
        weight = float(name.split(":")[1])
        return -(weight - 0.35) ** 2

    monkeypatch.setattr(AC, "panel", fake_panel)
    assert AC.select_convex_w({}, [42, 43]) == pytest.approx(0.35)
    assert calls and set(calls) == {AC.CAL_SPLIT}


def test_strict_scoring_manifests_are_rehashed(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    split_records = {}
    for filename in AC.paper_a_analysis.SCORING_SPLIT_FILES.values():
        path = manifests / filename
        path.write_text(json.dumps({"sample_id": filename}) + "\n")
        split_records[filename] = {"sha256": AC.C.sha256_file(path), "rows": 1}
    lock = {
        "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
        "artifact_paths": {"manifests": str(manifests)},
        "manifests": {"splits": split_records},
    }

    report = AC.verify_locked_scoring_manifests(lock)
    assert set(report) == set(AC.paper_a_analysis.SCORING_SPLIT_FILES.values())

    first = manifests / next(iter(AC.paper_a_analysis.SCORING_SPLIT_FILES.values()))
    first.write_text(first.read_text() + json.dumps({"sample_id": "tampered"}) + "\n")
    with pytest.raises(AC.C.ArtifactContractError, match="hash mismatch"):
        AC.verify_locked_scoring_manifests(lock)


def test_release_cache_uses_public_identity_and_explicit_reduced_artifact_validation(
        monkeypatch, tmp_path):
    lock = {
        "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
        "execution_sources": {"aggregate_sha256": "a" * 64},
    }
    release_report = {
        "release_cache_only": True,
        "release_contract": {
            "release_sha256": "6" * 64,
            "release_file_sha256": "7" * 64,
            "anchor_path": "configs/paper_a_sft_v2_release_anchor.json",
        },
        "public_release": {
            "sha256": "b" * 64,
            "splits": {"calibration": {"sha256": "c" * 64, "rows": 1}},
        },
        "execution_sources": {
            "aggregate_sha256": "a" * 64,
            "original_paper_a_execution_source_verification":
                "separate_immutable_source_bundle",
        },
    }
    verified_score = {
        "metadata": {"kind": "strict-score-metadata"},
        "scores_sha256": "d" * 64,
        "metadata_sha256": "e" * 64,
        "metadata_filename": "metadata.json",
        "bound": True,
        "legacy": False,
    }
    df = object()
    public_rows = [{"sample_id": "public-row"}]
    calls = {"runtime": []}

    monkeypatch.setattr(AC.C, "load_lock", lambda *args, **kwargs: lock)
    monkeypatch.setattr(AC.C, "verify_release_cache_lock", lambda value: release_report)
    monkeypatch.setattr(
        AC.paper_a_analysis, "validate_analysis_runtime",
        lambda value, **kwargs: calls["runtime"].append((value, kwargs)))
    monkeypatch.setattr(
        AC.C, "verify_score_artifact", lambda *args, **kwargs: verified_score)
    monkeypatch.setattr(AC, "load", lambda _path, **_kwargs: df)
    monkeypatch.setattr(
        AC.paper_a_analysis, "load_public_scoring_manifest_rows",
        lambda value: public_rows)
    monkeypatch.setattr(
        AC.paper_a_analysis, "load_locked_scoring_manifest_rows",
        lambda _value: pytest.fail("release-cache read raw manifest rows"))
    monkeypatch.setattr(
        AC, "verify_locked_scoring_manifests",
        lambda _value: pytest.fail("release-cache rehashed raw manifests"))

    def validate(value, lock_value, metadata, *, manifest_rows, release_cache):
        calls["matrix"] = (value, lock_value, metadata, manifest_rows, release_cache)
        return {"seeds": [42, 43, 44, 45, 46]}

    monkeypatch.setattr(AC.paper_a_analysis, "validate_score_artifacts", validate)
    monkeypatch.setattr(
        AC.C, "execution_source_hashes",
        lambda **kwargs: {
            "files": {name: "f" * 64 for name in kwargs["required_files"]},
            "aggregate_sha256": "1" * 64,
        })

    got_df, got_lock, report = AC.load_verified(
        tmp_path / "scores.parquet", tmp_path / "LOCK.json", release_cache=True)
    assert got_df is df and got_lock is lock
    assert calls["runtime"] == [(
        lock, {"nonfinal": False, "release_cache": True})]
    assert calls["matrix"][-2:] == (public_rows, True)
    assert report["release_cache"] is True
    assert report["scoring_manifests"] == release_report["public_release"]["splits"]
    release = report["release_cache_verification"]
    assert release["score_and_metadata_hashes_reverified"] is True
    assert release["release_contract"] == release_report["release_contract"]
    assert set(release["current_analysis_source_hashes"]["files"]) == set(
        AC.COMPOSITION_ANALYSIS_SOURCE_FILES)
    assert release["raw_manifest_files_locally_reverified"] is False
    assert release["run_metadata_and_adapter_bytes_locally_reverified"] is False

    monkeypatch.setattr(
        AC, "verify_locked_scoring_manifests",
        lambda _value: (_ for _ in ()).throw(
            AC.C.ArtifactContractError("locked scoring manifest is missing")))
    with pytest.raises(AC.C.ArtifactContractError, match="manifest is missing"):
        AC.load_verified(tmp_path / "scores.parquet", tmp_path / "LOCK.json")
    assert calls["runtime"][-1] == (
        lock, {"nonfinal": False, "release_cache": False})


def test_release_cache_rejects_legacy_nonfinal_and_legacy_opt_in(monkeypatch, tmp_path):
    lock_path = tmp_path / "LOCK.json"
    scores_path = tmp_path / "scores.parquet"
    with pytest.raises(AC.C.ArtifactContractError, match="cannot be combined"):
        AC.load_verified(
            scores_path, lock_path, allow_legacy=True, release_cache=True)

    monkeypatch.setattr(
        AC.C, "load_lock", lambda *args, **kwargs: {
            "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
            "finalization_status": "development_unverified",
        })
    with pytest.raises(AC.C.ArtifactContractError, match="strict final v2"):
        AC.load_verified(scores_path, lock_path, release_cache=True)

    monkeypatch.setattr(
        AC.C, "load_lock", lambda *args, **kwargs: {"lock_contract_version": 1})
    with pytest.raises(AC.C.ArtifactContractError, match="strict final v2"):
        AC.load_verified(scores_path, lock_path, release_cache=True)


def test_strict_composition_runtime_drift_fails_before_score_loading(monkeypatch, tmp_path):
    locked = {key: f"locked-{key}" for key in AC.C.PROTOCOL_SOFTWARE_KEYS}
    runtime = dict(locked)
    runtime["numpy"] = "drifted"
    lock = {
        "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
        "software_versions": locked,
    }
    monkeypatch.setattr(AC.C, "load_lock", lambda *args, **kwargs: lock)
    monkeypatch.setattr(AC.C, "software_versions", lambda: runtime)
    monkeypatch.setattr(
        AC.C, "verify_score_artifact",
        lambda *args, **kwargs: pytest.fail("runtime drift reached score loading"))
    with pytest.raises(AC.paper_a_analysis.ScoreValidationError, match="numpy_mismatch"):
        AC.load_verified(tmp_path / "scores.parquet", tmp_path / "LOCK.json")


def test_release_cache_propagates_public_and_score_tampering(monkeypatch, tmp_path):
    lock = {
        "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
    }
    monkeypatch.setattr(AC.C, "load_lock", lambda *args, **kwargs: lock)
    monkeypatch.setattr(
        AC.C, "verify_release_cache_lock",
        lambda _lock: (_ for _ in ()).throw(
            AC.C.ArtifactContractError("public split commitment mismatch")))
    with pytest.raises(AC.C.ArtifactContractError, match="public split commitment"):
        AC.load_verified(
            tmp_path / "scores.parquet", tmp_path / "LOCK.json", release_cache=True)

    release_report = {
        "release_contract": {
            "release_sha256": "6" * 64,
            "release_file_sha256": "7" * 64,
            "anchor_path": "configs/paper_a_sft_v2_release_anchor.json",
        },
        "public_release": {"sha256": "a" * 64, "splits": {}},
        "execution_sources": {
            "aggregate_sha256": "b" * 64,
            "original_paper_a_execution_source_verification":
                "separate_immutable_source_bundle",
        },
    }
    monkeypatch.setattr(
        AC.C, "verify_release_cache_lock", lambda _lock: release_report)
    monkeypatch.setattr(
        AC.paper_a_analysis, "validate_analysis_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        AC.C, "verify_score_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AC.C.ArtifactContractError("combined score hash mismatch")))
    with pytest.raises(AC.C.ArtifactContractError, match="score hash mismatch"):
        AC.load_verified(
            tmp_path / "scores.parquet", tmp_path / "LOCK.json", release_cache=True)


def test_release_cache_rejects_public_identity_mismatch(monkeypatch, tmp_path):
    lock = {
        "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
    }
    monkeypatch.setattr(AC.C, "load_lock", lambda *args, **kwargs: lock)
    monkeypatch.setattr(AC.C, "verify_release_cache_lock", lambda _lock: {
        "release_contract": {
            "release_sha256": "6" * 64,
            "release_file_sha256": "7" * 64,
            "anchor_path": "configs/paper_a_sft_v2_release_anchor.json",
        },
        "public_release": {"sha256": "a" * 64, "splits": {}},
        "execution_sources": {
            "aggregate_sha256": "b" * 64,
            "original_paper_a_execution_source_verification":
                "separate_immutable_source_bundle",
        },
    })
    monkeypatch.setattr(
        AC.paper_a_analysis, "validate_analysis_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(AC.C, "verify_score_artifact", lambda *args, **kwargs: {
        "metadata": {}, "scores_sha256": "c" * 64,
        "metadata_sha256": "d" * 64, "metadata_filename": "metadata.json",
        "bound": True, "legacy": False,
    })
    monkeypatch.setattr(AC, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        AC.paper_a_analysis, "load_public_scoring_manifest_rows",
        lambda _lock: [{"sample_id": "public"}])
    monkeypatch.setattr(
        AC.paper_a_analysis, "validate_score_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AC.paper_a_analysis.ScoreValidationError("manifest identity differs")))
    with pytest.raises(
            AC.paper_a_analysis.ScoreValidationError, match="manifest identity differs"):
        AC.load_verified(
            tmp_path / "scores.parquet", tmp_path / "LOCK.json", release_cache=True)

def test_explicit_legacy_load_path_remains_separate(monkeypatch, tmp_path):
    lock = {"lock_contract_version": 1, "lock_sha256": "a" * 64}
    calls = {}

    def load_lock(*args, **kwargs):
        calls["load_lock"] = kwargs
        return lock

    def verify_score(*args, **kwargs):
        calls["verify_score"] = kwargs
        return {
            "metadata": {}, "scores_sha256": "b" * 64,
            "metadata_sha256": "c" * 64, "metadata_filename": "metadata.json",
            "bound": True, "legacy": True,
        }

    monkeypatch.setattr(AC.C, "load_lock", load_lock)
    monkeypatch.setattr(AC.C, "verify_score_artifact", verify_score)
    monkeypatch.setattr(AC, "load", lambda _path, **_kwargs: object())
    monkeypatch.setattr(
        AC.paper_a_analysis, "validate_score_artifacts",
        lambda *args, **kwargs: {"seeds": [42, 43, 44, 45, 46]})
    monkeypatch.setattr(
        AC.C, "verify_release_cache_lock",
        lambda _lock: pytest.fail("legacy path entered release verification"))

    _, _, report = AC.load_verified(
        tmp_path / "scores.parquet", tmp_path / "LOCK.json", allow_legacy=True)
    assert calls["load_lock"]["allow_legacy"] is True
    assert calls["verify_score"]["allow_legacy"] is True
    assert report["legacy"] is True
    assert report["release_cache"] is False
    assert report["paper_a_source_tree_verification"] == "unrecoverable_legacy_source"
    assert "release_cache_verification" not in report


def test_composition_make_targets_are_v2_primary_and_output_disjoint():
    def dry_run(target):
        return subprocess.run(
            ["make", "-n", target], cwd=_ROOT, check=True,
            text=True, capture_output=True).stdout

    release = dry_run("composition")
    full = dry_run("composition-full")
    legacy = dry_run("composition-legacy")
    assert "--release-cache" in release
    assert "--lock artifacts/paper_a_sft_v2/LOCK.json" in release
    assert "--out artifacts/paper_a_sft_v2/analysis/composition" in release
    assert "--release-cache" not in full and "--allow-legacy-lock" not in full
    assert "--out artifacts/paper_a_sft_v2/analysis/composition-full" in full
    assert "--allow-legacy-lock" in legacy
    assert "--out artifacts/paper_a_sft/analysis/composition" in legacy


def test_composition_paths_are_lock_authoritative_and_mode_disjoint(tmp_path):
    v2_root = tmp_path / "paper_a_sft_v2"
    v2 = {
        "lock_contract_version": AC.C.LOCK_CONTRACT_VERSION,
        "artifact_paths": AC.C.artifact_paths_for_root(v2_root),
    }
    scores = v2_root / "scores/scores.parquet"
    release_out = v2_root / "analysis/composition"
    full_out = v2_root / "analysis/composition-full"
    AC.validate_composition_paths(v2, scores, release_out, release_cache=True)
    AC.validate_composition_paths(v2, scores, full_out, release_cache=False)
    with pytest.raises(AC.C.ArtifactContractError, match="mode-authoritative"):
        AC.validate_composition_paths(v2, scores, release_out, release_cache=False)
    with pytest.raises(AC.C.ArtifactContractError, match="lock-authoritative"):
        AC.validate_composition_paths(
            v2, tmp_path / "copied-scores.parquet", release_out, release_cache=True)
    diagnostic = tmp_path / "diagnostic-composition"
    AC.validate_composition_paths(
        v2, scores, diagnostic, release_cache=True, nonfinal=True)
    with pytest.raises(AC.C.ArtifactContractError, match="outside canonical"):
        AC.validate_composition_paths(
            v2, scores, release_out, release_cache=True, nonfinal=True)

    legacy_root = tmp_path / "paper_a_sft"
    legacy = {
        "lock_contract_version": 1,
        "artifact_paths": AC.C.artifact_paths_for_root(legacy_root),
    }
    AC.validate_composition_paths(
        legacy, legacy_root / "scores/scores.parquet",
        legacy_root / "analysis/composition")


def test_bootstrap_rejects_unimplemented_combiner(tiny_world):
    with pytest.raises(ValueError, match="only the primary calibrated_avg"):
        AC.bootstrap(tiny_world, [42, 43], reps=5, rng_seed=1, name="raw_avg")


# --------------------------------------------------------------------------- end-to-end determinism
@pytest.fixture()
def tiny_world(monkeypatch):
    """A 2-model, 1-source-per-regime, 2-seed synthetic world with both gold classes."""
    monkeypatch.setattr(AC, "MODELS", ["m1", "m2"])
    monkeypatch.setattr(AC, "REP", ["toxicchat"])
    monkeypatch.setattr(AC, "TR", ["jailbreakbench"])
    monkeypatch.setattr(AC, "REGIMES", {"represented": ("id_test", ["toxicchat"]),
                                        "transfer": ("transfer_test", ["jailbreakbench"])})
    rng = np.random.default_rng(7)
    data = {}
    for mk in ("m1", "m2"):
        data[mk] = {}
        for split, src in (("id_test", "toxicchat"), ("transfer_test", "jailbreakbench")):
            n = 40
            gold = np.array([0, 1] * (n // 2))
            fam = [f"{split}_{i // 2}" for i in range(n)]  # 2 rows per family
            def sig(strength):
                return np.clip(0.5 + strength * (gold - 0.5) + 0.15 * rng.standard_normal(n), 0, 1)
            base = {"cal": sig(0.4), "raw": sig(0.4), "logit": sig(0.4) * 4 - 2}
            sft = {s: {"cal": sig(0.6), "raw": sig(0.6), "logit": sig(0.6) * 4 - 2} for s in (42, 43)}
            data[mk][split] = {src: {"gold": gold, "fam": fam, "base": base, "sft": sft}}
        data[mk]["calibration"] = {
            "toxicchat": data[mk]["id_test"]["toxicchat"]
        }
    return data


def test_bootstrap_is_deterministic(tiny_world):
    a = AC.bootstrap({k: {s: {src: dict(e) for src, e in d.items()} for s, d in v.items()} for k, v in tiny_world.items()},
                     [42, 43], reps=100, rng_seed=123)
    b = AC.bootstrap({k: {s: {src: dict(e) for src, e in d.items()} for s, d in v.items()} for k, v in tiny_world.items()},
                     [42, 43], reps=100, rng_seed=123)
    for regime in ("represented", "transfer"):
        assert a[regime]["ens_minus_sft"]["panel"]["ci95"] == b[regime]["ens_minus_sft"]["panel"]["ci95"]
        assert a[regime]["ens_minus_base"]["panel"]["mean"] == b[regime]["ens_minus_base"]["panel"]["mean"]
        source = AC.REGIMES[regime][1][0]
        assert source in a[regime]["ens_minus_sft"]["per_benchmark"]
        assert len(a[regime]["ens_minus_sft"]["per_benchmark"][source]["ci95"]) == 2


def test_bootstrap_panel_mean_matches_benchmark_macro_mean(tiny_world):
    result = AC.bootstrap(tiny_world, [42, 43], reps=100, rng_seed=321)
    for regime in AC.REGIMES:
        for contrast in ("ens_minus_sft", "ens_minus_base"):
            benchmark_means = [
                value["mean"]
                for value in result[regime][contrast]["per_benchmark"].values()
            ]
            assert result[regime][contrast]["panel"]["mean"] == pytest.approx(
                float(np.mean(benchmark_means)))


def test_point_estimates_run_and_are_finite(tiny_world):
    pe = AC.point_estimates(tiny_world, [42, 43], ["base", "sft", "calibrated_avg"], pit=None)
    for name in ("base", "sft", "calibrated_avg"):
        for regime in ("represented", "transfer"):
            assert np.isfinite(pe[name][regime]["panel"])


def test_pit_maps_are_fit_per_adapter_seed(tiny_world):
    pit = AC.fit_pit(tiny_world, [42, 43])
    for mk in AC.MODELS:
        for src in AC.REP + AC.TR:
            assert (mk, src, 42) in pit
            assert (mk, src, 43) in pit


def test_shuffle_real_uses_same_mean_of_per_seed_estimand(tiny_world):
    got = AC.shuffle_null(tiny_world, [42, 43], rng_seed=9)
    for regime, (split, sources) in AC.REGIMES.items():
        expected = (
            AC.panel(tiny_world, split, sources, [42, 43], "calibrated_avg")
            - AC.panel(tiny_world, split, sources, [42, 43], "base")
        )
        assert got[regime]["real_ens_minus_base"] == pytest.approx(expected)


def test_operating_point_averages_per_seed_metrics(monkeypatch, tiny_world):
    monkeypatch.setattr(
        AC, "select_threshold",
        lambda scores, labels, target_fpr: {
            "status": "ok", "threshold": 0.5, "target_fpr": target_fpr,
        },
    )
    got = AC.operating_point(tiny_world, [42, 43], target_fpr=0.05)
    assert got["base"]["represented"]["n_seed_units_per_model"] == 1
    assert got["sft"]["represented"]["n_seed_units_per_model"] == 2

    expected_tprs = []
    for mk in AC.MODELS:
        entry = tiny_world[mk]["id_test"]["toxicchat"]
        pos = entry["gold"] == 1
        for seed in (42, 43):
            expected_tprs.append(float((entry["sft"][seed]["cal"][pos] >= 0.5).mean()))
    assert got["sft"]["represented"]["macro_tpr"] == pytest.approx(
        float(np.mean(expected_tprs)))
