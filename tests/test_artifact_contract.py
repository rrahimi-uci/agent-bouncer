"""Fail-closed provenance and artifact-contract tests for Paper A."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import paper_a_common as C  # noqa: E402
import lock_paper_a_sft as lock_script  # noqa: E402
import run_paper_a_sft as run_script  # noqa: E402
import eval_paper_a_sft as eval_script  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rehash(lock: dict) -> None:
    lock["lock_sha256"] = C.canonical_obj_sha256(
        {key: value for key, value in lock.items() if key != "lock_sha256"})


def _strict_fixture(tmp_path: Path) -> tuple[dict, Path]:
    artifact_paths = C.artifact_paths_for_root("artifacts/contract-test-v2")
    models = {}
    probes = {}
    for model_key in C.MODEL_KEYS:
        models[model_key] = {
            "model_id": f"fixture/{model_key}",
            "model_revision": "a" * 40,
            "tokenizer_revision": "b" * 40,
            "dtype": "bfloat16",
            "attn_implementation": None,
            "trust_remote_code": False,
        }
        probes[model_key] = {
            "status": "ok",
            "safe_token_id": 1,
            "unsafe_token_id": 2,
            "prompt_template_sha256": "c" * 64,
        }

    for rel in C.EXECUTION_SOURCE_FILES:
        path = tmp_path / rel
        if rel == "configs/paper_a_sft.yaml":
            C.write_json(path, {
                "study_id": "contract-test",
                "analysis_mode": "precision_focused",
                "models": models,
            })
        else:
            _write(path, f"fixture source for {rel}\n")

    manifest_root = tmp_path / artifact_paths["manifests"]
    manifest_root.mkdir(parents=True)
    _write(manifest_root / "manifest.json", '{"study_id":"contract-test"}\n')
    split_records = {}
    for index, filename in enumerate(C.LOCK_MANIFEST_FILES):
        path = manifest_root / filename
        _write(path, json.dumps({"sample_id": f"sample-{index}"}) + "\n")
        split_records[filename] = {
            "path": f"{artifact_paths['manifests']}/{filename}",
            "sha256": C.sha256_file(path),
            "rows": 1,
        }

    public_root = tmp_path / artifact_paths["public_manifests"]
    public_root.mkdir(parents=True)
    public_files = {}
    raw_commitments = {}
    for index, stem in enumerate(C.MANIFEST_SPLITS):
        public_path = public_root / f"{stem}.jsonl"
        _write(public_path, json.dumps({"sample_id": f"sample-{index}"}) + "\n")
        public_files[stem] = {
            "path": str(public_path.relative_to(tmp_path)),
            "sha256": C.sha256_file(public_path),
            "n_rows": 1,
        }
        raw = split_records[f"{stem}.jsonl"]
        raw_commitments[stem] = {"sha256": raw["sha256"], "n_rows": raw["rows"]}
    supplemental_files = {}
    for name, filename in C.PUBLIC_SUPPLEMENTAL_FILES.items():
        supplemental_path = public_root / filename
        C.write_json(supplemental_path, {"schema_version": 1, "name": name})
        supplemental_files[name] = {
            "path": str(supplemental_path.relative_to(tmp_path)),
            "sha256": C.sha256_file(supplemental_path),
        }
    public_manifest_path = public_root / "manifest.json"
    C.write_json(public_manifest_path, {
        "source_contract": "pinned_hf_v2",
        "clean_rerun_compatible": True,
        "files": public_files,
        "supplemental_files": supplemental_files,
        "raw_artifact_commitment": {
            "manifest_sha256": C.sha256_file(manifest_root / "manifest.json"),
            "splits": raw_commitments,
        },
    })

    audit_path = tmp_path / artifact_paths["audit"] / "audit.json"
    C.write_json(audit_path, {
        "audit_contract_version": C.AUDIT_CONTRACT_VERSION,
        "all_hard_assertions_pass": True,
        "hard_assertions": {key: True for key in C.AUDIT_HARD_ASSERTION_KEYS},
        "manifest_index": {
            "observed_sha256": C.sha256_file(manifest_root / "manifest.json")},
        "file_integrity": {
            stem: {
                "observed_sha256": split_records[f"{stem}.jsonl"]["sha256"],
                "observed_rows": 1,
                "sha256_matches": True,
                "row_count_matches": True,
            }
            for stem in C.MANIFEST_SPLITS
        },
        "public_release_validation": {
            "ok": True, "manifest_sha256": C.sha256_file(public_manifest_path)},
    })

    config_path = tmp_path / "configs" / "paper_a_sft.yaml"
    sources = C.execution_source_hashes(repo_root=tmp_path)
    lock = {
        "lock_contract_version": C.LOCK_CONTRACT_VERSION,
        "finalization_status": "final",
        "development_issues": [],
        "schema_version": 1,
        "study_id": "contract-test",
        "config": {
            "path": "configs/paper_a_sft.yaml",
            "sha256": C.sha256_file(config_path),
            "obj_sha256": C.canonical_obj_sha256(C.load_config(config_path)),
        },
        "git": {
            "git_sha": "d" * 40,
            "execution_clean": True,
            "execution_dirty": False,
            "dirty_state_policy": "require_clean_execution_state",
        },
        "execution_sources": sources,
        "software_versions": {
            key: f"fixture-{key}" for key in C.PROTOCOL_SOFTWARE_KEYS},
        "models": models,
        "tokenizer_probe": probes,
        "prompt": {
            "prompt_spec_sha256": "e" * 64,
            "per_model_template_sha256": {
                model_key: probes[model_key]["prompt_template_sha256"]
                for model_key in C.MODEL_KEYS},
        },
        "recipe": dict(C.DEFAULT_RECIPE),
        "seeds": list(C.DEFAULT_SEEDS),
        "n_checkpoints": len(C.MODEL_KEYS),
        "n_seeds": len(C.DEFAULT_SEEDS),
        "n_final_cells": len(C.MODEL_KEYS) * len(C.DEFAULT_SEEDS),
        "data": dict(C.DEFAULT_DATA_CONTRACT),
        "operating_point": dict(C.DEFAULT_OPERATING_POINT),
        "resampling_rules": dict(C.DEFAULT_RESAMPLING_RULES),
        "analysis_mode": "precision_focused",
        "descriptive_criteria": {"formal_rejection_authorized": False},
        "claim_gates": None,
        "artifact_paths": artifact_paths,
        "power_report": None,
        "seed_count_decision": None,
        "manifests": {
            "dir": artifact_paths["manifests"],
            "index": {
                "path": f"{artifact_paths['manifests']}/manifest.json",
                "sha256": C.sha256_file(manifest_root / "manifest.json"),
            },
            "splits": split_records,
        },
        "train_manifest_sha256": split_records["train.jsonl"]["sha256"],
        "audit": {
            "path": f"{artifact_paths['audit']}/audit.json",
            "sha256": C.sha256_file(audit_path),
        },
        "public_release": {
            "manifest_path": str(public_manifest_path.relative_to(tmp_path)),
            "manifest_sha256": C.sha256_file(public_manifest_path),
        },
    }
    _rehash(lock)
    lock_path = tmp_path / artifact_paths["lock"]
    C.write_json(lock_path, lock)
    return lock, lock_path


def test_strict_lock_verifies_self_config_manifests_audit_and_sources(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    report = C.verify_lock(lock, verify_files=True, repo_root=tmp_path)
    assert report["legacy"] is False
    assert report["files_verified"] is True
    assert report["execution_sources"]["n_files"] == len(C.EXECUTION_SOURCE_FILES)
    assert set(report["manifests"]["splits"]) == set(C.LOCK_MANIFEST_FILES)


def test_legacy_lock_requires_explicit_opt_in(tmp_path):
    legacy = {"schema_version": 1, "study_id": "legacy"}
    _rehash(legacy)
    path = tmp_path / "legacy.json"
    C.write_json(path, legacy)
    with pytest.raises(C.ArtifactContractError, match="allow_legacy"):
        C.load_lock(path)
    assert C.load_lock(path, allow_legacy=True)["study_id"] == "legacy"


def test_lock_self_hash_tamper_fails_even_in_legacy_mode(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    lock["study_id"] = "tampered"
    with pytest.raises(C.ArtifactContractError, match="self-hash mismatch"):
        C.verify_lock(lock, allow_legacy=True)


def test_comment_only_config_byte_drift_fails_strict_lock(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    config = tmp_path / lock["config"]["path"]
    config.write_text(config.read_text() + "# comment-only drift\n", encoding="utf-8")
    assert C.canonical_obj_sha256(C.load_config(config)) == lock["config"]["obj_sha256"]
    with pytest.raises(C.ArtifactContractError, match="config byte hash mismatch"):
        C.verify_lock(lock, verify_files=True, repo_root=tmp_path)


@pytest.mark.parametrize("kind", ["manifest", "audit", "source"])
def test_bound_file_tampering_fails(tmp_path, kind):
    lock, _ = _strict_fixture(tmp_path)
    if kind == "manifest":
        target = tmp_path / lock["manifests"]["splits"]["train.jsonl"]["path"]
        target.write_text(target.read_text() + "{}\n", encoding="utf-8")
        match = "manifest hash mismatch"
    elif kind == "audit":
        target = tmp_path / lock["audit"]["path"]
        target.write_text('{"all_hard_assertions_pass":false,"hard_assertions":{"x":false}}\n')
        lock["audit"]["sha256"] = C.sha256_file(target)
        _rehash(lock)
        match = "does not report"
    else:
        target = tmp_path / "guard_research" / "metrics.py"
        target.write_text("tampered\n", encoding="utf-8")
        match = "execution source hash mismatch"
    with pytest.raises(C.ArtifactContractError, match=match):
        C.verify_lock(lock, verify_files=True, repo_root=tmp_path)


def test_stale_passing_audit_cannot_bind_changed_manifest(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    audit_path = tmp_path / lock["audit"]["path"]
    audit = C.read_json(audit_path)
    audit["file_integrity"]["train"]["observed_sha256"] = "f" * 64
    C.write_json(audit_path, audit)
    lock["audit"]["sha256"] = C.sha256_file(audit_path)
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="audit digest for train"):
        C.verify_lock(lock, verify_files=True, repo_root=tmp_path)


def test_incomplete_passing_audit_schema_is_rejected(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    audit_path = tmp_path / lock["audit"]["path"]
    audit = C.read_json(audit_path)
    audit["hard_assertions"] = {"fixture_integrity": True}
    C.write_json(audit_path, audit)
    lock["audit"]["sha256"] = C.sha256_file(audit_path)
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="assertion schema"):
        C.verify_lock(lock, verify_files=True, repo_root=tmp_path)


def test_manifest_directory_and_file_symlink_redirects_are_rejected(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    lock["manifests"]["dir"] = str(tmp_path.parent / "external-manifests")
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="manifest directory"):
        C.verify_lock(lock, repo_root=tmp_path)

    lock, _ = _strict_fixture(tmp_path / "symlink")
    repo = tmp_path / "symlink"
    train_path = repo / lock["manifests"]["splits"]["train.jsonl"]["path"]
    outside = tmp_path / "outside-manifest" / "train.jsonl"
    _write(outside, train_path.read_text(encoding="utf-8"))
    train_path.unlink()
    train_path.symlink_to(outside)
    with pytest.raises(C.ArtifactContractError, match="resolves outside"):
        C.verify_lock(lock, verify_files=True, repo_root=repo)


def test_public_release_drift_fails_locked_verification(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    public_manifest = tmp_path / lock["public_release"]["manifest_path"]
    public_split = public_manifest.parent / "train.jsonl"
    public_split.write_text(public_split.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(C.ArtifactContractError, match="public split commitment mismatch"):
        C.verify_lock(lock, verify_files=True, repo_root=tmp_path)


def test_final_structure_requires_successful_tokenizer_probes_and_disables_powered_mode(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    lock["tokenizer_probe"][C.MODEL_KEYS[0]]["status"] = "error"
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="Tokenizer probe|tokenizer probe"):
        C.verify_lock(lock)

    lock, _ = _strict_fixture(tmp_path / "powered")
    lock["analysis_mode"] = "powered_confirmatory"
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="disabled"):
        C.verify_lock(lock)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [("model_revision", "main", "40-hex"),
     ("tokenizer_revision", "latest", "40-hex"),
     ("dtype", "auto", "unsupported dtype")],
)
def test_strict_model_runtime_requires_commit_shas_and_supported_dtype(
        tmp_path, field, value, match):
    lock, _ = _strict_fixture(tmp_path)
    lock["models"][C.MODEL_KEYS[0]][field] = value
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match=match):
        C.verify_lock(lock)


def test_powered_lock_is_rejected_even_with_a_power_report(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    power = tmp_path / "design" / "power.json"
    _write(power, '{"seed_count_decision":"five-seeds"}\n')
    lock["analysis_mode"] = "powered_confirmatory"
    lock["power_report"] = {
        "path": "design/power.json", "sha256": C.sha256_file(power)}
    lock["seed_count_decision"] = "five-seeds"
    lock["claim_gates"] = {"gate_a": "fixture", "gate_b": "fixture"}
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="disabled"):
        C.verify_lock(lock, verify_files=True, repo_root=tmp_path)


def test_default_lock_creation_rejects_missing_inputs(tmp_path):
    config = tmp_path / "config.yaml"
    _write(config, "study_id: missing-inputs\nanalysis_mode: precision_focused\n")
    args = argparse.Namespace(
        config=str(config), manifest=None, manifests_dir=str(tmp_path / "missing"),
        audit=None, power=None, analysis_mode="precision_focused",
        development_override=False, probe_tokenizers=False,
        require_tokenizer_probe=False, require_clean=True,
    )
    with pytest.raises(C.ArtifactContractError, match="config must|manifest index"):
        lock_script.build_lock(args)

    args.development_override = True
    development_lock = lock_script.build_lock(args)
    assert development_lock["finalization_status"] == "development_unverified"
    assert development_lock["artifact_paths"]["root"] == C.DEFAULT_ARTIFACTS_V2["root"]
    assert development_lock["artifact_paths"]["root"] != C.DEFAULT_ARTIFACTS["root"]
    assert development_lock["development_issues"]
    with pytest.raises(C.ArtifactContractError, match="not final"):
        C.verify_lock(development_lock)
    C.verify_lock(development_lock, allow_development=True)


def test_development_lock_handles_public_manifest_without_audit(tmp_path):
    config = tmp_path / "config.yaml"
    artifact_root = tmp_path / "v2"
    _write(config, "study_id: missing-audit\nanalysis_mode: precision_focused\n")
    _write(artifact_root / "public_manifests" / "manifest.json", "{}\n")
    args = argparse.Namespace(
        config=str(config), manifest=None,
        manifests_dir=str(artifact_root / "manifests"),
        audit=None, power=None, analysis_mode="precision_focused",
        development_override=True, probe_tokenizers=False,
        require_tokenizer_probe=False, require_clean=True,
        artifact_root=str(artifact_root), out=str(artifact_root / "LOCK.json"),
    )

    lock = lock_script.build_lock(args)

    assert lock["finalization_status"] == "development_unverified"
    assert lock["public_release"]["manifest_path"].endswith(
        "public_manifests/manifest.json")
    assert "final lock requires a passing public-release audit" in lock["development_issues"]
    assert any("audit public-manifest digest differs" in issue
               for issue in lock["development_issues"])


def test_external_development_root_is_quarantined_until_explicitly_allowed(tmp_path):
    config = tmp_path / "config.yaml"
    _write(config, "study_id: external-development\nanalysis_mode: precision_focused\n")
    artifact_root = tmp_path / "external-v2"
    args = argparse.Namespace(
        config=str(config), manifest=None, manifests_dir=None,
        audit=None, power=None, analysis_mode="precision_focused",
        development_override=True, probe_tokenizers=False,
        require_tokenizer_probe=False, require_clean=True,
        artifact_root=str(artifact_root), out=str(artifact_root / "LOCK.json"),
    )
    lock = lock_script.build_lock(args)
    assert lock["finalization_status"] == "development_unverified"
    assert any("inside the repository" in issue for issue in lock["development_issues"])
    with pytest.raises(C.ArtifactContractError, match="not final"):
        C.verify_lock(lock)
    C.verify_lock(lock, allow_development=True)


def test_strict_v2_namespace_cannot_collide_with_historical_artifacts(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    assert C.artifact_paths(lock)["root"] != C.DEFAULT_ARTIFACTS["root"]
    lock["artifact_paths"] = dict(C.DEFAULT_ARTIFACTS)
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="historical v1 artifact root"):
        C.verify_lock(lock)


def test_strict_fixed_panel_seed_and_recipe_contract_cannot_drift(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    lock["seeds"] = [1, 2]
    lock["n_seeds"] = 2
    lock["n_final_cells"] = len(C.MODEL_KEYS) * 2
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="seeds 42--46"):
        C.verify_lock(lock, repo_root=tmp_path)

    lock, _ = _strict_fixture(tmp_path / "recipe")
    lock["recipe"]["max_steps"] = 1
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="frozen recipe"):
        C.verify_lock(lock, repo_root=tmp_path / "recipe")


def test_strict_v2_rejects_absolute_and_child_symlink_escapes(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    lock["artifact_paths"] = C.artifact_paths_for_root(
        tmp_path / C.DEFAULT_ARTIFACTS["root"])
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="historical v1 artifact root"):
        C.verify_lock(lock, repo_root=tmp_path)

    external_repo = tmp_path / "external-case"
    lock, _ = _strict_fixture(external_repo)
    lock["artifact_paths"] = C.artifact_paths_for_root(tmp_path / "outside-v2")
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="inside the repository"):
        C.verify_lock(lock, repo_root=external_repo)

    linked_repo = tmp_path / "root-symlink-case"
    lock, _ = _strict_fixture(linked_repo)
    outside_root = tmp_path / "outside-linked-v2"
    outside_root.mkdir()
    linked_root = linked_repo / "artifacts" / "linked-v2"
    linked_root.symlink_to(outside_root, target_is_directory=True)
    lock["artifact_paths"] = C.artifact_paths_for_root("artifacts/linked-v2")
    _rehash(lock)
    with pytest.raises(C.ArtifactContractError, match="inside the repository"):
        C.verify_lock(lock, repo_root=linked_repo)

    lock, _ = _strict_fixture(tmp_path / "child")
    root = tmp_path / "child" / "artifacts" / "contract-test-v2"
    legacy_scores = tmp_path / "child" / C.DEFAULT_ARTIFACTS["scores"]
    legacy_scores.mkdir(parents=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "scores").symlink_to(legacy_scores, target_is_directory=True)
    with pytest.raises(C.ArtifactContractError, match="child resolves outside"):
        C.verify_lock(lock, repo_root=tmp_path / "child")


def test_strict_eval_rejects_output_override_into_historical_namespace(tmp_path, monkeypatch):
    _lock, lock_path = _strict_fixture(tmp_path)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    legacy_out = tmp_path / C.DEFAULT_ARTIFACTS["scores"]
    rc = eval_script.main([
        "--lock", str(lock_path),
        "--out", str(legacy_out),
    ])
    assert rc == 2
    assert not legacy_out.exists()


def test_nonfinal_eval_and_train_cannot_write_canonical_namespaces(tmp_path, monkeypatch):
    lock, lock_path = _strict_fixture(tmp_path)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    canonical_scores = tmp_path / lock["artifact_paths"]["scores"]
    rc = eval_script.main([
        "--lock", str(lock_path), "--nonfinal", "--synthetic",
        "--out", str(canonical_scores),
    ])
    assert rc == 2
    assert not canonical_scores.exists()

    canonical_cell = tmp_path / lock["artifact_paths"]["runs"] / C.MODEL_KEYS[0] \
        / "sft" / f"seed_{C.DEFAULT_SEEDS[0]}"
    args = argparse.Namespace(
        lock=str(lock_path), allow_legacy_lock=False, manifest=None,
        max_steps=None, nonfinal=True, dry_run=True, out=str(canonical_cell),
        model_key=C.MODEL_KEYS[0], seed=C.DEFAULT_SEEDS[0], seeds=None,
        device=None, force=False,
    )
    assert run_script.cmd_train(args) == 2
    assert not canonical_cell.exists()


def test_final_training_rejects_max_step_override(tmp_path, monkeypatch):
    _lock, lock_path = _strict_fixture(tmp_path)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    args = argparse.Namespace(
        lock=str(lock_path), allow_legacy_lock=False, manifest=None,
        max_steps=1, nonfinal=False, dry_run=False, out=None,
        model_key=None, seed=None, seeds=None, device=None, force=False,
    )
    assert run_script.cmd_train(args) == 2


def test_dry_run_cannot_write_canonical_final_run_metadata(tmp_path, monkeypatch):
    _lock, lock_path = _strict_fixture(tmp_path)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    args = argparse.Namespace(
        lock=str(lock_path), allow_legacy_lock=False, manifest=None,
        max_steps=None, nonfinal=False, dry_run=True, out=None,
        model_key=None, seed=None, seeds=None, device=None, force=False,
    )
    assert run_script.cmd_train(args) == 2
    canonical_runs = tmp_path / "artifacts" / "contract-test-v2" / "runs"
    assert not canonical_runs.exists()


def test_model_panel_preserves_locked_runtime_fields(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    panel = C.lock_model_panel(lock)
    assert panel[C.MODEL_KEYS[0]]["dtype"] == "bfloat16"
    assert panel[C.MODEL_KEYS[0]]["trust_remote_code"] is False
    assert "attn_implementation" in panel[C.MODEL_KEYS[0]]


def _valid_run_meta(lock: dict, model_key: str, seed: int) -> dict:
    model = C.lock_model_panel(lock)[model_key]
    return {
        "status": "completed",
        "model_key": model_key,
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "model_runtime": {key: model.get(key) for key in (
            "model_id", "model_revision", "tokenizer_revision", "dtype",
            "attn_implementation", "trust_remote_code")},
        "condition": "sft",
        "run_kind": "final",
        "lock_contract_status": "final",
        "seed": seed,
        "training_seed": seed,
        "data_order_seed": lock["data"]["data_order_seed"],
        "train_manifest_sha256": lock["train_manifest_sha256"],
        "config_sha256": lock["config"]["sha256"],
        "config_obj_sha256": lock["config"]["obj_sha256"],
        "prompt_spec_sha256": lock["prompt"]["prompt_spec_sha256"],
        "prompt_template_sha256": lock[
            "prompt"]["per_model_template_sha256"][model_key],
        "lock_sha256": lock["lock_sha256"],
        "git_sha": lock["git"]["git_sha"],
        "execution_sources_sha256": lock["execution_sources"]["aggregate_sha256"],
        "software_versions": dict(lock["software_versions"]),
        "recipe": lock["recipe"],
        "global_steps": lock["recipe"]["max_steps"],
    }


def test_completed_adapter_is_rehashed_before_reuse(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    model_key, seed = C.MODEL_KEYS[0], C.DEFAULT_SEEDS[0]
    run_dir = tmp_path / "runs" / model_key / "sft" / f"seed_{seed}"
    adapter = run_dir / "adapter"
    lora = lock["recipe"]["lora"]
    _write(adapter / "adapter_config.json", json.dumps({
        "r": lora["r"], "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "target_modules": lora["target_modules"], "task_type": "CAUSAL_LM",
    }) + "\n")
    _write(adapter / "adapter_model.bin", "weights-v1")
    meta = _valid_run_meta(lock, model_key, seed)
    meta["adapter_sha256"] = C.sha256_dir(adapter)
    C.write_json(run_dir / "run_meta.json", meta)
    assert C.validate_run_artifact(lock, model_key, seed, run_dir)["valid"]
    meta["data_order_seed"] = 99
    C.write_json(run_dir / "run_meta.json", meta)
    result = C.validate_run_artifact(lock, model_key, seed, run_dir)
    assert not result["valid"]
    assert "data_order_seed_mismatch" in result["issues"]
    meta["data_order_seed"] = lock["data"]["data_order_seed"]
    meta["software_versions"]["torch"] = "drifted"
    C.write_json(run_dir / "run_meta.json", meta)
    result = C.validate_run_artifact(lock, model_key, seed, run_dir)
    assert not result["valid"]
    assert "software_versions_torch_mismatch" in result["issues"]
    meta["software_versions"] = dict(lock["software_versions"])
    C.write_json(run_dir / "run_meta.json", meta)
    _write(adapter / "adapter_model.bin", "weights-tampered")
    result = C.validate_run_artifact(lock, model_key, seed, run_dir)
    assert not result["valid"]
    assert "adapter_sha256_mismatch" in result["issues"]


def test_completed_adapter_with_wrong_serialized_lora_config_is_rejected(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    model_key, seed = C.MODEL_KEYS[0], C.DEFAULT_SEEDS[0]
    run_dir = tmp_path / "runs" / model_key / "sft" / f"seed_{seed}"
    adapter = run_dir / "adapter"
    lora = lock["recipe"]["lora"]
    _write(adapter / "adapter_config.json", json.dumps({
        "r": 8, "lora_alpha": lora["alpha"], "lora_dropout": lora["dropout"],
        "target_modules": lora["target_modules"], "task_type": "CAUSAL_LM",
    }) + "\n")
    _write(adapter / "adapter_model.bin", "weights-v1")
    meta = _valid_run_meta(lock, model_key, seed)
    meta["adapter_sha256"] = C.sha256_dir(adapter)
    C.write_json(run_dir / "run_meta.json", meta)
    result = C.validate_run_artifact(lock, model_key, seed, run_dir)
    assert not result["valid"]
    assert "adapter_config_r_mismatch" in result["issues"]


def test_combined_score_digest_is_bound_and_verified(tmp_path):
    lock, _ = _strict_fixture(tmp_path)
    scores = tmp_path / "scores.parquet"
    scores.write_bytes(b"fixture parquet bytes")
    metadata = tmp_path / "metadata.json"
    C.write_json(metadata, {
        "score_artifact_contract_version": 2,
        "finalization_status": "final",
        "lock_sha256": lock["lock_sha256"],
        "execution_sources_sha256": lock["execution_sources"]["aggregate_sha256"],
        "software_versions": dict(lock["software_versions"]),
        "scores_sha256": C.sha256_file(scores),
    })
    first = C.verify_score_artifact(scores, metadata, lock)
    assert first["bound"]
    assert first["metadata_sha256"] == C.sha256_file(metadata)
    metadata_obj = C.read_json(metadata)
    metadata_obj["attestation_note"] = "digest must change"
    C.write_json(metadata, metadata_obj)
    second = C.verify_score_artifact(scores, metadata, lock)
    assert second["metadata_sha256"] != first["metadata_sha256"]
    scores.write_bytes(b"tampered")
    with pytest.raises(C.ArtifactContractError, match="combined score hash mismatch"):
        C.verify_score_artifact(scores, metadata, lock)


def _tiny_score_frame():
    import math
    import pandas as pd

    rows = []
    for condition, seed, score, threshold in (
            ("base", -1, 0.0, 0.6), ("sft", 42, 2.0, 0.7)):
        calibrated = 1.0 / (1.0 + math.exp(-(score / 2.0)))
        rows.append({
            "sample_id": "sample-1", "content_sha256": "a" * 64,
            "source": "fixture", "split": "calibration", "gold": 1,
            "family_id": "b" * 64, "model_key": "fixture-model",
            "model_revision": "c" * 40, "condition": condition, "seed": seed,
            "adapter_sha256": None if condition == "base" else "d" * 64,
            "prompt_sha256": "e" * 64, "safe_token_id": 1, "unsafe_token_id": 2,
            "safe_logit": -score / 2.0, "unsafe_logit": score / 2.0,
            "score_raw": score, "probability_raw": 1.0 / (1.0 + math.exp(-score)),
            "probability_calibrated": calibrated,
            "threshold_id": f"fixture:{condition}",
            "prediction": int(calibrated >= threshold),
            "original_token_count": 10, "scored_token_count": 10,
            "truncated": False, "truncation_strategy": "none", "latency_ms": 1.0,
        })
    frame = pd.DataFrame(rows, columns=eval_script.SCORE_COLUMNS)
    bundle_meta = {
        "fixture-model:base": {
            "adapter_sha256": None,
            "calibration": {"status": "ok", "optim_success": True, "temperature": 2.0},
            "threshold": {"threshold_id": "fixture:base", "threshold_value": 0.6}},
        "fixture-model:sft:seed_42": {
            "adapter_sha256": "d" * 64,
            "calibration": {"status": "ok", "optim_success": True, "temperature": 2.0},
            "threshold": {"threshold_id": "fixture:sft", "threshold_value": 0.7}},
    }
    manifest_rows = [{"sample_id": "sample-1"}]
    return frame, bundle_meta, manifest_rows


def test_evaluator_recomputes_calibration_and_prediction_before_publish():
    frame, bundle_meta, manifest_rows = _tiny_score_frame()
    eval_script.validate_combined_scores(
        frame, manifest_rows, ["fixture-model"], [42], bundle_meta=bundle_meta)

    bad_probability = frame.copy()
    bad_probability.loc[1, "probability_calibrated"] = 0.01
    with pytest.raises(C.ArtifactContractError, match="probability_calibrated"):
        eval_script.validate_combined_scores(
            bad_probability, manifest_rows, ["fixture-model"], [42],
            bundle_meta=bundle_meta)

    bad_prediction = frame.copy()
    bad_prediction.loc[1, "prediction"] = 0
    with pytest.raises(C.ArtifactContractError, match="prediction is inconsistent"):
        eval_script.validate_combined_scores(
            bad_prediction, manifest_rows, ["fixture-model"], [42],
            bundle_meta=bundle_meta)


def test_cache_identity_includes_batch_size_and_producer_runtime():
    expected = {key: f"fixture-{key}" for key in C.CACHE_KEYS}
    expected["batch_size"] = 16
    cached = dict(expected)
    assert C.cache_is_valid(cached, expected) == (True, [])
    cached["batch_size"] = 8
    assert C.cache_is_valid(cached, expected) == (False, ["batch_size"])
    cached = dict(expected)
    cached["producer_runtime_sha256"] = "different"
    assert C.cache_is_valid(cached, expected) == (
        False, ["producer_runtime_sha256"])


def test_temperature_fit_rejects_unsuccessful_optimizer(monkeypatch):
    import numpy as np
    import scipy.optimize

    class FailedResult:
        success = False
        x = np.array([0.0])
        message = "synthetic failure"

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *args, **kwargs: FailedResult())
    result = eval_script.fit_temperature(
        np.array([-1.0, 1.0]), np.array([0, 1]))
    assert result["status"] == "optimization_failed"
    assert result["optim_success"] is False


class _WhitespaceTokenizer:
    def __init__(self):
        self.to_id = {}
        self.to_token = {}

    def _id(self, token):
        if token not in self.to_id:
            value = len(self.to_id) + 1
            self.to_id[token] = value
            self.to_token[value] = token
        return self.to_id[token]

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        return {"input_ids": [self._id(token) for token in str(text).split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.to_token[value] for value in ids)


def test_budgeted_prompt_truncates_user_before_rendering_and_preserves_wrapper():
    tokenizer = _WhitespaceTokenizer()
    system = C.prompt_identity()["system"]

    def render(_tok, user):
        return f"{system}\nUSER {user}\nASSISTANT verdict"

    long_user = " ".join(f"token-{index}" for index in range(300))
    fixed_tokens = len(tokenizer(render(tokenizer, ""))["input_ids"])
    prompt, stats = C.budgeted_prompt(
        tokenizer, render, long_user, max_length=fixed_tokens + 30)
    assert stats["truncated"] is True
    assert stats["truncation_strategy"] == C.TRUNCATION_STRATEGY
    assert stats["wrapper_preserved"] is True
    assert system in prompt
    assert "ASSISTANT verdict" in prompt
    assert len(tokenizer(prompt)["input_ids"]) <= fixed_tokens + 30


def test_runtime_environment_records_requested_provenance_keys():
    env = C.runtime_environment("cpu")
    versions = env["software_versions"]
    for key in ("platform", "os", "python", "torch", "cuda_runtime", "cudnn",
                "transformers", "peft", "trl", "accelerate"):
        assert key in versions
    assert env["requested_device"] == "cpu"
