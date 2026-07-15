from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


PAPER_B = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_B.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = _load_module("paper_b_renderer", PAPER_B / "code/build_pilot_artifacts.py")
protocol_validator = _load_module(
    "paper_b_protocol_validator", PAPER_B / "code/validate_protocol.py"
)


def test_current_pilot_evidence_validates_and_renders_observed_contrasts():
    result, _metadata = renderer.load_and_validate(
        REPO_ROOT / "artifacts/paper_a_sft_v2/analysis/composition/composition.json",
        REPO_ROOT
        / "artifacts/paper_a_sft_v2/analysis/composition/composition_metadata.json",
    )
    outputs = renderer.render_all(result)

    assert set(outputs) == set(renderer.OUTPUT_NAMES)
    assert r"\newcommand{\PilotTransferDeltaSFT}{+0.076}" in outputs["pilot_macros.tex"]
    assert (
        r"\newcommand{\PilotTransferDeltaSFTBootstrapMean}{+0.075}"
        in outputs["pilot_macros.tex"]
    )
    assert "Qwen3-4B" in outputs["pilot_per_model_table.tex"]
    manifest = json.loads(outputs["MANIFEST.json"])
    assert manifest["evidence"]["prospective_confirmatory"] is False
    assert manifest["evidence"]["composition_sha256"] == renderer.EXPECTED_COMPOSITION_SHA256


def test_generated_pilot_inputs_are_current():
    result, _metadata = renderer.load_and_validate(
        renderer.DEFAULT_COMPOSITION, renderer.DEFAULT_METADATA
    )
    renderer.write_or_check(renderer.render_all(result), renderer.DEFAULT_OUTPUT, check=True)


def test_publication_anchor_rejects_changed_evidence(tmp_path: Path):
    changed = json.loads(renderer.DEFAULT_COMPOSITION.read_text(encoding="utf-8"))
    changed["prospective_confirmatory"] = True
    changed_path = tmp_path / "composition.json"
    changed_path.write_text(json.dumps(changed, indent=2, sort_keys=True), encoding="utf-8")
    metadata = json.loads(renderer.DEFAULT_METADATA.read_text(encoding="utf-8"))
    metadata_path = tmp_path / "composition_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(renderer.EvidenceError, match="differs from the reviewed"):
        renderer.load_and_validate(changed_path, metadata_path)


def _draft_protocol() -> dict:
    return json.loads(
        (PAPER_B / "config/prospective_protocol.template.json").read_text(encoding="utf-8")
    )


def test_protocol_template_is_valid_only_as_a_draft():
    warnings = protocol_validator.validate_protocol(_draft_protocol())
    assert any("draft only" in warning for warning in warnings)
    with pytest.raises(protocol_validator.ProtocolError, match="requires a locked protocol"):
        protocol_validator.validate_protocol(_draft_protocol(), require_locked=True)


def test_protocol_rejects_missing_equal_compute_control():
    protocol = copy.deepcopy(_draft_protocol())
    protocol["baselines"] = [
        item for item in protocol["baselines"] if item["id"] != "sft_sft_output"
    ]
    with pytest.raises(protocol_validator.ProtocolError, match="sft_sft_output"):
        protocol_validator.validate_protocol(protocol)


def test_protocol_rejects_same_cohort_competence_design():
    protocol = copy.deepcopy(_draft_protocol())
    protocol["competence_hypothesis"]["cohorts_disjoint"] = False
    with pytest.raises(protocol_validator.ProtocolError, match="must be disjoint"):
        protocol_validator.validate_protocol(protocol)
