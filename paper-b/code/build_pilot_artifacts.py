#!/usr/bin/env python3
"""Build Paper B's retrospective pilot tables from the frozen composition result.

This script is intentionally narrower than ``experiments/analyze_composition.py``.
The analyzer computes the statistics; this publication adapter verifies the exact
reviewed evidence object and renders deterministic LaTeX inputs.  It refuses to
silently promote the pilot to prospective or confirmatory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSITION = (
    REPO_ROOT
    / "artifacts/paper_a_sft_v2/analysis/composition/composition.json"
)
DEFAULT_METADATA = DEFAULT_COMPOSITION.with_name("composition_metadata.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "generated"

EXPECTED_LOCK_SHA256 = (
    "cabc8dee9b158773ce0be86f799ec3833c33c18787a2aa74d05ed1a261682c25"
)
EXPECTED_SCORES_SHA256 = (
    "b941ddbaea7057ab1f224c510687ec5748916f5eca6a78e1d1f429e0ede5a1c3"
)
EXPECTED_COMPOSITION_SHA256 = (
    "92c2cbc3ea71d5e6c72bf0e6f7eb0d3ef15f0e61f9fffaada885dade460e3ccc"
)
EXPECTED_ANALYSIS_SOURCE_SHA256 = (
    "8e2ca6f79d5115d3015114b0ab025a8636b70cc534c8e190aee68d619516244c"
)
EXPECTED_STATUS = "clean_v2_retrospective_estimation"
EXPECTED_PRIMARY = "calibrated_avg"
EXPECTED_MODELS = ("qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b")
DISPLAY_ORDER = ("smollm2_17b", "qwen25_15b", "smollm3_3b", "qwen3_4b")
DISPLAY_NAMES = {
    "qwen25_15b": "Qwen2.5-1.5B",
    "smollm2_17b": "SmolLM2-1.7B",
    "smollm3_3b": "SmolLM3-3B",
    "qwen3_4b": "Qwen3-4B",
}
EXPECTED_SEEDS = [42, 43, 44, 45, 46]
OUTPUT_NAMES = (
    "pilot_macros.tex",
    "pilot_summary_table.tex",
    "pilot_per_model_table.tex",
    "pilot_operating_point_table.tex",
    "MANIFEST.json",
)


class EvidenceError(ValueError):
    """Raised when a manuscript input violates the reviewed evidence contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path} must be an object")
    return value


def _at(root: dict[str, Any], *keys: str) -> Any:
    value: Any = root
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        value = _mapping(value, ".".join(walked[:-1]) or "root").get(key)
        _require(value is not None, f"missing required field: {'.'.join(walked)}")
    return value


def _number(value: Any, path: str, *, unit_interval: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{path} must be finite")
    if unit_interval:
        _require(0.0 <= result <= 1.0, f"{path} must lie in [0, 1]")
    return result


def _ci(record: dict[str, Any], path: str) -> tuple[float, float, float]:
    mean = _number(_at(record, "mean"), f"{path}.mean")
    interval = _at(record, "ci95")
    _require(
        isinstance(interval, list) and len(interval) == 2,
        f"{path}.ci95 must contain two values",
    )
    low = _number(interval[0], f"{path}.ci95[0]")
    high = _number(interval[1], f"{path}.ci95[1]")
    _require(low <= mean <= high, f"{path} mean must lie inside ci95")
    return mean, low, high


def load_and_validate(
    composition_path: Path, metadata_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fail closed on the exact reviewed clean-v2 pilot artifact."""

    _require(composition_path.is_file(), f"missing composition result: {composition_path}")
    _require(metadata_path.is_file(), f"missing composition metadata: {metadata_path}")
    try:
        result = json.loads(composition_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON evidence: {exc}") from exc
    result = _mapping(result, "composition")
    metadata = _mapping(metadata, "composition_metadata")

    observed_sha = sha256_file(composition_path)
    _require(
        observed_sha == EXPECTED_COMPOSITION_SHA256,
        "composition.json differs from the reviewed Paper B pilot; review and update "
        "the publication evidence contract before regenerating the manuscript",
    )
    _require(_at(result, "analysis") == "composition_v2", "unexpected analysis id")
    _require(_at(result, "analysis_mode") == "precision_focused", "unexpected analysis mode")
    _require(_at(result, "analysis_status") == EXPECTED_STATUS, "unexpected analysis status")
    _require(
        _at(result, "prospective_confirmatory") is False,
        "pilot evidence must remain non-confirmatory",
    )
    _require(_at(result, "legacy") is False, "legacy evidence is not permitted")
    _require(_at(result, "lock_contract_version") == 2, "v2 lock contract is required")
    _require(_at(result, "lock_sha256") == EXPECTED_LOCK_SHA256, "Paper A lock hash mismatch")
    _require(
        _at(result, "scores_sha256") == EXPECTED_SCORES_SHA256,
        "Paper A score hash mismatch",
    )
    parameters = _mapping(_at(result, "analysis_parameters"), "analysis_parameters")
    _require(
        _at(parameters, "primary_combiner") == EXPECTED_PRIMARY,
        "reviewed primary combiner must be calibrated_avg",
    )
    _require(
        _number(_at(parameters, "target_fpr"), "analysis_parameters.target_fpr") == 0.05,
        "reviewed operating-point target must remain 0.05",
    )
    _require(_at(parameters, "reps") == 4000, "reviewed bootstrap repetition count changed")
    _require(_at(parameters, "rng_seed") == 20260712, "reviewed bootstrap RNG seed changed")
    _require(
        _at(parameters, "shuffle_rng_seed") == 20260714,
        "reviewed shuffle RNG seed changed",
    )
    _require(
        _at(parameters, "status") == "fixed_prototype_constants_not_paper_a_lock",
        "prototype parameter status changed",
    )
    _require(_at(result, "seeds") == EXPECTED_SEEDS, "unexpected SFT seed panel")
    _require(
        _at(result, "combiner_order")
        == [
            "base",
            "sft",
            "calibrated_avg",
            "raw_avg",
            "logit_avg",
            "max_cal",
            "pit_avg",
            "convex_blind",
        ],
        "combiner order differs from the reviewed pilot",
    )
    _require(_at(result, "convex_selection_split") == "calibration", "convex selector leak")
    _require(
        abs(_number(_at(result, "convex_selected_w"), "convex_selected_w") - 0.95) < 1e-12,
        "reviewed convex weight changed",
    )

    _require(
        _at(metadata, "composition_metadata_contract_version") == 1,
        "unsupported composition metadata contract",
    )
    _require(_at(metadata, "analysis") == "composition_v2", "metadata analysis id mismatch")
    _require(_at(metadata, "execution_mode") == "release_cache", "release-cache evidence required")
    _require(_at(metadata, "nonfinal") is False, "nonfinal composition evidence is not allowed")
    _require(_at(metadata, "lock_sha256") == EXPECTED_LOCK_SHA256, "metadata lock mismatch")
    _require(
        _at(metadata, "scores_sha256") == EXPECTED_SCORES_SHA256,
        "metadata score hash mismatch",
    )
    _require(
        _at(metadata, "analysis_source_sha256") == EXPECTED_ANALYSIS_SOURCE_SHA256,
        "composition analyzer source differs from the reviewed pilot source",
    )
    _require(
        _at(metadata, "outputs", "composition.json") == observed_sha,
        "metadata does not bind the supplied composition.json",
    )

    points = _mapping(_at(result, "point_estimates"), "point_estimates")
    for guard in ("base", "sft", EXPECTED_PRIMARY, "logit_avg"):
        guard_record = _mapping(_at(points, guard), f"point_estimates.{guard}")
        for regime in ("represented", "transfer"):
            regime_record = _mapping(
                _at(guard_record, regime), f"point_estimates.{guard}.{regime}"
            )
            _number(
                _at(regime_record, "panel"),
                f"point_estimates.{guard}.{regime}.panel",
                unit_interval=True,
            )
            per_model = _mapping(
                _at(regime_record, "per_model"),
                f"point_estimates.{guard}.{regime}.per_model",
            )
            _require(
                set(per_model) == set(EXPECTED_MODELS),
                f"point_estimates.{guard}.{regime}.per_model has the wrong panel",
            )
            for model, value in per_model.items():
                _number(
                    value,
                    f"point_estimates.{guard}.{regime}.per_model.{model}",
                    unit_interval=True,
                )

    bootstrap = _mapping(_at(result, "bootstrap"), "bootstrap")
    _require(_at(bootstrap, "combiner") == EXPECTED_PRIMARY, "bootstrap combiner mismatch")
    for regime in ("represented", "transfer"):
        for contrast in ("ens_minus_sft", "ens_minus_base"):
            record = _mapping(
                _at(bootstrap, regime, contrast),
                f"bootstrap.{regime}.{contrast}",
            )
            _ci(
                _mapping(_at(record, "panel"), f"bootstrap.{regime}.{contrast}.panel"),
                f"bootstrap.{regime}.{contrast}.panel",
            )
            per_model = _mapping(
                _at(record, "per_model"), f"bootstrap.{regime}.{contrast}.per_model"
            )
            _require(
                set(per_model) == set(EXPECTED_MODELS),
                f"bootstrap.{regime}.{contrast}.per_model has the wrong panel",
            )
            for model in EXPECTED_MODELS:
                _ci(
                    _mapping(
                        _at(per_model, model),
                        f"bootstrap.{regime}.{contrast}.per_model.{model}",
                    ),
                    f"bootstrap.{regime}.{contrast}.per_model.{model}",
                )

    operating = _mapping(_at(result, "operating_point"), "operating_point")
    _require(
        _number(_at(operating, "target_fpr"), "operating_point.target_fpr") == 0.05,
        "operating-point target mismatch",
    )
    for guard in ("base", "sft", EXPECTED_PRIMARY):
        for regime in ("represented", "transfer"):
            record = _mapping(
                _at(operating, guard, regime), f"operating_point.{guard}.{regime}"
            )
            for field in ("macro_fpr", "macro_tpr", "pooled_fpr", "pooled_tpr"):
                _number(
                    _at(record, field),
                    f"operating_point.{guard}.{regime}.{field}",
                    unit_interval=True,
                )
    return result, metadata


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _panel(result: dict[str, Any], guard: str, regime: str) -> float:
    return float(_at(result, "point_estimates", guard, regime, "panel"))


def _contrast(
    result: dict[str, Any], regime: str, contrast: str, level: str, key: str | None = None
) -> tuple[float, float, float]:
    record = _at(result, "bootstrap", regime, contrast, level)
    if key is not None:
        record = _at(_mapping(record, level), key)
    return _ci(_mapping(record, f"{regime}.{contrast}.{level}"), f"{regime}.{contrast}.{level}")


def render_macros(result: dict[str, Any]) -> str:
    rep_sft_boot = _contrast(result, "represented", "ens_minus_sft", "panel")
    tr_sft_boot = _contrast(result, "transfer", "ens_minus_sft", "panel")
    tr_base_boot = _contrast(result, "transfer", "ens_minus_base", "panel")
    rep_sft_observed = (
        _panel(result, EXPECTED_PRIMARY, "represented")
        - _panel(result, "sft", "represented")
    )
    tr_sft_observed = (
        _panel(result, EXPECTED_PRIMARY, "transfer") - _panel(result, "sft", "transfer")
    )
    tr_base_observed = (
        _panel(result, EXPECTED_PRIMARY, "transfer") - _panel(result, "base", "transfer")
    )
    operating = _mapping(_at(result, "operating_point"), "operating_point")
    lines = [
        "% Generated by code/build_pilot_artifacts.py; do not edit by hand.",
        r"\newcommand{\PilotEvidenceStatus}{\texttt{clean\_v2\_retrospective\_estimation}}",
        rf"\newcommand{{\PilotNModels}}{{{len(EXPECTED_MODELS)}}}",
        rf"\newcommand{{\PilotNSeeds}}{{{len(EXPECTED_SEEDS)}}}",
        rf"\newcommand{{\PilotBaseRepresented}}{{{_fmt(_panel(result, 'base', 'represented'))}}}",
        rf"\newcommand{{\PilotBaseTransfer}}{{{_fmt(_panel(result, 'base', 'transfer'))}}}",
        rf"\newcommand{{\PilotSFTRepresented}}{{{_fmt(_panel(result, 'sft', 'represented'))}}}",
        rf"\newcommand{{\PilotSFTTransfer}}{{{_fmt(_panel(result, 'sft', 'transfer'))}}}",
        rf"\newcommand{{\PilotCompositionRepresented}}{{{_fmt(_panel(result, EXPECTED_PRIMARY, 'represented'))}}}",
        rf"\newcommand{{\PilotCompositionTransfer}}{{{_fmt(_panel(result, EXPECTED_PRIMARY, 'transfer'))}}}",
        rf"\newcommand{{\PilotLogitTransfer}}{{{_fmt(_panel(result, 'logit_avg', 'transfer'))}}}",
        rf"\newcommand{{\PilotRepDeltaSFT}}{{{_signed(rep_sft_observed)}}}",
        rf"\newcommand{{\PilotRepDeltaSFTBootstrapMean}}{{{_signed(rep_sft_boot[0])}}}",
        rf"\newcommand{{\PilotRepDeltaSFTLow}}{{{_signed(rep_sft_boot[1])}}}",
        rf"\newcommand{{\PilotRepDeltaSFTHigh}}{{{_signed(rep_sft_boot[2])}}}",
        rf"\newcommand{{\PilotTransferDeltaSFT}}{{{_signed(tr_sft_observed)}}}",
        rf"\newcommand{{\PilotTransferDeltaSFTBootstrapMean}}{{{_signed(tr_sft_boot[0])}}}",
        rf"\newcommand{{\PilotTransferDeltaSFTLow}}{{{_signed(tr_sft_boot[1])}}}",
        rf"\newcommand{{\PilotTransferDeltaSFTHigh}}{{{_signed(tr_sft_boot[2])}}}",
        rf"\newcommand{{\PilotTransferDeltaBase}}{{{_signed(tr_base_observed)}}}",
        rf"\newcommand{{\PilotTransferDeltaBaseBootstrapMean}}{{{_signed(tr_base_boot[0])}}}",
        rf"\newcommand{{\PilotTransferDeltaBaseLow}}{{{_signed(tr_base_boot[1])}}}",
        rf"\newcommand{{\PilotTransferDeltaBaseHigh}}{{{_signed(tr_base_boot[2])}}}",
        rf"\newcommand{{\PilotTargetFPRPct}}{{{100 * float(_at(operating, 'target_fpr')):.0f}}}",
        rf"\newcommand{{\PilotBaseTransferFPRPct}}{{{100 * float(_at(operating, 'base', 'transfer', 'macro_fpr')):.1f}}}",
        rf"\newcommand{{\PilotSFTTransferFPRPct}}{{{100 * float(_at(operating, 'sft', 'transfer', 'macro_fpr')):.1f}}}",
        rf"\newcommand{{\PilotCompositionTransferFPRPct}}{{{100 * float(_at(operating, EXPECTED_PRIMARY, 'transfer', 'macro_fpr')):.1f}}}",
        rf"\newcommand{{\PilotConvexWeight}}{{{float(_at(result, 'convex_selected_w')):.2f}}}",
        rf"\newcommand{{\PilotLockHashShort}}{{{EXPECTED_LOCK_SHA256[:8]}}}",
        rf"\newcommand{{\PilotScoreHashShort}}{{{EXPECTED_SCORES_SHA256[:8]}}}",
        rf"\newcommand{{\PilotCompositionHashShort}}{{{EXPECTED_COMPOSITION_SHA256[:8]}}}",
        "",
    ]
    return "\n".join(lines)


def render_summary_table(result: dict[str, Any]) -> str:
    rows = []
    labels = {
        "base": "Unadapted base",
        "sft": "SFT adapter",
        EXPECTED_PRIMARY: "Base+SFT calibrated average",
        "logit_avg": "Base+SFT logit average (ablation)",
    }
    for guard in ("base", "sft", EXPECTED_PRIMARY, "logit_avg"):
        rep = _panel(result, guard, "represented")
        transfer = _panel(result, guard, "transfer")
        rows.append(
            f"{labels[guard]} & {_fmt(rep)} & {_fmt(transfer)} & {_fmt(min(rep, transfer))} \\\\"
        )
    return "\n".join(
        [
            "% Generated by code/build_pilot_artifacts.py; do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Retrospective clean-v2 fixed-panel macro-AP. The calibrated average is the fixed primary operator. Logit averaging is an exposed ablation and cannot be promoted after observing transfer results.}",
            r"\label{tab:pilot-summary}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Guard & Represented & Transfer & $\min$(both) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_per_model_table(result: dict[str, Any]) -> str:
    rows = []
    for model in DISPLAY_ORDER:
        base = float(_at(result, "point_estimates", "base", "transfer", "per_model", model))
        sft = float(_at(result, "point_estimates", "sft", "transfer", "per_model", model))
        composed = float(
            _at(result, "point_estimates", EXPECTED_PRIMARY, "transfer", "per_model", model)
        )
        _mean_base, low_base, high_base = _contrast(
            result, "transfer", "ens_minus_base", "per_model", model
        )
        _mean_sft, low_sft, high_sft = _contrast(
            result, "transfer", "ens_minus_sft", "per_model", model
        )
        observed_sft = composed - sft
        observed_base = composed - base
        rows.append(
            f"{DISPLAY_NAMES[model]} & {_fmt(base)} & {_fmt(sft)} & {_fmt(composed)} & "
            f"{_signed(observed_sft)} [{_signed(low_sft)}, {_signed(high_sft)}] & "
            f"{_signed(observed_base)} [{_signed(low_base)}, {_signed(high_base)}] \\\\"
        )
    return "\n".join(
        [
            "% Generated by code/build_pilot_artifacts.py; do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Transfer macro-AP by checkpoint. Deltas are observed contrasts with descriptive paired-bootstrap percentile intervals. Recovery relative to SFT is positive for every checkpoint, but comparison with base is heterogeneous.}",
            r"\label{tab:pilot-models}",
            r"\small",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Checkpoint & Base & SFT & Base+SFT & $\Delta$ vs. SFT [95\% CI] & $\Delta$ vs. base [95\% CI] \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )


def render_operating_point_table(result: dict[str, Any]) -> str:
    operating = _mapping(_at(result, "operating_point"), "operating_point")
    labels = {
        "base": "Unadapted base",
        "sft": "SFT adapter",
        EXPECTED_PRIMARY: "Base+SFT calibrated average",
    }
    rows = []
    for guard in ("base", "sft", EXPECTED_PRIMARY):
        record = _mapping(_at(operating, guard, "transfer"), f"operating_point.{guard}.transfer")
        rows.append(
            f"{labels[guard]} & {_fmt(float(_at(record, 'macro_tpr')))} & "
            f"{_fmt(float(_at(record, 'macro_fpr')))} & "
            f"{_fmt(float(_at(record, 'pooled_fpr')))} \\\\"
        )
    return "\n".join(
        [
            "% Generated by code/build_pilot_artifacts.py; do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Realized transfer operating points after choosing thresholds for a 5\% FPR target on calibration negatives. Rank recovery does not imply calibration transfer.}",
            r"\label{tab:pilot-operating}",
            r"\small",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Guard & Macro TPR & Macro FPR & Pooled FPR \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_all(result: dict[str, Any]) -> dict[str, str]:
    outputs = {
        "pilot_macros.tex": render_macros(result),
        "pilot_summary_table.tex": render_summary_table(result),
        "pilot_per_model_table.tex": render_per_model_table(result),
        "pilot_operating_point_table.tex": render_operating_point_table(result),
    }
    manifest = {
        "paper_b_generated_contract_version": 1,
        "evidence": {
            "analysis_status": EXPECTED_STATUS,
            "analysis_source_sha256": EXPECTED_ANALYSIS_SOURCE_SHA256,
            "composition_sha256": EXPECTED_COMPOSITION_SHA256,
            "lock_sha256": EXPECTED_LOCK_SHA256,
            "prospective_confirmatory": False,
            "scores_sha256": EXPECTED_SCORES_SHA256,
        },
        "outputs": {
            name: hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            for name, rendered in sorted(outputs.items())
        },
    }
    outputs["MANIFEST.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return outputs


def write_or_check(outputs: dict[str, str], output_dir: Path, *, check: bool) -> None:
    if check:
        failures = []
        for name, expected in outputs.items():
            path = output_dir / name
            if not path.is_file():
                failures.append(f"missing {path}")
            elif path.read_text(encoding="utf-8") != expected:
                failures.append(f"stale {path}")
        if failures:
            raise EvidenceError("; ".join(failures))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rendered in outputs.items():
        (output_dir / name).write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition", type=Path, default=DEFAULT_COMPOSITION)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="fail if checked-in generated files are absent or stale"
    )
    args = parser.parse_args(argv)
    try:
        result, _metadata = load_and_validate(args.composition, args.metadata)
        outputs = render_all(result)
        write_or_check(outputs, args.out_dir, check=args.check)
    except EvidenceError as exc:
        parser.error(str(exc))
    action = "verified" if args.check else "generated"
    print(f"[paper-b] {action} {len(OUTPUT_NAMES)} pilot LaTeX inputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
