#!/usr/bin/env python3
"""Validate the Paper B prospective-study contract.

Draft contracts may contain explicit null placeholders.  A claim-bearing lock must
pass ``--require-locked``; that mode rejects missing cohort identities, post-hoc
operator choices, incomplete baselines, unsupported noninferiority margins, and
missing systems or software provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


REQUIRED_BASELINES = {
    "base",
    "sft",
    "base_sft_output",
    "sft_sft_output",
    "wise_ft",
    "kl_or_replay",
}
REQUIRED_SYSTEM_METRICS = {
    "latency_ms",
    "throughput_examples_per_second",
    "peak_accelerator_memory_bytes",
    "training_accelerator_hours",
    "inference_flops_or_proxy",
}
LOCKED_STATUSES = {"locked_not_executed", "executed"}
ALL_STATUSES = {"draft_not_executed", *LOCKED_STATUSES}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProtocolError(ValueError):
    """Raised when a protocol is internally unsafe or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path} must be an object")
    return value


def _field(root: dict[str, Any], *keys: str, allow_null: bool = False) -> Any:
    value: Any = root
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        parent = _mapping(value, ".".join(walked[:-1]) or "root")
        _require(key in parent, f"missing required field: {'.'.join(walked)}")
        value = parent[key]
    if not allow_null:
        _require(value is not None, f"{'.'.join(keys)} must not be null")
    return value


def canonical_sha256(protocol: dict[str, Any]) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonempty_string(value: Any, path: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{path} must be a nonempty string")
    _require("TO_BE" not in value, f"{path} still contains a draft placeholder")
    return value


def _sha(value: Any, path: str) -> str:
    text = _nonempty_string(value, path)
    _require(bool(HEX64.fullmatch(text)), f"{path} must be a lowercase SHA-256 digest")
    return text


def validate_protocol(protocol: dict[str, Any], *, require_locked: bool = False) -> list[str]:
    """Validate a draft or claim-bearing protocol and return draft warnings."""

    protocol = _mapping(protocol, "protocol")
    _require(_field(protocol, "contract_version") == 1, "unsupported contract_version")
    _require(
        _field(protocol, "study_id") == "paper_b_compose_dont_tune",
        "unexpected study_id",
    )
    status = _field(protocol, "status")
    _require(status in ALL_STATUSES, f"unsupported protocol status: {status!r}")
    if require_locked:
        _require(status in LOCKED_STATUSES, "claim-bearing validation requires a locked protocol")

    claim_scope = _mapping(_field(protocol, "claim_scope"), "claim_scope")
    forbidden = _field(claim_scope, "forbidden_claims")
    _require(isinstance(forbidden, list) and forbidden, "forbidden_claims must be a nonempty list")
    required_forbidden = {"pareto_dominance", "universal_remedy", "causal_mechanism"}
    _require(
        required_forbidden.issubset(set(forbidden)),
        "forbidden_claims must include Pareto, universal-remedy, and mechanism claims",
    )

    inputs = _mapping(_field(protocol, "inputs"), "inputs")
    _sha(_field(inputs, "paper_a_lock_sha256"), "inputs.paper_a_lock_sha256")
    _sha(_field(inputs, "development_scores_sha256"), "inputs.development_scores_sha256")
    cohort = _mapping(_field(inputs, "prospective_cohort"), "inputs.prospective_cohort")

    operator = _mapping(_field(protocol, "operator"), "operator")
    _require(_field(operator, "primary") == "calibrated_avg", "primary operator must be calibrated_avg")
    base_weight = _field(operator, "base_weight")
    sft_weight = _field(operator, "sft_weight")
    _require(
        isinstance(base_weight, (int, float))
        and isinstance(sft_weight, (int, float))
        and not isinstance(base_weight, bool)
        and not isinstance(sft_weight, bool),
        "operator weights must be numeric",
    )
    _require(
        math.isfinite(float(base_weight)) and math.isfinite(float(sft_weight)),
        "operator weights must be finite",
    )
    _require(abs(float(base_weight) - 0.5) < 1e-12, "base_weight must remain fixed at 0.5")
    _require(abs(float(sft_weight) - 0.5) < 1e-12, "sft_weight must remain fixed at 0.5")
    _require(
        _field(operator, "uses_reported_test_for_selection") is False,
        "the reported test cohort cannot select or tune the operator",
    )
    _nonempty_string(_field(operator, "calibration_split"), "operator.calibration_split")

    baselines = _field(protocol, "baselines")
    _require(isinstance(baselines, list), "baselines must be a list")
    baseline_records: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(baselines):
        record = _mapping(raw, f"baselines[{index}]")
        baseline_id = _nonempty_string(_field(record, "id"), f"baselines[{index}].id")
        _require(baseline_id not in baseline_records, f"duplicate baseline id: {baseline_id}")
        _nonempty_string(_field(record, "role"), f"baselines[{index}].role")
        _field(record, "implementation_status")
        baseline_records[baseline_id] = record
    missing = sorted(REQUIRED_BASELINES - set(baseline_records))
    _require(not missing, "missing required baselines: " + ", ".join(missing))

    estimands = _mapping(_field(protocol, "estimands"), "estimands")
    _require(
        _field(estimands, "primary") == "transfer_macro_ap_composition_minus_sft",
        "unexpected primary estimand",
    )
    _nonempty_string(
        _field(estimands, "represented_retention"), "estimands.represented_retention"
    )
    margin = _field(estimands, "noninferiority_margin", allow_null=True)
    margin_rationale = _field(estimands, "noninferiority_margin_rationale", allow_null=True)

    statistics = _mapping(_field(protocol, "statistics"), "statistics")
    _require(
        _field(statistics, "primary_metric") == "benchmark_macro_average_precision",
        "primary metric must be benchmark-macro AP",
    )
    _require(
        _field(statistics, "bootstrap_unit") == "family_x_seed_hierarchical_paired",
        "unexpected bootstrap unit",
    )
    _require(
        _field(statistics, "seed_pair_aggregation")
        == "checkpoint_level_shared_adapter_aware",
        "SFT seed pairs must be aggregated without pseudoreplication",
    )
    target_fpr = _field(statistics, "operating_point_target_fpr")
    _require(
        isinstance(target_fpr, (int, float))
        and not isinstance(target_fpr, bool)
        and math.isfinite(float(target_fpr))
        and 0.0 < float(target_fpr) < 1.0,
        "operating-point target FPR must lie in (0, 1)",
    )

    competence = _mapping(_field(protocol, "competence_hypothesis"), "competence_hypothesis")
    _require(
        _field(competence, "cohorts_disjoint") is True,
        "competence measurement and outcome cohorts must be disjoint",
    )
    _require(
        _field(competence, "outcome") == "composition_minus_sft_transfer_macro_ap",
        "competence outcome must avoid a same-row ensemble-minus-base coupling",
    )

    systems = _mapping(_field(protocol, "systems"), "systems")
    metrics = _field(systems, "metrics")
    _require(isinstance(metrics, list), "systems.metrics must be a list")
    missing_metrics = sorted(REQUIRED_SYSTEM_METRICS - set(metrics))
    _require(not missing_metrics, "missing systems metrics: " + ", ".join(missing_metrics))

    software = _mapping(_field(protocol, "software_lock"), "software_lock")
    execution = _mapping(_field(protocol, "execution"), "execution")
    warnings: list[str] = []
    if status == "draft_not_executed":
        warnings.extend(
            [
                "draft only: no claim-bearing Paper B lock exists",
                "prospective cohort is not yet frozen or proven uninspected",
                "noninferiority margin and rationale are not yet fixed",
                "baseline implementations and systems protocol are not yet complete",
            ]
        )
        return warnings

    _nonempty_string(_field(claim_scope, "primary_claim"), "claim_scope.primary_claim")
    cohort_id = _nonempty_string(_field(cohort, "cohort_id"), "inputs.prospective_cohort.cohort_id")
    _sha(_field(cohort, "content_sha256"), "inputs.prospective_cohort.content_sha256")
    _sha(_field(cohort, "label_policy_sha256"), "inputs.prospective_cohort.label_policy_sha256")
    _require(
        _field(cohort, "selected_before_lock") is True,
        "prospective cohort must be selected before the lock",
    )
    _require(
        _field(cohort, "inspected_before_lock") is False,
        "prospective cohort must be genuinely uninspected before the lock",
    )

    _require(
        isinstance(margin, (int, float))
        and not isinstance(margin, bool)
        and math.isfinite(float(margin))
        and 0.0 < float(margin) < 1.0,
        "a locked protocol needs a numeric noninferiority margin in (0, 1)",
    )
    _nonempty_string(margin_rationale, "estimands.noninferiority_margin_rationale")

    for baseline_id, record in baseline_records.items():
        _require(
            _field(record, "implementation_status") == "ready",
            f"baseline {baseline_id} is not ready at lock time",
        )

    reps = _field(statistics, "bootstrap_reps")
    rng_seed = _field(statistics, "rng_seed")
    _require(isinstance(reps, int) and not isinstance(reps, bool) and reps >= 4000,
             "locked bootstrap_reps must be an integer >= 4000")
    _require(isinstance(rng_seed, int) and not isinstance(rng_seed, bool) and rng_seed >= 0,
             "locked rng_seed must be a nonnegative integer")
    _nonempty_string(_field(statistics, "multiplicity_policy"), "statistics.multiplicity_policy")

    development_cohort = _nonempty_string(
        _field(competence, "development_measure_cohort"),
        "competence_hypothesis.development_measure_cohort",
    )
    outcome_cohort = _nonempty_string(
        _field(competence, "prospective_outcome_cohort"),
        "competence_hypothesis.prospective_outcome_cohort",
    )
    _require(development_cohort != outcome_cohort, "competence cohorts must have distinct ids")
    _require(outcome_cohort == cohort_id, "competence outcome must use the locked prospective cohort")
    panel_size = _field(competence, "minimum_checkpoint_panel")
    _require(
        isinstance(panel_size, int) and not isinstance(panel_size, bool) and panel_size > 4,
        "locked competence test must expand beyond the four-checkpoint pilot",
    )

    _sha(_field(systems, "hardware_spec_sha256"), "systems.hardware_spec_sha256")
    _nonempty_string(_field(systems, "measurement_protocol"), "systems.measurement_protocol")
    for field in (
        "source_tree_sha256",
        "environment_sha256",
        "analysis_config_sha256",
    ):
        _sha(_field(software, field), f"software_lock.{field}")
    _nonempty_string(_field(software, "container_digest"), "software_lock.container_digest")
    _nonempty_string(_field(execution, "lock_id"), "execution.lock_id")
    _nonempty_string(_field(execution, "lock_utc"), "execution.lock_utc")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument(
        "--require-locked",
        action="store_true",
        help="reject draft placeholders and require a claim-bearing locked contract",
    )
    args = parser.parse_args(argv)
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        warnings = validate_protocol(protocol, require_locked=args.require_locked)
    except (OSError, json.JSONDecodeError, ProtocolError) as exc:
        parser.error(str(exc))
    print(f"[paper-b] protocol valid: status={protocol['status']}")
    print(f"[paper-b] canonical protocol sha256={canonical_sha256(protocol)}")
    for warning in warnings:
        print(f"[paper-b] warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
