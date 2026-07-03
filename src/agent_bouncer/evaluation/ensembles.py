"""Offline ensemble evaluation over dumped per-sample predictions.

Guards are expensive to run but cheap to combine: once each guard's per-sample
predictions are dumped (``outputs/predictions/<guard>.json`` from the benchmark
run), any ensemble of them can be scored instantly with no model inference. This
module is the single source of truth for that combine-and-score logic — used by
both the ``eval_ensembles.py`` CLI and the Studio ``/api/ensemble`` endpoint.

Prediction file format (from ``dump_predictions`` / the benchmark run)::

    {benchmark: [[y, u, s, ms], ...]}   # y=gold-unsafe, u=pred-unsafe, s=score, ms=latency

Sample order matches the cached benchmark subset, so members align by index.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

from agent_bouncer.core.schema import Decision
from agent_bouncer.evaluation.curves import roc_auc
from agent_bouncer.evaluation.metrics import compute_metrics
from agent_bouncer.models.ensemble import STRATEGIES, combine

PRED_DIR = "outputs/predictions"


def load_predictions(pred_dir: str = PRED_DIR) -> dict[str, dict]:
    """Load every ``<guard>.json`` prediction file in ``pred_dir`` → {guard: {bench: rows}}."""
    preds: dict[str, dict] = {}
    if os.path.isdir(pred_dir):
        for fname in sorted(os.listdir(pred_dir)):
            if fname.endswith(".json"):
                try:
                    preds[fname[:-5]] = json.load(open(os.path.join(pred_dir, fname)))
                except (ValueError, OSError):
                    continue
    return preds


def available_members(pred_dir: str = PRED_DIR) -> list[str]:
    """Guard names that have dumped predictions (candidate ensemble members)."""
    return sorted(load_predictions(pred_dir))


def _auc(gold: Sequence[Decision], scores: Sequence[float], m: dict) -> float:
    """True swept AUC when scores vary, else the single-operating-point estimate."""
    auc = roc_auc([1 if g == Decision.UNSAFE else 0 for g in gold], list(scores))
    return auc if auc is not None else (m["recall"] + 1 - m["fpr_on_benign"]) / 2


def evaluate_ensemble(
    preds: dict[str, dict],
    members: Sequence[str],
    strategy: str = "majority",
    *,
    weights: Sequence[float] | None = None,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """Score an ensemble of ``members`` over their shared benchmarks.

    Returns ``{benchmark: metrics_dict}`` (metrics include ``roc_auc``). Raises
    ``ValueError`` with an actionable message for the API to surface on bad input.
    """
    members = list(members)
    if not members:
        raise ValueError("select at least one member guard")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
    missing = [m for m in members if m not in preds]
    if missing:
        raise ValueError(f"no dumped predictions for: {', '.join(missing)} — run the benchmark suite first")
    if weights is not None and len(list(weights)) != len(members):
        raise ValueError("weights length must match the number of members")

    benches = set.intersection(*[set(preds[m]) for m in members])
    if not benches:
        raise ValueError("the selected members share no common benchmark")

    out: dict[str, dict] = {}
    for bench in sorted(benches):
        rows = [preds[m][bench] for m in members]
        n = len(rows[0])
        if any(len(r) != n for r in rows):  # misaligned dumps — skip this benchmark
            continue
        gold, pred, lat, scores = [], [], [], []
        for i in range(n):
            gold.append(Decision.UNSAFE if rows[0][i][0] == 1 else Decision.SAFE)
            unsafe, sc = combine([(bool(r[i][1]), r[i][2]) for r in rows],
                                 strategy, weights=weights, threshold=threshold)
            pred.append(Decision.UNSAFE if unsafe else Decision.SAFE)
            scores.append(sc)
            lat.append(sum(r[i][3] for r in rows))  # members run sequentially
        metrics = compute_metrics(gold, pred, lat).to_dict()
        metrics["roc_auc"] = _auc(gold, scores, metrics)
        out[bench] = metrics
    if not out:
        raise ValueError("members have mismatched sample counts on every shared benchmark")
    return out


_MACRO_KEYS = ("precision", "recall", "f1", "roc_auc", "fpr_on_benign",
               "latency_p50_ms", "latency_p90_ms", "throughput_per_s")


def macro_average(per_bench: dict[str, dict]) -> dict[str, float]:
    """Mean of each metric across the benchmarks scored."""
    if not per_bench:
        return {}
    return {k: round(sum(per_bench[b][k] for b in per_bench) / len(per_bench), 4) for k in _MACRO_KEYS}
