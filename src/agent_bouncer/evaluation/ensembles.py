"""Offline ensemble evaluation over dumped per-sample predictions.

Guards are expensive to run but cheap to combine: once each guard's per-sample
predictions are dumped (``outputs/predictions/<guard>.json`` from the benchmark
run), any ensemble of them can be scored instantly with no model inference. This
module is the single source of truth for that combine-and-score logic — used by
both the ``eval_ensembles.py`` CLI and the Workbench ``/api/ensemble`` endpoint.

Prediction file format (from ``dump_predictions`` / the benchmark run)::

    {benchmark: [[y, u, s, ms], ...]}   # y=gold-unsafe, u=pred-unsafe, s=score, ms=latency

Sample order matches the cached benchmark subset, so members align by index.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

from agent_bouncer.core.schema import Decision
from agent_bouncer.evaluation.curves import auc_with_fallback
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
    return auc_with_fallback(
        [1 if g == Decision.UNSAFE else 0 for g in gold], list(scores),
        recall=m["recall"], fpr=m["fpr_on_benign"],
    )


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


# Strategies that ignore the threshold (no sweep needed).
_HARD_STRATEGIES = ("union", "intersection", "majority")
_MAX_CANDIDATES = 2500  # keep the search responsive (a few seconds, offline)


def _objective_key(macro: dict, objective: str, fpr_cap: float):
    """Sort key (higher is better) for ranking candidate ensembles."""
    f1 = macro.get("f1") or 0.0
    fpr = macro.get("fpr_on_benign")
    fpr = 1.0 if fpr is None else fpr
    p50 = macro.get("latency_p50_ms") or 0.0
    if objective == "f1":            # best quality, latency as tiebreak
        return (f1, -fpr, -p50)
    if objective == "fpr":           # least over-blocking, then quality
        return (-fpr, f1, -p50)
    # "balanced" (default): best F1 among those within the over-blocking budget
    within = 1 if fpr <= fpr_cap else 0
    return (within, f1, -fpr, -p50)


def optimize_ensemble(
    preds: dict[str, dict],
    *,
    objective: str = "balanced",
    fpr_cap: float = 0.2,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    min_members: int = 2,
    max_members: int | None = None,
    top_k: int = 5,
    pool: Sequence[str] | None = None,
) -> dict:
    """Search member subsets × strategies for the best ensemble, scored offline.

    ``objective`` — ``"balanced"`` (best macro-F1 with FPR@benign ≤ ``fpr_cap``),
    ``"f1"`` (best macro-F1), or ``"fpr"`` (least over-blocking). ``pool`` restricts the
    candidate members (e.g. the small models only); defaults to every guard with predictions.
    Returns the winning config + macro/per-benchmark metrics and the top-``top_k`` candidates.
    The search is bounded (``_MAX_CANDIDATES``) so it stays responsive.
    """
    import itertools

    if objective not in ("balanced", "f1", "fpr"):
        raise ValueError(f"unknown objective {objective!r}; choose balanced, f1, or fpr")
    allowed = set(pool) if pool is not None else None
    names = sorted(n for n in preds if allowed is None or n in allowed)
    if len(names) < min_members:
        raise ValueError(
            f"need at least {min_members} guards with dumped predictions to build an ensemble; "
            f"have {len(names)}{' in the selected pool' if allowed is not None else ''} — "
            "test more models on the benchmarks first"
        )
    hi = min(max_members or len(names), len(names))
    subsets = [list(c) for r in range(min_members, hi + 1) for c in itertools.combinations(names, r)]
    thr = list(thresholds)
    # If the space is too big, shrink it (drop the threshold sweep, cap subset size) so the
    # search stays fast — exhaustive for the usual handful of guards, bounded beyond that.
    if len(subsets) * (len(_HARD_STRATEGIES) + len(thr)) > _MAX_CANDIDATES:
        thr = [0.5]
        subsets = [s for s in subsets if len(s) <= 5]

    scored: list[tuple] = []
    for members in subsets:
        for strat in _HARD_STRATEGIES:
            try:
                pb = evaluate_ensemble(preds, members, strat)
            except ValueError:
                continue
            scored.append((members, strat, None, macro_average(pb), pb))
        for t in thr:
            try:
                pb = evaluate_ensemble(preds, members, "mean", threshold=t)
            except ValueError:
                continue
            scored.append((members, "mean", t, macro_average(pb), pb))
    if not scored:
        raise ValueError("no scorable ensembles from the available predictions")

    scored.sort(key=lambda r: _objective_key(r[3], objective, fpr_cap), reverse=True)
    best = scored[0]
    candidates = [
        {"members": m, "strategy": s, "threshold": t,
         **{k: macro.get(k) for k in ("f1", "roc_auc", "fpr_on_benign", "latency_p50_ms")}}
        for (m, s, t, macro, _pb) in scored[:top_k]
    ]
    return {
        "best": {"members": best[0], "strategy": best[1], "threshold": best[2],
                 "macro": best[3], "per_bench": best[4]},
        "candidates": candidates,
        "objective": objective,
        "n_evaluated": len(scored),
    }
