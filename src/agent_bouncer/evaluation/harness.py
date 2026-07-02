"""The evaluation harness — runs any `Guard` over a labeled set.

This works today with the reference KeywordGuard, so `make eval` is green from
day one; swap in a trained guard once training lands. MLflow is optional: if it is
installed, metrics are also logged; otherwise they are just computed and returned.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_bouncer.core.guard import Guard
from agent_bouncer.core.schema import Decision, Surface
from agent_bouncer.evaluation.metrics import GuardMetrics, compute_metrics


def evaluate(
    guard: Guard,
    samples: Sequence[dict],
    *,
    surface: Surface = Surface.USER_PROMPT,
    run_name: str | None = None,
) -> GuardMetrics:
    """`samples`: iterable of ``{"text": str, "label": "safe"|"unsafe"}``."""
    gold: list[Decision] = []
    pred: list[Decision] = []
    latencies: list[float] = []
    for sample in samples:
        verdict = guard.predict(sample["text"], surface=surface)
        gold.append(Decision(sample["label"]))
        pred.append(verdict.decision)
        if verdict.latency_ms is not None:
            latencies.append(verdict.latency_ms)

    metrics = compute_metrics(gold, pred, latencies)
    _maybe_log_mlflow(guard, metrics, run_name)
    return metrics


def _maybe_log_mlflow(guard: Guard, metrics: GuardMetrics, run_name: str | None) -> None:
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name=run_name or guard.name):
        mlflow.log_param("guard", guard.name)
        mlflow.log_metrics(metrics.to_dict())
