"""Guardrail metrics — pure Python, no heavy deps so CI stays fast and green.

The headline metric is `fpr_on_benign`: the rate at which the guard blocks
*benign* traffic. Over-blocking is what makes guardrails unusable in production,
and it is the number incumbents underreport — so we put it front and center.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .schema import Decision


@dataclass
class GuardMetrics:
    n: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    fpr_on_benign: float  # ← the differentiator: over-blocking of benign inputs
    latency_p50_ms: float
    latency_p95_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def compute_metrics(
    gold: Sequence[Decision],
    pred: Sequence[Decision],
    latencies_ms: Sequence[float] | None = None,
) -> GuardMetrics:
    """`UNSAFE` is the positive class. `gold`/`pred` must be equal-length."""
    if len(gold) != len(pred):
        raise ValueError(f"gold ({len(gold)}) and pred ({len(pred)}) length mismatch")

    tp = fp = tn = fn = 0
    benign_total = benign_fp = 0
    for g, p in zip(gold, pred, strict=True):
        g_unsafe, p_unsafe = g == Decision.UNSAFE, p == Decision.UNSAFE
        if g_unsafe and p_unsafe:
            tp += 1
        elif not g_unsafe and p_unsafe:
            fp += 1
        elif not g_unsafe and not p_unsafe:
            tn += 1
        else:
            fn += 1
        if not g_unsafe:
            benign_total += 1
            benign_fp += int(p_unsafe)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(gold) if gold else 0.0
    fpr_on_benign = benign_fp / benign_total if benign_total else 0.0

    lat = list(latencies_ms or [])
    return GuardMetrics(
        n=len(gold),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        fpr_on_benign=fpr_on_benign,
        latency_p50_ms=_percentile(lat, 0.5),
        latency_p95_ms=_percentile(lat, 0.95),
    )
