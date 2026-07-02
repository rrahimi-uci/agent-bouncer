"""ROC / precision-recall curves and AUC — pure Python, no sklearn dependency.

A guard's `Verdict.score` is its unsafe-probability, so we can plot threshold
curves and compute AUC from ``(label, score)`` pairs. `roc_auc` uses the
tie-corrected Mann–Whitney U statistic (exact, order-independent); the curve
functions return point lists ready to feed a chart.

`UNSAFE` is the positive class. All functions accept labels as ``Decision``,
the strings ``"safe"/"unsafe"``, ints ``0/1``, or bools.
"""

from __future__ import annotations

from collections.abc import Sequence


def _binarize(labels: Sequence) -> list[int]:
    out = []
    for label in labels:
        value = getattr(label, "value", label)
        out.append(1 if value in (1, True, "unsafe") else 0)
    return out


def roc_auc(labels: Sequence, scores: Sequence[float]) -> float | None:
    """Area under the ROC curve via the tie-corrected rank (Mann–Whitney U).

    Returns ``None`` when AUC is undefined (only one class present)."""
    y = _binarize(labels)
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # Average ranks (1-based), sharing rank across tied scores.
    paired = sorted(zip(scores, y, strict=True), key=lambda t: t[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_ranks_pos = sum(r for r, (_, lbl) in zip(ranks, paired, strict=True) if lbl == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(labels: Sequence, scores: Sequence[float]) -> list[tuple[float, float]]:
    """Return ROC points ``[(fpr, tpr), ...]`` swept high→low threshold."""
    y = _binarize(labels)
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    points = [(0.0, 0.0)]
    prev_score = None
    for idx in order:
        if prev_score is not None and scores[idx] != prev_score:
            points.append((fp / n_neg, tp / n_pos))
        if y[idx] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = scores[idx]
    points.append((fp / n_neg, tp / n_pos))  # (1.0, 1.0)
    return points


def pr_curve(labels: Sequence, scores: Sequence[float]) -> list[tuple[float, float]]:
    """Return precision-recall points ``[(recall, precision), ...]``."""
    y = _binarize(labels)
    n_pos = sum(y)
    if n_pos == 0:
        return [(0.0, 1.0), (1.0, 0.0)]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = fp = 0
    points = []
    prev_score = None
    for idx in order:
        if prev_score is not None and scores[idx] != prev_score:
            points.append((tp / n_pos, tp / (tp + fp) if (tp + fp) else 1.0))
        if y[idx] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = scores[idx]
    points.append((tp / n_pos, tp / (tp + fp) if (tp + fp) else 1.0))
    return points


def downsample(points: list[tuple[float, float]], max_points: int = 60) -> list[tuple[float, float]]:
    """Evenly thin a point list for compact charting (keeps first + last)."""
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    idxs = sorted({round(k * step) for k in range(max_points)} | {0, len(points) - 1})
    return [points[i] for i in idxs]
