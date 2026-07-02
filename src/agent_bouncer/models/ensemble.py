"""Ensemble guards — combine several guards into one to beat any single SLM.

Different guards fail differently (the encoder is fast + in-domain, the decoder
generalizes, OpenAI Moderation is conservative), so combining their votes can lift
quality and/or cut over-blocking. `combine()` is pure (used both live and for offline
evaluation over cached per-sample predictions), so ensembles are cheap to explore.

Strategies (positive class = unsafe):
- ``union``        — flag if ANY member flags (max recall; more over-blocking).
- ``intersection`` — flag only if ALL flag (max precision; less over-blocking).
- ``majority``     — flag if a strict majority flag (balanced).
- ``mean``         — average the members' unsafe-scores, threshold (soft vote).
- ``weighted``     — weighted average of scores, threshold (upweight strong members).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from agent_bouncer.core.guard import Guard
from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.core.taxonomy import Hazard

STRATEGIES = ("union", "intersection", "majority", "mean", "weighted")


def combine(
    members: Sequence[tuple[bool, float]],
    strategy: str = "majority",
    *,
    weights: Sequence[float] | None = None,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Combine per-member ``(unsafe, score)`` into an ensemble ``(unsafe, score)``."""
    if not members:
        return False, 0.0
    flags = [bool(m[0]) for m in members]
    scores = [float(m[1]) for m in members]
    n = len(members)
    if strategy == "union":
        return any(flags), max(scores)
    if strategy == "intersection":
        return all(flags), min(scores)
    if strategy == "majority":
        votes = sum(flags)
        return votes > n / 2, votes / n
    if strategy == "mean":
        sc = sum(scores) / n
        return sc >= threshold, sc
    if strategy == "weighted":
        w = list(weights) if weights else [1.0] * n
        if len(w) != n:
            raise ValueError("weights length must match members")
        total = sum(w) or 1.0
        sc = sum(wi * si for wi, si in zip(w, scores, strict=True)) / total
        return sc >= threshold, sc
    raise ValueError(f"unknown strategy {strategy!r}; known: {STRATEGIES}")


class EnsembleGuard:
    """A `Guard` that queries several member guards and combines their verdicts."""

    def __init__(self, guards: list[Guard], *, strategy: str = "majority",
                 weights: Sequence[float] | None = None, threshold: float = 0.5,
                 name: str | None = None) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; known: {STRATEGIES}")
        self.guards = guards
        self.strategy = strategy
        self.weights = weights
        self.threshold = threshold
        self.name = name or "ensemble-" + "+".join(getattr(g, "name", "?") for g in guards)

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        start = time.perf_counter()
        verdicts = [g.predict(text, surface=surface) for g in self.guards]
        unsafe, score = combine([(v.blocked, v.score) for v in verdicts],
                                self.strategy, weights=self.weights, threshold=self.threshold)
        # carry the hazard from the first flagging member (best-effort)
        hazard = Hazard.NONE
        if unsafe:
            hazard = next((v.hazard for v in verdicts if v.blocked and v.hazard is not Hazard.NONE),
                          Hazard.NON_VIOLENT_CRIMES)
        return Verdict(
            decision=Decision.UNSAFE if unsafe else Decision.SAFE,
            hazard=hazard, score=float(score), surface=surface,
            latency_ms=(time.perf_counter() - start) * 1000, model=self.name,
        )
