"""Verifiable reward functions for GRPO / RLVR.

Key insight: a guardrail has ground-truth labels, so the *label is the reward* —
no reward model is needed. Each function is a pure function of (prediction, gold),
so it is deterministic and unit-testable. The false-positive penalty bakes our
headline metric (don't over-block benign traffic) directly into the objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Decision, Verdict


@dataclass(frozen=True)
class RewardWeights:
    correctness: float = 1.0
    category: float = 0.5
    format: float = 0.2
    false_positive_penalty: float = 1.0
    brevity: float = 0.1
    rationale_budget: int = 256  # characters; longer rationales are penalized (latency)


def correctness_reward(pred: Verdict, gold: Verdict) -> float:
    """+1 for correct safe/unsafe decision."""
    return 1.0 if pred.decision == gold.decision else 0.0


def category_reward(pred: Verdict, gold: Verdict) -> float:
    """Partial credit for the right hazard category (only when gold is unsafe)."""
    if gold.decision != Decision.UNSAFE:
        return 0.0
    return 1.0 if pred.hazard == gold.hazard else 0.0


def format_reward(parsed: Verdict | None) -> float:
    """+1 if the model produced a parseable, schema-valid verdict."""
    return 1.0 if parsed is not None else 0.0


def false_positive_penalty(pred: Verdict, gold: Verdict) -> float:
    """1.0 when a benign input is blocked — subtracted in the composite reward."""
    return 1.0 if (gold.decision == Decision.SAFE and pred.decision == Decision.UNSAFE) else 0.0


def brevity_reward(rationale: str | None, budget: int) -> float:
    """Encourage short rationales (latency matters on the request path)."""
    n = len(rationale or "")
    if budget <= 0:
        return 1.0 if n == 0 else 0.0
    if n <= budget:
        return 1.0
    return max(0.0, 1.0 - (n - budget) / budget)


def composite_reward(
    pred: Verdict | None,
    gold: Verdict,
    weights: RewardWeights | None = None,
) -> float:
    """Total scalar reward for one rollout. `pred` is None when parsing failed."""
    if weights is None:
        weights = RewardWeights()
    if pred is None:  # unparseable output: only the (zero) format reward applies
        return weights.format * format_reward(None)
    reward = 0.0
    reward += weights.correctness * correctness_reward(pred, gold)
    reward += weights.category * category_reward(pred, gold)
    reward += weights.format * format_reward(pred)
    reward -= weights.false_positive_penalty * false_positive_penalty(pred, gold)
    reward += weights.brevity * brevity_reward(pred.rationale, weights.rationale_budget)
    return reward
