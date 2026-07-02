"""Train/test separation with **anti-leakage safeguards**.

A guardrail benchmark is only meaningful if the model was never trained on the test
items. These helpers produce deterministic, *disjoint* splits and provide an explicit
leakage check that training/eval code (and tests) can assert on.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


def _key(rec: dict) -> str:
    """Normalized text key used for overlap detection (case/space-insensitive)."""
    return " ".join((rec.get("text") or "").lower().split())


def find_leakage(train: Iterable[dict], test: Iterable[dict]) -> list[str]:
    """Return the normalized texts that appear in BOTH splits (empty = clean)."""
    train_keys = {_key(r) for r in train}
    seen: set[str] = set()
    leaked: list[str] = []
    for r in test:
        k = _key(r)
        if k and k in train_keys and k not in seen:
            seen.add(k)
            leaked.append(k)
    return leaked


def assert_no_leakage(train: Iterable[dict], test: Iterable[dict]) -> None:
    """Raise if any test item also appears in train."""
    train = list(train)
    test = list(test)
    leaked = find_leakage(train, test)
    if leaked:
        raise ValueError(
            f"data leakage: {len(leaked)} test item(s) appear in train "
            f"(e.g. {leaked[0][:60]!r}). Splits must be disjoint."
        )


def train_test_split(
    records: Sequence[dict],
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
    dedup: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Deterministic, guaranteed-disjoint split.

    De-duplicates by normalized text first (so the same prompt can't land in both
    splits), shuffles with ``seed``, then carves off ``test_ratio`` as the test set.
    The result is asserted leakage-free before returning.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    seen: set[str] = set()
    uniq: list[dict] = []
    for r in records:
        k = _key(r)
        if dedup and k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    random.Random(seed).shuffle(uniq)
    n_test = int(len(uniq) * test_ratio)
    test, train = uniq[:n_test], uniq[n_test:]
    assert_no_leakage(train, test)
    return train, test
