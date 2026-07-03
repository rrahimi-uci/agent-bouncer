"""Sampling + splitting strategies for building train/test sets from benchmark records.

The training workflow separates two orthogonal choices:

* **Sampling strategy** — *how* to draw examples from the selected benchmark(s):
  - ``random``     — uniform shuffle then take N.
  - ``stratified`` — draw N while preserving the class (safe/unsafe) proportions.
* **Split strategy** — *how* to divide the drawn examples for train/test:
  - a single **ratio** split (e.g. 70/30, 80/20), optionally stratified, or
  - **k-fold** cross-validation (K disjoint folds, optionally stratified).

Everything here is pure and deterministic (seeded), reuses the anti-leakage machinery
in :mod:`agent_bouncer.data.split`, and returns plain records — so it is trivial to test
and to drive from the API/UI.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from agent_bouncer.data.split import assert_no_leakage, dedup

#: Sampling strategies surfaced to the UI/CLI.
SAMPLING_STRATEGIES: dict[str, dict] = {
    "random": {
        "label": "Random",
        "desc": "Uniform shuffle, then take N — the simple, unbiased default.",
    },
    "stratified": {
        "label": "Stratified",
        "desc": "Preserve the safe/unsafe class balance of the source when sampling and "
                "splitting — recommended for imbalanced benchmarks.",
    },
}

#: Split strategies surfaced to the UI/CLI.
SPLIT_STRATEGIES: dict[str, dict] = {
    "ratio": {"label": "Train/test ratio", "desc": "One split, e.g. 70/30 or 80/20."},
    "kfold": {"label": "K-fold cross-validation", "desc": "K disjoint folds; each is held out once."},
}


def _group_by(records: Sequence[dict], key: str) -> dict:
    groups: dict = defaultdict(list)
    for r in records:
        groups[r.get(key)].append(r)
    return groups


def random_sample(records: Sequence[dict], n: int, *, seed: int = 42,
                  deduplicate: bool = True) -> list[dict]:
    """Uniformly sample up to ``n`` records (deterministic)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    pool = dedup(records) if deduplicate else list(records)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def stratified_sample(records: Sequence[dict], n: int, *, key: str = "label",
                      seed: int = 42, deduplicate: bool = True) -> list[dict]:
    """Sample up to ``n`` records preserving each class's proportion of the pool.

    Uses largest-remainder rounding so the per-class counts sum to exactly
    ``min(n, len(pool))``. Deterministic under ``seed``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    pool = dedup(records) if deduplicate else list(records)
    total = len(pool)
    target = min(n, total)
    if target == 0:
        return []
    rng = random.Random(seed)
    groups = _group_by(pool, key)
    # largest-remainder apportionment of `target` across classes by their share
    quotas = {c: target * len(recs) / total for c, recs in groups.items()}
    base = {c: int(q) for c, q in quotas.items()}
    remainder = target - sum(base.values())
    # hand out the remaining slots to the largest fractional parts
    for c in sorted(groups, key=lambda c: quotas[c] - base[c], reverse=True)[:remainder]:
        base[c] += 1
    out: list[dict] = []
    for c, recs in groups.items():
        picks = list(recs)
        rng.shuffle(picks)
        out.extend(picks[: min(base[c], len(picks))])
    rng.shuffle(out)
    return out


def ratio_split(records: Sequence[dict], *, test_ratio: float = 0.3, key: str = "label",
                stratified: bool = False, seed: int = 42,
                deduplicate: bool = True) -> tuple[list[dict], list[dict]]:
    """Split into (train, test) by ``test_ratio``; ``stratified`` preserves class balance.

    Always de-duplicated (unless disabled) and asserted leakage-free before returning.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    pool = dedup(records) if deduplicate else list(records)
    rng = random.Random(seed)
    if not stratified:
        pool = list(pool)
        rng.shuffle(pool)
        n_test = int(len(pool) * test_ratio)
        test, train = pool[:n_test], pool[n_test:]
    else:
        train, test = [], []
        for recs in _group_by(pool, key).values():
            recs = list(recs)
            rng.shuffle(recs)
            n_test = int(len(recs) * test_ratio)
            test += recs[:n_test]
            train += recs[n_test:]
        rng.shuffle(train)
        rng.shuffle(test)
    assert_no_leakage(train, test)
    return train, test


def k_fold(records: Sequence[dict], *, k: int = 5, key: str = "label",
           stratified: bool = False, seed: int = 42,
           deduplicate: bool = True) -> list[tuple[list[dict], list[dict]]]:
    """Return ``k`` disjoint (train, test) folds; each record is held out exactly once.

    ``stratified`` distributes each class evenly across folds. Every fold is asserted
    leakage-free. Requires at least ``k`` records.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    pool = dedup(records) if deduplicate else list(records)
    if len(pool) < k:
        raise ValueError(f"need at least k={k} records, got {len(pool)}")
    rng = random.Random(seed)
    rng.shuffle(pool)
    folds: list[list[dict]] = [[] for _ in range(k)]
    if stratified:
        for recs in _group_by(pool, key).values():
            for i, r in enumerate(recs):
                folds[i % k].append(r)
    else:
        for i, r in enumerate(pool):
            folds[i % k].append(r)
    result: list[tuple[list[dict], list[dict]]] = []
    for i in range(k):
        test = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        assert_no_leakage(train, test)
        result.append((train, test))
    return result


def sample_and_split(
    records: Sequence[dict],
    *,
    sampling: str = "stratified",
    split: str = "ratio",
    n: int | None = None,
    test_ratio: float = 0.3,
    k: int = 5,
    key: str = "label",
    seed: int = 42,
) -> dict:
    """One entry point the workflow/API can call.

    Draws a sample with ``sampling`` (``random`` | ``stratified``), then divides it with
    ``split`` (``ratio`` | ``kfold``). Returns a dict with either ``train``/``test`` (ratio)
    or ``folds`` (kfold), plus the config used — so results are self-describing.
    """
    if sampling not in SAMPLING_STRATEGIES:
        raise ValueError(f"unknown sampling {sampling!r}; known: {sorted(SAMPLING_STRATEGIES)}")
    if split not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split {split!r}; known: {sorted(SPLIT_STRATEGIES)}")
    strat = sampling == "stratified"
    pool = list(records) if n is None else (
        stratified_sample(records, n, key=key, seed=seed) if strat
        else random_sample(records, n, seed=seed)
    )
    meta = {"sampling": sampling, "split": split, "n": len(pool), "seed": seed,
            "stratified": strat, "key": key}
    if split == "ratio":
        train, test = ratio_split(pool, test_ratio=test_ratio, key=key,
                                  stratified=strat, seed=seed)
        return {**meta, "test_ratio": test_ratio, "train": train, "test": test,
                "n_train": len(train), "n_test": len(test)}
    folds = k_fold(pool, k=k, key=key, stratified=strat, seed=seed)
    return {**meta, "k": k,
            "folds": [{"train": tr, "test": te, "n_train": len(tr), "n_test": len(te)}
                      for tr, te in folds]}
