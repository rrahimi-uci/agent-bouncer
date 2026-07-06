"""Train/test separation with **anti-leakage safeguards**.

A guardrail benchmark is only meaningful if the model was never trained on the test
items. These helpers produce deterministic, *disjoint* splits and provide an explicit
leakage check that training/eval code (and tests) can assert on.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


def holdout_count(n: int, ratio: float) -> int:
    """Number of held-out (test) rows for ``n`` records at ``ratio``.

    Guarantees at least ONE held-out row whenever a non-zero ratio is requested and there are at
    least two rows to split — so a requested split never silently returns an empty test set — while
    always leaving at least one training row. For the usual larger inputs this is exactly
    ``int(n * ratio)`` (the ``max``/``min`` clamps are no-ops)."""
    if n < 2 or ratio <= 0:
        return 0
    return min(n - 1, max(1, int(n * ratio)))


def _key(rec: dict) -> str:
    """Normalized text key used for overlap detection (case/space-insensitive)."""
    return " ".join((rec.get("text") or "").lower().split())


def _tokens(rec: dict) -> frozenset:
    """Word set of the normalized text — used for near-duplicate (Jaccard) matching."""
    return frozenset(_key(rec).split())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dedup(records: Iterable[dict]) -> list[dict]:
    """Drop records with a duplicate normalized text (keeps the first occurrence)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        k = _key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def find_leakage(
    train: Iterable[dict],
    test: Iterable[dict],
    *,
    fuzzy: bool = False,
    threshold: float = 0.9,
    min_tokens: int = 5,
) -> list[str]:
    """Return the normalized test texts that also appear in train (empty = clean).

    Exact matches (case/whitespace-insensitive) are always caught. With ``fuzzy=True``
    near-duplicates are caught too: a test item is flagged when its word-set Jaccard
    similarity to some train item is ``>= threshold``. Only items with ``>= min_tokens``
    words are fuzzy-matched, so short prompts aren't dropped on incidental word overlap.
    """
    train_list = list(train)
    train_keys = {_key(r) for r in train_list}
    seen: set[str] = set()
    leaked: list[str] = []
    test_list = list(test)

    # 1) exact (normalized) overlap — fast, zero false positives.
    for r in test_list:
        k = _key(r)
        if k and k in train_keys and k not in seen:
            seen.add(k)
            leaked.append(k)
    if not fuzzy:
        return leaked

    # 2) near-duplicates via an inverted index over train word-sets, so we only compare
    #    candidate pairs that share a discriminating token (very common tokens are skipped
    #    — they don't discriminate and would explode the candidate set).
    train_tokens = [_tokens(r) for r in train_list]
    n_train = len(train_list) or 1
    df: dict[str, int] = {}
    for ts in train_tokens:
        for tok in ts:
            df[tok] = df.get(tok, 0) + 1
    df_cap = max(20, int(0.10 * n_train))  # ignore tokens appearing in >10% of train rows
    inverted: dict[str, list[int]] = {}
    for i, ts in enumerate(train_tokens):
        for tok in ts:
            if df[tok] <= df_cap:
                inverted.setdefault(tok, []).append(i)

    for r in test_list:
        k = _key(r)
        if not k or k in seen:
            continue
        tt = _tokens(r)
        if len(tt) < min_tokens:
            continue
        candidates: set[int] = set()
        for tok in tt:
            candidates.update(inverted.get(tok, ()))
        if any(_jaccard(tt, train_tokens[i]) >= threshold for i in candidates):
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
    deduplicate: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Deterministic, guaranteed-disjoint split.

    De-duplicates by normalized text first (so the same prompt can't land in both
    splits), shuffles with ``seed``, then carves off ``test_ratio`` as the test set.
    The result is asserted leakage-free before returning.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    uniq = dedup(records) if deduplicate else list(records)
    random.Random(seed).shuffle(uniq)
    n_test = holdout_count(len(uniq), test_ratio)
    test, train = uniq[:n_test], uniq[n_test:]
    assert_no_leakage(train, test)
    return train, test
