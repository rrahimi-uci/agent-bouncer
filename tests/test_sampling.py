import pytest

from agent_bouncer.data.sampling import (
    SAMPLING_STRATEGIES,
    SPLIT_STRATEGIES,
    k_fold,
    random_sample,
    ratio_split,
    sample_and_split,
    stratified_sample,
)


def _recs(n_safe, n_unsafe):
    return (
        [{"text": f"safe-{i}", "label": "safe"} for i in range(n_safe)]
        + [{"text": f"bad-{i}", "label": "unsafe"} for i in range(n_unsafe)]
    )


def _counts(recs):
    return (sum(r["label"] == "safe" for r in recs), sum(r["label"] == "unsafe" for r in recs))


# ---- random_sample ---------------------------------------------------------

def test_random_sample_size_and_determinism():
    recs = _recs(50, 50)
    a = random_sample(recs, 30, seed=1)
    b = random_sample(recs, 30, seed=1)
    assert len(a) == 30
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert [r["text"] for r in random_sample(recs, 30, seed=2)] != [r["text"] for r in a]


def test_random_sample_caps_at_pool_and_dedups():
    recs = _recs(3, 0) + [{"text": "safe-0", "label": "safe"}]  # dup of safe-0
    assert len(random_sample(recs, 100)) == 3  # deduped to 3, n capped
    assert random_sample(recs, 0) == []


def test_random_sample_negative_n_raises():
    with pytest.raises(ValueError):
        random_sample(_recs(2, 2), -1)


# ---- stratified_sample -----------------------------------------------------

def test_stratified_sample_preserves_proportions():
    recs = _recs(100, 20)  # 120 total, 5:1
    out = stratified_sample(recs, 60, seed=3)
    assert len(out) == 60
    assert _counts(out) == (50, 10)  # proportional, sums exactly to 60


def test_stratified_sample_largest_remainder_sums_exactly():
    recs = _recs(7, 3)  # 10 total
    out = stratified_sample(recs, 5, seed=1)
    assert len(out) == 5  # 3.5 -> 4 safe, 1.5 -> 1 unsafe by largest remainder (=5)


def test_stratified_sample_edge_cases():
    assert stratified_sample([], 5) == []
    assert stratified_sample(_recs(2, 2), 0) == []
    assert len(stratified_sample(_recs(2, 2), 100)) == 4  # capped at pool
    with pytest.raises(ValueError):
        stratified_sample(_recs(1, 1), -3)


# ---- ratio_split -----------------------------------------------------------

def test_ratio_split_disjoint_and_deterministic():
    recs = _recs(60, 40)
    tr1, te1 = ratio_split(recs, test_ratio=0.3, seed=5)
    tr2, te2 = ratio_split(recs, test_ratio=0.3, seed=5)
    assert len(te1) == 30 and len(tr1) == 70
    assert [r["text"] for r in te1] == [r["text"] for r in te2]
    assert set(r["text"] for r in tr1).isdisjoint(r["text"] for r in te1)


def test_ratio_split_stratified_preserves_balance():
    recs = _recs(80, 40)
    tr, te = ratio_split(recs, test_ratio=0.25, stratified=True, seed=1)
    assert _counts(te) == (20, 10)  # 25% of each class
    assert _counts(tr) == (60, 30)


def test_ratio_split_bad_ratio_raises():
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            ratio_split(_recs(5, 5), test_ratio=bad)


# ---- k_fold ----------------------------------------------------------------

def test_k_fold_partitions_exactly_once():
    recs = _recs(50, 50)
    folds = k_fold(recs, k=5, seed=2)
    assert len(folds) == 5
    test_texts = [r["text"] for _, te in folds for r in te]
    assert len(test_texts) == 100 and len(set(test_texts)) == 100  # each held out once
    for train, test in folds:
        assert set(r["text"] for r in train).isdisjoint(r["text"] for r in test)
        assert len(train) + len(test) == 100


def test_k_fold_stratified_balances_classes_per_fold():
    recs = _recs(50, 50)
    folds = k_fold(recs, k=5, stratified=True, seed=1)
    for _, test in folds:
        s, u = _counts(test)
        assert abs(s - u) <= 1  # each fold ~balanced


def test_k_fold_validation():
    with pytest.raises(ValueError):
        k_fold(_recs(10, 10), k=1)
    with pytest.raises(ValueError):
        k_fold(_recs(1, 0), k=5)  # fewer records than folds


# ---- sample_and_split (orchestrator) --------------------------------------

def test_sample_and_split_ratio():
    out = sample_and_split(_recs(80, 40), sampling="stratified", split="ratio",
                           n=60, test_ratio=0.25, seed=1)
    assert out["split"] == "ratio" and out["n"] == 60 and out["stratified"] is True
    assert out["n_train"] + out["n_test"] == 60
    assert _counts(out["test"]) == (10, 5)   # 25% of the 40 safe / 20 unsafe sampled
    assert _counts(out["train"]) == (30, 15)


def test_sample_and_split_kfold_random_full_pool():
    out = sample_and_split(_recs(25, 25), sampling="random", split="kfold", k=5, seed=1)
    assert out["split"] == "kfold" and out["n"] == 50  # n=None -> full pool
    assert len(out["folds"]) == 5
    assert sum(f["n_test"] for f in out["folds"]) == 50


def test_sample_and_split_rejects_unknown():
    with pytest.raises(ValueError):
        sample_and_split(_recs(5, 5), sampling="nope")
    with pytest.raises(ValueError):
        sample_and_split(_recs(5, 5), split="nope")


def test_strategy_catalogs_are_exposed():
    assert set(SAMPLING_STRATEGIES) == {"random", "stratified"}
    assert set(SPLIT_STRATEGIES) == {"ratio", "kfold"}
