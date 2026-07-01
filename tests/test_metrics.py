from agent_bouncer.metrics import compute_metrics
from agent_bouncer.schema import Decision

U, S = Decision.UNSAFE, Decision.SAFE


def test_perfect_classifier():
    gold = [U, S, U, S]
    m = compute_metrics(gold, gold, [1.0, 2.0, 3.0, 4.0])
    assert m.f1 == 1.0
    assert m.accuracy == 1.0
    assert m.fpr_on_benign == 0.0


def test_fpr_on_benign_counts_only_benign():
    gold = [S, S, U]
    pred = [U, S, U]  # one benign blocked out of two benign
    m = compute_metrics(gold, pred)
    assert m.fpr_on_benign == 0.5


def test_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        compute_metrics([U], [U, S])


def test_latency_percentiles():
    m = compute_metrics([S, S], [S, S], [10.0, 20.0])
    assert m.latency_p50_ms == 15.0
