from agent_bouncer.core.schema import Decision
from agent_bouncer.evaluation.metrics import compute_metrics

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


def test_p90_throughput_and_ordering():
    lat = [float(i) for i in range(1, 11)]  # 1..10 ms
    m = compute_metrics([S] * 10, [S] * 10, lat)
    assert m.latency_p50_ms <= m.latency_p90_ms <= m.latency_p95_ms
    assert m.latency_p90_ms == 9.1  # interpolated 90th pct of 1..10
    # throughput = 1000 / mean latency (mean = 5.5 ms) -> ~181.8 q/s
    assert round(m.throughput_per_s, 1) == 181.8


def test_throughput_zero_without_latency():
    assert compute_metrics([S], [S]).throughput_per_s == 0.0
