import pytest

from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.evaluation import benchmarks as B


def _recs(n_safe, n_unsafe):
    recs = [{"text": f"safe{i}", "label": "safe", "hazard": "none"} for i in range(n_safe)]
    recs += [{"text": f"bad{i}", "label": "unsafe", "hazard": "non_violent_crimes"} for i in range(n_unsafe)]
    return recs


class _KeywordFakeGuard:
    """Deterministic guard: flags any text containing 'bad'. No model/network."""

    name = "fake"

    def predict(self, text, *, surface=Surface.USER_PROMPT):
        unsafe = "bad" in text
        return Verdict(
            decision=Decision.UNSAFE if unsafe else Decision.SAFE,
            score=1.0 if unsafe else 0.0,
            surface=surface,
            latency_ms=0.1,
        )


def test_registry_covers_both_axes():
    axes = {b.axis for b in B.BENCHMARKS.values()}
    assert {"guardrail", "red_team", "over_refusal"} <= axes
    # every benchmark has a callable loader and a description
    for name, b in B.BENCHMARKS.items():
        assert callable(b.loader) and b.description and b.name == name


def test_balanced_subset_is_balanced_and_deterministic():
    recs = _recs(100, 20)
    sub = B.balanced_subset(recs, per_class=30)
    n_safe, n_unsafe = B.class_counts(sub)
    # only 20 unsafe available -> 20 per class
    assert n_safe == 20 and n_unsafe == 20
    assert [r["text"] for r in sub] == [r["text"] for r in B.balanced_subset(recs, per_class=30)]


def test_subsample_caps_and_preserves_when_small():
    recs = _recs(5, 5)
    assert len(B.subsample(recs, 3)) == 3
    assert len(B.subsample(recs, 100)) == 10


def test_load_benchmark_uses_registry_loader(monkeypatch):
    monkeypatch.setitem(
        B.BENCHMARKS, "fake_bench",
        B.Benchmark("fake_bench", lambda: _recs(50, 50), "guardrail", "test"),
    )
    out = B.load_benchmark("fake_bench", balanced=True, per_class=10)
    assert len(out) == 20
    with pytest.raises(ValueError):
        B.load_benchmark("does-not-exist")


def test_run_suite_scores_all_pairs():
    datasets = {"b1": _recs(10, 10), "b2": _recs(8, 12)}
    results = B.run_suite([("fake", _KeywordFakeGuard())], datasets)
    # perfect guard -> F1 == 1.0 on both benchmarks
    assert results["b1"]["fake"]["f1"] == pytest.approx(1.0)
    assert results["b2"]["fake"]["recall"] == pytest.approx(1.0)
    assert results["b2"]["fake"]["fpr_on_benign"] == pytest.approx(0.0)


def test_run_suite_isolates_failing_guard():
    class _Boom:
        name = "boom"

        def predict(self, text, *, surface=Surface.USER_PROMPT):
            raise RuntimeError("model down")

    errors = []
    results = B.run_suite(
        [("boom", _Boom()), ("fake", _KeywordFakeGuard())],
        {"b1": _recs(4, 4)},
        on_error=lambda bench, guard, exc: errors.append((bench, guard)),
    )
    assert "boom" not in results["b1"] and "fake" in results["b1"]
    assert errors == [("b1", "boom")]
