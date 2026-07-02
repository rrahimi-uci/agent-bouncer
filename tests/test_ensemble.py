import pytest

from agent_bouncer.ensemble import EnsembleGuard, combine
from agent_bouncer.schema import Decision, Surface, Verdict


def test_union_and_intersection():
    m = [(True, 0.9), (False, 0.2), (False, 0.1)]
    assert combine(m, "union") == (True, 0.9)
    assert combine(m, "intersection")[0] is False
    assert combine([(True, 0.9), (True, 0.6)], "intersection") == (True, 0.6)


def test_majority():
    assert combine([(True, 1), (True, 1), (False, 0)], "majority")[0] is True   # 2/3
    assert combine([(True, 1), (False, 0), (False, 0)], "majority")[0] is False  # 1/3
    assert combine([(True, 1), (True, 1)], "majority")[1] == 1.0                 # 2/2


def test_mean_and_threshold():
    assert combine([(True, 0.8), (False, 0.2)], "mean", threshold=0.6)[0] is False  # mean .5 < .6
    assert combine([(True, 0.9), (True, 0.7)], "mean", threshold=0.5)[0] is True     # mean .8
    assert combine([(False, 0.3), (False, 0.1)], "mean", threshold=0.5)[0] is False  # mean .2


def test_weighted_upweights_member():
    # weight the confident member heavily -> flags unsafe
    unsafe, sc = combine([(True, 1.0), (False, 0.0)], "weighted", weights=[4, 1], threshold=0.5)
    assert unsafe is True and round(sc, 2) == 0.8
    with pytest.raises(ValueError):
        combine([(True, 1.0)], "weighted", weights=[1, 2])


def test_empty_and_unknown():
    assert combine([], "union") == (False, 0.0)
    with pytest.raises(ValueError):
        combine([(True, 1.0)], "nope")


class _Fake:
    def __init__(self, name, unsafe, score):
        self.name = name
        self._u = unsafe
        self._s = score

    def predict(self, text, *, surface=Surface.USER_PROMPT):
        return Verdict(decision=Decision.UNSAFE if self._u else Decision.SAFE, score=self._s, surface=surface)


def test_ensemble_guard_predict_majority():
    members = [_Fake("a", True, .9), _Fake("b", True, .8), _Fake("c", False, .1)]
    g = EnsembleGuard(members, strategy="majority")
    v = g.predict("x")
    assert v.blocked and v.latency_ms is not None and "ensemble" in g.name


def test_ensemble_guard_rejects_bad_strategy():
    with pytest.raises(ValueError):
        EnsembleGuard([_Fake("a", True, 1.0)], strategy="bogus")
