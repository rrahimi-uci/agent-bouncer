"""Tests for the offline ensemble evaluator (agent_bouncer.evaluation.ensembles)."""

import json

import pytest

from agent_bouncer.evaluation.ensembles import (
    available_members,
    evaluate_ensemble,
    load_predictions,
    macro_average,
)

# Prediction rows are [y, u, score, latency_ms]; 4 unsafe (y=1) then 2 safe (y=0).
_A = {"b1": [[1, 1, 0.9, 10], [1, 1, 0.8, 10], [1, 0, 0.4, 10],
             [1, 1, 0.7, 10], [0, 0, 0.1, 10], [0, 1, 0.6, 10]]}
_B = {"b1": [[1, 1, 0.95, 20], [1, 0, 0.2, 20], [1, 1, 0.7, 20],
             [1, 1, 0.6, 20], [0, 0, 0.3, 20], [0, 0, 0.15, 20]]}
_C = {"b1": [[1, 0, 0.3, 5], [1, 1, 0.9, 5], [1, 1, 0.85, 5],
             [1, 1, 0.55, 5], [0, 0, 0.05, 5], [0, 0, 0.2, 5]]}
_PREDS = {"guard-a": _A, "guard-b": _B, "guard-c": _C}


def test_majority_beats_a_single_flaky_member():
    # Each unsafe sample is flagged by >=2 of 3 members -> majority recovers all unsafe.
    out = evaluate_ensemble(_PREDS, ["guard-a", "guard-b", "guard-c"], "majority")
    assert out["b1"]["recall"] == 1.0
    assert out["b1"]["fpr_on_benign"] == 0.0


def test_latency_is_summed_across_members():
    # Members run sequentially, so ensemble latency = sum of member latencies (10+20+5=35).
    out = evaluate_ensemble(_PREDS, ["guard-a", "guard-b", "guard-c"], "majority")
    assert out["b1"]["latency_p50_ms"] == 35.0


def test_union_maximises_recall_intersection_maximises_precision():
    union = evaluate_ensemble(_PREDS, ["guard-a", "guard-b"], "union")["b1"]
    inter = evaluate_ensemble(_PREDS, ["guard-a", "guard-b"], "intersection")["b1"]
    assert union["recall"] >= inter["recall"]
    assert inter["fpr_on_benign"] <= union["fpr_on_benign"]


def test_weighted_requires_matching_weights():
    with pytest.raises(ValueError, match="weights length"):
        evaluate_ensemble(_PREDS, ["guard-a", "guard-b"], "weighted", weights=[1.0])


@pytest.mark.parametrize("members,strategy,match", [
    ([], "majority", "at least one"),
    (["guard-a", "missing"], "majority", "no dumped predictions"),
    (["guard-a", "guard-b"], "bogus", "unknown strategy"),
])
def test_bad_input_raises_actionable_valueerror(members, strategy, match):
    with pytest.raises(ValueError, match=match):
        evaluate_ensemble(_PREDS, members, strategy)


def test_members_sharing_no_benchmark_raise():
    preds = {"x": {"b1": _A["b1"]}, "y": {"b2": _B["b1"]}}
    with pytest.raises(ValueError, match="no common benchmark"):
        evaluate_ensemble(preds, ["x", "y"], "majority")


def test_macro_average_is_mean_across_benchmarks():
    out = {"b1": {k: 0.4 for k in _macro_keys()}, "b2": {k: 0.6 for k in _macro_keys()}}
    macro = macro_average(out)
    assert macro["f1"] == 0.5
    assert macro_average({}) == {}


def _macro_keys():
    from agent_bouncer.evaluation.ensembles import _MACRO_KEYS
    return _MACRO_KEYS


def test_load_and_available_members_roundtrip(tmp_path):
    (tmp_path / "guard-a.json").write_text(json.dumps(_A))
    (tmp_path / "guard-b.json").write_text(json.dumps(_B))
    (tmp_path / "notjson.txt").write_text("ignore me")
    preds = load_predictions(str(tmp_path))
    assert set(preds) == {"guard-a", "guard-b"}
    assert available_members(str(tmp_path)) == ["guard-a", "guard-b"]
    assert available_members(str(tmp_path / "does-not-exist")) == []


# --------------------------------------------------------- optimizer (auto best ensemble)
from agent_bouncer.evaluation.ensembles import optimize_ensemble  # noqa: E402


def test_optimize_finds_a_valid_best():
    out = optimize_ensemble(_PREDS, objective="balanced", top_k=3)
    best = out["best"]
    assert len(best["members"]) >= 2
    assert best["strategy"] in ("union", "intersection", "majority", "mean")
    assert 0.0 <= best["macro"]["f1"] <= 1.0
    assert len(out["candidates"]) <= 3 and out["n_evaluated"] > 0
    # candidates are ranked (first is the best for the objective)
    assert out["candidates"][0]["members"] == best["members"]


def test_optimize_f1_objective_maximizes_f1():
    out = optimize_ensemble(_PREDS, objective="f1")
    f1s = [c["f1"] for c in out["candidates"]]
    assert out["best"]["macro"]["f1"] >= max(f1s)   # best is the top-ranked
    # majority over the 3 members recovers all unsafe with no over-block on this fixture
    assert out["best"]["macro"]["f1"] == 1.0


def test_optimize_balanced_respects_fpr_cap_when_possible():
    out = optimize_ensemble(_PREDS, objective="balanced", fpr_cap=0.2)
    assert out["best"]["macro"]["fpr_on_benign"] <= 0.2


def test_optimize_needs_two_members():
    with pytest.raises(ValueError, match="at least 2"):
        optimize_ensemble({"only-one": _A})


def test_optimize_rejects_bad_objective():
    with pytest.raises(ValueError, match="unknown objective"):
        optimize_ensemble(_PREDS, objective="nonsense")
