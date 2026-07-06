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
    # each candidate carries calculated precision + recall (for the UI tables), not just F1
    for c in out["candidates"]:
        assert 0.0 <= c["precision"] <= 1.0 and 0.0 <= c["recall"] <= 1.0


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


def test_optimize_pool_restricts_members():
    """`pool` confines the search to the given members (e.g. small models only), so the winning
    ensemble never includes an excluded guard."""
    preds = {**_PREDS, "openai-x": _B}  # a "GPT baseline" that must be excluded
    out = optimize_ensemble(preds, objective="f1", pool=["guard-a", "guard-b", "guard-c"])
    assert set(out["best"]["members"]).issubset({"guard-a", "guard-b", "guard-c"})
    assert "openai-x" not in out["best"]["members"]


def test_optimize_pool_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        optimize_ensemble(_PREDS, pool=["guard-a"])  # only one member in the pool


# --------------------------------------------------------- AB-003: sample-identity alignment
from agent_bouncer.evaluation.ensembles import sample_key  # noqa: E402


def _krow(y, u, sc, ms, text):
    return [y, u, sc, ms, sample_key(text)]


def test_keyed_members_align_by_identity_not_position():
    """Members dumped in different sample ORDER (but with keys) are aligned by prompt, so union of
    two identical guards equals either one — position is irrelevant."""
    a = {"b1": [_krow(1, 1, 0.9, 10, "t1"), _krow(0, 0, 0.1, 10, "t2"), _krow(1, 1, 0.8, 10, "t3")]}
    b = {"b1": [_krow(1, 1, 0.7, 20, "t3"), _krow(0, 0, 0.2, 20, "t2"), _krow(1, 1, 0.6, 20, "t1")]}
    out = evaluate_ensemble({"a": a, "b": b}, ["a", "b"], "union")
    assert out["b1"]["recall"] == 1.0 and out["b1"]["fpr_on_benign"] == 0.0


def test_keyed_members_use_only_shared_samples():
    """Members scored on different (leakage-filtered) subsets are combined on the INTERSECTION of
    samples, not blindly by index."""
    a = {"b1": [_krow(1, 1, 0.9, 10, "t1"), _krow(0, 0, 0.1, 10, "t2"), _krow(1, 1, 0.8, 10, "t3")]}
    b = {"b1": [_krow(0, 0, 0.2, 20, "t2"), _krow(1, 1, 0.7, 20, "t3"), _krow(1, 0, 0.4, 20, "t4")]}
    out = evaluate_ensemble({"a": a, "b": b}, ["a", "b"], "majority")
    assert out["b1"]["n"] == 2   # only t2 + t3 are shared


def test_legacy_positional_rows_with_mismatched_gold_are_rejected():
    """Legacy 4-element dumps (no key) with the SAME length but DIFFERENT gold columns must not be
    silently combined — the benchmark is skipped, and with no scorable benchmark it raises."""
    a = {"b1": [[1, 1, 0.9, 10], [0, 0, 0.1, 10]]}
    b = {"b1": [[0, 0, 0.2, 20], [1, 1, 0.8, 20]]}   # gold column reversed → misaligned
    with pytest.raises(ValueError, match="mismatched sample counts|share no common"):
        evaluate_ensemble({"a": a, "b": b}, ["a", "b"], "union")


def test_duplicate_prompts_are_preserved_not_collapsed():
    """AB-012: two rows sharing the same prompt text must BOTH be scored (aligned by occurrence),
    not collapsed to one by a text-only key."""
    a = {"b": [_krow(1, 1, 0.9, 10, "dup"), _krow(0, 0, 0.1, 10, "dup"), _krow(1, 1, 0.8, 10, "x")]}
    b = {"b": [_krow(1, 1, 0.7, 20, "dup"), _krow(0, 0, 0.2, 20, "dup"), _krow(1, 1, 0.6, 20, "x")]}
    out = evaluate_ensemble({"a": a, "b": b}, ["a", "b"], "union")
    assert out["b"]["n"] == 3   # both "dup" rows kept + "x"


def test_duplicate_occurrences_align_across_reordered_members():
    """The k-th occurrence of a duplicated prompt in one member aligns with the k-th in another even
    when the members' overall row order differs."""
    a = {"b": [_krow(1, 1, 0.9, 10, "dup"), _krow(0, 0, 0.1, 10, "dup")]}
    b = {"b": [_krow(0, 0, 0.2, 20, "dup"), _krow(1, 1, 0.8, 20, "dup")]}  # reversed
    # occurrence-0 (gold 1) pairs with occurrence-0 (gold 0) -> gold mismatch -> benchmark skipped
    with pytest.raises(ValueError):
        evaluate_ensemble({"a": a, "b": b}, ["a", "b"], "union")


# --------------------------------------------------------- recall→precision cascade
from agent_bouncer.evaluation.ensembles import evaluate_cascade, optimize_cascade  # noqa: E402


def test_cascade_decision_is_gate_and_filter():
    """Final unsafe = gate AND filter, so the cascade takes the gate's recall and the filter's
    precision. Gate flags t1,t2,t3; filter flags only t1 → cascade flags only t1."""
    gate = {"b": [_krow(1, 1, 0.9, 100, "t1"), _krow(1, 1, 0.8, 100, "t2"),
                  _krow(0, 1, 0.6, 100, "t3"), _krow(0, 0, 0.2, 100, "t4")]}
    filt = {"b": [_krow(1, 1, 0.95, 500, "t1"), _krow(1, 0, 0.3, 500, "t2"),
                  _krow(0, 0, 0.1, 500, "t3"), _krow(0, 0, 0.1, 500, "t4")]}
    m = evaluate_cascade({"g": gate, "f": filt}, "g", "f")["b"]
    assert m["precision"] == 1.0 and m["recall"] == 0.5 and m["fpr_on_benign"] == 0.0


def test_cascade_latency_runs_filter_only_on_gate_flagged():
    """The filter's latency is charged ONLY to inputs the gate flagged — t4 (gate-safe) costs the
    gate's 100ms alone, so the cascade is cheaper than running both on everything."""
    gate = {"b": [_krow(1, 1, 0.9, 100, "t1"), _krow(0, 0, 0.2, 100, "t4")]}   # t4 not flagged
    filt = {"b": [_krow(1, 1, 0.95, 500, "t1"), _krow(0, 0, 0.1, 500, "t4")]}
    m = evaluate_cascade({"g": gate, "f": filt}, "g", "f")["b"]
    # per-sample latency: t1 = 100+500 = 600 (flagged), t4 = 100 (filter skipped). If the filter had
    # run on t4 too, p50 would be 600; it's 350, proving the skip.
    assert m["latency_p50_ms"] == 350.0


def test_cascade_rejects_same_model():
    g = {"b": [_krow(1, 1, 0.9, 10, "t1")]}
    with pytest.raises(ValueError, match="two different models"):
        evaluate_cascade({"g": g}, "g", "g")


def test_optimize_cascade_picks_recall_gate_and_precision_filter():
    """Gate = highest-recall model; filter = highest-precision (different) model."""
    high_recall = {"b": [_krow(1, 1, 0.9, 10, "t1"), _krow(1, 1, 0.8, 10, "t2"),
                         _krow(0, 1, 0.6, 10, "t3")]}                     # recall 1.0, precision .67
    high_prec = {"b": [_krow(1, 1, 0.9, 10, "t1"), _krow(1, 0, 0.3, 10, "t2"),
                       _krow(0, 0, 0.1, 10, "t3")]}                       # recall .5, precision 1.0
    res = optimize_cascade({"recall_m": high_recall, "prec_m": high_prec})
    assert res["stage1"] == "recall_m" and res["stage2"] == "prec_m"


def test_optimize_cascade_needs_two_models():
    with pytest.raises(ValueError, match="at least 2"):
        optimize_cascade({"only": {"b": [_krow(1, 1, 0.9, 10, "t1")]}})


# --------------------------------------------------------- AB-011: eval_tuned keyed alignment
def test_eval_tuned_aligns_reordered_keyed_members():
    from eval_ensembles import eval_tuned
    a = {"b": [_krow(1, 1, 0.9, 10, "t1"), _krow(1, 1, 0.8, 10, "t2"),
               _krow(0, 0, 0.1, 10, "t3"), _krow(0, 0, 0.2, 10, "t4")]}
    b_fwd = {"b": [_krow(0, 0, 0.2, 20, "t4"), _krow(0, 0, 0.1, 20, "t3"),
                   _krow(1, 1, 0.8, 20, "t2"), _krow(1, 1, 0.9, 20, "t1")]}  # reversed order
    b_ord = {"b": [_krow(1, 1, 0.9, 20, "t1"), _krow(1, 1, 0.8, 20, "t2"),
                   _krow(0, 0, 0.1, 20, "t3"), _krow(0, 0, 0.2, 20, "t4")]}  # same order as a
    out_rev, t_rev = eval_tuned(["a", "b"], {"a": a, "b": b_fwd})
    out_ord, t_ord = eval_tuned(["a", "b"], {"a": a, "b": b_ord})
    # aligned by identity, so B's row order is irrelevant — tuned threshold + metrics match
    assert out_rev and out_rev["b"]["n"] == out_ord["b"]["n"] and t_rev == t_ord
    assert out_rev["b"]["f1"] == out_ord["b"]["f1"]
