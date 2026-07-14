"""Unit + self-tests for the composition ("compose, don't tune") analyzer.

Covers the parts that must be correct for the paper's numbers: the weighted tie-aware
AP wrapper, the combiner math, seed-averaged benchmark-macro AP, and end-to-end
determinism of the paired hierarchical bootstrap (same rng_seed -> identical CIs).
"""

import os
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.join(_ROOT, "experiments")
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze_composition as AC  # noqa: E402


# --------------------------------------------------------------------------- wap
def test_wap_matches_sklearn_and_arg_order():
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 0, 1, 1, 0])
    s = rng.random(6)
    # AC.wap(scores, labels): scores first (the guard_research convention)
    assert AC.wap(s, y) == pytest.approx(average_precision_score(y, s))


def test_wap_single_class_is_nan():
    assert np.isnan(AC.wap([0.1, 0.2, 0.3], [0, 0, 0]))
    assert np.isnan(AC.wap([], []))


def test_wap_weighted_matches_sklearn_sample_weight():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.2, 0.9, 0.4, 0.6])
    w = np.array([1.0, 2.0, 1.0, 3.0])
    assert AC.wap(s, y, weights=w) == pytest.approx(
        average_precision_score(y, s, sample_weight=w))


def test_wap_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])  # all positives ranked above negatives
    assert AC.wap(s, y) == pytest.approx(1.0)


# --------------------------------------------------------------------------- combiners
def _entry(seed_ids=(42, 43)):
    base = {"cal": np.array([0.2, 0.8, 0.4, 0.6]),
            "raw": np.array([0.1, 0.7, 0.3, 0.5]),
            "logit": np.array([-1.0, 2.0, 0.0, 1.0])}
    sft = {s: {"cal": np.array([0.9, 0.9, 0.1, 0.1]),
               "raw": np.array([0.8, 0.85, 0.2, 0.15]),
               "logit": np.array([3.0, 3.0, -3.0, -3.0])} for s in seed_ids}
    return {"gold": np.array([0, 1, 0, 1]), "fam": ["a", "b", "c", "d"],
            "base": base, "sft": sft}


def test_combiner_math():
    e = _entry()
    b, s = e["base"], e["sft"][42]
    assert np.allclose(AC.combiner_score(e, 42, "base"), b["cal"])
    assert np.allclose(AC.combiner_score(e, 42, "sft"), s["cal"])
    assert np.allclose(AC.combiner_score(e, 42, "calibrated_avg"), 0.5 * (b["cal"] + s["cal"]))
    assert np.allclose(AC.combiner_score(e, 42, "logit_avg"), 0.5 * (b["logit"] + s["logit"]))
    assert np.allclose(AC.combiner_score(e, 42, "max_cal"), np.maximum(b["cal"], s["cal"]))
    # convex:w = w*sft + (1-w)*base
    assert np.allclose(AC.combiner_score(e, 42, "convex:0.25"), 0.25 * s["cal"] + 0.75 * b["cal"])


# --------------------------------------------------------------------------- macro
def test_macro_is_mean_of_per_seed_ap():
    e = _entry()
    data = {"m": {"id_test": {"toxicchat": e}}}
    # both seeds identical here -> macro == single-seed AP of the calibrated_avg score
    got = AC.macro(data, "m", "id_test", ["toxicchat"], [42, 43], "calibrated_avg")
    exp = AC.wap(0.5 * (e["base"]["cal"] + e["sft"][42]["cal"]), e["gold"])
    assert got == pytest.approx(exp)


def test_macro_skips_single_class_source():
    e = _entry()
    bad = _entry(); bad["gold"] = np.array([1, 1, 1, 1])  # single-class -> nan, skipped
    data = {"m": {"id_test": {"toxicchat": e, "prompt_injections": bad}}}
    got = AC.macro(data, "m", "id_test", ["toxicchat", "prompt_injections"], [42], "sft")
    exp = AC.wap(e["sft"][42]["cal"], e["gold"])  # only the valid source counts
    assert got == pytest.approx(exp)


# --------------------------------------------------------------------------- end-to-end determinism
@pytest.fixture()
def tiny_world(monkeypatch):
    """A 2-model, 1-source-per-regime, 2-seed synthetic world with both gold classes."""
    monkeypatch.setattr(AC, "MODELS", ["m1", "m2"])
    monkeypatch.setattr(AC, "REP", ["toxicchat"])
    monkeypatch.setattr(AC, "TR", ["jailbreakbench"])
    monkeypatch.setattr(AC, "REGIMES", {"represented": ("id_test", ["toxicchat"]),
                                        "transfer": ("transfer_test", ["jailbreakbench"])})
    rng = np.random.default_rng(7)
    data = {}
    for mk in ("m1", "m2"):
        data[mk] = {}
        for split, src in (("id_test", "toxicchat"), ("transfer_test", "jailbreakbench")):
            n = 40
            gold = np.array([0, 1] * (n // 2))
            fam = [f"{split}_{i // 2}" for i in range(n)]  # 2 rows per family
            def sig(strength):
                return np.clip(0.5 + strength * (gold - 0.5) + 0.15 * rng.standard_normal(n), 0, 1)
            base = {"cal": sig(0.4), "raw": sig(0.4), "logit": sig(0.4) * 4 - 2}
            sft = {s: {"cal": sig(0.6), "raw": sig(0.6), "logit": sig(0.6) * 4 - 2} for s in (42, 43)}
            data[mk][split] = {src: {"gold": gold, "fam": fam, "base": base, "sft": sft}}
        data[mk]["calibration"] = {}
    return data


def test_bootstrap_is_deterministic(tiny_world):
    a = AC.bootstrap({k: {s: {src: dict(e) for src, e in d.items()} for s, d in v.items()} for k, v in tiny_world.items()},
                     [42, 43], reps=100, rng_seed=123)
    b = AC.bootstrap({k: {s: {src: dict(e) for src, e in d.items()} for s, d in v.items()} for k, v in tiny_world.items()},
                     [42, 43], reps=100, rng_seed=123)
    for regime in ("represented", "transfer"):
        assert a[regime]["ens_minus_sft"]["panel"]["ci95"] == b[regime]["ens_minus_sft"]["panel"]["ci95"]
        assert a[regime]["ens_minus_base"]["panel"]["mean"] == b[regime]["ens_minus_base"]["panel"]["mean"]


def test_point_estimates_run_and_are_finite(tiny_world):
    pe = AC.point_estimates(tiny_world, [42, 43], ["base", "sft", "calibrated_avg"], pit=None)
    for name in ("base", "sft", "calibrated_avg"):
        for regime in ("represented", "transfer"):
            assert np.isfinite(pe[name][regime]["panel"])
