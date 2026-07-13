#!/usr/bin/env python
"""Emit the in-house threshold-free AUPRC poolings the paper reports in tab:auprc-poolings and the AUPRC
column of tab:main-inhouse. Those values (guard in-dist 0.846, held-out 0.860 [0.803,0.909], pooled
0.844 [0.825,0.866]; ShieldGemma / Llama-Guard / base poolings) were previously not emitted by any
producer -- a provenance gap. This script closes it, reading ONLY committed per-row caches (no model
scoring): guard/shieldgemma/llama continuous scores from preds_corrected.json and the untuned base's
continuous scores from base_smollm3_inhouse.json, split by strata into in-distribution vs in-house
held-out (JailbreakBench, XSTest). Writes summary_auprc_poolings.json.
Run from repo root:  .venv/bin/python legacy/experiments/emit_inhouse_auprc_poolings.py
"""
import json
import numpy as np

ND = "notebooks/outputs/nb-smollm3-guard"
HELD = {"jailbreakbench", "xstest"}

def auprc(s, g):
    s = np.asarray(s, float); g = np.asarray(g, float)
    o = np.argsort(-s); g = g[o]
    tp = np.cumsum(g); fp = np.cumsum(1 - g); P = g.sum()
    if P == 0:
        return float("nan")
    pr = tp / (tp + fp); rc = tp / P
    rc = np.r_[0, rc]; pr = np.r_[1, pr]
    return float(np.sum((rc[1:] - rc[:-1]) * pr[1:]))

def ci(s, g, B=2000, seed=0):
    s = np.asarray(s, float); g = np.asarray(g); rng = np.random.default_rng(seed); n = len(g)
    v = [auprc(s[i], g[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]

tu = json.load(open(f"{ND}/preds_corrected.json"))
ba = json.load(open(f"{ND}/base_smollm3_inhouse.json"))
assert tu["gold"] == ba["gold"] and tu["strata"] == ba["strata"], "row drift between corrected and base caches"
gold = np.array(tu["gold"]); strata = np.array(tu["strata"])
ind = ~np.isin(strata, list(HELD)); hld = np.isin(strata, list(HELD))
scores = {
    "guard":       np.array(tu["guard_test_cont"]),
    "base":        np.array(ba["test_scores"]),
    "shieldgemma": np.array(tu["shield_test_cont"]),
    "llama-guard": np.array(tu["llama_test_cont"]),
}

res = {"n_test": int(len(gold)), "n_in_dist": int(ind.sum()), "n_held_out": int(hld.sum()),
       "held_out_benchmarks": sorted(HELD), "auprc": {}}
for k, s in scores.items():
    res["auprc"][k] = {
        "pooled":   [round(auprc(s, gold), 3), *ci(s, gold)],
        "in_dist":  [round(auprc(s[ind], gold[ind]), 3), *ci(s[ind], gold[ind])],
        "held_out": [round(auprc(s[hld], gold[hld]), 3), *ci(s[hld], gold[hld])],
    }

json.dump(res, open(f"{ND}/summary_auprc_poolings.json", "w"), indent=2, default=float)
print(f"n_test={res['n_test']} (in-dist {res['n_in_dist']} / held-out {res['n_held_out']})")
for k in scores:
    a = res["auprc"][k]
    print(f"  {k:12s} pooled {a['pooled'][0]:.3f} [{a['pooled'][1]:.3f},{a['pooled'][2]:.3f}]"
          f"  in-dist {a['in_dist'][0]:.3f}  held-out {a['held_out'][0]:.3f} [{a['held_out'][1]:.3f},{a['held_out'][2]:.3f}]")
print("saved -> summary_auprc_poolings.json\nDONE_AUPRC_POOLINGS")
