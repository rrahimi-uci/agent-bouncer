#!/usr/bin/env python
"""Evaluate ensembles offline from dumped per-sample predictions and compare to GPT-5.2.

Reads outputs/predictions/<guard>.json (from dump_predictions.py), combines members with
each strategy, and computes P/R/F1/ROC-AUC/FPR + latency (summed member latency) per
benchmark + macro. Writes outputs/ensemble_results.json and merges the chosen ensembles
into outputs/benchmark_results.json so they show up in the scoreboard / Studio.

Usage:
    python scripts/eval/eval_ensembles.py [--merge ensemble-maj3 ensemble-mean3]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from agent_bouncer.core.schema import Decision
from agent_bouncer.evaluation.curves import roc_auc
from agent_bouncer.evaluation.ensembles import evaluate_ensemble, load_predictions, macro_average
from agent_bouncer.evaluation.metrics import compute_metrics
from agent_bouncer.models.ensemble import combine

PRED = "outputs/predictions"
RESULTS = "outputs/benchmark_results.json"
OUT = "outputs/ensemble_results.json"

# Ensemble specs: name -> (members, strategy, kwargs). Members must have dumped predictions.
_ENC, _SFT, _GRPO, _MOD, _KW = ("encoder-distilbert", "decoder-sft-0.6B",
                                "decoder-grpo-0.6B", "openai-moderation", "keyword-baseline")
_SFT17 = "decoder-sft-1.7B"
ENSEMBLES = {
    "ensemble-maj3-local": ([_ENC, _SFT, _GRPO], "majority", {}),
    "ensemble-maj3-diverse": ([_ENC, _SFT, _MOD], "majority", {}),
    "ensemble-union2": ([_ENC, _SFT], "union", {}),
    "ensemble-inter2": ([_ENC, _SFT], "intersection", {}),
    "ensemble-mean3": ([_ENC, _SFT, _GRPO], "mean", {"threshold": 0.5}),
    "ensemble-maj5": ([_KW, _ENC, _SFT, _GRPO, _MOD], "majority", {}),
    "ensemble-wtd": ([_ENC, _SFT, _GRPO, _MOD], "weighted", {"weights": [2, 1, 1, 1], "threshold": 0.5}),
    # variants that add the stronger, less-correlated 1.7B decoder
    "ensemble-maj3-17b": ([_ENC, _SFT17, _MOD], "majority", {}),
    "ensemble-union-17b": ([_ENC, _SFT17], "union", {}),
    "ensemble-maj5-17b": ([_ENC, _SFT, _SFT17, _GRPO, _MOD], "majority", {}),
}


def load_preds() -> dict[str, dict]:
    return load_predictions(PRED)


def macro(metrics: dict) -> dict:
    return macro_average(metrics)


def eval_ensemble(members, strategy, kwargs, preds) -> dict | None:
    """Thin wrapper over the shared evaluator: returns None (skip) on any bad-input error."""
    try:
        return evaluate_ensemble(preds, members, strategy, **kwargs)
    except ValueError:
        return None


def eval_tuned(members, preds, *, weights=None, seed=42, fpr_cap=0.20) -> tuple[dict, float]:
    """Weighted-mean ensemble with a GLOBAL threshold tuned on a val split, scored on a
    disjoint test split (per benchmark) — exploits AUC headroom without overfitting.

    Objective: maximize F1 subject to val-FPR <= ``fpr_cap`` (so the tuned operating point
    is comparable to GPT-5.2's over-blocking budget, not just F1-gaming via recall)."""
    if any(m not in preds for m in members):
        return {}, 0.5
    import random
    benches = sorted(set.intersection(*[set(preds[m]) for m in members]))
    w = weights or [1.0] * len(members)
    # build (bench, idx, gold, score, latency), deterministically split val/test
    val, test = [], []
    for b in benches:
        rows = [preds[m][b] for m in members]
        n = len(rows[0])
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        cut = n // 2
        for j, i in enumerate(idx):
            _, sc = combine([(bool(r[i][1]), r[i][2]) for r in rows], "weighted",
                            weights=w, threshold=0.5)
            rec = (b, rows[0][i][0], sc, sum(r[i][3] for r in rows))
            (val if j < cut else test).append(rec)
    # sweep threshold on val: maximize F1 subject to val-FPR <= fpr_cap
    # (fallback to the min-FPR threshold if the cap is unreachable)
    best_t, best_f1, fb_t, fb_fpr = 0.5, -1.0, 0.5, 1.0
    for t in [x / 100 for x in range(5, 100, 5)]:
        tp = sum(1 for _, y, s, _ in val if s >= t and y == 1)
        fp = sum(1 for _, y, s, _ in val if s >= t and y == 0)
        fn = sum(1 for _, y, s, _ in val if s < t and y == 1)
        tn = sum(1 for _, y, s, _ in val if s < t and y == 0)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        if fpr < fb_fpr:
            fb_fpr, fb_t = fpr, t
        if fpr <= fpr_cap and f1 > best_f1:
            best_f1, best_t = f1, t
    if best_f1 < 0:  # cap unreachable -> lowest-FPR operating point
        best_t = fb_t
    # score the held-out test split with the tuned threshold, per benchmark
    out = {}
    for b in benches:
        rows = [(y, s, ms) for bb, y, s, ms in test if bb == b]
        if not rows:
            continue
        gold = [Decision.UNSAFE if y == 1 else Decision.SAFE for y, _, _ in rows]
        pred = [Decision.UNSAFE if s >= best_t else Decision.SAFE for _, s, _ in rows]
        m = compute_metrics(gold, pred, [ms for _, _, ms in rows]).to_dict()
        auc = roc_auc([y for y, _, _ in rows], [s for _, s, _ in rows])
        m["roc_auc"] = auc if auc is not None else (m["recall"] + 1 - m["fpr_on_benign"]) / 2
        out[b] = m
    return out, best_t


def member_metrics(preds: dict) -> dict[str, dict]:
    """Per-benchmark metrics for each individual guard, derived from its dumped predictions.

    Lets a single prediction pass populate both the ensemble rows AND each member's own
    scoreboard row (so slow local decoders are scored only once)."""
    out: dict[str, dict] = {}
    for guard, benches in preds.items():
        for b, rows in benches.items():
            gold = [Decision.UNSAFE if r[0] == 1 else Decision.SAFE for r in rows]
            pred = [Decision.UNSAFE if r[1] == 1 else Decision.SAFE for r in rows]
            lat = [r[3] for r in rows]
            m = compute_metrics(gold, pred, lat).to_dict()
            auc = roc_auc([r[0] for r in rows], [r[2] for r in rows])
            m["roc_auc"] = auc if auc is not None else (m["recall"] + 1 - m["fpr_on_benign"]) / 2
            out.setdefault(guard, {})[b] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge", nargs="*", default=None, help="ensemble names to merge into the scoreboard")
    ap.add_argument("--merge-members", action="store_true",
                    help="also merge each member's own metrics (from its predictions) into the scoreboard")
    args = ap.parse_args()

    preds = load_preds()
    print(f"loaded predictions for: {sorted(preds)}\n")
    results, summary = {}, {}
    for name, (members, strat, kw) in ENSEMBLES.items():
        r = eval_ensemble(members, strat, kw, preds)
        if not r:
            print(f"-- skip {name} (missing member predictions)")
            continue
        results[name] = r
        summary[name] = macro(r)

    # threshold-tuned weighted ensemble (tune on val split, score on disjoint test split).
    # use the richest member set available (prefer the one including the 1.7B decoder).
    tuned_members = ([_ENC, _SFT, _SFT17, _GRPO, _MOD] if _SFT17 in preds
                     else [_ENC, _SFT, _GRPO, _MOD])
    tuned_wts = [2, 1, 2, 1, 1] if _SFT17 in preds else [2, 1, 1, 1]
    tuned, thr = eval_tuned(tuned_members, preds, weights=tuned_wts)
    if tuned:
        results["ensemble-tuned"] = tuned
        summary["ensemble-tuned"] = macro(tuned)
        print(f"tuned weighted-ensemble ({len(tuned_members)} members) threshold = {thr}\n")

    # reference: gpt-5.2 macro from the scoreboard
    ref = {}
    if os.path.exists(RESULTS):
        blob = json.load(open(RESULTS))
        for g in ("openai-gpt-5.2-low", "openai-gpt-4o-mini", "decoder-grpo-0.6B", "encoder-distilbert"):
            vals = [blob["results"][b][g] for b in blob["results"] if g in blob["results"][b]]
            if vals:
                ref[g] = {k: round(sum(v[k] for v in vals) / len(vals), 4)
                          for k in ("f1", "roc_auc", "fpr_on_benign", "latency_p50_ms")}

    def _row(label, m):
        print(f"{label:<26} {m['f1']:>8.3f} {m['roc_auc']:>6.3f} "
              f"{m['fpr_on_benign']:>6.3f} {m['latency_p50_ms']:>8.0f}")

    print(f"{'guard/ensemble':<26} {'macroF1':>8} {'AUC':>6} {'FPR':>6} {'p50ms':>8}")
    for g, m in ref.items():
        _row(g, m)
    print("  " + "-" * 54)
    for name, m in sorted(summary.items(), key=lambda kv: -kv[1]["f1"]):
        _row(name, m)

    os.makedirs("outputs", exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

    to_merge = args.merge if args.merge is not None else _auto_pick(summary)
    members = member_metrics(preds) if args.merge_members else {}
    if to_merge or members:
        blob = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"meta": {}, "results": {}}
        for name in to_merge or []:
            for b, m in results.get(name, {}).items():
                blob.setdefault("results", {}).setdefault(b, {})[name] = m
        for name, benches in members.items():
            for b, m in benches.items():
                blob.setdefault("results", {}).setdefault(b, {})[name] = m
        fd, tmp = tempfile.mkstemp(dir="outputs", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:   # ensure flush+close before the atomic rename
            json.dump(blob, fh, indent=2)
        os.replace(tmp, RESULTS)
        print(f"merged into scoreboard: ensembles={to_merge or []} members={sorted(members)}")


def _auto_pick(summary: dict) -> list[str]:
    """Best macro-F1 ensemble + the lowest-FPR one (both useful stories)."""
    if not summary:
        return []
    best_f1 = max(summary, key=lambda n: summary[n]["f1"])
    low_fpr = min(summary, key=lambda n: summary[n]["fpr_on_benign"])
    return list(dict.fromkeys([best_f1, low_fpr]))


if __name__ == "__main__":
    main()
