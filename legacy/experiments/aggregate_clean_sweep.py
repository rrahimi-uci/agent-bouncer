#!/usr/bin/env python
"""Aggregate the clean, source-family-disjoint multi-seed sweep into the paper's
tables. Reads summary_{base}-clean-{obj}-s{seed}.json (emitted by guard_eval_pipeline.py,
tie-aware AUPRC after the metric correction) and the HPO best-config JSONs, and prints:

  * tab:slm-sweep   -- six bases under the identical SFT recipe: base->tuned in-house
                       and novel AUPRC, Delta_novel, matched-FPR@0.10 F1, latency,
                       as seed-mean with [min,max] over seeds.
  * tab:rl-vs-ft    -- same bases x {SFT,DPO,GRPO}: in-house and novel AUPRC (seed-mean),
                       novel_base, with the DPO>SFT-novel bold rule applied.
  * tab:hpo         -- per-objective HPO on the primary base: best dev/novel AUPRC + params.

Usage:  RESULT_DIR=notebooks/outputs/nb-smollm3-guard python3 legacy/experiments/aggregate_clean_sweep.py
        (RESULT_DIR may also hold hpo_best_{sft,dpo,grpo}_*.json under an hpo/ subdir or flat.)

It is read-only and defensive: any (base,obj) with missing seeds is reported with the
seeds it has, and the seed count is printed so partial runs are obvious.
"""
import os, json, glob, re, math
from collections import defaultdict

RD = os.environ.get("RESULT_DIR", "notebooks/outputs/nb-smollm3-guard")

# display order + labels; * reasoning-distilled, + primary base
BASES = [
    ("smollm3-3b",       "SmolLM3-3B$^\\dagger$"),
    ("qwen2.5-1.5b",     "Qwen2.5-1.5B"),
    ("smollm2-1.7b",     "SmolLM2-1.7B"),
    ("deepseek-r1-1.5b", "DeepSeek-R1-1.5B$^\\ast$"),
    ("qwen3-4b",         "Qwen3-4B"),
    ("qwen3-8b",         "Qwen3-8B"),
]
OBJS = ["sft", "dpo", "grpo"]

def load_all():
    """tag -> summary dict, for every summary_*-clean-*.json under RD (+ sm3 alias)."""
    out = {}
    for p in glob.glob(os.path.join(RD, "summary_*clean-*.json")):
        name = os.path.basename(p)
        if "expguard" in name or "name_fairness" in name:
            continue
        try:
            out[name] = json.load(open(p))
        except Exception as e:
            print(f"  !! skip {name}: {e}")
    return out

def parse_tag(fn):
    # summary_<base>-clean-<obj>-s<seed>.json (primary base uses the joined tag "sm3clean-...")
    m = re.match(r"summary_(.+?)-?clean-(sft|dpo|grpo)-s(\d+)\.json", fn)
    if not m:
        return None
    base, obj, seed = m.group(1), m.group(2), int(m.group(3))
    if base == "sm3":
        base = "smollm3-3b"
    return base, obj, seed

def getf(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return {"mean": m, "min": min(vals), "max": max(vals), "n": len(vals),
            "std": (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0}

def collect():
    """(base,obj) -> dict of stat() over seeds for each metric."""
    raw = load_all()
    by = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for fn, d in raw.items():
        t = parse_tag(fn)
        if not t:
            continue
        base, obj, seed = t
        seeds[(base, obj)].add(seed)
        by[(base, obj)]["inhouse_tuned"].append(getf(d, "base_vs_tuned_inhouse", "guard_auprc"))
        by[(base, obj)]["inhouse_base"].append(getf(d, "base_vs_tuned_inhouse", "base_auprc"))
        by[(base, obj)]["novel_tuned"].append(getf(d, "novel_heldout", "guard", "auprc"))
        by[(base, obj)]["novel_base"].append(getf(d, "novel_heldout", "base", "auprc"))
        by[(base, obj)]["mf1"].append(getf(d, "matched_fpr_and_auprc", "guard", "f1"))
        by[(base, obj)]["lat"].append(getf(d, "latency_batch1", "p50"))
    agg = {}
    for k, metrics in by.items():
        agg[k] = {m: stat(v) for m, v in metrics.items()}
        agg[k]["_seeds"] = sorted(seeds[k])
    return agg

def f3(x):
    return "---" if x is None else f"{x:.3f}"

def mm(s):
    """seed-mean [min,max] compact."""
    if s is None:
        return "---"
    if s["n"] == 1:
        return f"{s['mean']:.3f}"
    return f"{s['mean']:.3f}\\,[{s['min']:.3f},{s['max']:.3f}]"

def main():
    agg = collect()
    print(f"# RESULT_DIR = {RD}")
    print(f"# (base,obj) present: {sorted((b,o) for (b,o) in agg)}\n")

    print("=" * 70)
    print("tab:slm-sweep  (SFT only; base -> tuned; seed-mean [min,max])")
    print("=" * 70)
    print("base | seeds | inhouse_base->tuned | novel_base | novel_tuned | dNovel | mF1 | lat_ms")
    for bkey, blabel in BASES:
        s = agg.get((bkey, "sft"))
        if not s:
            print(f"{blabel:24s} | (no sft runs yet)")
            continue
        nb = getf(s, "novel_base", "mean")
        nt = getf(s, "novel_tuned", "mean")
        dnov = None if (nb is None or nt is None) else nt - nb
        print(f"{blabel:24s} | {s['_seeds']} | "
              f"{f3(getf(s,'inhouse_base','mean'))}->{mm(s.get('inhouse_tuned'))} | "
              f"{f3(nb)} | {mm(s.get('novel_tuned'))} | {f3(dnov)} | "
              f"{mm(s.get('mf1'))} | {f3(getf(s,'lat','mean'))}")

    print("\n" + "=" * 70)
    print("tab:rl-vs-ft  (obj x base; seed-mean [min,max]; bold DPO>SFT novel)")
    print("=" * 70)
    for bkey, blabel in BASES:
        nb = None
        for o in OBJS:
            s = agg.get((bkey, o))
            if s and getf(s, "novel_base", "mean") is not None:
                nb = getf(s, "novel_base", "mean"); break
        sft_nov = getf(agg.get((bkey, "sft"), {}), "novel_tuned", "mean")
        print(f"\n{blabel}  novel_base={f3(nb)}")
        for o in OBJS:
            s = agg.get((bkey, o))
            if not s:
                print(f"  {o.upper():5s} | ---")
                continue
            nt = getf(s, "novel_tuned", "mean")
            bold = (o == "dpo" and nt is not None and sft_nov is not None and nt > sft_nov)
            tag = "  <-- DPO>SFT novel" if bold else ""
            print(f"  {o.upper():5s} | seeds={s['_seeds']} inhouse={mm(s.get('inhouse_tuned'))} "
                  f"novel={mm(s.get('novel_tuned'))}{tag}")

    print("\n" + "=" * 70)
    print("tab:hpo  (primary base per-objective HPO best config)")
    print("=" * 70)
    for o in OBJS:
        cands = glob.glob(os.path.join(RD, f"hpo_best_{o}_*.json")) + \
                glob.glob(os.path.join(RD, "hpo", f"hpo_best_{o}_*.json")) + \
                glob.glob(os.path.join(RD, "..", "..", "..", "outputs", "hpo", f"hpo_best_{o}_*.json"))
        if not cands:
            print(f"  {o.upper():5s} | (no hpo json found)")
            continue
        d = json.load(open(cands[0]))
        print(f"  {o.upper():5s} | dev={f3(d.get('best_dev_auprc') or d.get('dev_auprc'))} "
              f"novel={f3(d.get('best_novel_auprc') or d.get('novel_auprc') or d.get('novel_tracked'))} "
              f"params={d.get('best_params') or d.get('params')}")

if __name__ == "__main__":
    main()
