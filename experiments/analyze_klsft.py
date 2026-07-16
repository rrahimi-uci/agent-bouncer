#!/usr/bin/env python
"""Compare KL-regularized SFT against vanilla SFT and the base, on the same panel/splits as Act I.

Reads:
  --committed  artifacts/paper_a_sft_v2/scores/scores.parquet   (base + vanilla SFT, canonical env)
  --klsft-dir  dir of klsft_scores_<mk>.parquet (from run_klsft_sweep; has a `kl_beta` column,
               condition='sft'; kl_beta==0 is the in-env vanilla cross-check)

Metric: macro-AP = mean over the benchmarks in a regime of guard_research.metrics.average_precision,
using the SAME split->regime map as analyze_paper_a_sft (represented=id_test, transfer=transfer_test).
Per checkpoint we report base, vanilla SFT (committed + in-env beta0), and KL at each beta>0, with the
transfer delta vs vanilla -- the anti-forgetting question.

Usage: python experiments/analyze_klsft.py --klsft-dir /tmp/klsft_results [--out /tmp/klsft_summary.json]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REGIME = {"represented": "id_test", "transfer": "transfer_test"}
ORDER = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]
PRETTY = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
          "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}


def macro_ap(frame, split):
    """mean over that split's sources of average_precision(score_raw, gold)."""
    from guard_research.metrics import average_precision as AP
    sub = frame[frame.split == split]
    aps = []
    for s in sorted(sub.source.unique()):
        g = sub[sub.source == s]
        if g.gold.nunique() < 2:
            continue
        aps.append(AP(g.score_raw.values, g.gold.values.astype(int)))
    return float(sum(aps) / len(aps)) if aps else float("nan")


def seed_mean_macro(frame, split, seeds):
    vals = [macro_ap(frame[frame.seed == sd], split) for sd in seeds]
    vals = [v for v in vals if v == v]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--committed", default="artifacts/paper_a_sft_v2/scores/scores.parquet")
    ap.add_argument("--klsft-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    import numpy as np
    import pandas as pd

    comm = pd.read_parquet(args.committed,
                           columns=["split", "source", "gold", "model_key", "condition", "seed", "score_raw"])
    kl_files = sorted(glob.glob(os.path.join(args.klsft_dir, "klsft_scores_*.parquet")))
    if not kl_files:
        print(f"[analyze] no klsft parquets in {args.klsft_dir}", file=sys.stderr)
        return 2
    kl = pd.concat([pd.read_parquet(f) for f in kl_files], ignore_index=True)
    print(f"[analyze] committed rows={len(comm)} | klsft rows={len(kl)} from {len(kl_files)} files "
          f"| checkpoints={sorted(kl.model_key.unique())} | betas={sorted(kl.kl_beta.unique())}")

    rows = []
    for mk in ORDER:
        if mk not in kl.model_key.unique():
            continue
        b = comm[(comm.model_key == mk) & (comm.condition == "base")]
        s = comm[(comm.model_key == mk) & (comm.condition == "sft")]
        s_seeds = sorted(s.seed.unique())
        rec = {"model_key": mk, "pretty": PRETTY[mk]}
        for reg, split in REGIME.items():
            rec[f"base_{reg}"] = macro_ap(b, split)
            rec[f"sft_committed_{reg}"] = seed_mean_macro(s, split, s_seeds)
        kmk = kl[kl.model_key == mk]
        for beta in sorted(kmk.kl_beta.unique()):
            kb = kmk[kmk.kl_beta == beta]
            kseeds = sorted(kb.seed.unique())
            tag = "sft_inenv" if beta == 0 else f"kl{beta:g}"
            for reg, split in REGIME.items():
                rec[f"{tag}_{reg}"] = seed_mean_macro(kb, split, kseeds)
        rows.append(rec)

    # print a readable table: transfer + represented, KL vs in-env vanilla
    def f3(x):
        return "  n/a" if (x is None or (isinstance(x, float) and x != x)) else f"{x:.3f}"
    def d3(a, b):
        return " n/a " if (a is None or b is None or a != a or b != b) else f"{a-b:+.3f}"
    print("\n=== KL-SFT vs vanilla vs base (macro-AP; delta_transfer vs in-env vanilla beta0) ===")
    for r in rows:
        print(f"\n{r['pretty']}")
        v_t = r.get("sft_inenv_transfer"); v_r = r.get("sft_inenv_represented")
        print(f"  base       : transfer={f3(r.get('base_transfer'))}  repr={f3(r.get('base_represented'))}")
        print(f"  SFT (b=0)  : transfer={f3(v_t)}  repr={f3(v_r)}   [in-env vanilla; "
              f"committed SFT transfer={f3(r.get('sft_committed_transfer'))}]")
        for beta in (0.5, 1.0):
            kt = r.get(f"kl{beta:g}_transfer"); kr = r.get(f"kl{beta:g}_represented")
            if kt is not None and kt == kt:
                print(f"  KL b={beta:<3}: transfer={f3(kt)}  repr={f3(kr)}   "
                      f"dtransfer_vs_SFT={d3(kt, v_t)}  vs_base={d3(kt, r.get('base_transfer'))}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"\n[analyze] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
