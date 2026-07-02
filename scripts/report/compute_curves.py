#!/usr/bin/env python
"""Compute ROC / precision-recall curves + ROC-AUC for **every** guard in
outputs/benchmark_results.json, and write:

- outputs/curves.json — {benchmark: {guard: {auc, kind, roc, pr, n}}}
- the same `roc_auc` merged back into each cell of benchmark_results.json

Two cases, both honest:
- **Continuous score** (the fine-tuned encoder) → threshold-swept ROC/PR + real AUC.
- **Hard decision** (keyword · decoders · OpenAI) → a single operating point, so the ROC is
  ``[(0,0),(FPR,TPR),(1,1)]`` and ``AUC = (recall + 1 - FPR) / 2`` (exact, derived from the
  stored recall/FPR — no re-running the API/decoders).

Usage:
    python scripts/report/compute_curves.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402

from agent_bouncer.core.schema import Surface  # noqa: E402
from agent_bouncer.data import read_jsonl  # noqa: E402
from agent_bouncer.evaluation.curves import downsample, pr_curve, roc_auc, roc_curve  # noqa: E402

CACHE_DIR = "data/benchmarks"
RESULTS_JSON = "outputs/benchmark_results.json"
OUT = "outputs/curves.json"

# Guards that emit a continuous unsafe-probability → scored live for a real swept curve.
CONTINUOUS = {"encoder-distilbert": ("outputs/demo-encoder", "encoder")}


def _live_guard(name):
    path, kind = CONTINUOUS[name]
    if not os.path.isdir(path):
        return None
    from agent_bouncer.models.encoder import EncoderGuard
    return EncoderGuard(path, name=name)


def _derive_point(m: dict) -> dict:
    """Single-operating-point ROC/PR/AUC from a hard classifier's stored metrics."""
    tpr = float(m.get("recall", 0.0))
    fpr = float(m.get("fpr_on_benign", 0.0))
    prec = float(m.get("precision", 0.0))
    auc = (tpr + 1.0 - fpr) / 2.0
    return {
        "auc": auc, "kind": "point", "n": int(m.get("n", 0)),
        "roc": [[0.0, 0.0], [fpr, tpr], [1.0, 1.0]],
        "pr": [[0.0, 1.0], [tpr, prec], [1.0, 0.5]],
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    if not os.path.exists(RESULTS_JSON):
        raise SystemExit(f"{RESULTS_JSON} not found — run the benchmark suite first")
    blob = json.load(open(RESULTS_JSON))
    results = blob["results"]

    live_cache: dict[str, object] = {}
    curves: dict[str, dict] = {}
    for bench, guard_map in results.items():
        curves[bench] = {}
        cache = f"{CACHE_DIR}/{bench}.jsonl"
        recs = read_jsonl(cache) if os.path.exists(cache) else None
        labels = [r["label"] for r in recs] if recs else None
        for gname, m in guard_map.items():
            if gname in CONTINUOUS and labels is not None:
                if gname not in live_cache:
                    live_cache[gname] = _live_guard(gname)
                guard = live_cache[gname]
                if guard is not None:
                    scores = [guard.predict(r["text"], surface=Surface.USER_PROMPT).score for r in recs]
                    auc = roc_auc(labels, scores)
                    entry = {
                        "auc": auc, "kind": "swept", "n": len(recs),
                        "roc": [list(p) for p in downsample(roc_curve(labels, scores))],
                        "pr": [list(p) for p in downsample(pr_curve(labels, scores))],
                    }
                    curves[bench][gname] = entry
                    m["roc_auc"] = auc
                    print(f"  [{bench}] {gname}: AUC={auc:.3f} (swept)")
                    continue
            entry = _derive_point(m)          # hard-decision guard
            curves[bench][gname] = entry
            m["roc_auc"] = entry["auc"]
            print(f"  [{bench}] {gname}: AUC={entry['auc']:.3f} (point)")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(curves, fh, indent=2)
    # merge roc_auc back into the results file (atomic)
    fd, tmp = tempfile.mkstemp(dir="outputs", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, RESULTS_JSON)
    print(f"\nwrote {OUT} and merged roc_auc into {RESULTS_JSON}")


if __name__ == "__main__":
    main()
