#!/usr/bin/env python
"""Rebuild outputs/benchmark_results.json by parsing the printed result lines in the
run logs. Recovery utility: values are at the 3-dp precision the logs print (which is
exactly what the report renders), and it never invents numbers — only lines actually
emitted by a scoring run are used. `meta` (axis/description/class counts) is taken
from the benchmark registry + the cached subsets in data/benchmarks/.

Usage:
    python scripts/report/rebuild_results_from_logs.py outputs/logs/run_benchmarks.log \
        outputs/logs/eval_decoder.log [--per-class 100]
"""

from __future__ import annotations

import argparse
import json
import re

from agent_bouncer.data import read_jsonl
from agent_bouncer.evaluation.benchmarks import BENCHMARKS, class_counts

CACHE_DIR = "data/benchmarks"
RESULTS_JSON = "outputs/benchmark_results.json"

LINE = re.compile(
    r"\[([a-z_]+)\]\s+([\w.\-]+):\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+FPR=([\d.]+)\s+p50=([\d.]+)ms"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--per-class", type=int, default=100)
    args = ap.parse_args()

    results: dict[str, dict[str, dict]] = {}
    n_lines = 0
    for log in args.logs:
        with open(log) as fh:
            for line in fh:
                m = LINE.search(line)
                if not m:
                    continue
                bench, guard, p, r, f1, fpr, p50 = m.groups()
                results.setdefault(bench, {})[guard] = {
                    "precision": float(p), "recall": float(r), "f1": float(f1),
                    "fpr_on_benign": float(fpr), "latency_p50_ms": float(p50),
                }
                n_lines += 1

    meta: dict[str, dict] = {}
    for bench in results:
        b = BENCHMARKS[bench]
        recs = read_jsonl(f"{CACHE_DIR}/{bench}.jsonl")
        n_safe, n_unsafe = class_counts(recs)
        meta[bench] = {"axis": b.axis, "description": b.description, "n_safe": n_safe, "n_unsafe": n_unsafe}

    blob = {"per_class": args.per_class, "meta": meta, "results": results}
    with open(RESULTS_JSON, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"rebuilt {RESULTS_JSON}: {n_lines} result cells across {len(results)} benchmarks")
    for bench, gm in results.items():
        print(f"  {bench}: {len(gm)} guards")


if __name__ == "__main__":
    main()
