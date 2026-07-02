#!/usr/bin/env python
"""Download the full-size ungated benchmark suite.

Writes one JSONL per benchmark to ``data/benchmarks/full`` by default so the
balanced cache in ``data/benchmarks`` remains untouched.

Usage:
    python scripts/data/download_full_benchmarks.py
    python scripts/data/download_full_benchmarks.py --out-dir data/benchmarks/full
    python scripts/data/download_full_benchmarks.py --benchmarks beavertails xstest
"""

from __future__ import annotations

import argparse

from agent_bouncer import data as D
from agent_bouncer.evaluation.benchmarks import BENCHMARKS, GATED_BENCHMARKS, class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="data/benchmarks/full",
        help="directory where the full benchmark JSONL files will be written",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(BENCHMARKS),
        help="subset of ungated benchmark names to download",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.benchmarks) - set(BENCHMARKS))
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {unknown}; known: {sorted(BENCHMARKS)}")

    out_dir = args.out_dir.rstrip("/")
    total_rows = 0
    for name in args.benchmarks:
        bench = BENCHMARKS[name]
        print(f"== loading {name} ({bench.hf_id}) ...")
        records = bench.loader()
        n_safe, n_unsafe = class_counts(records)
        path = f"{out_dir}/{name}.jsonl"
        n = D.write_jsonl(records, path)
        total_rows += n
        print(f"   wrote {n} rows ({n_safe} safe / {n_unsafe} unsafe) -> {path}")

    if GATED_BENCHMARKS:
        print("\n-- gated benchmarks not downloaded:")
        for name, desc in GATED_BENCHMARKS.items():
            print(f"   {name}: {desc}")

    print(f"\n== wrote {total_rows} rows across {len(args.benchmarks)} benchmark files to {out_dir}")


if __name__ == "__main__":
    main()
