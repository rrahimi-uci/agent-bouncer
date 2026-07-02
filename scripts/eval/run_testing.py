#!/usr/bin/env python
"""Test a trained model version against the benchmark data — with **train/test leakage
guards** — and record an evaluation experiment (P/R/F1, ROC-AUC, latency, throughput, P90).

Usage:
    python scripts/eval/run_testing.py --exp smollm2-1.7b-sft-20260702-1200 --per-class 40
    python scripts/eval/run_testing.py --exp <id> --benchmarks beavertails xstest --device mps
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from agent_bouncer.training.runner import evaluate_and_record  # noqa: E402


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, help="training experiment id to evaluate")
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    load_dotenv()
    exp = evaluate_and_record(args.exp, benchmarks=args.benchmarks,
                              per_class=args.per_class, device=args.device)
    print("EVAL_EXPERIMENT_ID=" + exp["id"])


if __name__ == "__main__":
    main()
