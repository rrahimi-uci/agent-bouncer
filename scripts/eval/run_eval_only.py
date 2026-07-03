#!/usr/bin/env python
"""Evaluation-only: score an **already-saved** model on benchmarks — no training.

Resolves the model from the model store (``--model-id``) or a raw ``--path``, scores it
(leakage-guarded), records an ``eval`` experiment, and refreshes the saved record's metrics.

Usage:
    python scripts/eval/run_eval_only.py --model-id qwen3-0.6b-sft-20260702-1200
    python scripts/eval/run_eval_only.py --path outputs/models/qwen3-0.6b/v1 --arch decoder \
        --technique sft --benchmarks beavertails xstest --device mps
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402

import agent_bouncer  # noqa: E402,F401  (auto-loads .env)
from agent_bouncer.evaluation.benchmarks import BENCHMARKS  # noqa: E402
from agent_bouncer.training.runner import eval_only  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=None, help="saved-model id in the model store")
    ap.add_argument("--path", default=None, help="raw model directory (uploaded / on disk)")
    ap.add_argument("--arch", default="decoder", choices=["encoder", "decoder"])
    ap.add_argument("--technique", default="sft")
    ap.add_argument("--model-key", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()

    benchmarks = args.benchmarks or list(BENCHMARKS)
    res = eval_only(
        benchmarks=benchmarks, model_id=args.model_id, path=args.path,
        arch=args.arch, technique=args.technique, model_key=args.model_key,
        per_class=args.per_class, device=args.device,
    )
    macro = res.get("macro", {})
    print(f"[eval-only] {res['model_key']} · macro F1={macro.get('f1')} "
          f"FPR={macro.get('fpr_on_benign')}", flush=True)
    if res.get("experiment_id"):
        print("EVAL_EXPERIMENT_ID=" + res["experiment_id"])


if __name__ == "__main__":
    main()
