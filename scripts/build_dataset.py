#!/usr/bin/env python
"""Build a training set with a chosen strategy (train/test split guaranteed disjoint).

Usage:
    python scripts/build_dataset.py --strategy balanced --name bt-balanced --sources beavertails
    python scripts/build_dataset.py --strategy mixed --name broad --sources beavertails toxicchat xstest
    python scripts/build_dataset.py --strategy over_refusal_aware --name low-fpr --sources beavertails
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402

import agent_bouncer  # noqa: E402,F401  (auto-loads .env → HF_TOKEN)
from agent_bouncer.training_sets import STRATEGIES, build_training_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--name", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[build] strategy={args.strategy} sources={args.sources} → {args.name}", flush=True)
    meta = build_training_set(
        args.strategy, args.sources, name=args.name,
        per_class=args.per_class, holdout_ratio=args.holdout, seed=args.seed,
    )
    print(f"[build] train={meta['n_train']} ({meta['train_safe']} safe / {meta['train_unsafe']} unsafe) "
          f"· held-out test={meta['n_test']} · leakage_checked={meta['leakage_checked']}", flush=True)
    print(f"[build] wrote {meta['train_path']}", flush=True)
    print("DATASET_BUILT=" + args.name)


if __name__ == "__main__":
    main()
