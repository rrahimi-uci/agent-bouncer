#!/usr/bin/env python
"""Prepare a balanced binary (safe/unsafe) demo dataset from BeaverTails (ungated).

Groups rows by prompt (a prompt is 'unsafe' if ANY of its responses is unsafe),
balances the classes, and writes train/val/test JSONL to data/demo/.

Usage:
    python scripts/prepare_beavertails_demo.py [--per-class 2500]
"""

from __future__ import annotations

import argparse
import random

from agent_bouncer.data import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=2500)
    parser.add_argument("--split", default="30k_train")
    parser.add_argument("--out", default="data/demo")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    print(f"loaded {len(ds)} rows; columns={ds.column_names}")

    unsafe_prompts: set[str] = set()
    safe_candidates: set[str] = set()
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        (safe_candidates if bool(row.get("is_safe", False)) else unsafe_prompts).add(prompt)
    safe_prompts = safe_candidates - unsafe_prompts  # drop ambiguous prompts

    rng = random.Random(42)
    unsafe = sorted(unsafe_prompts)
    safe = sorted(safe_prompts)
    rng.shuffle(unsafe)
    rng.shuffle(safe)
    n = min(len(unsafe), len(safe), args.per_class)
    print(f"unsafe={len(unsafe)} safe={len(safe)} -> {n} per class")

    records = [{"text": t, "label": "unsafe", "hazard": "non_violent_crimes", "source": "beavertails"} for t in unsafe[:n]]
    records += [{"text": t, "label": "safe", "hazard": "none", "source": "beavertails"} for t in safe[:n]]
    rng.shuffle(records)

    n_test = int(len(records) * 0.15)
    n_val = int(len(records) * 0.15)
    test, val, train = records[:n_test], records[n_test : n_test + n_val], records[n_test + n_val :]

    write_jsonl(train, f"{args.out}/train.jsonl")
    write_jsonl(val, f"{args.out}/validation.jsonl")
    write_jsonl(test, f"{args.out}/test.jsonl")
    print(f"wrote {len(train)} train / {len(val)} val / {len(test)} test to {args.out}/")


if __name__ == "__main__":
    main()
