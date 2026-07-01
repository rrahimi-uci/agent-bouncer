#!/usr/bin/env python
"""Download and unify the training datasets defined in configs/data.yaml.

Writes one JSONL per source plus combined train/validation splits, all normalized
to the unified taxonomy. Requires the train extra: pip install -e '.[train]'.

Usage:
    python scripts/download_data.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse

import yaml

from agent_bouncer import data as D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = cfg.get("output_dir", "data/")
    seed = cfg.get("seed", 42)
    val_ratio = cfg.get("splits", {}).get("validation", 0.1)

    all_records: list[dict] = []
    for source in cfg.get("sources", []) + cfg.get("benign", []):
        name = source["name"]
        if not source.get("enabled", True):
            print(f"-- skip {name} (disabled)")
            continue
        loader = D.LOADERS.get(name)
        if loader is None:
            print(f"-- skip {name} (no loader)")
            continue
        print(f"== loading {name} ({source.get('hf_id')}) ...")
        records = loader()
        n = D.write_jsonl(records, f"{out_dir.rstrip('/')}/{name}.jsonl")
        print(f"   {n} examples -> {out_dir.rstrip('/')}/{name}.jsonl")
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No records produced — check dataset access (HF_TOKEN?) and field names.")

    train, val = D.train_val_split(all_records, val_ratio=val_ratio, seed=seed)
    D.write_jsonl(train, f"{out_dir.rstrip('/')}/train.jsonl")
    D.write_jsonl(val, f"{out_dir.rstrip('/')}/validation.jsonl")
    print(f"== wrote {len(train)} train / {len(val)} validation to {out_dir}")


if __name__ == "__main__":
    main()
