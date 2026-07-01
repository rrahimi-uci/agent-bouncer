#!/usr/bin/env python
"""Download and unify the training datasets defined in configs/data.yaml.

Usage:
    python scripts/download_data.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    print(f"Loaded config: {len(cfg.get('sources', []))} source(s) -> {cfg.get('output_dir')}")
    # TODO: for each enabled source, datasets.load_dataset(...) then
    #       agent_bouncer.data.unify_to_taxonomy(...) and write JSONL to output_dir.
    raise SystemExit("TODO: implement dataset download + unification (see docs/datasets.md).")


if __name__ == "__main__":
    main()
