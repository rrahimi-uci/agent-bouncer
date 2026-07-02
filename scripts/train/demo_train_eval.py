#!/usr/bin/env python
"""End-to-end demo: fine-tune an encoder guard and check it beats the baseline.

Trains Regime A (encoder) on data/demo, then evaluates the trained EncoderGuard
and the reference KeywordGuard on the held-out test set through the same harness,
and reports whether our approach moved the baseline.

Usage:
    python scripts/train/demo_train_eval.py [--config configs/model/demo_encoder.yaml]
"""

from __future__ import annotations

import argparse
import json

import yaml

from agent_bouncer.core.guard import KeywordGuard
from agent_bouncer.data import read_jsonl
from agent_bouncer.evaluation.harness import evaluate
from agent_bouncer.evaluation.report import render_results_table
from agent_bouncer.models.encoder import EncoderGuard
from agent_bouncer.training.sft import train_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/demo_encoder.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = train_encoder(cfg)

    test = read_jsonl(cfg["data"]["test"])
    keyword = KeywordGuard()
    encoder = EncoderGuard(out_dir, name="agent-bouncer-encoder")

    m_keyword = evaluate(keyword, test, run_name="keyword-baseline")
    m_encoder = evaluate(encoder, test, run_name="agent-bouncer-encoder")

    rows = [
        {"guard": keyword.name, "params": "0", **m_keyword.to_dict()},
        {"guard": encoder.name, "params": "66M", **m_encoder.to_dict()},
    ]
    print("\n" + render_results_table(rows) + "\n")

    results = {"test_n": len(test), "keyword": m_keyword.to_dict(), "encoder": m_encoder.to_dict()}
    with open("outputs/demo_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    moved = m_encoder.f1 > m_keyword.f1
    print(f"F1: baseline {m_keyword.f1:.3f} -> encoder {m_encoder.f1:.3f}")
    print(f"Recall: baseline {m_keyword.recall:.3f} -> encoder {m_encoder.recall:.3f}")
    print(f"APPROACH MOVED BASELINE: {'YES' if moved else 'NO'}")


if __name__ == "__main__":
    main()
