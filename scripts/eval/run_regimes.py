#!/usr/bin/env python
"""Train a decoder (Qwen3-*) SFT regime and compare all learned regimes to the
baseline on a fixed, balanced held-out subset — through the same harness.

Results merge into outputs/size_sweep.json (keyed by guard name), so running this
across model sizes builds one combined size-comparison table.

Usage:
    python scripts/eval/run_regimes.py --base Qwen/Qwen3-0.6B --decoder-params 0.6B
    python scripts/eval/run_regimes.py --base Qwen/Qwen3-1.7B --decoder-params 1.7B --bf16
    python scripts/eval/run_regimes.py --base Qwen/Qwen3-4B --decoder-params 4B --bf16 --batch 4
"""

from __future__ import annotations

import argparse
import json
import os
import random

from agent_bouncer.core.guard import KeywordGuard
from agent_bouncer.data import read_jsonl
from agent_bouncer.evaluation.harness import evaluate
from agent_bouncer.evaluation.report import render_results_table
from agent_bouncer.models.decoder import DecoderGuard
from agent_bouncer.models.encoder import EncoderGuard
from agent_bouncer.training.sft import train_decoder

SWEEP_PATH = "outputs/size_sweep.json"


def balanced_subset(records: list[dict], per_class: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    safe = [r for r in records if r["label"] == "safe"]
    unsafe = [r for r in records if r["label"] == "unsafe"]
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    k = min(per_class, len(safe), len(unsafe))
    subset = safe[:k] + unsafe[:k]
    rng.shuffle(subset)
    return subset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--decoder-params", default="0.6B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--test-per-class", type=int, default=125)
    args = ap.parse_args()

    short = args.base.split("/")[-1]
    cfg = {
        "base_model": args.base,
        "mode": "sft",
        "data": {"train": "data/demo/train.jsonl"},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "grad_accum": args.grad_accum,
            "lr": 2e-4,
            "max_seq_len": 256,
            "bf16": args.bf16,
        },
        "output_dir": f"outputs/decoder-sft-{short}",
        "seed": 42,
    }
    decoder_dir = train_decoder(cfg)

    test = balanced_subset(read_jsonl("data/demo/test.jsonl"), args.test_per_class)
    print(f"eval subset: {len(test)} examples")

    guards = [
        ("keyword-baseline", "0", KeywordGuard()),
        ("encoder-distilbert", "66M", EncoderGuard("outputs/demo-encoder", name="encoder-distilbert")),
        (
            f"decoder-sft-{short}",
            args.decoder_params,
            DecoderGuard(decoder_dir, mode="sft", name=f"decoder-sft-{short}"),
        ),
    ]

    data = json.load(open(SWEEP_PATH)) if os.path.exists(SWEEP_PATH) else {"test_n": len(test), "results": {}}
    for name, params, guard in guards:
        m = evaluate(guard, test, run_name=name)
        data["results"][name] = {"params": params, **m.to_dict()}
        print(f"  {name}: F1={m.f1:.3f} recall={m.recall:.3f} fpr_benign={m.fpr_on_benign:.3f}")

    os.makedirs("outputs", exist_ok=True)
    with open(SWEEP_PATH, "w") as fh:
        json.dump(data, fh, indent=2)

    rows = [{"guard": k, **v} for k, v in data["results"].items()]
    print("\n" + render_results_table(rows) + "\n")
    print(f"wrote {SWEEP_PATH}")


if __name__ == "__main__":
    main()
