#!/usr/bin/env python
"""Train a registered model (SFT / GRPO / DPO) to a versioned directory and record an
experiment. Works for any model in the registry — the existing Qwen3 SLMs plus
DeepSeek-R1-1.5B, SmolLM2-1.7B, and Gemma-1B.

Usage:
    python scripts/train/run_training.py --model smollm2-1.7b --technique sft --max-steps 30
    python scripts/train/run_training.py --model distilbert --technique sft --epochs 2
    python scripts/train/run_training.py --model deepseek-r1-1.5b --technique grpo --max-steps 40
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from agent_bouncer.models.registry import BASE_MODELS  # noqa: E402
from agent_bouncer.training.runner import train_and_record  # noqa: E402


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
    ap.add_argument("--model", required=True, choices=list(BASE_MODELS))
    ap.add_argument("--technique", default="sft", choices=["sft", "grpo", "dpo"])
    ap.add_argument("--train-data", default="data/demo/train.jsonl")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--lora-r", type=int)
    ap.add_argument("--lora-alpha", type=int)
    ap.add_argument("--lora-dropout", type=float)
    ap.add_argument("--max-seq-len", type=int)
    ap.add_argument("--max-length", type=int, help="encoder input length")
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--beta", type=float, help="DPO beta")
    ap.add_argument("--num-generations", type=int, help="GRPO rollouts per step")
    ap.add_argument("--max-completion-len", type=int, help="GRPO max completion tokens")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_dotenv()
    params: dict = {}
    for arg, dest in [("epochs", "epochs"), ("lr", "lr"), ("batch", "batch_size"),
                      ("max_steps", "max_steps"), ("lora_r", "lora_r"), ("lora_alpha", "lora_alpha"),
                      ("lora_dropout", "lora_dropout"), ("max_seq_len", "max_seq_len"),
                      ("max_length", "max_length"), ("grad_accum", "grad_accum"), ("beta", "beta"),
                      ("num_generations", "num_generations"),
                      ("max_completion_len", "max_completion_len")]:
        val = getattr(args, arg)
        if val is not None:
            params[dest] = val
    if args.bf16:
        params["bf16"] = True

    exp = train_and_record(args.model, args.technique, train_data=args.train_data,
                           params=params, seed=args.seed)
    print("EXPERIMENT_ID=" + exp["id"])


if __name__ == "__main__":
    main()
