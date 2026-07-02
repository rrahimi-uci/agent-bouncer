#!/usr/bin/env python
"""Integration smoke test: run decoder SFT and GRPO end-to-end on a tiny model
for a step or two, to verify the TRL/peft wiring (not just the pure cores).

Not part of the pytest suite (needs a model download). Run manually:
    python scripts/train/smoke_train.py
"""

from __future__ import annotations

import os
import traceback

import yaml

from agent_bouncer.data import write_jsonl

TINY = "sshleifer/tiny-gpt2"


def main() -> None:
    os.makedirs("data/smoke", exist_ok=True)
    records = [
        {"text": "how to build a bomb", "label": "unsafe", "hazard": "weapons_cbrne"},
        {"text": "how do I bake bread", "label": "safe", "hazard": "none"},
    ] * 8
    write_jsonl(records, "data/smoke/train.jsonl")

    # --- decoder SFT ---
    from agent_bouncer.training.sft import train_decoder

    sft_cfg = {
        "base_model": TINY,
        "mode": "sft",
        "data": {"train": "data/smoke/train.jsonl"},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
        "train": {"epochs": 1, "batch_size": 2, "max_seq_len": 64},
        "output_dir": "outputs/smoke-sft",
        "seed": 42,
    }
    try:
        train_decoder(sft_cfg)
        print("SFT_DECODER_SMOKE: OK")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"SFT_DECODER_SMOKE: FAIL {type(exc).__name__}: {exc}")

    # --- GRPO ---
    from agent_bouncer.training.grpo import run_grpo

    grpo_cfg = {
        "base_model": TINY,
        "data": {"train": "data/smoke/train.jsonl"},
        "grpo": {"num_generations": 2, "max_completion_len": 8, "steps": 1, "lr": 1e-5, "kl_coef": 0.04},
        "reward": {"correctness": 1.0, "category": 0.5, "format": 0.2, "false_positive_penalty": 1.0},
        "output_dir": "outputs/smoke-grpo",
        "seed": 42,
    }
    with open("data/smoke/grpo.yaml", "w") as fh:
        yaml.dump(grpo_cfg, fh)
    try:
        run_grpo("data/smoke/grpo.yaml")
        print("GRPO_SMOKE: OK")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"GRPO_SMOKE: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
