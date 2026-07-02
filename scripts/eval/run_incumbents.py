#!/usr/bin/env python
"""Compare Agent Bouncer against incumbents on a held-out subset and report
precision/recall. Runs whatever is reachable given the credentials in .env:

- always: keyword baseline + our trained encoder/decoder (if present in outputs/)
- if OPENAI_API_KEY set: OpenAI Moderation API + a mini chat model
- if HF_TOKEN set (and model licenses accepted): Llama Guard 3 / ShieldGemma

Missing credentials are skipped with a clear message — never fabricated.

Usage:
    python scripts/eval/run_incumbents.py [--limit 150] [--chat-model gpt-4o-mini]
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

COLUMNS = [
    ("guard", "Guard"),
    ("params", "Params"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("fpr_on_benign", "FPR@benign"),
    ("latency_p50_ms", "p50 ms"),
]


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from .env into the environment (no logging)."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


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


def build_guards(chat_model: str) -> list[tuple[str, str, object]]:
    guards: list[tuple[str, str, object]] = [("keyword-baseline", "0", KeywordGuard())]

    if os.path.isdir("outputs/demo-encoder"):
        guards.append(
            ("encoder-distilbert", "66M", EncoderGuard("outputs/demo-encoder", name="encoder-distilbert"))
        )
    if os.path.isdir("outputs/demo-decoder-sft"):
        guards.append(
            ("decoder-sft-0.6B", "0.6B",
             DecoderGuard("outputs/demo-decoder-sft", mode="sft", name="decoder-sft-0.6B"))
        )

    if os.environ.get("OPENAI_API_KEY"):
        from agent_bouncer.evaluation.openai_guards import OpenAIChatGuard, OpenAIModerationGuard

        guards.append(("openai-moderation", "api", OpenAIModerationGuard()))
        guards.append((f"openai-{chat_model}", chat_model, OpenAIChatGuard(chat_model)))
    else:
        print("!! OPENAI_API_KEY not set -> skipping OpenAI Moderation + mini chat guards")

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        from agent_bouncer.evaluation.baselines import load_llama_guard, load_shieldgemma

        for name, params, loader in [
            ("llama-guard-3-1b", "1B", load_llama_guard),
            ("shieldgemma-2b", "2B", load_shieldgemma),
        ]:
            try:
                guards.append((name, params, loader()))
            except Exception as exc:  # noqa: BLE001 - gated/license/network
                print(f"!! {name} unavailable ({type(exc).__name__}); skipping")
    else:
        print("!! HF_TOKEN not set -> skipping gated incumbents (Llama Guard / ShieldGemma)")

    return guards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=150, help="total eval examples (balanced)")
    ap.add_argument("--chat-model", default="gpt-4o-mini")
    args = ap.parse_args()

    load_dotenv()
    test = balanced_subset(read_jsonl("data/demo/test.jsonl"), args.limit // 2)
    print(f"eval subset: {len(test)} examples\n")

    rows, results = [], {}
    for name, params, guard in build_guards(args.chat_model):
        try:
            m = evaluate(guard, test, run_name=name)
        except Exception as exc:  # noqa: BLE001 - one bad guard shouldn't kill the run
            print(f"!! {name} eval failed ({type(exc).__name__}: {exc}); skipping")
            continue
        rows.append({"guard": name, "params": params, **m.to_dict()})
        results[name] = m.to_dict()
        print(f"  {name}: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} FPR={m.fpr_on_benign:.3f}")

    print("\n" + render_results_table(rows, COLUMNS) + "\n")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/incumbents_results.json", "w") as fh:
        json.dump({"test_n": len(test), "results": results}, fh, indent=2)
    print("wrote outputs/incumbents_results.json")


if __name__ == "__main__":
    main()
