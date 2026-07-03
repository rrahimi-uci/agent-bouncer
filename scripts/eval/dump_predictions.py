#!/usr/bin/env python
"""Dump a guard's PER-SAMPLE predictions on the cached benchmark subsets, so ensembles
can be combined offline (score each guard once, then explore combinations for free).

Run one guard per process (avoids co-loading a BERT encoder + a Qwen decoder, which
deadlocks the tokenizer threadpool). Output: outputs/predictions/<guard>.json =
{benchmark: [[y, u, s, ms], ...]} where y=gold-unsafe, u=pred-unsafe, s=score, ms=latency.
Sample order matches the cached subset, so guards align by index.

Usage:
    python scripts/eval/dump_predictions.py --guard encoder-distilbert
    python scripts/eval/dump_predictions.py --guard decoder-sft-0.6B --device mps
    python scripts/eval/dump_predictions.py --guard openai-moderation --workers 8
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import agent_bouncer  # noqa: E402,F401  (auto-loads .env)
from agent_bouncer.core.guard import KeywordGuard  # noqa: E402
from agent_bouncer.core.schema import Decision, Surface  # noqa: E402
from agent_bouncer.data import read_jsonl  # noqa: E402

CACHE = "data/benchmarks"
OUT = "outputs/predictions"


def build_guard(name: str, device: str, path: str | None = None, arch: str = "decoder",
                mode: str = "sft"):
    if name == "keyword-baseline":
        return KeywordGuard()
    # Explicit --path wins: score a real trained checkpoint under the given --name.
    if path:
        if arch == "encoder":
            from agent_bouncer.models.encoder import EncoderGuard
            return EncoderGuard(path, name=name)
        from agent_bouncer.models.decoder import DecoderGuard
        return DecoderGuard(path, mode=mode, name=name, device=device)
    if name == "encoder-distilbert":
        from agent_bouncer.models.encoder import EncoderGuard
        return EncoderGuard("outputs/demo-encoder", name=name)
    decoders = {"decoder-sft-0.6B": ("outputs/demo-decoder-sft", "sft"),
                "decoder-grpo-0.6B": ("outputs/grpo-qwen3-0.6b", "reasoning"),
                "decoder-sft-1.7B": ("outputs/decoder-sft-Qwen3-1.7B", "sft")}
    if name in decoders:
        from agent_bouncer.models.decoder import DecoderGuard
        path, mode = decoders[name]
        return DecoderGuard(path, mode=mode, name=name, device=device)
    if name.startswith("openai-"):
        import re

        from agent_bouncer.evaluation.openai_guards import OpenAIChatGuard, OpenAIModerationGuard
        if name == "openai-moderation":
            return OpenAIModerationGuard()
        if name == "openai-gpt-4o-mini":
            return OpenAIChatGuard("gpt-4o-mini")
        m = re.match(r"^openai-(gpt-[\d.]+)-(low|medium|high)$", name)
        if m:
            return OpenAIChatGuard(m.group(1), reasoning_effort=m.group(2))
    raise ValueError(f"unknown guard {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--path", default=None, help="explicit checkpoint path (score a real trained model)")
    ap.add_argument("--arch", choices=["decoder", "encoder"], default="decoder")
    ap.add_argument("--mode", default="sft", help="decoder mode: sft|reasoning")
    args = ap.parse_args()

    guard = build_guard(args.guard, args.device, path=args.path, arch=args.arch, mode=args.mode)
    benches = sorted(f[:-6] for f in os.listdir(CACHE) if f.endswith(".jsonl"))
    out: dict[str, list] = {}
    for bench in benches:
        recs = read_jsonl(f"{CACHE}/{bench}.jsonl")
        texts = [r["text"] for r in recs]
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                vs = list(ex.map(lambda t: guard.predict(t, surface=Surface.USER_PROMPT), texts))
        else:
            vs = [guard.predict(t, surface=Surface.USER_PROMPT) for t in texts]
        rows = []
        for r, v in zip(recs, vs, strict=True):
            y = 1 if r["label"] == "unsafe" else 0
            u = 1 if v.decision == Decision.UNSAFE else 0
            rows.append([y, u, round(float(v.score), 4), round(float(v.latency_ms or 0.0), 3)])
        out[bench] = rows
        print(f"  [{bench}] {args.guard}: {len(rows)} preds", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{args.guard}.json", "w") as fh:
        json.dump(out, fh)
    print(f"wrote {OUT}/{args.guard}.json")


if __name__ == "__main__":
    main()
