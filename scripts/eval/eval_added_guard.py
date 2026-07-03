#!/usr/bin/env python
"""Score ONE extra guard on the already-cached benchmark subsets and merge it into
outputs/benchmark_results.json + re-render outputs/BENCHMARKS.md.

Lets you add a newly-trained guard (e.g. the GRPO/RL model) to the scoreboard
*without* re-running the paid OpenAI guards — it reuses the exact same cached
subsets in data/benchmarks/ that the main suite scored, so numbers stay comparable.

Usage:
    python scripts/eval/eval_added_guard.py --path outputs/grpo-qwen3-0.6b \
        --arch decoder --mode reasoning --name decoder-grpo-0.6B --params 0.6B
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from run_benchmarks import (  # noqa: E402
    CACHE_DIR,
    REPORT_MD,
    RESULTS_JSON,
    dump_prediction_rows,
    evaluate_guard,
    prediction_rows,
)

from agent_bouncer.data import read_jsonl
from agent_bouncer.evaluation.benchmarks import GATED_BENCHMARKS
from agent_bouncer.evaluation.report import render_benchmark_report
from agent_bouncer.models.decoder import DecoderGuard
from agent_bouncer.models.encoder import EncoderGuard

# Canonical scoreboard order (present guards are filtered from this).
CANON_ORDER = [
    "keyword-baseline", "encoder-distilbert", "encoder-modernbert-large",
    "decoder-sft-0.6B", "decoder-sft-1.7B", "decoder-grpo-0.6B",
    "openai-moderation", "openai-gpt-4o-mini",
    "openai-gpt-5.2-low", "openai-gpt-5.2-medium", "openai-gpt-5.2-high",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True)
    ap.add_argument("--arch", choices=["decoder", "encoder"], default="decoder")
    ap.add_argument("--mode", default="reasoning", help="decoder mode: sft|reasoning")
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", default="")
    ap.add_argument("--device", default="cpu", help="decoder device: cpu (reproducible) | mps | cuda")
    args = ap.parse_args()

    with open(RESULTS_JSON) as fh:
        blob = json.load(fh)
    meta, results = blob["meta"], blob["results"]

    if args.arch == "encoder":
        guard = EncoderGuard(args.path, name=args.name)
    else:
        # This process runs ONLY the decoder (no co-resident encoder), which sidesteps
        # the torch-threadpool deadlock. Default CPU for reproducible latency; --device mps
        # is much faster for larger decoders (latency then reflects the Apple GPU, not CPU).
        guard = DecoderGuard(args.path, mode=args.mode, name=args.name, device=args.device)

    scored: dict[str, dict] = {}
    for bench in results:
        path = f"{CACHE_DIR}/{bench}.jsonl"
        if not os.path.exists(path):
            print(f"!! no cached subset for {bench}; skipping")
            continue
        records = read_jsonl(path)
        m, verdicts = evaluate_guard(guard, records, workers=1)
        scored[bench] = m.to_dict()
        dump_prediction_rows(args.name, bench, prediction_rows(records, verdicts))
        print(f"  [{bench}] {args.name}: P={m.precision:.3f} R={m.recall:.3f} "
              f"F1={m.f1:.3f} FPR={m.fpr_on_benign:.3f} p50={m.latency_p50_ms:.0f}ms")

    # Merge-on-write: re-read the latest file and graft our cells onto it, then write
    # atomically (temp + rename). This never shrinks the file or clobbers a concurrent
    # writer — only adds this guard's results.
    latest = json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else blob
    for bench, metrics in scored.items():
        latest.setdefault("results", {}).setdefault(bench, {})[args.name] = metrics
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(RESULTS_JSON) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(latest, fh, indent=2)
    os.replace(tmp, RESULTS_JSON)
    meta, results = latest.get("meta", meta), latest["results"]

    # Rebuild the params map from what's present, adding the new guard.
    params = {g: "" for bench in results for g in results[bench]}
    from run_benchmarks import GUARD_PARAMS

    params.update(GUARD_PARAMS)
    params[args.name] = args.params
    order = [g for g in CANON_ORDER if any(g in results[b] for b in results)]

    report = render_benchmark_report(results, meta, params, guard_order=order, gated=GATED_BENCHMARKS)
    header = (
        "# Agent Bouncer — standard benchmark suite\n\n"
        f"All guards scored through the **same harness** on class-balanced subsets "
        f"(≤{blob.get('per_class', '?')}/class) of each benchmark. Positive class = `unsafe`. "
        "`fpr_on_benign` (over-blocking) is the headline usability metric.\n\n"
    )
    with open(REPORT_MD, "w") as fh:
        fh.write(header + report)
    print(f"\nmerged {args.name} into {RESULTS_JSON} and re-rendered {REPORT_MD}")


if __name__ == "__main__":
    main()
