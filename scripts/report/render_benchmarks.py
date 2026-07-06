#!/usr/bin/env python
"""Re-render outputs/BENCHMARKS.md from outputs/benchmark_results.json.

Pure formatting (no eval, no network) — handy after merging an extra guard, or to
tweak the table without re-scoring. Parameter counts and guard order are canonical.
"""

from __future__ import annotations

import argparse
import json

from agent_bouncer.evaluation.benchmarks import GATED_BENCHMARKS
from agent_bouncer.evaluation.report import render_benchmark_report

CANON_ORDER = [
    "keyword-baseline", "encoder-distilbert", "encoder-modernbert-large",
    "decoder-sft-0.6B", "decoder-sft-1.7B", "decoder-grpo-0.6B",
    "openai-moderation", "openai-gpt-4o-mini",
    "openai-gpt-5.4-mini", "openai-gpt-5.2-low", "openai-gpt-5.2-medium", "openai-gpt-5.2-high",
]
PARAMS = {
    "keyword-baseline": "0", "encoder-distilbert": "66M", "encoder-modernbert-large": "395M",
    "decoder-sft-0.6B": "0.6B", "decoder-sft-1.7B": "1.7B", "decoder-grpo-0.6B": "0.6B",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="outputs/benchmark_results.json")
    ap.add_argument("--out", default="outputs/BENCHMARKS.md")
    args = ap.parse_args()

    blob = json.load(open(args.results))
    meta, results = blob["meta"], blob["results"]

    params = dict(PARAMS)
    for bench in results:
        for g in results[bench]:
            params.setdefault(g, "api" if g.startswith("openai-") else "")

    present = {g for b in results for g in results[b]}
    # canonical order first, then any other scored guards (ensembles, custom names) so nothing is dropped
    order = [g for g in CANON_ORDER if g in present] + sorted(present - set(CANON_ORDER))
    report = render_benchmark_report(results, meta, params, guard_order=order, gated=GATED_BENCHMARKS)
    header = (
        "# Agent Bouncer — standard benchmark suite\n\n"
        f"All guards scored through the **same harness** on class-balanced subsets "
        f"(≤{blob.get('per_class', '?')}/class) of each benchmark. Positive class = `unsafe`. "
        "`fpr_on_benign` (over-blocking) is the headline usability metric. "
        "Local guards (keyword/encoder/decoder) run on CPU; OpenAI guards are live API calls.\n\n"
    )
    with open(args.out, "w") as fh:
        fh.write(header + report)
    print(f"rendered {args.out} from {args.results}")


if __name__ == "__main__":
    main()
