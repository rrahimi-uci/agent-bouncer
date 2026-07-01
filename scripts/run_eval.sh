#!/usr/bin/env bash
# Run the standard benchmarks and write the results table.
# Requires the eval extra: pip install -e '.[eval]'
set -euo pipefail

echo "==> Agent Bouncer benchmark suite (GuardBench / PINT / XSTest)"
echo "    TODO: wire eval/benchmarks.py adapters, then loop over configs/eval.yaml guards."
echo "    For now, smoke-test the harness on the reference guard:"
agent-bouncer eval tests/data/smoke.jsonl --run-name keyword-baseline
