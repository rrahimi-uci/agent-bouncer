#!/usr/bin/env bash
# Run the standard guardrail + red-teaming benchmark suite end to end.
# Requires the eval extra (pip install -e '.[eval]'); OpenAI/HF keys are optional
# and auto-detected from .env (missing keys are skipped, never faked).
set -euo pipefail

echo "==> Agent Bouncer benchmark suite"
echo "    Downloads + scores every reachable guard on the ungated standard benchmarks"
echo "    (BeaverTails / OpenAI-Moderation / ToxicChat / prompt-injections /"
echo "     jailbreak-classification / JailbreakBench / XSTest) -> outputs/BENCHMARKS.md"
python scripts/eval/run_benchmarks.py "$@"
