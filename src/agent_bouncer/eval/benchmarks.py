"""Adapters for the standard guardrail benchmarks.

- GuardBench : 40-dataset guardrail benchmark (the standard; pip-installable).
- Lakera PINT: prompt-injection / jailbreak detection.
- XSTest      : over-refusal — feeds the false-positive-on-benign headline metric.

Each adapter returns our `GuardMetrics` so results are comparable to `harness.evaluate`.
"""

from __future__ import annotations

from ..guard import Guard
from ..metrics import GuardMetrics


def run_guardbench(guard: Guard) -> GuardMetrics:
    raise NotImplementedError("TODO: wrap the guardbench eval pipeline; map its datasets to Verdict.")


def run_pint(guard: Guard) -> GuardMetrics:
    raise NotImplementedError("TODO: load lakeraai/pint-benchmark; score injection/jailbreak detection.")


def run_xstest(guard: Guard) -> GuardMetrics:
    raise NotImplementedError("TODO: load XSTest; report over-refusal (fpr_on_benign) on safe prompts.")
