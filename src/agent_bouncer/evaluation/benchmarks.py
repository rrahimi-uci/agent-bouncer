"""The standard-benchmark suite: a registry of guardrail / red-teaming benchmarks,
each normalized to our unified ``{text, label, hazard, source}`` schema so every
guard is scored through the *same* harness and the numbers are apples-to-apples.

Two things live here:

* ``BENCHMARKS`` — a registry mapping a benchmark name to its loader, axis, and a
  one-line description. Loaders are the ungated Hugging Face datasets from
  ``agent_bouncer.data`` (gated sets — WildGuardMix, HarmBench, Lakera PINT —
  need ``HF_TOKEN`` and are intentionally *not* fabricated here).
* runners — ``load_benchmark`` (download + optional balanced subsample) and
  ``run_benchmark`` / ``run_suite`` (score one or many guards). Everything but the
  download is pure and unit-tested (see ``tests/test_benchmarks.py``).

Axes:
  - ``guardrail``    : content-safety / harmful-request detection.
  - ``red_team``     : adversarial prompt-injection / jailbreak detection.
  - ``over_refusal`` : benign-but-scary prompts — feeds the ``fpr_on_benign`` metric.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from agent_bouncer import data as D
from agent_bouncer.core.guard import Guard
from agent_bouncer.core.schema import Surface
from agent_bouncer.evaluation.harness import evaluate
from agent_bouncer.evaluation.metrics import GuardMetrics

#: Local cache of the **full-size** downloaded benchmarks (see
#: ``scripts/data/download_full_benchmarks.py``). Preferred over a network fetch when present.
FULL_CACHE_DIR = "data/benchmarks/full"


def cached_full(name: str) -> list[dict] | None:
    """Return the locally-cached full dataset for ``name`` if it was downloaded, else None."""
    path = os.path.join(FULL_CACHE_DIR, f"{name}.jsonl")
    return D.read_jsonl(path) if os.path.exists(path) else None


@dataclass(frozen=True)
class Benchmark:
    name: str
    loader: Callable[[], list[dict]]
    axis: str
    description: str
    hf_id: str = ""
    #: kwargs the driver uses by default (e.g. balance/limit for cost control).
    defaults: dict = field(default_factory=dict)


#: Ungated standard benchmarks that download + run today (no HF_TOKEN needed).
BENCHMARKS: dict[str, Benchmark] = {
    "beavertails": Benchmark(
        # 30k_test is held out from 30k_train (our demo training source) — no leakage.
        "beavertails", lambda: D.load_beavertails(split="30k_test"), "guardrail",
        "PKU-Alignment/BeaverTails (30k_test) — 14-category harmful-QA prompt safety.",
        hf_id="PKU-Alignment/BeaverTails",
    ),
    "openai_moderation": Benchmark(
        "openai_moderation", D.load_openai_moderation, "guardrail",
        "OpenAI Moderation eval — 8-category content-moderation gold set.",
        hf_id="mmathys/openai-moderation-api-evaluation",
    ),
    "toxicchat": Benchmark(
        "toxicchat", D.load_toxicchat, "guardrail",
        "LMSYS ToxicChat (toxicchat0124) — real user-input toxicity detection.",
        hf_id="lmsys/toxic-chat",
    ),
    "prompt_injections": Benchmark(
        "prompt_injections", D.load_prompt_injections, "red_team",
        "deepset/prompt-injections — prompt-injection attack detection.",
        hf_id="deepset/prompt-injections",
    ),
    "jailbreak_classification": Benchmark(
        "jailbreak_classification", D.load_jailbreak_classification, "red_team",
        "jackhhao/jailbreak-classification — jailbreak vs. benign prompts.",
        hf_id="jackhhao/jailbreak-classification",
    ),
    "jailbreakbench": Benchmark(
        "jailbreakbench", D.load_jailbreakbench, "red_team",
        "JailbreakBench/JBB-Behaviors — 100 harmful + 100 benign red-team behaviors.",
        hf_id="JailbreakBench/JBB-Behaviors",
    ),
    "xstest": Benchmark(
        "xstest", D.load_xstest, "over_refusal",
        "XSTest v2 — safe prompts that look unsafe; measures over-blocking (FPR).",
        hf_id="natolambert/xstest-v2-copy",
    ),
}

#: Standard benchmarks we would include but that are gated (need HF_TOKEN +
#: license acceptance). Listed so reports state what was *not* run, never faked.
GATED_BENCHMARKS: dict[str, str] = {
    "wildguardmix": "allenai/wildguardmix (WildGuard test)",
    "harmbench": "walledai/HarmBench",
    "advbench": "walledai/AdvBench",
    "pint": "Lakera PINT (private benchmark)",
}


# --------------------------------------------------------------------- sampling
def balanced_subset(records: list[dict], per_class: int, seed: int = 42) -> list[dict]:
    """A class-balanced, deterministically shuffled subset (``per_class`` each)."""
    rng = random.Random(seed)
    safe = [r for r in records if r["label"] == "safe"]
    unsafe = [r for r in records if r["label"] == "unsafe"]
    rng.shuffle(safe)
    rng.shuffle(unsafe)
    k = min(per_class, len(safe), len(unsafe))
    subset = safe[:k] + unsafe[:k]
    rng.shuffle(subset)
    return subset


def subsample(records: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """A deterministic (unbalanced) subset preserving the natural class ratio."""
    if limit >= len(records):
        return list(records)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


# ------------------------------------------------------------------- loaders/run
def load_benchmark(
    name: str,
    *,
    balanced: bool = False,
    per_class: int | None = None,
    limit: int | None = None,
    seed: int = 42,
    prefer_cache: bool = True,
) -> list[dict]:
    """Load a registered benchmark and (optionally) subsample it.

    Source: the locally-cached **full** dataset (``data/benchmarks/full``) when present,
    otherwise the Hugging Face loader. With neither ``balanced``+``per_class`` nor ``limit``
    set, the **entire** benchmark is returned. ``balanced`` takes precedence (``per_class``
    safe + ``per_class`` unsafe); otherwise ``limit`` caps at the natural class ratio.
    """
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    records = (cached_full(name) if prefer_cache else None) or BENCHMARKS[name].loader()
    if balanced and per_class:
        return balanced_subset(records, per_class, seed=seed)
    if limit is not None:
        return subsample(records, limit, seed=seed)
    return records


def class_counts(records: Sequence[dict]) -> tuple[int, int]:
    """Return ``(n_safe, n_unsafe)`` for a record list."""
    unsafe = sum(1 for r in records if r["label"] == "unsafe")
    return len(records) - unsafe, unsafe


def run_benchmark(
    guard: Guard,
    records: Sequence[dict],
    *,
    surface: Surface = Surface.USER_PROMPT,
    run_name: str | None = None,
) -> GuardMetrics:
    """Score one guard on one already-loaded benchmark (delegates to the harness)."""
    return evaluate(guard, records, surface=surface, run_name=run_name)


def run_suite(
    guards: Sequence[tuple[str, Guard]],
    datasets: dict[str, Sequence[dict]],
    *,
    surface: Surface = Surface.USER_PROMPT,
    on_error: Callable[[str, str, Exception], None] | None = None,
) -> dict[str, dict[str, dict]]:
    """Score every ``(name, guard)`` on every ``{benchmark: records}``.

    Returns ``{benchmark: {guard_name: metrics_dict}}``. A guard that raises on a
    benchmark is skipped (reported via ``on_error``) so one failure never kills the
    whole run — matching the "never fabricate, always report what ran" contract.
    """
    results: dict[str, dict[str, dict]] = {}
    for bench_name, records in datasets.items():
        results[bench_name] = {}
        for guard_name, guard in guards:
            run_name = f"{guard_name}:{bench_name}"
            try:
                metrics = run_benchmark(guard, records, surface=surface, run_name=run_name)
            except Exception as exc:  # noqa: BLE001 - isolate a bad guard/benchmark pair
                if on_error is not None:
                    on_error(bench_name, guard_name, exc)
                continue
            results[bench_name][guard_name] = metrics.to_dict()
    return results


# Convenience wrappers kept for the documented `make bench` axes.
def run_xstest(guard: Guard, **kw) -> GuardMetrics:
    return run_benchmark(guard, load_benchmark("xstest", **kw))
