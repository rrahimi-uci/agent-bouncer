#!/usr/bin/env python
"""Download and run the standard guardrail + red-teaming benchmark suite end to end,
scoring every reachable guard through the SAME harness and writing a Markdown report.

Guards scored (whatever is reachable given outputs/ + .env credentials):
  - keyword-baseline                       (always)
  - encoder-distilbert (66M)               if outputs/demo-encoder exists
  - decoder-sft-0.6B  (Qwen3)              if outputs/demo-decoder-sft exists
  - decoder-grpo-0.6B (Qwen3, RL)          if outputs/grpo-qwen3-0.6b exists
  - openai-moderation                      if OPENAI_API_KEY set
  - openai-gpt-4o-mini                      if OPENAI_API_KEY set
  - openai-gpt-5.2 (reasoning_effort=low)   if OPENAI_API_KEY set

Benchmarks (ungated; gated ones are reported as "not run", never faked). Each is
downloaded once and cached (balanced subset) to data/benchmarks/<name>.jsonl so
re-runs are deterministic and offline.

Usage:
    python scripts/eval/run_benchmarks.py [--per-class 75] [--chat-model gpt-4o-mini]
                                     [--reasoning-model gpt-5.2] [--no-openai] [--refresh]
"""

from __future__ import annotations

import os

# Disable HF fast-tokenizer Rust parallelism BEFORE importing transformers: its
# rayon thread pool deadlocks when a tokenizer is called repeatedly in a loop
# alongside other torch models, which hangs the decoder guard. Must precede imports.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from agent_bouncer.core.guard import KeywordGuard  # noqa: E402
from agent_bouncer.core.schema import Decision, Surface  # noqa: E402
from agent_bouncer.data import read_jsonl, write_jsonl  # noqa: E402
from agent_bouncer.evaluation.benchmarks import (  # noqa: E402
    BENCHMARKS,
    GATED_BENCHMARKS,
    class_counts,
    load_benchmark,
)
from agent_bouncer.evaluation.ensembles import PRED_DIR  # noqa: E402
from agent_bouncer.evaluation.metrics import compute_metrics  # noqa: E402
from agent_bouncer.evaluation.report import render_benchmark_report  # noqa: E402
from agent_bouncer.models.decoder import DecoderGuard  # noqa: E402
from agent_bouncer.models.encoder import EncoderGuard  # noqa: E402

CACHE_DIR = "data/benchmarks"
RESULTS_JSON = "outputs/benchmark_results.json"
REPORT_MD = "outputs/BENCHMARKS.md"


def resolve_bench_cache(name: str) -> str:
    """Path to a benchmark's cached records: the balanced subset if present, else the full
    set (``data/benchmarks/full/``). Lets add-guard / prediction / curve steps still find the
    data a ``--full`` run produced instead of silently scoring nothing."""
    subset = f"{CACHE_DIR}/{name}.jsonl"
    return subset if os.path.exists(subset) else f"{CACHE_DIR}/full/{name}.jsonl"

# Canonical guard display order for the report.
GUARD_ORDER = [
    "keyword-baseline", "encoder-distilbert", "encoder-modernbert-large",
    "decoder-sft-0.6B", "decoder-sft-1.7B", "decoder-grpo-0.6B",
    "openai-moderation", "openai-gpt-4o-mini",
    "openai-gpt-5.2-low", "openai-gpt-5.2-medium", "openai-gpt-5.2-high",
]

# Guard display order + parameter counts for the report.
GUARD_PARAMS = {
    "keyword-baseline": "0",
    "encoder-distilbert": "66M",
    "decoder-sft-0.6B": "0.6B",
    "decoder-sft-1.7B": "1.7B",
    "decoder-grpo-0.6B": "0.6B",
    "openai-moderation": "api",
}


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _merge_persist(per_class: int, meta: dict, results: dict) -> None:
    """Merge this run's meta/results into the existing scoreboard and write atomically."""
    import tempfile

    blob = {"per_class": per_class, "meta": {}, "results": {}}
    if os.path.exists(RESULTS_JSON):
        try:
            blob = json.load(open(RESULTS_JSON))
        except (ValueError, OSError):
            pass
    blob["per_class"] = per_class
    blob.setdefault("meta", {}).update(meta)
    blob.setdefault("results", {})
    for bench, guard_map in results.items():
        blob["results"].setdefault(bench, {}).update(guard_map)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(RESULTS_JSON) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, RESULTS_JSON)


def get_benchmark_records(name: str, per_class: int, refresh: bool) -> list[dict]:
    """Return a benchmark's records, cached to JSONL.

    ``per_class > 0`` → a balanced subset (``data/benchmarks/<name>.jsonl``).
    ``per_class <= 0`` → the **full** benchmark (``data/benchmarks/full/<name>.jsonl``).
    """
    if per_class and per_class > 0:
        path = f"{CACHE_DIR}/{name}.jsonl"
        if os.path.exists(path) and not refresh:
            return read_jsonl(path)
        records = load_benchmark(name, balanced=True, per_class=per_class)
        write_jsonl(records, path)
        return records
    # full mode
    path = f"{CACHE_DIR}/full/{name}.jsonl"
    if os.path.exists(path) and not refresh:
        return read_jsonl(path)
    records = load_benchmark(name, prefer_cache=False)  # entire set, fetched fresh
    write_jsonl(records, path)
    return records


def evaluate_guard(guard, records: list[dict], *, workers: int = 1):
    """Score a guard on records; run I/O-bound (API) guards concurrently.

    Per-sample latency is measured around each individual request, so p50/p95
    stay meaningful even under concurrency. Returns ``(GuardMetrics, verdicts)`` so the
    caller can also persist per-sample predictions for offline ensembling."""
    texts = [r["text"] for r in records]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            verdicts = list(ex.map(lambda t: guard.predict(t, surface=Surface.USER_PROMPT), texts))
    else:
        verdicts = [guard.predict(t, surface=Surface.USER_PROMPT) for t in texts]
    gold = [Decision(r["label"]) for r in records]
    pred = [v.decision for v in verdicts]
    lat = [v.latency_ms for v in verdicts if v.latency_ms is not None]
    return compute_metrics(gold, pred, lat), verdicts


def prediction_rows(records: list[dict], verdicts: list) -> list[list]:
    """Per-sample ``[y, u, score, latency_ms]`` rows aligned to the benchmark subset."""
    rows = []
    for r, v in zip(records, verdicts, strict=True):
        y = 1 if r["label"] == "unsafe" else 0
        u = 1 if v.decision == Decision.UNSAFE else 0
        rows.append([y, u, round(float(v.score), 4), round(float(v.latency_ms or 0.0), 3)])
    return rows


def dump_prediction_rows(guard_name: str, bench: str, rows: list[list]) -> None:
    """Merge one benchmark's prediction rows into ``outputs/predictions/<guard>.json`` atomically."""
    import tempfile

    os.makedirs(PRED_DIR, exist_ok=True)
    path = f"{PRED_DIR}/{guard_name}.json"
    blob = {}
    if os.path.exists(path):
        try:
            blob = json.load(open(path))
        except (ValueError, OSError):
            blob = {}
    blob[bench] = rows
    fd, tmp = tempfile.mkstemp(dir=PRED_DIR, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(blob, fh)
    os.replace(tmp, path)


def build_guards(args) -> list[tuple[str, object, int]]:
    """Return (name, guard, workers). workers>1 => concurrent (API) guard."""
    guards: list[tuple[str, object, int]] = [("keyword-baseline", KeywordGuard(), 1)]

    if os.path.isdir("outputs/demo-encoder"):
        enc = EncoderGuard("outputs/demo-encoder", name="encoder-distilbert")
        guards.append(("encoder-distilbert", enc, 1))
    if os.path.isdir("outputs/demo-decoder-sft"):
        # Pin to CPU for reproducible latency (the "small guard on a laptop" story).
        dec = DecoderGuard("outputs/demo-decoder-sft", mode="sft", name="decoder-sft-0.6B", device="cpu")
        guards.append(("decoder-sft-0.6B", dec, 1))
    if os.path.isdir("outputs/decoder-sft-Qwen3-1.7B"):
        dec17 = DecoderGuard(
            "outputs/decoder-sft-Qwen3-1.7B", mode="sft", name="decoder-sft-1.7B", device="cpu"
        )
        guards.append(("decoder-sft-1.7B", dec17, 1))
    if os.path.isdir("outputs/grpo-qwen3-0.6b"):
        grpo = DecoderGuard(
            "outputs/grpo-qwen3-0.6b", mode="reasoning", name="decoder-grpo-0.6B", device="cpu"
        )
        guards.append(("decoder-grpo-0.6B", grpo, 1))

    if args.skip_decoder:
        # Local decoder guards are scored in a SEPARATE process (scripts/eval/eval_added_guard.py):
        # co-residing a BERT encoder + a Qwen decoder in one process deadlocks torch's
        # threadpool on this toolchain. Isolation makes the run reliable.
        guards = [g for g in guards if not g[0].startswith("decoder-")]

    if not args.no_openai and os.environ.get("OPENAI_API_KEY"):
        from agent_bouncer.evaluation.openai_guards import OpenAIChatGuard, OpenAIModerationGuard

        guards.append(("openai-moderation", OpenAIModerationGuard(), args.workers))
        chat = OpenAIChatGuard(args.chat_model)
        guards.append((chat.name, chat, args.workers))
        GUARD_PARAMS.setdefault(chat.name, "api")
        # One reasoning guard per effort tier (low/medium/high) — same model, different budget.
        for effort in args.reasoning_efforts:
            reasoning = OpenAIChatGuard(args.reasoning_model, reasoning_effort=effort)
            guards.append((reasoning.name, reasoning, args.workers))
            GUARD_PARAMS.setdefault(reasoning.name, "api")
    else:
        print("!! OpenAI guards skipped (no key or --no-openai)")

    if args.guards is not None:  # UI can pin the exact guard set
        keep = set(args.guards)
        guards = [g for g in guards if g[0] in keep]
    return guards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=75,
                    help="balanced eval size per class per benchmark (0 or --full = entire benchmark)")
    ap.add_argument("--full", action="store_true", help="evaluate on the FULL benchmarks (no subsampling)")
    ap.add_argument("--chat-model", default="gpt-4o-mini")
    ap.add_argument("--reasoning-model", default="gpt-5.2")
    ap.add_argument("--reasoning-efforts", nargs="*", default=["low", "medium", "high"],
                    help="reasoning_effort tiers to score for the reasoning model (default: all three)")
    ap.add_argument("--workers", type=int, default=8, help="concurrency for API guards")
    ap.add_argument("--no-openai", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true",
                    help="skip local decoder guards (score them separately for process isolation)")
    ap.add_argument("--guards", nargs="*", default=None,
                    help="pin the exact guard names to score (default: all reachable)")
    ap.add_argument("--refresh", action="store_true", help="re-download benchmark subsets")
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS), help="subset of benchmark names")
    args = ap.parse_args()
    if args.full:
        args.per_class = 0

    load_dotenv()
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(f"{CACHE_DIR}/full", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1) Download + cache the benchmark subsets.
    datasets: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    for name in args.benchmarks:
        print(f"== loading benchmark {name} ...")
        records = get_benchmark_records(name, args.per_class, args.refresh)
        n_safe, n_unsafe = class_counts(records)
        datasets[name] = records
        b = BENCHMARKS[name]
        meta[name] = {"axis": b.axis, "description": b.description, "n_safe": n_safe, "n_unsafe": n_unsafe}
        print(f"   {name}: {n_safe} safe + {n_unsafe} unsafe  ({b.axis})")

    # 2) Build guards and score every guard on every benchmark (same records).
    guards = build_guards(args)
    print(f"\nguards: {[g[0] for g in guards]}\n")

    results: dict[str, dict[str, dict]] = {n: {} for n in datasets}
    for name, records in datasets.items():
        for guard_name, guard, workers in guards:
            try:
                m, verdicts = evaluate_guard(guard, records, workers=workers)
            except Exception as exc:  # noqa: BLE001 - one bad pair shouldn't kill the run
                print(f"!! {guard_name} on {name} failed ({type(exc).__name__}: {exc}); skipping")
                continue
            results[name][guard_name] = m.to_dict()
            # Persist per-sample predictions so the ensemble builder can combine guards offline.
            dump_prediction_rows(guard_name, name, prediction_rows(records, verdicts))
            print(f"  [{name}] {guard_name}: P={m.precision:.3f} R={m.recall:.3f} "
                  f"F1={m.f1:.3f} FPR={m.fpr_on_benign:.3f} p50={m.latency_p50_ms:.0f}ms")
        # Merge-on-write after each benchmark: augment the existing scoreboard (so a
        # partial run — e.g. from the dashboard — updates only its cells) and persist
        # atomically so a crash never truncates the file.
        _merge_persist(args.per_class, meta, results)

    # 3) Render the Markdown report from the MERGED scoreboard (so a partial run still
    #    produces a complete report including previously-scored guards/benchmarks).
    blob = json.load(open(RESULTS_JSON))
    all_guards = {g for gm in blob["results"].values() for g in gm}
    order = [g for g in GUARD_ORDER if g in all_guards] + sorted(all_guards - set(GUARD_ORDER))
    params = dict(GUARD_PARAMS)
    for g in all_guards:
        params.setdefault(g, "api" if g.startswith("openai-") else "")
    report = render_benchmark_report(
        blob["results"], blob["meta"], params, guard_order=order, gated=GATED_BENCHMARKS
    )
    header = (
        "# Agent Bouncer — standard benchmark suite\n\n"
        "All guards scored through the **same harness** on class-balanced subsets "
        f"(≤{blob.get('per_class', args.per_class)}/class) of each benchmark. Positive class = `unsafe`. "
        "`fpr_on_benign` (over-blocking) is the headline usability metric.\n\n"
    )
    with open(REPORT_MD, "w") as fh:
        fh.write(header + report)
    print(f"\nwrote {RESULTS_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
