"""End-to-end training & testing orchestration with versioning + experiment tracking.

- ``train_and_record`` resolves a base model from the registry, trains it (SFT/GRPO/DPO)
  to a **versioned** directory, and records a ``kind="train"`` experiment (params, data,
  hardware, git, timing).
- ``evaluate_and_record`` loads a trained version, **checks train/test leakage** for every
  benchmark (dropping any leaked test items), scores it through the harness (incl. p90 +
  throughput), and records a ``kind="eval"`` experiment linked back to the training run.

The CLI scripts (`scripts/run_training.py`, `scripts/run_testing.py`) and the FastAPI
server both call these — the server runs them as subprocesses to avoid co-loading models.
"""

from __future__ import annotations

import os
import tempfile
import time

import yaml

from agent_bouncer.core.schema import Decision, Surface
from agent_bouncer.data import read_jsonl
from agent_bouncer.data.split import find_leakage
from agent_bouncer.evaluation.metrics import compute_metrics
from agent_bouncer.models.registry import get_base_model
from agent_bouncer.tracking import experiments as X
from agent_bouncer.tracking.hardware import hardware_info

RESULTS_JSON = "outputs/benchmark_results.json"


# --------------------------------------------------------------------------- train
def build_config(model_key: str, technique: str, train_data: str, out_dir: str,
                 params: dict, seed: int) -> dict:
    """Translate UI/CLI params into a trainer config dict for the given base model."""
    bm = get_base_model(model_key)
    cfg: dict = {"base_model": bm.hf_id, "output_dir": out_dir, "seed": seed}
    if bm.arch == "encoder":
        cfg["arch"] = "encoder"
        cfg["data"] = {"train": train_data, "validation": params.get("validation", train_data)}
        cfg["train"] = {
            "epochs": params.get("epochs", 2), "lr": params.get("lr", 2e-5),
            "batch_size": params.get("batch_size", 16), "max_length": params.get("max_length", 128),
        }
    else:
        cfg["arch"] = "decoder"
        cfg["mode"] = "reasoning" if technique == "grpo" else params.get("mode", "sft")
        cfg["data"] = {"train": train_data}
        cfg["lora"] = {"r": params.get("lora_r", 16), "alpha": params.get("lora_alpha", 32),
                       "dropout": params.get("lora_dropout", 0.05)}
        cfg["train"] = {
            "epochs": params.get("epochs", 1), "lr": params.get("lr", 2e-4),
            "batch_size": params.get("batch_size", 8), "grad_accum": params.get("grad_accum", 1),
            "max_seq_len": params.get("max_seq_len", 512),
        }
        if "bf16" in params:
            cfg["train"]["bf16"] = bool(params["bf16"])
        if params.get("max_steps"):
            cfg["train"]["max_steps"] = int(params["max_steps"])
        if technique == "grpo":
            cfg["grpo"] = {"num_generations": params.get("num_generations", 4),
                           "batch_size": params.get("batch_size", 4),
                           "max_completion_len": params.get("max_completion_len", 96),
                           "steps": params.get("max_steps", 60), "lr": params.get("lr", 1e-6),
                           "log_steps": 2}
    return cfg


def train_and_record(model_key: str, technique: str, *, train_data: str,
                     params: dict | None = None, seed: int = 42, notes: str = "") -> dict:
    bm = get_base_model(model_key)
    if technique not in bm.techniques:
        raise ValueError(f"{model_key} supports {list(bm.techniques)}, not {technique!r}")
    params = dict(params or {})
    stamp, created = X.now()
    version = stamp
    out_dir = X.version_dir(model_key, version)
    cfg = build_config(model_key, technique, train_data, out_dir, params, seed)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    print(f"[train] {model_key} ({bm.hf_id}) · {technique} · v{version}", flush=True)

    t0 = time.perf_counter()
    if technique == "sft":
        from agent_bouncer.training.sft import run_sft
        run_sft(tmp.name)
    elif technique == "grpo":
        from agent_bouncer.training.grpo import run_grpo
        run_grpo(tmp.name)
    else:
        from agent_bouncer.training.dpo import run_dpo
        run_dpo(tmp.name)
    elapsed = round(time.perf_counter() - t0, 1)
    os.unlink(tmp.name)

    n_train = len(read_jsonl(train_data)) if os.path.exists(train_data) else None
    exp = X.Experiment(
        id=X.make_id(model_key, technique, stamp), kind="train", model_key=model_key,
        base_hf_id=bm.hf_id, technique=technique, version=version, created=created,
        params={**params, "seed": seed, "arch": bm.arch}, output_dir=out_dir,
        data={"train": train_data, "n_train": n_train, "leakage_checked": False},
        hardware=hardware_info(), git_commit=X.git_commit(),
        metrics_summary={"train_seconds": elapsed}, notes=notes,
    )
    X.record(exp)
    print(f"[train] done in {elapsed}s → {out_dir}  (experiment {exp.id})", flush=True)
    return exp.to_dict()


# ---------------------------------------------------------------------------- test
def _load_guard(model_key: str, arch: str, out_dir: str, technique: str, device: str):
    if arch == "encoder":
        from agent_bouncer.models.encoder import EncoderGuard
        return EncoderGuard(out_dir, name=model_key)
    from agent_bouncer.models.decoder import DecoderGuard
    mode = "reasoning" if technique == "grpo" else "sft"
    return DecoderGuard(out_dir, mode=mode, name=model_key, device=device)


_MACRO_KEYS = ("precision", "recall", "f1", "roc_auc", "fpr_on_benign",
               "latency_p50_ms", "latency_p90_ms", "throughput_per_s")


def macro_average(metrics: dict, keys: tuple[str, ...] = _MACRO_KEYS) -> dict:
    """Average per-benchmark metrics into one macro row (empty in → empty out)."""
    if not metrics:
        return {}
    return {k: round(sum(metrics[b][k] for b in metrics) / len(metrics), 4) for k in keys}


def _norm(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def score_guard(guard, benchmarks: list[str], *, per_class: int = 40,
                train_recs: list[dict] | None = None, loader=None) -> tuple[dict, dict]:
    """Score a guard over benchmarks with leakage guards; returns (metrics, leakage).

    ``loader(bench)`` returns that benchmark's records (defaults to the balanced registry
    loader). Pure w.r.t. the guard object, so tests can pass a fake guard + loader.
    """
    if loader is None:
        from agent_bouncer.evaluation.benchmarks import load_benchmark
        def loader(bench):  # noqa: E306
            return load_benchmark(bench, balanced=True, per_class=per_class)

    train_recs = train_recs or []
    metrics: dict = {}
    leakage: dict = {}
    for bench in benchmarks:
        recs = loader(bench)
        leaked = set(find_leakage(train_recs, recs)) if train_recs else set()
        clean = [r for r in recs if _norm(r.get("text")) not in leaked]
        leakage[bench] = {"n": len(recs), "dropped_leaked": len(recs) - len(clean)}
        verdicts = [guard.predict(r["text"], surface=Surface.USER_PROMPT) for r in clean]
        gold = [Decision(r["label"]) for r in clean]
        lat = [v.latency_ms for v in verdicts if v.latency_ms is not None]
        m = compute_metrics(gold, [v.decision for v in verdicts], lat).to_dict()
        m["roc_auc"] = (m["recall"] + 1.0 - m["fpr_on_benign"]) / 2.0  # single-operating-point AUC
        metrics[bench] = m
        # format kept in sync with the Studio's live test-result parser (_TEST_RE)
        print(f"  [{bench}] {getattr(guard, 'name', '?')}: F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} FPR={m['fpr_on_benign']:.3f} p90={m['latency_p90_ms']:.0f}ms "
              f"thr={m['throughput_per_s']:.1f}/s (dropped {leakage[bench]['dropped_leaked']} leaked)",
              flush=True)
    return metrics, leakage


def evaluate_and_record(train_exp_id: str, *, benchmarks: list[str] | None = None,
                        per_class: int = 40, device: str = "cpu", merge_scoreboard: bool = False) -> dict:
    """Load a trained version and score it on benchmarks with leakage guards."""
    from agent_bouncer.evaluation.benchmarks import BENCHMARKS, load_benchmark

    train_exp = X.get(train_exp_id)
    if train_exp is None:
        raise ValueError(f"unknown training experiment {train_exp_id!r}")
    model_key = train_exp["model_key"]
    arch = train_exp.get("params", {}).get("arch", "decoder")
    out_dir = train_exp["output_dir"]
    train_recs = read_jsonl(train_exp["data"]["train"]) if os.path.exists(train_exp["data"]["train"]) else []

    guard = _load_guard(model_key, arch, out_dir, train_exp.get("technique", "sft"), device)
    benchmarks = benchmarks or list(BENCHMARKS)
    stamp, created = X.now()
    metrics, leakage = score_guard(
        guard, benchmarks, per_class=per_class, train_recs=train_recs,
        loader=lambda b: load_benchmark(b, balanced=True, per_class=per_class),
    )
    macro = macro_average(metrics)
    exp = X.Experiment(
        id=X.make_id(model_key, "eval", stamp), kind="eval", model_key=model_key,
        base_hf_id=train_exp.get("base_hf_id", ""), technique=train_exp.get("technique", ""),
        version=train_exp.get("version", ""), created=created,
        params={"train_exp": train_exp_id, "per_class": per_class, "device": device},
        data={"benchmarks": benchmarks, "leakage": leakage, "leakage_checked": True},
        output_dir=out_dir, hardware=hardware_info(), git_commit=X.git_commit(),
        metrics=metrics, metrics_summary=macro,
    )
    X.record(exp)
    if merge_scoreboard and metrics:
        _merge_scoreboard(model_key, train_exp.get("params", {}).get("params", ""), metrics)
    print(f"[test] recorded experiment {exp.id} (macro F1 {macro.get('f1')})", flush=True)
    return exp.to_dict()


def _merge_scoreboard(guard_name: str, params: str, metrics: dict) -> None:
    """Add a trained model's per-benchmark metrics to outputs/benchmark_results.json."""
    import json

    blob = {"per_class": None, "meta": {}, "results": {}}
    if os.path.exists(RESULTS_JSON):
        try:
            blob = json.load(open(RESULTS_JSON))
        except (ValueError, OSError):
            pass
    for bench, m in metrics.items():
        blob.setdefault("results", {}).setdefault(bench, {})[guard_name] = m
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(RESULTS_JSON) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(blob, fh, indent=2)
    os.replace(tmp, RESULTS_JSON)


# ------------------------------------------------------------------ eval-only mode
def eval_only(
    *,
    benchmarks: list[str],
    model_id: str | None = None,
    store=None,
    path: str | None = None,
    arch: str = "decoder",
    technique: str = "sft",
    model_key: str | None = None,
    per_class: int = 40,
    device: str = "cpu",
    guard_loader=None,
    bench_loader=None,
    record_exp: bool = True,
    update_store: bool = True,
) -> dict:
    """Evaluate an **already-trained** model on benchmarks — no training.

    Resolve the model from a saved :class:`ModelStore` record (``model_id``) or a raw
    ``path`` (an uploaded / on-disk model). Scores it (leakage-guarded), optionally records
    an ``eval`` experiment, and refreshes the store record's metrics. ``guard_loader`` and
    ``bench_loader`` are injectable so this is testable without loading real weights.
    """
    if model_id is not None:
        if store is None:
            from agent_bouncer.tracking.model_store import ModelStore
            store = ModelStore()
        rec = store.get(model_id)
        if rec is None:
            raise ValueError(f"unknown saved model {model_id!r}")
        path = path or rec.path
        arch = rec.arch or arch
        technique = rec.technique or technique
        model_key = model_key or rec.base_model or model_id
    if not path:
        raise ValueError("eval_only needs a model_id or a path")
    model_key = model_key or "uploaded"

    load = guard_loader or _load_guard
    guard = load(model_key, arch, path, technique, device)
    metrics, leakage = score_guard(guard, benchmarks, per_class=per_class, loader=bench_loader)
    macro = macro_average(metrics)
    result = {
        "eval_only": True, "model_key": model_key, "model_id": model_id, "path": path,
        "technique": technique, "arch": arch, "benchmarks": benchmarks,
        "metrics": metrics, "macro": macro, "leakage": leakage,
    }
    if record_exp:
        stamp, created = X.now()
        exp = X.Experiment(
            id=X.make_id(model_key, "evalonly", stamp), kind="eval", model_key=model_key,
            technique=technique, created=created,
            params={"eval_only": True, "model_id": model_id, "path": path,
                    "per_class": per_class, "device": device},
            data={"benchmarks": benchmarks, "leakage": leakage, "leakage_checked": True},
            output_dir=path, hardware=hardware_info(), git_commit=X.git_commit(),
            metrics=metrics, metrics_summary=macro,
        )
        X.record(exp)
        result["experiment_id"] = exp.id
    if model_id and update_store and store is not None:
        rec = store.get(model_id)
        if rec is not None:
            rec.metrics = macro
            rec.per_benchmark = metrics
            store.save(rec)
    return result


def save_trained_model(exp: dict, *, store=None, sampling: str = "", split: str = "",
                       benchmarks: list[str] | None = None, test_ratio: float | None = None,
                       k: int | None = None, n_test: int = 0, metrics: dict | None = None,
                       per_benchmark: dict | None = None) -> str:
    """Persist a trained model (from a train experiment dict) into the model store with the
    full workflow metadata (source benchmarks, sampling, split, metrics). Returns its id."""
    from agent_bouncer.tracking.model_store import ModelRecord, ModelStore
    store = store or ModelStore()
    rec = ModelRecord(
        base_model=exp.get("model_key", ""),
        arch=exp.get("params", {}).get("arch", ""),
        technique=exp.get("technique", ""),
        version=exp.get("version", ""),
        benchmarks=benchmarks or [],
        sampling=sampling, split=split, test_ratio=test_ratio, k=k,
        n_train=exp.get("data", {}).get("n_train") or 0, n_test=n_test,
        metrics=metrics or {}, per_benchmark=per_benchmark or {},
        path=exp.get("output_dir", ""), notes=exp.get("notes", ""),
        git_commit=exp.get("git_commit"),
    )
    return store.save(rec)
