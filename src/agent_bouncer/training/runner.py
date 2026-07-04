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
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from agent_bouncer.core.schema import Decision, Surface
from agent_bouncer.data import read_jsonl
from agent_bouncer.data.split import find_leakage
from agent_bouncer.evaluation.metrics import compute_metrics
from agent_bouncer.models.registry import get_base_model
from agent_bouncer.tracking import experiments as X
from agent_bouncer.tracking.hardware import hardware_info
from agent_bouncer.training.progress import Throttle, progress_line

RESULTS_JSON = "outputs/benchmark_results.json"


def dataset_name(train_data: str) -> str:
    """Human name of the training set from its path, e.g. data/train_sets/bt-balanced/train.jsonl
    → 'bt-balanced'; data/demo/train.jsonl → 'demo'."""
    parent = os.path.basename(os.path.dirname(train_data or ""))
    return parent or os.path.splitext(os.path.basename(train_data or "dataset"))[0] or "dataset"


def descriptive_name(model_key: str, technique: str, dataset: str) -> str:
    """A clear name for a trained model: ``<model>-<params>-<technique>-<dataset>`` (params are
    added only when the model key doesn't already encode them, avoiding e.g. qwen3-0.6b-0.6b)."""
    params = get_base_model(model_key).params
    base = model_key if params.lower() in model_key.lower() else f"{model_key}-{params}"
    return f"{base}-{technique}-{dataset}"


# --------------------------------------------------------- live-console helpers
def _params_billions(params: str) -> float:
    """Parameter count in billions: '0.6B'→0.6, '1.7B'→1.7, '66M'→0.066; unknown→1.0."""
    s = (params or "").strip().upper()
    try:
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1000.0
        return float(s)
    except ValueError:
        return 1.0


def fmt_duration(sec: float) -> str:
    """Human duration: 45→'45s', 344→'5m 44s', 3900→'1h 05m'."""
    s = int(round(max(0.0, sec)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _class_balance(recs: list[dict]) -> tuple[int, int]:
    """(n_safe, n_unsafe) from a list of labelled records."""
    unsafe = sum(1 for r in recs if str(r.get("label", "")).strip().lower() in ("unsafe", "1", "true"))
    return len(recs) - unsafe, unsafe


def _plan_steps(cfg: dict, technique: str, n_train: int) -> int:
    """Rough number of optimizer steps the run will take (for the ETA estimate)."""
    import math

    t = cfg.get("train", {})
    if technique == "grpo":
        return max(1, int(cfg.get("grpo", {}).get("steps", 60) or 60))
    ms = int(t.get("max_steps", 0) or 0)
    if ms > 0:
        return ms
    batch = max(1, int(t.get("batch_size", 8) or 8))
    epochs = max(1, int(t.get("epochs", 1) or 1))
    return max(1, math.ceil(max(1, n_train) / batch) * epochs)


def _eta_seconds(arch: str, params_b: float, device: str, steps: int, technique: str,
                 num_gen: int = 4) -> float:
    """Very rough wall-clock estimate (order-of-magnitude only) for the training run."""
    if arch == "encoder":
        per = {"cuda": 0.15, "mps": 0.4, "cpu": 1.2}.get(device, 1.2)
    else:
        base = {"cuda": 0.6, "mps": 2.0, "cpu": 9.0}.get(device, 9.0)
        per = base * max(0.5, params_b / 0.6)
        if technique == "grpo":
            per *= max(2, num_gen)   # generation rollouts dominate GRPO
        elif technique == "dpo":
            per *= 1.6               # chosen + rejected pair per step
    return per * max(1, steps)


def _device_from_hw(hw: dict) -> str:
    """Map hardware_info()'s gpu field to a training device label ('cuda'|'mps'|'cpu')."""
    return {"cuda": "cuda", "mps": "mps"}.get(hw.get("gpu"), "cpu")


def _print_training_header(model_key: str, bm, technique: str, ds: str, recs: list[dict],
                           cfg: dict, hw: dict, name: str, seed: int) -> None:
    """A beautiful, informative console banner shown before a training run starts —
    what's being trained, on which dataset, with what config, and a rough ETA."""
    n_train = len(recs)
    n_safe, n_unsafe = _class_balance(recs)
    device = _device_from_hw(hw)
    steps = _plan_steps(cfg, technique, n_train)
    eta = _eta_seconds(bm.arch, _params_billions(bm.params), device, steps, technique,
                       int(cfg.get("grpo", {}).get("num_generations", 4)))
    dev_label = {"cuda": "cuda · GPU", "mps": "mps · Apple Silicon", "cpu": "cpu"}[device]

    t = cfg.get("train", {})
    bits = [f"epochs {t.get('epochs', 1)}", f"batch {t.get('batch_size', 8)}"]
    if bm.arch == "decoder":
        bits.append(f"LoRA r={cfg.get('lora', {}).get('r', 16)}")
    if t.get("max_steps"):
        bits.append(f"max-steps {t['max_steps']}")
    if technique == "grpo":
        bits.append(f"rollouts {cfg.get('grpo', {}).get('num_generations', 4)}")
    bits.append(f"seed {seed}")

    lo, hi = fmt_duration(eta * 0.6), fmt_duration(eta * 1.9)
    bal = f"⚖️ {n_safe} safe / {n_unsafe} unsafe" if n_train else "⚠️ no examples found"
    for line in (
        "",
        f"🚀 Training {model_key} · {bm.params} {bm.arch} · {technique.upper()}",
        f"📚 Dataset: {ds} — {n_train} examples ({bal})",
        f"🎛️ Config: {' · '.join(bits)}",
        f"🖥️ Device: {dev_label}",
        f"⏱️ Estimated: ~{lo}–{hi} for ~{steps} steps (rough)",
        f"🏷️ Saves as: {name}",
        "",
    ):
        print(line, flush=True)


def _print_training_footer(model_key: str, technique: str, name: str, elapsed: float,
                           out_dir: str, exp_id: str) -> None:
    """Closing banner once training finishes — actual duration + where it was saved."""
    print("", flush=True)
    print(f"✅ Trained {model_key} · {technique.upper()} in {fmt_duration(elapsed)}", flush=True)
    print(f"🏷️ Saved as {name} → {out_dir}  (experiment {exp_id})", flush=True)


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
        elif technique == "dpo":
            # DPOTrainer reads its own section — mirror the user's hyperparameters here so the
            # CLI/UI flags actually take effect (run_dpo reads cfg["dpo"], not cfg["train"]).
            # Small batch + bounded lengths keep the concatenated chosen/rejected logits under
            # Apple MPS's INT_MAX element limit on large-vocab models (see dpo.run_dpo).
            cfg["dpo"] = {
                "epochs": params.get("epochs", 1), "lr": params.get("lr", 5e-6),
                "beta": params.get("beta", 0.1),
                "batch_size": params.get("batch_size", 2),
                "grad_accum": params.get("grad_accum", 4),
                "max_length": params.get("max_seq_len", 1024),
            }
            if params.get("max_steps"):
                cfg["dpo"]["max_steps"] = int(params["max_steps"])
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

    # Read the training set once — reused for the console banner and the experiment record.
    recs = read_jsonl(train_data) if os.path.exists(train_data) else []
    n_train = len(recs)
    ds = dataset_name(train_data)
    name = descriptive_name(model_key, technique, ds)   # <model>-<params>-<technique>-<dataset>
    hw = hardware_info()
    _print_training_header(model_key, bm, technique, ds, recs, cfg, hw, name, seed)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()

    t0 = time.perf_counter()
    try:
        if technique == "sft":
            from agent_bouncer.training.sft import run_sft
            run_sft(tmp.name)
        elif technique == "grpo":
            from agent_bouncer.training.grpo import run_grpo
            run_grpo(tmp.name)
        else:
            from agent_bouncer.training.dpo import run_dpo
            run_dpo(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
    elapsed = round(time.perf_counter() - t0, 1)

    exp = X.Experiment(
        id=f"{name}-{stamp}", kind="train", model_key=model_key,
        base_hf_id=bm.hf_id, technique=technique, version=version, created=created,
        params={**params, "seed": seed, "arch": bm.arch, "name": name}, output_dir=out_dir,
        data={"train": train_data, "dataset": ds, "n_train": n_train, "leakage_checked": False},
        hardware=hw, git_commit=X.git_commit(),
        metrics_summary={"train_seconds": elapsed}, notes=notes,
    )
    X.record(exp)
    _print_training_footer(model_key, technique, name, elapsed, out_dir, exp.id)
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


def _auto_test_workers(arch: str, device: str, requested: int) -> int:
    """Worker-count policy for test scoring.

    Safety default: parallelize only encoder-on-CPU runs (decoder/mps/cuda stay single-stream).
    ``requested <= 0`` means auto.
    """
    if requested > 0:
        return requested
    if arch == "encoder" and device == "cpu":
        cpu = os.cpu_count() or 4
        return min(8, max(2, cpu // 2))
    return 1


def _warm_guard_for_parallel(guard) -> None:
    """Load the model once on the main thread before spawning worker threads."""
    if getattr(guard, "_model", None) is None:
        load = getattr(guard, "_load", None)
        if callable(load):
            load()


def score_guard(guard, benchmarks: list[str], *, per_class: int = 40,
                train_recs: list[dict] | None = None, loader=None,
                workers: int = 1, fuzzy_leakage: bool = False) -> tuple[dict, dict]:
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
        print(f"loading benchmark {bench}", flush=True)
        recs = loader(bench)
        if not recs:
            raise ValueError(f"benchmark/test set {bench!r} is empty")
        leaked = set(find_leakage(train_recs, recs, fuzzy=fuzzy_leakage)) if train_recs else set()
        clean = [r for r in recs if _norm(r.get("text")) not in leaked]
        leakage[bench] = {"n": len(recs), "dropped_leaked": len(recs) - len(clean)}
        n = len(clean)
        if not clean:
            raise ValueError(
                f"no clean samples left for {bench!r} after leakage filtering "
                f"({leakage[bench]['dropped_leaked']}/{len(recs)} dropped)"
            )
        verdicts = []
        worker_count = max(1, min(int(workers or 1), n or 1))
        throttle, t0 = Throttle(2.0), time.perf_counter()
        if n:
            print(progress_line(0, n, phase="test", label=bench), flush=True)
        else:
            print(f"⚠️ No clean samples left for {bench} after leakage filtering.", flush=True)

        if worker_count > 1 and n > 0:
            print(f"🧠 Testing in parallel with {worker_count} workers ({bench})", flush=True)
            _warm_guard_for_parallel(guard)
            verdicts = [None] * n
            done = 0
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(guard.predict, r["text"], surface=Surface.USER_PROMPT): i
                    for i, r in enumerate(clean)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    verdicts[i] = fut.result()
                    done += 1
                    now = time.perf_counter()
                    if throttle.ready(now, force=(done == n)):
                        rate = done / (now - t0) if now > t0 else None
                        eta = (n - done) / rate if rate else None
                        print(progress_line(done, n, phase="test", label=bench, rate=rate, eta=eta),
                              flush=True)
            verdicts = [v for v in verdicts if v is not None]
        else:
            # Predict one at a time so the console shows steady progress (a full benchmark can
            # be thousands of prompts — minutes for a decoder guard — otherwise silent until done).
            for i, r in enumerate(clean, 1):
                verdicts.append(guard.predict(r["text"], surface=Surface.USER_PROMPT))
                now = time.perf_counter()
                if throttle.ready(now, force=(i == n)):
                    rate = i / (now - t0) if now > t0 else None
                    eta = (n - i) / rate if rate else None
                    print(progress_line(i, n, phase="test", label=bench, rate=rate, eta=eta),
                          flush=True)
        gold = [Decision(r["label"]) for r in clean]
        lat = [v.latency_ms for v in verdicts if v.latency_ms is not None]
        m = compute_metrics(gold, [v.decision for v in verdicts], lat).to_dict()
        m["roc_auc"] = (m["recall"] + 1.0 - m["fpr_on_benign"]) / 2.0  # single-operating-point AUC
        metrics[bench] = m
        # format kept in sync with the Workbench's live test-result parser (_TEST_RE)
        print(f"  [{bench}] {getattr(guard, 'name', '?')}: F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} FPR={m['fpr_on_benign']:.3f} p90={m['latency_p90_ms']:.0f}ms "
              f"thr={m['throughput_per_s']:.1f}/s (dropped {leakage[bench]['dropped_leaked']} leaked)",
              flush=True)
    return metrics, leakage


def evaluate_and_record(train_exp_id: str, *, benchmarks: list[str] | None = None,
                        test_set: str | None = None, per_class: int = 40,
                        device: str = "cpu", workers: int = 0,
                        merge_scoreboard: bool = False) -> dict:
    """Load a trained version and score it — on a **created test set** (``test_set`` JSONL)
    or on benchmarks — with train→test leakage guards (overlapping prompts are dropped)."""
    train_exp = X.get(train_exp_id)
    if train_exp is None:
        raise ValueError(f"unknown training experiment {train_exp_id!r}")
    model_key = train_exp["model_key"]
    arch = train_exp.get("params", {}).get("arch", "decoder")
    out_dir = train_exp["output_dir"]
    train_path = train_exp.get("data", {}).get("train", "")
    train_available = bool(train_path) and os.path.exists(train_path)
    train_recs = read_jsonl(train_path) if train_available else []
    if not train_available or not train_recs:
        train_available = False
        # Never silently skip the leakage guard: if the model's training file is gone
        # (e.g. deleted train set), say so loudly and record leakage_checked=False so the
        # metrics are not mistaken for leakage-verified.
        print("⚠️ ⚠️  LEAKAGE CHECK SKIPPED — training file not found: "
              f"{train_path or '(unset)'}. Test metrics are NOT verified leakage-free; "
              "re-run with the model's training set present to enforce the guard.", flush=True)
    test_recs: list[dict] | None = None
    test_name: str | None = None
    if test_set:
        if not os.path.exists(test_set):
            raise ValueError(f"test set not found: {test_set}")
        test_name = os.path.basename(os.path.dirname(test_set)) or "test-set"
        test_recs = read_jsonl(test_set)
        if not test_recs:
            raise ValueError(f"test set is empty: {test_set}")

    print(f"🔧 Loading guard {model_key} ({arch}) from {out_dir} on {device}", flush=True)
    guard = _load_guard(model_key, arch, out_dir, train_exp.get("technique", "sft"), device)
    worker_count = _auto_test_workers(arch, device, int(workers or 0))
    if int(workers or 0) > 1 and worker_count == 1:
        print("⚠️ Parallel workers disabled for this model/device (decoder or non-cpu run).", flush=True)
    elif worker_count > 1:
        source = "manual" if int(workers or 0) > 0 else "auto"
        print(f"⚙️ Test workers: {worker_count} ({source})", flush=True)
    stamp, created = X.now()
    if test_set:
        # test on a created dataset's held-out split (train on train-1 → test on test-1)
        benchmarks = [test_name or "test-set"]
        metrics, leakage = score_guard(guard, benchmarks, train_recs=train_recs,
                                       loader=lambda b: test_recs or [], workers=worker_count,
                                       fuzzy_leakage=True)
    else:
        from agent_bouncer.evaluation.benchmarks import BENCHMARKS, load_benchmark
        benchmarks = benchmarks or list(BENCHMARKS)
        metrics, leakage = score_guard(
            guard, benchmarks, per_class=per_class, train_recs=train_recs,
            loader=lambda b: load_benchmark(b, balanced=True, per_class=per_class),
            workers=worker_count, fuzzy_leakage=True,
        )
    macro = macro_average(metrics)
    exp = X.Experiment(
        id=X.make_id(model_key, "eval", stamp), kind="eval", model_key=model_key,
        base_hf_id=train_exp.get("base_hf_id", ""), technique=train_exp.get("technique", ""),
        version=train_exp.get("version", ""), created=created,
        params={"train_exp": train_exp_id, "per_class": per_class, "device": device,
                "test_set": test_set},
        data={"benchmarks": benchmarks, "test_set": test_set,
              "leakage": leakage, "leakage_checked": train_available},
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
            # eval-only has no reference to the model's training set, so leakage can't be
            # verified here — record that honestly rather than implying a clean check.
            data={"benchmarks": benchmarks, "leakage": leakage, "leakage_checked": False},
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
    ds = (exp.get("data", {}) or {}).get("dataset", "")
    model_key = exp.get("model_key", "")
    technique = exp.get("technique", "")
    # clear name: <model>-<params>-<technique>-<dataset>
    name = exp.get("params", {}).get("name") or (
        descriptive_name(model_key, technique, ds) if model_key and ds else exp.get("id", ""))
    rec = ModelRecord(
        name=name,
        base_model=model_key,
        arch=exp.get("params", {}).get("arch", ""),
        technique=technique,
        dataset=ds,
        version=exp.get("version", ""),
        benchmarks=benchmarks or [],
        sampling=sampling, split=split, test_ratio=test_ratio, k=k,
        n_train=exp.get("data", {}).get("n_train") or 0, n_test=n_test,
        metrics=metrics or {}, per_benchmark=per_benchmark or {},
        path=exp.get("output_dir", ""), notes=exp.get("notes", ""),
        git_commit=exp.get("git_commit"),
    )
    return store.save(rec)
