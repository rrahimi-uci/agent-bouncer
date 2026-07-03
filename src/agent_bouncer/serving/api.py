"""FastAPI app: the guard `/screen` endpoint **and** the Benchmark Studio dashboard.

The dashboard (served at ``/``) lets you pick benchmarks, models + tuning
techniques, and a test-set size, then launches the real pipeline as a subprocess
and streams each step's progress over Server-Sent Events — finishing with live
Precision / Recall / F1 / AUC charts.

Install the serve extra: ``pip install -e '.[serve]'``, then::

    uvicorn agent_bouncer.serving.api:app --reload      # http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

from agent_bouncer.core.guard import KeywordGuard
from agent_bouncer.core.schema import Surface, Verdict
from agent_bouncer.data import read_jsonl
from agent_bouncer.data.sampling import SAMPLING_STRATEGIES, SPLIT_STRATEGIES
from agent_bouncer.data.training_sets import STRATEGIES, list_training_sets, validate_strategy_sources
from agent_bouncer.evaluation.benchmarks import BENCHMARKS, GATED_BENCHMARKS
from agent_bouncer.models.registry import TECHNIQUES, catalog, technique_matrix
from agent_bouncer.tracking import experiments as X
from agent_bouncer.tracking.hardware import hardware_info, hardware_label
from agent_bouncer.tracking.model_store import ModelStore

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install the serve extra: pip install -e '.[serve]'") from exc

# repo root: src/agent_bouncer/serve/api.py -> parents[3]
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "outputs" / "benchmark_results.json"
CURVES_JSON = ROOT / "outputs" / "curves.json"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env so the OpenAI guards are reachable (never logged)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

app = FastAPI(title="Agent Bouncer — Benchmark Studio", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
_guard = KeywordGuard()


# --------------------------------------------------------------------- guard API
class ScreenRequest(BaseModel):
    text: str
    surface: Surface = Surface.USER_PROMPT


@app.post("/screen", response_model=Verdict)
def screen(req: ScreenRequest) -> Verdict:
    return _guard.predict(req.text, surface=req.surface)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "guard": _guard.name}


# ------------------------------------------------------------------ dashboard API
@app.get("/")
def dashboard() -> FileResponse:
    # no-store so the browser always serves the latest UI (no stale cached dashboard)
    return FileResponse(str(HERE / "dashboard.html"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/benchmarks")
def benchmark_browser() -> FileResponse:
    return dashboard()


@app.get("/benchmarks/{name}")
def benchmark_page(name: str) -> FileResponse:
    if name not in BENCHMARKS:
        raise HTTPException(404, "unknown benchmark")
    return dashboard()


#: Guard catalog surfaced to the UI: technique + params + how to run it.
GUARD_CATALOG = [
    {"name": "keyword-baseline", "label": "Keyword baseline", "technique": "heuristic",
     "params": "0", "kind": "local", "always": True},
    {"name": "encoder-distilbert", "label": "Encoder (DistilBERT)", "technique": "SFT · classifier",
     "params": "66M", "kind": "local", "needs": "outputs/demo-encoder"},
    {"name": "decoder-sft-0.6B", "label": "Decoder (Qwen3-0.6B)", "technique": "SFT · LoRA",
     "params": "0.6B", "kind": "local-decoder", "needs": "outputs/demo-decoder-sft"},
    {"name": "decoder-sft-1.7B", "label": "Decoder (Qwen3-1.7B)", "technique": "SFT · LoRA",
     "params": "1.7B", "kind": "local-decoder", "needs": "outputs/decoder-sft-Qwen3-1.7B"},
    {"name": "decoder-grpo-0.6B", "label": "Decoder (Qwen3-0.6B)", "technique": "GRPO · RL",
     "params": "0.6B", "kind": "local-decoder", "needs": "outputs/grpo-qwen3-0.6b"},
    {"name": "openai-moderation", "label": "OpenAI Moderation", "technique": "API classifier",
     "params": "api", "kind": "openai"},
    {"name": "openai-gpt-4o-mini", "label": "GPT-4o-mini", "technique": "LLM judge",
     "params": "api", "kind": "openai"},
    {"name": "openai-gpt-5.2-low", "label": "GPT-5.2 (low reasoning)", "technique": "reasoning judge",
     "params": "api", "kind": "openai"},
]


@app.get("/api/config")
def config() -> dict:
    guards = []
    for g in GUARD_CATALOG:
        available = True
        if g.get("needs"):
            available = (ROOT / g["needs"]).is_dir()
        if g["kind"] == "openai":
            available = bool(os.environ.get("OPENAI_API_KEY"))
        guards.append({**{k: v for k, v in g.items() if k != "needs"}, "available": available})
    benchmarks = [
        {"name": n, "axis": b.axis, "description": b.description, "hf_id": b.hf_id}
        for n, b in BENCHMARKS.items()
    ]
    return {
        "benchmarks": benchmarks,
        "guards": guards,
        "gated": GATED_BENCHMARKS,
        "openai_available": bool(os.environ.get("OPENAI_API_KEY")),
    }


@app.get("/api/results")
def results() -> JSONResponse:
    if RESULTS_JSON.exists():
        return JSONResponse(json.loads(RESULTS_JSON.read_text()))
    return JSONResponse({"per_class": None, "meta": {}, "results": {}})


@app.get("/api/curves")
def curves() -> JSONResponse:
    if CURVES_JSON.exists():
        return JSONResponse(json.loads(CURVES_JSON.read_text()))
    return JSONResponse({})


# ------------------------------------------------------- training / experiments API
@app.get("/api/models")
def models() -> dict:
    """Base-model catalog + tuning techniques + the model×technique validity matrix.

    ``matrix`` lets the UI grey out invalid combinations (encoders: SFT only)."""
    return {"models": catalog(), "techniques": TECHNIQUES, "matrix": technique_matrix()}


@app.get("/api/sampling")
def sampling() -> dict:
    """Sampling + split strategies for the guided workflow (random/stratified, ratio/k-fold)."""
    return {
        "sampling": [{"key": k, **v} for k, v in SAMPLING_STRATEGIES.items()],
        "split": [{"key": k, **v} for k, v in SPLIT_STRATEGIES.items()],
    }


@app.get("/api/hardware")
def hardware() -> dict:
    info = hardware_info()
    return {"info": info, "label": hardware_label(info)}


@app.get("/api/experiments")
def experiments() -> dict:
    """All experiment summaries (newest first) + trained versions grouped by model."""
    idx = X.list_experiments()
    versions: dict[str, list] = {}
    for e in idx:
        if e.get("kind") == "train":
            versions.setdefault(e["model_key"], []).append(
                {"id": e["id"], "version": e["version"],
                 "technique": e.get("technique"), "created": e["created"]})
    return {"experiments": idx, "versions": versions}


@app.get("/api/experiment/{exp_id}")
def experiment(exp_id: str) -> JSONResponse:
    exp = X.get(exp_id)
    if exp is None:
        raise HTTPException(404, "unknown experiment")
    return JSONResponse(exp)


# ------------------------------------------------------------ saved model store
def _store() -> ModelStore:
    return ModelStore(root=str(ROOT / "outputs" / "model_store"))


@app.get("/api/saved_models")
def saved_models() -> dict:
    """All persisted models (newest first) for the eval-only + comparison views."""
    return {"models": [r.to_dict() for r in _store().list()]}


@app.get("/api/saved_models/{model_id}")
def saved_model(model_id: str) -> JSONResponse:
    rec = _store().get(model_id)
    if rec is None:
        raise HTTPException(404, "unknown saved model")
    return JSONResponse(rec.to_dict())


@app.delete("/api/saved_models/{model_id}")
def delete_saved_model(model_id: str) -> dict:
    if not _store().delete(model_id):
        raise HTTPException(404, "unknown saved model")
    return {"deleted": model_id}


class SaveModelConfig(BaseModel):
    """Persist a trained version into the model store with its workflow metadata."""
    train_exp: str
    eval_exp: str | None = None
    sampling: str = ""
    split: str = ""
    benchmarks: list[str] = []
    test_ratio: float | None = None
    k: int | None = None


@app.post("/api/saved_models")
def save_model(cfg: SaveModelConfig) -> dict:
    from agent_bouncer.training.runner import save_trained_model
    exp = X.get(cfg.train_exp)
    if exp is None:
        raise HTTPException(404, "unknown training experiment")
    metrics, per_bench = {}, {}
    if cfg.eval_exp:
        ev = X.get(cfg.eval_exp)
        if ev:
            metrics = ev.get("metrics_summary", {})
            per_bench = ev.get("metrics", {})
    mid = save_trained_model(
        exp, store=_store(), sampling=cfg.sampling, split=cfg.split,
        benchmarks=cfg.benchmarks, test_ratio=cfg.test_ratio, k=cfg.k,
        metrics=metrics, per_benchmark=per_bench,
    )
    return {"saved": mid}


# ------------------------------------------------------- benchmark viewer + datasets
def _benchmark_path(name: str) -> tuple[Path, bool]:
    full = ROOT / "data" / "benchmarks" / "full" / f"{name}.jsonl"
    subset = ROOT / "data" / "benchmarks" / f"{name}.jsonl"
    return (full, True) if full.exists() else (subset, False)


def _hazard_value(raw: str | None) -> str:
    return str(raw or "none").strip().lower()


@app.get("/api/benchmark/{name}")
def benchmark_detail(
    name: str,
    limit: int = 100,
    offset: int = 0,
    q: str = "",
    label: str = "all",
    hazard: str = "all",
) -> dict:
    """Benchmark metadata + one filtered page of contents.

    The explorer always serves the **full** dataset when downloaded
    (``data/benchmarks/full``), otherwise the balanced cached subset. Browsing is
    intentionally capped at ``100`` rows/page so the UI stays responsive even on the
    larger benchmark files.
    """
    if name not in BENCHMARKS:
        raise HTTPException(404, "unknown benchmark")
    b = BENCHMARKS[name]
    path, is_full = _benchmark_path(name)
    recs = read_jsonl(str(path)) if path.exists() else []
    n_unsafe = sum(r.get("label") == "unsafe" for r in recs)
    n_safe = len(recs) - n_unsafe

    query = q.strip().lower()
    label = (label or "all").strip().lower()
    label = label if label in {"all", "safe", "unsafe"} else "all"
    hazard = _hazard_value(hazard)
    hazard = "all" if hazard == "" else hazard

    search_filtered = recs
    if query:
        search_filtered = [r for r in recs if query in str(r.get("text", "")).lower()]

    label_counts = {
        "all": len(search_filtered),
        "safe": sum(r.get("label") == "safe" for r in search_filtered),
        "unsafe": sum(r.get("label") == "unsafe" for r in search_filtered),
    }

    label_filtered = search_filtered
    if label != "all":
        label_filtered = [r for r in search_filtered if r.get("label") == label]

    hazard_counter = Counter(_hazard_value(r.get("hazard")) for r in label_filtered)
    hazards = [{"value": "all", "count": len(label_filtered)}]
    if "none" in hazard_counter:
        hazards.append({"value": "none", "count": hazard_counter["none"]})
    hazards.extend(
        {"value": hz, "count": hazard_counter[hz]}
        for hz in sorted(k for k in hazard_counter if k != "none")
    )

    filtered = label_filtered
    if hazard != "all":
        filtered = [r for r in label_filtered if _hazard_value(r.get("hazard")) == hazard]

    page_size = 100 if limit <= 0 else min(max(limit, 1), 100)
    filtered_total = len(filtered)
    if filtered_total and offset >= filtered_total:
        offset = ((filtered_total - 1) // page_size) * page_size
    offset = max(offset, 0)
    page = 1 if not filtered_total else (offset // page_size) + 1
    pages = max(1, (filtered_total + page_size - 1) // page_size)
    window = filtered[offset:offset + page_size]
    filtered_unsafe = sum(r.get("label") == "unsafe" for r in filtered)
    return {
        "name": name, "axis": b.axis, "description": b.description, "hf_id": b.hf_id,
        "cached": path.exists(), "is_full": is_full, "total": len(recs),
        "n_safe": n_safe, "n_unsafe": n_unsafe,
        "filtered_total": filtered_total,
        "filtered_safe": filtered_total - filtered_unsafe,
        "filtered_unsafe": filtered_unsafe,
        "shown": len(window), "offset": offset, "limit": page_size,
        "page": page, "pages": pages,
        "page_start": offset + 1 if filtered_total else 0,
        "page_end": offset + len(window),
        "has_prev": page > 1,
        "has_next": offset + len(window) < filtered_total,
        "filters": {"q": q, "label": label, "hazard": hazard},
        "label_counts": label_counts,
        "hazards": hazards,
        "samples": [
            {
                "text": r.get("text", ""),
                "label": r.get("label"),
                "hazard": _hazard_value(r.get("hazard")),
                "source": r.get("source", name),
            }
            for r in window
        ],
    }


@app.get("/api/datasets")
def datasets() -> dict:
    """Strategies, selectable source datasets, and already-built training sets."""
    sources = [{"name": n, "axis": b.axis} for n, b in BENCHMARKS.items()]
    strategies = [{"key": k, **v} for k, v in STRATEGIES.items()]
    return {"strategies": strategies, "sources": sources, "train_sets": list_training_sets()}


@app.get("/api/train_sets")
def train_sets() -> dict:
    return {"train_sets": list_training_sets()}


# ------------------------------------------------------------------- run pipeline
class RunConfig(BaseModel):
    benchmarks: list[str] = []
    guards: list[str] = []
    per_class: int = 40


_RUNS: dict[str, dict] = {}

_RESULT_RE = re.compile(
    r"\[([a-z_]+)\]\s+([\w.\-]+):\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+FPR=([\d.]+)\s+p50=([\d.]+)ms"
)
_TEST_RE = re.compile(
    r"\[(\w+)\]\s+([\w.\-]+):\s+F1=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)\s+FPR=([\d.]+)\s+p90=([\d.]+)ms"
)
_LOADING_RE = re.compile(r"loading benchmark (\w+)")
_EXP_RE = re.compile(r"(?:EXPERIMENT_ID|EVAL_EXPERIMENT_ID)=(\S+)")


def _parse_line(text: str) -> dict:
    m = _RESULT_RE.search(text)
    if m:
        b, g, p, r, f1, fpr, p50 = m.groups()
        return {"type": "result", "benchmark": b, "guard": g, "precision": float(p),
                "recall": float(r), "f1": float(f1), "fpr_on_benign": float(fpr),
                "latency_p50_ms": float(p50), "text": text}
    m = _TEST_RE.search(text)
    if m:
        b, g, f1, p, r, fpr, p90 = m.groups()
        return {"type": "test_result", "benchmark": b, "guard": g, "f1": float(f1),
                "precision": float(p), "recall": float(r), "fpr_on_benign": float(fpr),
                "latency_p90_ms": float(p90), "text": text}
    m = _EXP_RE.search(text)
    if m:
        return {"type": "experiment", "exp_id": m.group(1), "text": text}
    m = re.search(r"DATASET_BUILT=(\S+)", text)
    if m:
        return {"type": "dataset", "name": m.group(1), "text": text}
    m = _LOADING_RE.search(text)
    if m:
        return {"type": "loading", "benchmark": m.group(1), "text": text}
    return {"type": "log", "text": text}


def _build_commands(cfg: RunConfig) -> list[list[str]]:
    py = sys.executable
    benches = cfg.benchmarks or list(BENCHMARKS)
    guards = cfg.guards or ["keyword-baseline", "encoder-distilbert"]
    need_openai = any(g.startswith("openai-") for g in guards)
    fast = [g for g in guards if not g.startswith("decoder-")]

    main = [py, "scripts/eval/run_benchmarks.py", "--per-class", str(cfg.per_class),
            "--skip-decoder", "--benchmarks", *benches, "--guards", *fast]
    if not need_openai:
        main.append("--no-openai")
    cmds = [main]

    decoders = {
        "decoder-sft-0.6B": ("outputs/demo-decoder-sft", "sft"),
        "decoder-sft-1.7B": ("outputs/decoder-sft-Qwen3-1.7B", "sft"),
        "decoder-grpo-0.6B": ("outputs/grpo-qwen3-0.6b", "sft"),
    }
    for name, (path, mode) in decoders.items():
        if name in guards and (ROOT / path).is_dir():
            cmds.append([py, "scripts/eval/eval_added_guard.py", "--path", path, "--arch",
                         "decoder", "--mode", mode, "--name", name, "--params", "0.6B"])
    cmds.append([py, "scripts/report/compute_curves.py"])
    return cmds


async def _run(run_id: str, commands: list[list[str]]) -> None:
    run = _RUNS[run_id]
    q: asyncio.Queue = run["queue"]
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    try:
        for cmd in commands:
            await q.put({"type": "step", "text": " ".join(Path(c).name if "/" in c else c for c in cmd)})
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            run["proc"] = proc
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line and "\r" not in line:  # skip tqdm progress-bar carriage lines
                    await q.put(_parse_line(line))
            await proc.wait()
            if proc.returncode != 0:
                await q.put({"type": "error", "text": f"step exited with code {proc.returncode}"})
                break
    except Exception as exc:  # noqa: BLE001 - surface any launch failure to the UI
        await q.put({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
    finally:
        await q.put({"type": "done"})
        run["done"] = True


def _launch(commands: list[list[str]]) -> str:
    run_id = uuid.uuid4().hex[:8]
    _RUNS[run_id] = {"queue": asyncio.Queue(), "done": False, "proc": None}
    asyncio.create_task(_run(run_id, commands))
    return run_id


@app.post("/api/run")
async def start_run(cfg: RunConfig) -> dict:
    commands = _build_commands(cfg)
    return {"run_id": _launch(commands), "steps": len(commands)}


class TrainConfig(BaseModel):
    model: str | None = None       # single model (back-compat)
    technique: str = "sft"
    jobs: list[dict] = []          # [{model, technique}, …] — train many at once
    train_data: str = "data/demo/train.jsonl"
    params: dict = {}


class TestConfig(BaseModel):
    exp: str | None = None         # single training-experiment id (back-compat)
    exps: list[str] = []           # multiple trained versions to test at once
    benchmarks: list[str] = []
    test_set: str | None = None    # path to a created test split (JSONL) — overrides benchmarks
    per_class: int = 40
    device: str = "cpu"


_PARAM_FLAGS = {"epochs": "--epochs", "lr": "--lr", "batch_size": "--batch",
                "max_steps": "--max-steps", "lora_r": "--lora-r", "lora_alpha": "--lora-alpha",
                "max_seq_len": "--max-seq-len"}


def _param_flags(params: dict) -> list[str]:
    flags = []
    for key, flag in _PARAM_FLAGS.items():
        if params.get(key) is not None:
            flags += [flag, str(params[key])]
    if params.get("bf16"):
        flags.append("--bf16")
    return flags


@app.post("/api/train")
async def start_train(cfg: TrainConfig) -> dict:
    """Train one or many (model × technique) jobs — each on the same training set — as a
    sequential multi-step run."""
    jobs = cfg.jobs or ([{"model": cfg.model, "technique": cfg.technique}] if cfg.model else [])
    if not jobs:
        raise HTTPException(400, "no models selected to train")
    flags = _param_flags(cfg.params)
    cmds = [[sys.executable, "scripts/train/run_training.py", "--model", j["model"],
             "--technique", j.get("technique", "sft"), "--train-data", cfg.train_data, *flags]
            for j in jobs]
    return {"run_id": _launch(cmds), "steps": len(cmds)}


@app.post("/api/test")
async def start_test(cfg: TestConfig) -> dict:
    """Test one or many trained versions — each on the same test set / benchmarks."""
    exps = cfg.exps or ([cfg.exp] if cfg.exp else [])
    if not exps:
        raise HTTPException(400, "no trained versions selected to test")
    cmds = []
    for e in exps:
        cmd = [sys.executable, "scripts/eval/run_testing.py", "--exp", e,
               "--per-class", str(cfg.per_class), "--device", cfg.device]
        if cfg.test_set:                   # test on a created dataset's held-out split
            cmd += ["--test-set", cfg.test_set]
        elif cfg.benchmarks:
            cmd += ["--benchmarks", *cfg.benchmarks]
        cmds.append(cmd)
    return {"run_id": _launch(cmds), "steps": len(cmds)}


class BuildConfig(BaseModel):
    strategy: str
    name: str
    sources: list[str]
    per_class: int = 200
    holdout_ratio: float | None = None   # train/test split fraction (e.g. 0.2 = 80/20)


class EvalConfig(BaseModel):
    """Evaluation-only run: score a saved model on benchmarks, no training."""
    model_id: str
    benchmarks: list[str] = []
    per_class: int = 40
    device: str = "cpu"


@app.post("/api/eval")
async def start_eval(cfg: EvalConfig) -> dict:
    cmd = [sys.executable, "scripts/eval/run_eval_only.py", "--model-id", cfg.model_id,
           "--per-class", str(cfg.per_class), "--device", cfg.device]
    if cfg.benchmarks:
        cmd += ["--benchmarks", *cfg.benchmarks]
    return {"run_id": _launch([cmd]), "steps": 1}


@app.post("/api/dataset/build")
async def start_build(cfg: BuildConfig) -> dict:
    try:
        validate_strategy_sources(cfg.strategy, cfg.sources)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    cmd = [sys.executable, "scripts/data/build_dataset.py", "--strategy", cfg.strategy,
           "--name", cfg.name, "--per-class", str(cfg.per_class), "--sources", *cfg.sources]
    if cfg.holdout_ratio is not None:
        cmd += ["--holdout", str(cfg.holdout_ratio)]
    return {"run_id": _launch([cmd]), "steps": 1}


@app.get("/api/run/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")

    async def gen():
        q: asyncio.Queue = run["queue"]
        while True:
            evt = await q.get()
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("type") == "done":
                break

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
