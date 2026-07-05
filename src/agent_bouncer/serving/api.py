"""FastAPI app: the guard `/screen` endpoint **and** the Agent Bouncer Workbench dashboard.

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
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from agent_bouncer.core.guard import KeywordGuard
from agent_bouncer.core.schema import Surface, Verdict
from agent_bouncer.data import read_jsonl
from agent_bouncer.data.sampling import SAMPLING_STRATEGIES, SPLIT_STRATEGIES
from agent_bouncer.data.training_sets import (
    STRATEGIES,
    list_training_sets,
    validate_strategy_sources,
    validate_training_set_name,
)
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    # On shutdown (e.g. ./stop.sh → SIGTERM), kill every in-flight training/eval subprocess
    # tree so nothing keeps running orphaned in the background.
    await asyncio.gather(*(_terminate_run(r) for r in list(_RUNS.values())),
                         return_exceptions=True)


app = FastAPI(title="Agent Bouncer Workbench", version="0.1.0", lifespan=_lifespan)
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
        "hf_available": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
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


@app.get("/api/report")
def report(sort: str = "f1") -> FileResponse:
    """Render the leaderboard (macro-average scoreboard) to a PDF and return it for download."""
    import tempfile
    from datetime import datetime, timezone

    from starlette.background import BackgroundTask

    from agent_bouncer.serving.leaderboard_report import build_html, render_pdf

    if not RESULTS_JSON.exists():
        raise HTTPException(404, "No benchmark results yet — run the suite to populate the leaderboard.")
    blob = json.loads(RESULTS_JSON.read_text())
    if not blob.get("results"):
        raise HTTPException(404, "No benchmark results yet — run the suite to populate the leaderboard.")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_str = build_html(blob, sort=sort, generated=stamp)
    # Render to a temp file (deleted after the response is sent) so the report never
    # litters outputs/.
    fd, out = tempfile.mkstemp(prefix="ab-leaderboard-", suffix=".pdf")
    os.close(fd)
    try:
        render_pdf(html_str, out)
    except RuntimeError as exc:
        if os.path.exists(out):
            os.remove(out)
        raise HTTPException(501, str(exc)) from exc
    return FileResponse(out, media_type="application/pdf",
                        filename="agent-bouncer-leaderboard.pdf",
                        headers={"Cache-Control": "no-store"},
                        background=BackgroundTask(lambda: os.path.exists(out) and os.remove(out)))


# ---------------------------------------------------------------- ensemble builder
PRED_DIR = ROOT / "outputs" / "predictions"


#: Serialize scoreboard read-modify-write across concurrent API calls (sync endpoints run in
#: a threadpool, so two ensemble builds could otherwise interleave and lose each other's cells).
_SCOREBOARD_LOCK = threading.Lock()


def _merge_scoreboard(name: str, per_bench: dict) -> None:
    """Graft an ensemble's per-benchmark metrics into the scoreboard, atomically."""
    import tempfile

    with _SCOREBOARD_LOCK:
        blob = {"per_class": None, "meta": {}, "results": {}}
        if RESULTS_JSON.exists():
            try:
                blob = json.loads(RESULTS_JSON.read_text())
            except ValueError:
                pass
        for bench, metrics in per_bench.items():
            blob.setdefault("results", {}).setdefault(bench, {})[name] = metrics
        fd, tmp = tempfile.mkstemp(dir=str(RESULTS_JSON.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(blob, fh, indent=2)
        os.replace(tmp, RESULTS_JSON)


@app.get("/api/ensemble/members")
def ensemble_members() -> dict:
    """Guards with dumped predictions (candidate members) + the available strategies."""
    from agent_bouncer.evaluation.ensembles import available_members
    from agent_bouncer.models.ensemble import STRATEGIES

    return {"members": available_members(str(PRED_DIR)), "strategies": list(STRATEGIES)}


class EnsembleConfig(BaseModel):
    """Build + score an ensemble from dumped per-sample predictions."""
    members: list[str]
    strategy: str = "majority"
    threshold: float = 0.5
    weights: list[float] | None = None
    name: str | None = None


@app.post("/api/ensemble")
def build_ensemble(cfg: EnsembleConfig) -> dict:
    """Combine the selected members offline, merge the result into the scoreboard, return metrics."""
    from agent_bouncer.evaluation.ensembles import evaluate_ensemble, load_predictions, macro_average

    preds = load_predictions(str(PRED_DIR))
    try:
        per_bench = evaluate_ensemble(preds, cfg.members, cfg.strategy,
                                      weights=cfg.weights, threshold=cfg.threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    name = _ensemble_name(cfg.name, "ensemble-" + cfg.strategy + "-" + str(len(cfg.members)))
    _merge_scoreboard(name, per_bench)
    return {"name": name, "members": cfg.members, "strategy": cfg.strategy,
            "macro": macro_average(per_bench), "per_benchmark": per_bench}


def _ensemble_name(raw: str | None, fallback: str) -> str:
    """Sanitize a user name and keep the ``ensemble-`` prefix so the leaderboard groups it."""
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", (raw or "").strip() or fallback).strip("-") or "ensemble"
    return name if name.startswith("ensemble-") else "ensemble-" + name


class EnsembleOptimizeConfig(BaseModel):
    """Auto-search for the best ensemble over the dumped predictions."""
    objective: str = "balanced"          # "balanced" | "f1" | "fpr"
    fpr_cap: float = 0.2
    name: str | None = None


@app.post("/api/ensemble/optimize")
def optimize_ensemble_endpoint(cfg: EnsembleOptimizeConfig) -> dict:
    """Find the best ensemble by searching member subsets × strategies offline, merge the
    winner into the scoreboard, and return it plus the top candidates."""
    from agent_bouncer.evaluation.ensembles import load_predictions, optimize_ensemble

    preds = load_predictions(str(PRED_DIR))
    try:
        result = optimize_ensemble(preds, objective=cfg.objective, fpr_cap=cfg.fpr_cap)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    best = result["best"]
    name = _ensemble_name(cfg.name, "ensemble-optimized")
    _merge_scoreboard(name, best["per_bench"])
    return {"name": name, "objective": result["objective"], "n_evaluated": result["n_evaluated"],
            "best": {"members": best["members"], "strategy": best["strategy"],
                     "threshold": best["threshold"], "macro": best["macro"]},
            "candidates": result["candidates"]}


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
                {"id": e["id"], "version": e["version"], "technique": e.get("technique"),
                 "dataset": (e.get("data") or {}).get("dataset"), "created": e["created"]})
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


#: In-memory record of each run. Every event is appended to ``history`` (so a client that
#: reconnects — e.g. after a page refresh — can replay the whole console) and fanned out to any
#: live ``subscribers`` (open SSE streams). ``_MAX_RUNS`` finished runs are kept, then pruned.
_RUNS: dict[str, dict] = {}
_MAX_RUNS = 24


def _emit(run: dict, evt: dict) -> None:
    """Record an event in the run's history and fan it out to every open SSE subscriber.
    Each event carries a monotonic ``_seq`` so a reconnecting client can de-dup the overlap
    between the history it replays and the live stream it then follows."""
    evt = {**evt, "_seq": len(run["history"])}
    run["history"].append(evt)
    for q in list(run["subscribers"]):
        q.put_nowait(evt)


def _prune_runs() -> None:
    """Drop the oldest finished runs so history doesn't grow without bound."""
    if len(_RUNS) <= _MAX_RUNS:
        return
    finished = sorted((rid for rid, r in _RUNS.items() if r.get("done")),
                      key=lambda rid: _RUNS[rid].get("created", 0.0))
    for rid in finished[: len(_RUNS) - _MAX_RUNS]:
        _RUNS.pop(rid, None)


def _signal_group(proc, sig: int) -> None:
    """Send ``sig`` to the subprocess's whole process group — so dataloader workers and any
    other grandchildren die with it — falling back to the single process if the group can't
    be resolved (or on platforms without process groups)."""
    if proc is None or proc.pid is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)   # subprocess is a session leader (see _run)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


async def _terminate_run(run: dict, *, grace: float = 4.0) -> bool:
    """Stop a run's subprocess tree: SIGTERM, then SIGKILL if it hasn't exited within ``grace``
    seconds. Marks the run user-stopped. Returns True if a live process was signalled."""
    run["stopped"] = True
    proc = run.get("proc")
    if proc is None or proc.returncode is not None:
        return False
    _signal_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except asyncio.TimeoutError:
        _signal_group(proc, signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    return True


_RESULT_RE = re.compile(
    r"\[([\w\-]+)\]\s+([\w.\-]+):\s+P=([\d.]+)\s+R=([\d.]+)\s+F1=([\d.]+)\s+FPR=([\d.]+)\s+p50=([\d.]+)ms"
)
_TEST_RE = re.compile(
    r"\[([\w\-]+)\]\s+([\w.\-]+):\s+F1=([\d.]+)\s+P=([\d.]+)\s+R=([\d.]+)\s+FPR=([\d.]+)\s+p90=([\d.]+)ms"
)
_LOADING_RE = re.compile(r"loading benchmark ([\w\-]+)")
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
    if text.lstrip().startswith("PROGRESS "):  # live step/%/loss/ETA from train + eval loops
        return _parse_progress(text)
    if text.lstrip().startswith(_INFO_PREFIXES):  # beautiful training/eval banner lines
        return {"type": "info", "text": text}
    return {"type": "log", "text": text}


def _parse_progress(text: str) -> dict:
    """Parse a ``PROGRESS key=value ...`` marker into a progress event (numbers coerced)."""
    kv = dict(re.findall(r"(\w+)=(\S+)", text))

    def num(key, cast):
        try:
            return cast(kv[key])
        except (KeyError, ValueError):
            return None

    step, total, pct = num("step", int), num("total", int), num("pct", int)
    if pct is None and step is not None and total:
        pct = int(100 * step / total)
    return {"type": "progress", "phase": kv.get("phase", "train"), "label": kv.get("label"),
            "step": step, "total": total, "pct": pct, "loss": num("loss", float),
            "rate": num("rate", float), "eta": num("eta", int), "epoch": num("epoch", float),
            "text": text}


# Leading glyphs of the rich console banners emitted by the runner (train/eval headers,
# config, device, ETA, save name, done) — surfaced as highlighted "info" lines in the UI.
_INFO_PREFIXES = ("🚀", "📚", "🎛", "🖥", "⏱", "🏷", "✅", "🎯", "🧠", "🔧", "⚙", "📈", "⚖", "⚠", "⏹")


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

    # name -> (checkpoint path, decode mode, param count). GRPO is a reasoning model, so it
    # must be scored in "reasoning" mode (SFT mode truncates its <think> trace → fails closed).
    decoders = {
        "decoder-sft-0.6B": ("outputs/demo-decoder-sft", "sft", "0.6B"),
        "decoder-sft-1.7B": ("outputs/decoder-sft-Qwen3-1.7B", "sft", "1.7B"),
        "decoder-grpo-0.6B": ("outputs/grpo-qwen3-0.6b", "reasoning", "0.6B"),
    }
    for name, (path, mode, params) in decoders.items():
        if name in guards and (ROOT / path).is_dir():
            cmds.append([py, "scripts/eval/eval_added_guard.py", "--path", path, "--arch",
                         "decoder", "--mode", mode, "--name", name, "--params", params])
    cmds.append([py, "scripts/report/compute_curves.py"])
    return cmds


async def _run(run_id: str, commands: list[list[str]], stop_on_error: bool = False) -> None:
    """Run each command in sequence. By default a failed step (e.g. a gated model, an OOM, or
    a killed process) is reported but the remaining steps STILL run — so selecting many
    model×technique jobs trains every one that can, instead of aborting the whole batch on the
    first failure. Set ``stop_on_error`` for dependent pipelines."""
    run = _RUNS[run_id]
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8"}  # so the emoji banner lines pipe through cleanly
    total, failures = len(commands), 0
    try:
        for i, cmd in enumerate(commands, 1):
            if run.get("stopped"):        # user pressed Stop between jobs — run no more
                break
            _emit(run, {"type": "step", "index": i, "total": total,
                        "text": " ".join(Path(c).name if "/" in c else c for c in cmd)})
            rc, err = -1, None
            try:
                # start_new_session=True → the child leads its own process group, so a single
                # killpg tears down the whole tree (dataloader workers, tokenizers, …) on stop.
                proc = await asyncio.create_subprocess_exec(
                    *cmd, cwd=str(ROOT), env=env, start_new_session=True,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
                run["proc"] = proc
                if run.get("stopped"):    # Stop arrived before the process was recorded
                    _signal_group(proc, signal.SIGKILL)
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").rstrip()
                    if line and "\r" not in line:  # skip tqdm progress-bar carriage lines
                        _emit(run, _parse_line(line))
                await proc.wait()
                rc = proc.returncode
            except Exception as exc:  # noqa: BLE001 - a launch failure shouldn't kill the batch
                err = f"{type(exc).__name__}: {exc}"
            if run.get("stopped"):        # killed by the user — report it and stop the batch
                _emit(run, {"type": "info", "text": "⏹️ Stopped by user — training/testing killed."})
                break
            if rc != 0:
                failures += 1
                msg = f"step {i}/{total} failed" + (f": {err}" if err else f" (exit {rc})")
                _emit(run, {"type": "error", "text": msg + ("" if stop_on_error else " — continuing")})
                if stop_on_error:
                    break
    except Exception as exc:  # noqa: BLE001 - surface any unexpected failure to the UI
        _emit(run, {"type": "error", "text": f"{type(exc).__name__}: {exc}"})
    finally:
        _emit(run, {"type": "done", "failures": failures, "total": total,
                    "stopped": bool(run.get("stopped"))})
        run["done"] = True


def _launch(commands: list[list[str]], *, kind: str = "run") -> str:
    _prune_runs()
    run_id = uuid.uuid4().hex[:8]
    _RUNS[run_id] = {"history": [], "subscribers": set(), "done": False, "proc": None,
                     "stopped": False, "kind": kind, "created": time.time()}
    asyncio.create_task(_run(run_id, commands))
    return run_id


@app.post("/api/run")
async def start_run(cfg: RunConfig) -> dict:
    commands = _build_commands(cfg)
    return {"run_id": _launch(commands, kind="benchmark"), "steps": len(commands)}


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
    workers: int = 0               # 0=auto, >0 explicit worker count


_PARAM_FLAGS = {"epochs": "--epochs", "lr": "--lr", "batch_size": "--batch",
                "max_steps": "--max-steps", "lora_r": "--lora-r", "lora_alpha": "--lora-alpha",
                "lora_dropout": "--lora-dropout", "max_seq_len": "--max-seq-len",
                "max_length": "--max-length", "grad_accum": "--grad-accum", "beta": "--beta",
                "num_generations": "--num-generations", "max_completion_len": "--max-completion-len"}


def _param_flags(params: dict) -> list[str]:
    flags = []
    for key, flag in _PARAM_FLAGS.items():
        if params.get(key) is not None:
            flags += [flag, str(params[key])]
    if params.get("bf16"):
        flags.append("--bf16")
    return flags


@app.get("/api/hyperparams")
def hyperparams() -> dict:
    """Per (arch × technique) hyperparameter specs — recommended defaults + accepted values —
    for the Studio's training settings panel (keyed ``"<arch>:<technique>"``)."""
    from agent_bouncer.models.registry import get_base_model
    from agent_bouncer.training.hyperparams import param_spec

    archs = {get_base_model(m["key"]).arch for m in catalog()}
    specs = {}
    for arch in archs:
        techs = ["sft"] if arch == "encoder" else ["sft", "grpo", "dpo"]
        for tech in techs:
            specs[f"{arch}:{tech}"] = param_spec(arch, tech)
    return {"specs": specs}


@app.post("/api/train")
async def start_train(cfg: TrainConfig) -> dict:
    """Train one or many (model × technique) jobs — each on the same training set — as a
    sequential multi-step run."""
    from agent_bouncer.models.registry import get_base_model
    from agent_bouncer.training.hyperparams import validate_params

    jobs = cfg.jobs or ([{"model": cfg.model, "technique": cfg.technique}] if cfg.model else [])
    if not jobs:
        raise HTTPException(400, "no models selected to train")

    def job_params(j: dict) -> dict:
        # per-job params if provided, else the shared cfg.params (back-compat)
        return j.get("params") if isinstance(j.get("params"), dict) else (cfg.params or {})

    # Validate each job against its OWN hyperparameters + arch/technique, so a run can't
    # start with a value that model/technique doesn't accept.
    for j in jobs:
        try:
            arch = get_base_model(j["model"]).arch
        except Exception:  # noqa: BLE001 - unknown model is caught later by the trainer
            continue
        try:
            validate_params(arch, j.get("technique", "sft"), job_params(j))
        except ValueError as exc:
            raise HTTPException(400, f"{j['model']} ({j.get('technique', 'sft')}): {exc}") from exc
    # Each job gets its own command built from its own params.
    cmds = [[sys.executable, "scripts/train/run_training.py", "--model", j["model"],
             "--technique", j.get("technique", "sft"), "--train-data", cfg.train_data,
             *_param_flags(job_params(j))]
            for j in jobs]
    return {"run_id": _launch(cmds, kind="train"), "steps": len(cmds)}


@app.post("/api/test")
async def start_test(cfg: TestConfig) -> dict:
    """Test one or many trained versions — each on the same test set / benchmarks."""
    exps = cfg.exps or ([cfg.exp] if cfg.exp else [])
    if not exps:
        raise HTTPException(400, "no trained versions selected to test")
    cmds = []
    for e in exps:
        cmd = [sys.executable, "scripts/eval/run_testing.py", "--exp", e,
               "--per-class", str(cfg.per_class), "--device", cfg.device,
               "--workers", str(cfg.workers)]
        if cfg.test_set:                   # test on a created dataset's held-out split
            cmd += ["--test-set", cfg.test_set]
        elif cfg.benchmarks:
            cmd += ["--benchmarks", *cfg.benchmarks]
        cmds.append(cmd)
    return {"run_id": _launch(cmds, kind="test"), "steps": len(cmds)}


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
    return {"run_id": _launch([cmd], kind="eval"), "steps": 1}


@app.post("/api/dataset/build")
async def start_build(cfg: BuildConfig) -> dict:
    try:
        name = validate_training_set_name(cfg.name)
        validate_strategy_sources(cfg.strategy, cfg.sources)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    cmd = [sys.executable, "scripts/data/build_dataset.py", "--strategy", cfg.strategy,
           "--name", name, "--per-class", str(cfg.per_class), "--sources", *cfg.sources]
    if cfg.holdout_ratio is not None:
        cmd += ["--holdout", str(cfg.holdout_ratio)]
    return {"run_id": _launch([cmd], kind="dataset"), "steps": 1}


@app.post("/api/run/{run_id}/stop")
async def stop_run(run_id: str) -> dict:
    """Stop a run: kill its current training/testing subprocess tree and skip any remaining
    jobs. Idempotent — stopping an already-finished run is a no-op."""
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    killed = await _terminate_run(run)
    return {"run_id": run_id, "stopped": True, "killed_process": killed}


@app.get("/api/runs")
def list_runs() -> dict:
    """Active + recent runs (newest first) so the UI can reconnect its console after a refresh."""
    runs = sorted(_RUNS.items(), key=lambda kv: kv[1].get("created", 0.0), reverse=True)
    return {"runs": [
        {"run_id": rid, "kind": r.get("kind"), "done": r.get("done"),
         "stopped": r.get("stopped"), "created": r.get("created"),
         "events": len(r.get("history", []))}
        for rid, r in runs]}


def _sse(evt: dict) -> str:
    return f"data: {json.dumps({k: v for k, v in evt.items() if k != '_seq'})}\n\n"


@app.get("/api/run/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")

    async def gen():
        # Subscribe first, then snapshot history, so nothing emitted in between is lost. Replay
        # the snapshot, then follow live — skipping any queued event already in the snapshot.
        q: asyncio.Queue = asyncio.Queue()
        run["subscribers"].add(q)
        try:
            snapshot = list(run["history"])
            for evt in snapshot:
                yield _sse(evt)
            if any(e.get("type") == "done" for e in snapshot):
                return                        # run already finished — history had it all
            n = len(snapshot)
            while True:
                evt = await q.get()
                if evt["_seq"] < n:           # already replayed from the snapshot
                    continue
                yield _sse(evt)
                if evt.get("type") == "done":
                    break
        finally:
            run["subscribers"].discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
