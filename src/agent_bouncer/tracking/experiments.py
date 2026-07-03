"""Lightweight experiment tracking + model versioning — a JSON store, no server.

Every training or evaluation run is recorded as one JSON file under
``outputs/experiments/`` plus a summary line in ``index.json``. Each record captures
*what* ran (model, technique, params), *on what* (train/test datasets, with a leakage
flag), *the result* (metrics), and *where* (hardware + git commit) — so runs are
reproducible and comparable. MLflow is optional and orthogonal; this store always works.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

EXP_DIR = "outputs/experiments"
INDEX = os.path.join(EXP_DIR, "index.json")
MODELS_DIR = "outputs/models"

_SUMMARY_KEYS = (
    "id", "kind", "model_key", "base_hf_id", "technique", "version",
    "created", "params", "data", "hardware", "git_commit", "metrics_summary",
)


def now() -> tuple[str, str]:
    """Return (version_stamp, iso_timestamp). UTC, deterministic format."""
    t = datetime.now(timezone.utc)
    return t.strftime("%Y%m%d-%H%M%S"), t.isoformat(timespec="seconds")


def make_id(model_key: str, technique: str, stamp: str) -> str:
    return f"{model_key}-{technique}-{stamp}"


def version_dir(model_key: str, version: str) -> str:
    return os.path.join(MODELS_DIR, model_key, version)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Experiment:
    id: str
    kind: str  # "train" | "eval"
    model_key: str
    base_hf_id: str = ""
    technique: str = ""
    version: str = ""
    created: str = ""
    params: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)      # {train, test, n_train, n_test, leakage_checked}
    output_dir: str = ""
    hardware: dict = field(default_factory=dict)
    git_commit: str | None = None
    metrics: dict = field(default_factory=dict)   # per-benchmark or {"summary": {...}}
    metrics_summary: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _load_index() -> list[dict]:
    if not os.path.exists(INDEX):
        return []
    try:
        return json.load(open(INDEX))
    except (ValueError, OSError):
        return []


def _summary(exp: dict) -> dict:
    return {k: exp.get(k) for k in _SUMMARY_KEYS}


def _write_json_atomic(path: str, obj) -> None:
    """Write JSON via a temp file + atomic rename, so a crash mid-write never
    truncates or empties ``path`` (which ``_load_index`` would silently read as [])."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def record(exp: Experiment | dict) -> str:
    """Persist an experiment (full JSON) + update the index. Returns its id."""
    data = exp.to_dict() if isinstance(exp, Experiment) else dict(exp)
    os.makedirs(EXP_DIR, exist_ok=True)
    eid = data["id"]
    _write_json_atomic(os.path.join(EXP_DIR, f"{eid}.json"), data)
    index = [e for e in _load_index() if e.get("id") != eid]
    index.append(_summary(data))
    index.sort(key=lambda e: e.get("created", ""))
    _write_json_atomic(INDEX, index)
    return eid


def get(exp_id: str) -> dict | None:
    path = os.path.join(EXP_DIR, f"{exp_id}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def list_experiments() -> list[dict]:
    """Index summaries, newest first."""
    return sorted(_load_index(), key=lambda e: e.get("created", ""), reverse=True)


def versions_for(model_key: str) -> list[dict]:
    """Trained versions of a model (kind='train'), newest first."""
    return [e for e in list_experiments() if e.get("model_key") == model_key and e.get("kind") == "train"]
