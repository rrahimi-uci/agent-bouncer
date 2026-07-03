"""Persist trained guard models with rich metadata — in **SQLite** or the **filesystem**.

A :class:`ModelRecord` captures everything needed to reproduce and compare a saved model:
its source benchmark(s), sampling + split strategy, training technique, evaluation metrics,
version, timestamp, and the on-disk ``path`` to the weights. The weights themselves live on
disk (never inside the DB); this store keeps the *metadata* and a pointer to them.

Two interchangeable backends behind one API:
* ``fs`` (default) — one JSON document per model under the store root, so **every artifact
  lives as an inspectable file on disk** (alongside weights, experiments, and datasets).
* ``sqlite`` — a single ``models.db`` file; queryable by base model / technique.

Both are dependency-free (stdlib ``json`` / ``sqlite3``) and take a ``root`` dir, so tests
can point them at a temp directory.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field

from agent_bouncer.tracking.experiments import git_commit, make_id, now

DEFAULT_ROOT = "outputs/model_store"
BACKENDS = ("sqlite", "fs")


@dataclass
class ModelRecord:
    """A saved model + the metadata that makes it reproducible and comparable."""

    id: str = ""
    name: str = ""
    version: str = ""
    base_model: str = ""
    arch: str = ""             # "encoder" | "decoder"
    technique: str = ""        # sft | grpo | dpo
    dataset: str = ""          # training set it was trained on
    benchmarks: list = field(default_factory=list)   # source benchmark(s)
    sampling: str = ""         # random | stratified
    split: str = ""            # ratio | kfold
    test_ratio: float | None = None
    k: int | None = None
    n_train: int = 0
    n_test: int = 0
    metrics: dict = field(default_factory=dict)        # macro metrics
    per_benchmark: dict = field(default_factory=dict)  # per-benchmark metrics
    created: str = ""          # iso timestamp
    path: str = ""             # on-disk location of the weights
    git_commit: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ModelStore:
    def __init__(self, backend: str = "fs", root: str = DEFAULT_ROOT) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; use one of {BACKENDS}")
        self.backend = backend
        self.root = root
        os.makedirs(root, exist_ok=True)
        if backend == "sqlite":
            self.db = os.path.join(root, "models.db")
            with sqlite3.connect(self.db) as con:
                con.execute(
                    "CREATE TABLE IF NOT EXISTS models "
                    "(id TEXT PRIMARY KEY, base_model TEXT, technique TEXT, created TEXT, data TEXT)"
                )

    # -- public API ----------------------------------------------------------

    def save(self, record: ModelRecord | dict) -> str:
        """Persist a model record (auto-filling id/version/created/git if absent). Returns id."""
        rec = record if isinstance(record, ModelRecord) else ModelRecord(**record)
        stamp, ts = now()
        if not rec.version:
            rec.version = stamp
        if not rec.created:
            rec.created = ts
        if not rec.id:
            rec.id = make_id(rec.base_model or "model", rec.technique or "na", rec.version)
        if not rec.name:
            rec.name = rec.id
        if rec.git_commit is None:
            rec.git_commit = git_commit()
        if self.backend == "sqlite":
            with sqlite3.connect(self.db) as con:
                con.execute(
                    "INSERT OR REPLACE INTO models(id, base_model, technique, created, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rec.id, rec.base_model, rec.technique, rec.created, json.dumps(rec.to_dict())),
                )
        else:
            with open(os.path.join(self.root, f"{rec.id}.json"), "w") as fh:
                json.dump(rec.to_dict(), fh, indent=2)
        return rec.id

    def get(self, model_id: str) -> ModelRecord | None:
        if self.backend == "sqlite":
            with sqlite3.connect(self.db) as con:
                row = con.execute("SELECT data FROM models WHERE id = ?", (model_id,)).fetchone()
            return ModelRecord(**json.loads(row[0])) if row else None
        path = os.path.join(self.root, f"{model_id}.json")
        if not os.path.exists(path):
            return None
        return ModelRecord(**json.load(open(path)))

    def list(self, *, base_model: str | None = None,
             technique: str | None = None) -> list[ModelRecord]:
        """All saved models (newest first), optionally filtered by base model / technique."""
        if self.backend == "sqlite":
            q, args = "SELECT data FROM models", []
            conds = []
            if base_model:
                conds.append("base_model = ?")
                args.append(base_model)
            if technique:
                conds.append("technique = ?")
                args.append(technique)
            if conds:
                q += " WHERE " + " AND ".join(conds)
            with sqlite3.connect(self.db) as con:
                rows = con.execute(q, args).fetchall()
            recs = [ModelRecord(**json.loads(r[0])) for r in rows]
        else:
            recs = []
            for f in os.listdir(self.root):
                if not f.endswith(".json"):
                    continue
                rec = ModelRecord(**json.load(open(os.path.join(self.root, f))))
                if base_model and rec.base_model != base_model:
                    continue
                if technique and rec.technique != technique:
                    continue
                recs.append(rec)
        return sorted(recs, key=lambda r: r.created, reverse=True)

    def delete(self, model_id: str) -> bool:
        """Delete a record; returns True if something was removed."""
        if self.backend == "sqlite":
            with sqlite3.connect(self.db) as con:
                cur = con.execute("DELETE FROM models WHERE id = ?", (model_id,))
            return cur.rowcount > 0
        path = os.path.join(self.root, f"{model_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
