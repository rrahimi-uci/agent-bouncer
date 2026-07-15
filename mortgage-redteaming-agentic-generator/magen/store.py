"""Config loading + JSONL row (de)serialization + deterministic RNG."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Iterable

import yaml

from .schema import Row

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)          # the generator folder
REPO_ROOT = os.path.dirname(ROOT)      # the parent guard-ranking-fragility repo


def load_dotenv(path: str | None = None) -> list[str]:
    """Load KEY=VALUE lines from the parent repo's .env into os.environ (only if unset).

    Returns the names loaded (never the values). Used so the OpenAI/HF keys the user keeps in
    .env are available without exporting them. Secrets are never printed.
    """
    path = path or os.path.join(REPO_ROOT, ".env")
    loaded: list[str] = []
    if not os.path.exists(path):
        return loaded
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
                loaded.append(key)
    return loaded


def load_config(path: str | None = None) -> dict[str, Any]:
    path = path or os.path.join(ROOT, "config", "default.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def rng(seed: int, *salt: str) -> random.Random:
    """A namespaced deterministic RNG: same (seed, salt) → same stream."""
    h = 0
    for s in salt:
        h ^= int.from_bytes(
            __import__("hashlib").blake2b(s.encode(), digest_size=8).digest(), "big")
    return random.Random((seed ^ h) & 0xFFFFFFFFFFFFFFFF)


def abspath(cfg_root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(cfg_root, p))


# ------------------------------------------------------------------ JSONL row IO
def write_rows(rows: Iterable[Row], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_rows(path: str) -> list[Row]:
    rows: list[Row] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(Row(**json.loads(line)))
    return rows


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
