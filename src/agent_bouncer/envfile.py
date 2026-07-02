"""Tiny `.env` loader so credentials (OPENAI_API_KEY, HF_TOKEN) are picked up
consistently by the CLI, scripts, notebook, and server — without a dependency.

`datasets` / `huggingface_hub` read ``HF_TOKEN`` from the *environment*, so all this
does is copy ``.env`` KEY=VALUE lines into ``os.environ`` (with ``setdefault`` — a real
environment variable always wins). Never raises; a missing ``.env`` is a no-op.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | os.PathLike | None = None) -> str | None:
    """Load the nearest ``.env`` into ``os.environ``. Returns the file used, or None.

    Search order: an explicit ``path``; then ``.env`` in the cwd and each parent; then
    the repo root relative to this file. The first file found wins."""
    if path is not None:  # explicit path: use only that (no fallback search)
        candidates: list[Path] = [Path(path)]
    else:
        cwd = Path.cwd()
        candidates = [cwd / ".env", *[p / ".env" for p in cwd.parents]]
        candidates.append(Path(__file__).resolve().parents[2] / ".env")

    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            lines = resolved.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        return str(resolved)
    return None
