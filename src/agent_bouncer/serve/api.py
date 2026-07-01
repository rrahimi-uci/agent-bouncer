"""Minimal FastAPI app exposing the guard as a /screen endpoint.

Install the serve extra: `pip install -e '.[serve]'`, then:
    uvicorn agent_bouncer.serve.api:app --reload
"""

from __future__ import annotations

from ..guard import KeywordGuard
from ..schema import Surface, Verdict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install the serve extra: pip install -e '.[serve]'") from exc


class ScreenRequest(BaseModel):
    text: str
    surface: Surface = Surface.USER_PROMPT


app = FastAPI(title="Agent Bouncer", version="0.0.1")
_guard = KeywordGuard()  # TODO: load a trained guard via env/config


@app.post("/screen", response_model=Verdict)
def screen(req: ScreenRequest) -> Verdict:
    return _guard.predict(req.text, surface=req.surface)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "guard": _guard.name}
