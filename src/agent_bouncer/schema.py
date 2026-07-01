"""Typed I/O contract for a guard verdict.

Every guard — heuristic, encoder, or RL-tuned decoder — returns a `Verdict`, so
the eval harness, serving layer, and reward functions all speak one language.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .taxonomy import Hazard


class Decision(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class Surface(str, Enum):
    """What is being screened. Named because guarding *agents* means more than
    just the user's prompt."""

    USER_PROMPT = "user_prompt"
    TOOL_CALL = "tool_call"
    AGENT_OUTPUT = "agent_output"


class Verdict(BaseModel):
    """The result of screening one piece of text."""

    decision: Decision
    hazard: Hazard = Hazard.NONE
    score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence that input is unsafe")
    rationale: str | None = None
    surface: Surface = Surface.USER_PROMPT
    latency_ms: float | None = None
    model: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision == Decision.UNSAFE
