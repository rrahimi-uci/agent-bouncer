"""Agent Bouncer — a tiny, fast safety bouncer for LLMs and agents.

Screens user prompts, tool calls, and agent outputs *before* they reach (or leave)
your model, using a small, low-latency guardrail model. See README.md for the story.
"""

from .schema import Decision, Surface, Verdict
from .taxonomy import HAZARD_CATEGORIES, Hazard

__version__ = "0.0.1"

__all__ = [
    "Decision",
    "Surface",
    "Verdict",
    "Hazard",
    "HAZARD_CATEGORIES",
    "__version__",
]
