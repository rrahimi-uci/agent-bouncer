"""Agent Bouncer — a tiny, fast safety bouncer for LLMs and agents.

Screens user prompts, tool calls, and agent outputs *before* they reach (or leave)
your model, using a small, low-latency guardrail model. See README.md for the story.
"""

from .envfile import load_env as _load_env
from .schema import Decision, Surface, Verdict
from .taxonomy import HAZARD_CATEGORIES, Hazard

# Auto-load .env (OPENAI_API_KEY, HF_TOKEN, …) into the environment on import, so
# `datasets`/`huggingface_hub` authenticate everywhere. `setdefault` — real env wins.
_load_env()

__version__ = "0.0.1"

__all__ = [
    "Decision",
    "Surface",
    "Verdict",
    "Hazard",
    "HAZARD_CATEGORIES",
    "__version__",
]
