"""Agent Bouncer — a tiny, fast safety bouncer for LLMs and agents.

Screens user prompts, tool calls, and agent outputs *before* they reach (or leave)
your model, using a small, low-latency guardrail model. See README.md for the story.
"""

from agent_bouncer.config.envfile import load_env as _load_env
from agent_bouncer.config.logutil import quiet_load_report as _quiet_load_report
from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.core.taxonomy import HAZARD_CATEGORIES, Hazard

# Auto-load .env (OPENAI_API_KEY, HF_TOKEN, …) into the environment on import, so
# `datasets`/`huggingface_hub` authenticate everywhere. `setdefault` — real env wins.
_load_env()
# Drop transformers' benign "LOAD REPORT" warning so training/eval logs stay clean.
_quiet_load_report()

__version__ = "0.0.1"

__all__ = [
    "Decision",
    "Surface",
    "Verdict",
    "Hazard",
    "HAZARD_CATEGORIES",
    "__version__",
]
