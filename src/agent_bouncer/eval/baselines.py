"""Zero-shot incumbent guards, wrapped in the `Guard` interface for apples-to-apples
comparison in our harness. These are the bar Agent Bouncer must beat — on quality,
on false-positive-on-benign, and on latency.
"""

from __future__ import annotations

from ..guard import Guard


def load_llama_guard(model_id: str = "meta-llama/Llama-Guard-3-1B") -> Guard:
    raise NotImplementedError("TODO: wrap Llama Guard chat template -> Verdict.")


def load_shieldgemma(model_id: str = "google/shieldgemma-2b") -> Guard:
    raise NotImplementedError("TODO: wrap ShieldGemma -> Verdict.")


def load_promptguard(model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M") -> Guard:
    raise NotImplementedError("TODO: wrap PromptGuard2 (injection/jailbreak) -> Verdict.")
