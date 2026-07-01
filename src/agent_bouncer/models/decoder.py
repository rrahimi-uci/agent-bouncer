"""Decoder guard: a small instruct model (Qwen3-0.6B/1.7B, Llama-3.2-1B).

Two modes:
- ``sft``       : directly emits ``safe/unsafe + hazard`` (fast).
- ``reasoning`` : emits a short rationale then the verdict (GRPO-trained).

Implements the `Guard` protocol; parses model output into a `Verdict`.
"""

from __future__ import annotations

from ..schema import Surface, Verdict


class DecoderGuard:
    name = "decoder-guard"

    def __init__(self, model_path: str, *, mode: str = "sft") -> None:
        self.model_path = model_path
        self.mode = mode
        # TODO: lazy-load model (transformers or mlx-lm) here.

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        raise NotImplementedError(
            "TODO: build prompt -> generate -> parse_verdict() -> Verdict (+ latency_ms)."
        )

    @staticmethod
    def parse_verdict(raw_output: str) -> Verdict | None:
        """Parse model text into a `Verdict`; return None on malformed output
        (drives the format reward in rewards.py)."""
        raise NotImplementedError("TODO: parse structured output into Verdict.")
