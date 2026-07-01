"""Encoder guard: ModernBERT + classification head.

The latency hero — treats safety as text classification, so it is the fastest
option (targets <30 ms on CPU) and anchors the latency SOTA comparison.
Implements the `Guard` protocol.
"""

from __future__ import annotations

from ..schema import Surface, Verdict


class EncoderGuard:
    name = "modernbert-encoder"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        # TODO: lazy-load transformers model + tokenizer here.

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        raise NotImplementedError("TODO: tokenize -> forward -> map logits to Verdict (+ latency_ms).")
