"""Encoder guard: ModernBERT (or any BERT-family) + a classification head.

The latency hero — treats safety as text classification, so it is the fastest
option and anchors the latency SOTA comparison. Implements the `Guard` protocol.

`aggregate_unsafe` (pure) turns a label→prob distribution into a decision, so it
is unit-testable without a model, and works for both binary (safe/unsafe) and
multi-class (per-hazard) heads.
"""

from __future__ import annotations

import time

from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.core.taxonomy import Hazard

_SAFE_LABELS = {"safe", "none", "benign", "label_0", "0"}


def aggregate_unsafe(
    id2label: dict[int, str],
    probs: list[float],
    threshold: float = 0.5,
) -> tuple[Decision, Hazard, float]:
    """Collapse a class distribution into (decision, hazard, unsafe_probability)."""
    unsafe_prob = 0.0
    best_hazard_label: str | None = None
    best_hazard_prob = -1.0
    for idx, prob in enumerate(probs):
        label = str(id2label[idx]).strip().lower()
        if label in _SAFE_LABELS:
            continue
        unsafe_prob += prob
        if prob > best_hazard_prob:
            best_hazard_prob, best_hazard_label = prob, label

    if unsafe_prob < threshold:
        return Decision.SAFE, Hazard.NONE, unsafe_prob
    try:
        hazard = Hazard(best_hazard_label)
    except (ValueError, TypeError):
        hazard = Hazard.NON_VIOLENT_CRIMES  # binary head: "unsafe" isn't a hazard name
    return Decision.UNSAFE, hazard, unsafe_prob


class EncoderGuard:
    name = "encoder-guard"

    def __init__(self, model_path: str, *, name: str | None = None, threshold: float = 0.5,
                 max_length: int = 256) -> None:
        self.model_path = model_path
        self.threshold = threshold
        # Truncate inference inputs to the length the classifier was trained on (default 256,
        # matching the training default) so train/inference see the same window.
        self.max_length = max_length
        if name:
            self.name = name
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.eval()

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        import torch

        if self._model is None:
            self._load()
        start = time.perf_counter()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        decision, hazard, score = aggregate_unsafe(id2label, probs, self.threshold)
        return Verdict(
            decision=decision,
            hazard=hazard,
            score=score,
            surface=surface,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.name,
        )
