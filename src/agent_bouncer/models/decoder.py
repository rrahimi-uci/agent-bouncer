"""Decoder guard: a small instruct model (Qwen3-0.6B/1.7B, Llama-3.2-1B).

This module owns the *canonical* prompt/target/parse format shared by SFT, GRPO,
and DPO — one source of truth so training and inference never drift. The
`build_prompt`, `format_target`, and `parse_verdict` functions are pure and
heavily unit-tested; only `DecoderGuard.predict` needs a loaded model.

Two modes:
- ``sft``       : directly emits a JSON verdict (fast).
- ``reasoning`` : emits ``<think>...</think>`` then the JSON verdict (GRPO-trained).
"""

from __future__ import annotations

import json
import re
import time

from ..schema import Decision, Surface, Verdict
from ..taxonomy import Hazard

SYSTEM_INSTRUCTION = (
    "You are Agent Bouncer, a safety classifier. Decide whether the input is safe "
    'or unsafe and respond with a JSON object: {"decision": "safe" or "unsafe", '
    '"hazard": one hazard name or "none"}.'
)


def build_prompt(text: str, surface: Surface = Surface.USER_PROMPT, *, reasoning: bool = False) -> str:
    instruction = SYSTEM_INSTRUCTION
    if reasoning:
        instruction += " First reason briefly inside <think>...</think>, then output the JSON."
    return f"{instruction}\n\n[{surface.value}]\n{text}\n\nVerdict:"


def format_target(verdict: Verdict, *, reasoning: bool = False) -> str:
    """The training target string for a verdict (labels for SFT / DPO)."""
    body = json.dumps({"decision": verdict.decision.value, "hazard": verdict.hazard.value})
    if reasoning and verdict.rationale:
        return f"<think>{verdict.rationale}</think>\n{body}"
    return body


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_DECISION_RE = re.compile(r'"?decision"?\s*[:=]\s*"?(safe|unsafe)"?', re.I)
_HAZARD_RE = re.compile(r'"?hazard"?\s*[:=]\s*"?([a-z_]+)"?', re.I)


def parse_verdict(
    text: str,
    *,
    surface: Surface = Surface.USER_PROMPT,
    latency_ms: float | None = None,
    model: str = "decoder-guard",
) -> Verdict | None:
    """Parse model output into a `Verdict`; return None on malformed output
    (this drives the format reward in `rewards.py`)."""
    if not text:
        return None

    rationale = None
    think = _THINK_RE.search(text)
    if think:
        rationale = think.group(1).strip()
        text = _THINK_RE.sub("", text)

    decision_str: str | None = None
    hazard_str: str | None = None
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        decision_str = str(obj.get("decision", "")).lower()
        hazard_str = str(obj.get("hazard", "none")).lower()
    except (ValueError, json.JSONDecodeError):
        if (m := _DECISION_RE.search(text)):
            decision_str = m.group(1).lower()
        if (m := _HAZARD_RE.search(text)):
            hazard_str = m.group(1).lower()

    if decision_str not in {"safe", "unsafe"}:
        return None
    decision = Decision(decision_str)

    hazard = Hazard.NONE
    if decision == Decision.UNSAFE and hazard_str:
        try:
            hazard = Hazard(hazard_str)
        except ValueError:
            hazard = Hazard.NON_VIOLENT_CRIMES

    return Verdict(
        decision=decision,
        hazard=hazard,
        score=1.0 if decision == Decision.UNSAFE else 0.0,
        rationale=rationale,
        surface=surface,
        latency_ms=latency_ms,
        model=model,
    )


class DecoderGuard:
    name = "decoder-guard"

    def __init__(self, model_path: str, *, mode: str = "sft", name: str | None = None) -> None:
        self.model_path = model_path
        self.mode = mode
        if name:
            self.name = name
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self._model.eval()

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        import torch

        if self._model is None:
            self._load()
        prompt = build_prompt(text, surface, reasoning=(self.mode == "reasoning"))
        inputs = self._tokenizer(prompt, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        decoded = self._tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        latency = (time.perf_counter() - start) * 1000
        verdict = parse_verdict(decoded, surface=surface, latency_ms=latency, model=self.name)
        if verdict is None:  # fail closed: unparseable output is treated as unsafe
            return Verdict(
                decision=Decision.UNSAFE,
                hazard=Hazard.NONE,
                score=0.5,
                surface=surface,
                latency_ms=latency,
                model=self.name,
            )
        return verdict
