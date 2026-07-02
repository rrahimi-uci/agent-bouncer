"""Zero-shot incumbent guards, wrapped in the `Guard` interface for apples-to-apples
comparison in our harness. These are the bar Agent Bouncer must beat — on quality,
on false-positive-on-benign, and on latency.

The output *parsers* are pure functions (unit-tested with sample outputs, no
download). The model wrappers are lazy and require the gated checkpoints (set
`HF_TOKEN`); importing this module never triggers a download.
"""

from __future__ import annotations

import time

from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.core.taxonomy import Hazard

# Llama Guard hazard codes (S1..S13) -> our taxonomy.
LLAMA_GUARD_CODE_MAP: dict[str, Hazard] = {
    "S1": Hazard.VIOLENT_CRIMES,
    "S2": Hazard.NON_VIOLENT_CRIMES,
    "S3": Hazard.SEX_CRIMES,
    "S4": Hazard.CHILD_EXPLOITATION,
    "S5": Hazard.NON_VIOLENT_CRIMES,  # defamation
    "S6": Hazard.NON_VIOLENT_CRIMES,  # specialized advice
    "S7": Hazard.PRIVACY,
    "S8": Hazard.NON_VIOLENT_CRIMES,  # intellectual property
    "S9": Hazard.WEAPONS_CBRNE,
    "S10": Hazard.HATE,
    "S11": Hazard.SUICIDE_SELF_HARM,
    "S12": Hazard.SEXUAL_CONTENT,
    "S13": Hazard.NON_VIOLENT_CRIMES,  # elections
}


def parse_llama_guard_output(
    text: str,
    *,
    surface: Surface = Surface.USER_PROMPT,
    latency_ms: float | None = None,
    model: str = "llama-guard",
) -> Verdict:
    """Llama Guard emits `safe` or `unsafe\\n<comma-separated S-codes>`."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    decision, hazard = Decision.SAFE, Hazard.NONE
    if lines and lines[0].lower().startswith("unsafe"):
        decision = Decision.UNSAFE
        hazard = Hazard.NON_VIOLENT_CRIMES  # unsafe but no parseable code
        codes = []
        for ln in lines[1:]:
            codes += [tok.strip().upper() for tok in ln.replace(",", " ").split()]
        for code in codes:
            if code in LLAMA_GUARD_CODE_MAP:
                hazard = LLAMA_GUARD_CODE_MAP[code]
                break
    return Verdict(
        decision=decision,
        hazard=hazard,
        score=1.0 if decision == Decision.UNSAFE else 0.0,
        surface=surface,
        latency_ms=latency_ms,
        model=model,
    )


def parse_yes_no_output(
    text: str,
    *,
    surface: Surface = Surface.USER_PROMPT,
    latency_ms: float | None = None,
    model: str = "shieldgemma",
) -> Verdict:
    """ShieldGemma-style: `Yes` means the content violates the policy (unsafe)."""
    unsafe = (text or "").strip().lower().startswith("yes")
    return Verdict(
        decision=Decision.UNSAFE if unsafe else Decision.SAFE,
        hazard=Hazard.NON_VIOLENT_CRIMES if unsafe else Hazard.NONE,
        score=1.0 if unsafe else 0.0,
        surface=surface,
        latency_ms=latency_ms,
        model=model,
    )


class _HFGuardBase:
    """Shared lazy model/tokenizer loading for HF-based baseline guards."""

    hf_id: str
    name: str

    def __init__(self, hf_id: str | None = None) -> None:
        if hf_id:
            self.hf_id = hf_id
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.hf_id)
        return self._model, self._tokenizer


class LlamaGuard(_HFGuardBase):
    hf_id = "meta-llama/Llama-Guard-3-1B"
    name = "llama-guard-3-1b"

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        model, tok = self._ensure_loaded()
        start = time.perf_counter()
        messages = [{"role": "user", "content": text}]
        inputs = tok.apply_chat_template(messages, return_tensors="pt", return_dict=True)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        decoded = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return parse_llama_guard_output(
            decoded, surface=surface, latency_ms=(time.perf_counter() - start) * 1000, model=self.name
        )


#: ShieldGemma requires its content wrapped in a policy guideline; predicting the
#: "Yes" token then means the content violates the policy.
SHIELDGEMMA_GUIDELINE = (
    "The prompt must not contain or request harmful content: violence, illegal "
    "acts, hate, harassment, sexual content, self-harm, or dangerous instructions."
)


class ShieldGemma(_HFGuardBase):
    hf_id = "google/shieldgemma-2b"
    name = "shieldgemma-2b"

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        model, tok = self._ensure_loaded()
        start = time.perf_counter()
        messages = [{"role": "user", "content": text}]
        inputs = tok.apply_chat_template(
            messages, guideline=SHIELDGEMMA_GUIDELINE, return_tensors="pt", return_dict=True
        )
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return parse_yes_no_output(
            decoded, surface=surface, latency_ms=(time.perf_counter() - start) * 1000, model=self.name
        )


class PromptGuard:
    """Meta PromptGuard2 — a sequence classifier for injection/jailbreak."""

    hf_id = "meta-llama/Llama-Prompt-Guard-2-86M"
    name = "prompt-guard-2-86m"

    def __init__(self, hf_id: str | None = None) -> None:
        self.hf_id = hf_id or self.hf_id
        self._pipe = None

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline("text-classification", model=self.hf_id)
        start = time.perf_counter()
        result = self._pipe(text, truncation=True)[0]
        malicious = str(result["label"]).upper() not in {"BENIGN", "LABEL_0", "SAFE"}
        return Verdict(
            decision=Decision.UNSAFE if malicious else Decision.SAFE,
            hazard=Hazard.JAILBREAK if malicious else Hazard.NONE,
            score=float(result.get("score", 1.0)) if malicious else 0.0,
            surface=surface,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.name,
        )


def load_llama_guard(model_id: str = LlamaGuard.hf_id) -> LlamaGuard:
    return LlamaGuard(model_id)


def load_shieldgemma(model_id: str = ShieldGemma.hf_id) -> ShieldGemma:
    return ShieldGemma(model_id)


def load_promptguard(model_id: str = PromptGuard.hf_id) -> PromptGuard:
    return PromptGuard(model_id)
