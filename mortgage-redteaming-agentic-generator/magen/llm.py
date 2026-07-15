"""Provider-agnostic LLM client + a deterministic offline mock.

The pipeline talks to models only through `LLMClient.complete(system, user, *, json=…)`.
Three backends:

  - ``mock``      — no network, no keys. Deterministic pseudo-responses seeded by a hash of
                    the prompt, so the whole build is reproducible and testable offline.
  - ``openai``    — needs OPENAI_API_KEY + `pip install openai`.
  - ``anthropic`` — needs ANTHROPIC_API_KEY + `pip install anthropic`.

Agents (magen/agents.py) request JSON; the client is responsible for returning parseable
JSON (real providers via response-format / tool-use; the mock synthesises it).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Any, Callable


def _seed_int(*parts: str) -> int:
    h = hashlib.blake2b("".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


class LLMClient:
    """One object per run; tracks a call budget and dispatches to the chosen backend."""

    def __init__(self, cfg: dict[str, Any], *, mock_handler: Callable[[str, str], Any] | None = None):
        self.provider = cfg.get("provider", "mock")
        self.generator_model = cfg.get("generator_model", "")
        self.judge_model = cfg.get("judge_model", "")
        self.temperature = float(cfg.get("temperature", 0.7))
        self.max_tokens = int(cfg.get("max_tokens", 700))
        self.max_calls = int(cfg.get("max_llm_calls", 10_000))
        self.calls = 0
        self._lock = threading.Lock()          # guards the call counter under parallelism
        self._mock_handler = mock_handler
        self._client = None
        if self.provider != "mock":
            self._client = self._build_real_client()

    # ------------------------------------------------------------------ public API
    def complete(self, system: str, user: str, *, model: str | None = None,
                 temperature: float | None = None, want_json: bool = True) -> str:
        """Return the model's raw text (JSON string when want_json)."""
        with self._lock:
            if self.calls >= self.max_calls:
                raise RuntimeError(f"LLM call budget exhausted ({self.max_calls})")
            self.calls += 1
        temp = self.temperature if temperature is None else temperature
        mdl = model or self.generator_model
        if self.provider == "mock":
            return self._mock(system, user, want_json=want_json)
        if self.provider == "openai":
            return self._openai(system, user, mdl, temp, want_json)
        if self.provider == "anthropic":
            return self._anthropic(system, user, mdl, temp, want_json)
        raise ValueError(f"unknown provider {self.provider!r}")

    def complete_json(self, system: str, user: str, **kw: Any) -> Any:
        """complete() + robust JSON parse (tolerant of ```json fences / surrounding prose)."""
        raw = self.complete(system, user, want_json=True, **kw)
        return _parse_json(raw)

    # ------------------------------------------------------------------ mock backend
    def _mock(self, system: str, user: str, *, want_json: bool) -> str:
        if self._mock_handler is None:
            raise RuntimeError("mock provider selected but no mock_handler supplied")
        out = self._mock_handler(system, user)
        return out if isinstance(out, str) else json.dumps(out)

    # ------------------------------------------------------------------ real backends
    def _build_real_client(self):  # pragma: no cover - exercised only online
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic()
        raise ValueError(self.provider)

    def _openai(self, system: str, user: str, model: str, temp: float,
                want_json: bool) -> str:  # pragma: no cover - online only
        # GPT-5.x / o-series quirks: use max_completion_tokens, and they only accept the
        # default temperature (omit it). Older gpt-4* take max_tokens + temperature.
        reasoning = model.startswith("gpt-5") or model.startswith("o")
        base: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
        if reasoning:
            base["max_completion_tokens"] = self.max_tokens
        else:
            base["max_tokens"] = self.max_tokens
            base["temperature"] = temp
        attempts = ([{**base, "response_format": {"type": "json_object"}}] if want_json else [])
        attempts.append(base)
        last: Exception | None = None
        for kw in attempts:
            try:
                resp = self._client.chat.completions.create(**kw)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry without response_format on 400s
                last = e
        raise last  # type: ignore[misc]

    def _anthropic(self, system: str, user: str, model: str, temp: float,
                   want_json: bool) -> str:  # pragma: no cover - online only
        if want_json:
            user = user + "\n\nRespond with a single valid JSON object and nothing else."
        resp = self._client.messages.create(
            model=model, system=system, temperature=temp, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in resp.content)


# ------------------------------------------------------------------ JSON parsing helper
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # last resort: grab the outermost {...}
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"could not parse JSON from model output: {raw[:200]!r}")


def env_ready(provider: str) -> bool:
    """True if the env is configured for a real provider (used to warn early)."""
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return True
