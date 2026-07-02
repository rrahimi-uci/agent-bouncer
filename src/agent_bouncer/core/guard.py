"""The `Guard` interface and a dependency-free reference implementation.

`KeywordGuard` is intentionally simple: it makes the package runnable end-to-end
(`agent-bouncer predict ...`, `make eval`) *before* any model is trained, and it
serves as the trivial baseline every trained model must beat. It is NOT meant for
production.
"""

from __future__ import annotations

import re
import time
from typing import Protocol, runtime_checkable

from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.core.taxonomy import Hazard


@runtime_checkable
class Guard(Protocol):
    """Anything that can screen text and return a `Verdict`."""

    name: str

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict: ...


# Minimal, transparent signal patterns for the reference baseline only.
_PATTERNS: dict[Hazard, list[str]] = {
    Hazard.JAILBREAK: [
        r"ignore (all|any|the|your|previous|prior) .{0,20}instructions",
        r"\bDAN\b",
        r"do anything now",
        r"developer mode",
    ],
    Hazard.PROMPT_INJECTION: [
        r"disregard .{0,30}(system|above|previous)",
        r"you are now (a|an|the)\b",
        r"new instructions?\s*:",
    ],
    Hazard.SUICIDE_SELF_HARM: [r"kill myself", r"how to .{0,20}(suicide|self[- ]?harm)"],
    Hazard.WEAPONS_CBRNE: [
        r"how to (make|build|synthesi[sz]e) .{0,30}(bomb|explosive|nerve agent|bioweapon)",
    ],
    Hazard.MALICIOUS_CODE: [r"write .{0,20}(ransomware|keylogger|malware|a virus)"],
    Hazard.PRIVACY: [r"\b\d{3}-\d{2}-\d{4}\b"],  # naive SSN pattern
}


class KeywordGuard:
    """Reference heuristic guard: fast, transparent, and dependency-free."""

    name = "keyword-baseline"

    def __init__(self) -> None:
        self._compiled = {
            hazard: [re.compile(p, re.IGNORECASE) for p in patterns]
            for hazard, patterns in _PATTERNS.items()
        }

    def predict(self, text: str, *, surface: Surface = Surface.USER_PROMPT) -> Verdict:
        start = time.perf_counter()
        hazard, score = Hazard.NONE, 0.0
        for candidate, patterns in self._compiled.items():
            if any(p.search(text or "") for p in patterns):
                hazard, score = candidate, 1.0
                break
        decision = Decision.UNSAFE if hazard is not Hazard.NONE else Decision.SAFE
        return Verdict(
            decision=decision,
            hazard=hazard,
            score=score,
            surface=surface,
            latency_ms=(time.perf_counter() - start) * 1000,
            model=self.name,
        )
