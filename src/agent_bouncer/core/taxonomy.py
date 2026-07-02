"""Unified hazard taxonomy.

Aligned to the MLCommons AILuminate hazard set and extended with two
agent-specific surfaces — prompt injection and jailbreak — that generic content
moderators miss. Every training dataset (WildGuardMix, BeaverTails, Aegis 2.0) is
mapped onto *this* schema in `agent_bouncer.data` so results are comparable.
"""

from __future__ import annotations

from enum import Enum


class Hazard(str, Enum):
    """A single, canonical set of hazard categories."""

    NONE = "none"
    # --- MLCommons-aligned content hazards ---
    VIOLENT_CRIMES = "violent_crimes"
    NON_VIOLENT_CRIMES = "non_violent_crimes"
    SEX_CRIMES = "sex_crimes"
    CHILD_EXPLOITATION = "child_exploitation"
    WEAPONS_CBRNE = "weapons_cbrne"
    SUICIDE_SELF_HARM = "suicide_self_harm"
    HATE = "hate"
    PRIVACY = "privacy"  # PII / data leakage
    SEXUAL_CONTENT = "sexual_content"
    MALICIOUS_CODE = "malicious_code"
    # --- agent-specific surfaces (Agent Bouncer's differentiator) ---
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"

    @property
    def is_agentic(self) -> bool:
        """True for hazards that target the agent/tool layer, not just content."""
        return self in {Hazard.PROMPT_INJECTION, Hazard.JAILBREAK}


#: All hazards except the sentinel NONE — the label space for classification.
HAZARD_CATEGORIES: list[Hazard] = [h for h in Hazard if h is not Hazard.NONE]
