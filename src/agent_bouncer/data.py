"""Dataset loading and taxonomy unification.

All source datasets are Apache-2.0 (see docs/datasets.md). The job here is to map
each dataset's native labels onto our single `Hazard` taxonomy so training and
evaluation are comparable across sources.

Status: taxonomy maps are sketched; the actual HF download/normalization is a TODO
(kept out of import time so the core package has no heavy dependencies).
"""

from __future__ import annotations

from collections.abc import Iterable

from .taxonomy import Hazard

#: Native-label -> unified Hazard maps. Extend as datasets are wired in.
BEAVERTAILS_MAP: dict[str, Hazard] = {
    "violence,aiding_and_abetting,incitement": Hazard.VIOLENT_CRIMES,
    "hate_speech,offensive_language": Hazard.HATE,
    "self_harm": Hazard.SUICIDE_SELF_HARM,
    "privacy_violation": Hazard.PRIVACY,
    "sexually_explicit,adult_content": Hazard.SEXUAL_CONTENT,
    # ... complete against the BeaverTails 14-category schema
}

WILDGUARDMIX_MAP: dict[str, Hazard] = {
    "prompt_harm": Hazard.NON_VIOLENT_CRIMES,  # refine per WildGuard subcategory
    "jailbreak": Hazard.JAILBREAK,
}

AEGIS_MAP: dict[str, Hazard] = {
    "Suicide and Self Harm": Hazard.SUICIDE_SELF_HARM,
    "Guns / Illegal Weapons": Hazard.WEAPONS_CBRNE,
    "PII/Privacy": Hazard.PRIVACY,
    # ... complete against the Aegis 2.0 taxonomy
}


def unify_to_taxonomy(records: Iterable[dict], source: str) -> list[dict]:
    """Normalize raw dataset rows to `{'text': str, 'label': 'safe'|'unsafe',
    'hazard': Hazard, 'source': str}`."""
    raise NotImplementedError(
        "TODO: implement per-source normalization using the *_MAP tables above. "
        "See docs/datasets.md for field-by-field mapping."
    )


def load_wildguardmix(split: str = "train") -> list[dict]:
    raise NotImplementedError(
        "TODO: load_dataset('allenai/wildguardmix') then unify_to_taxonomy(...)"
    )


def load_beavertails(split: str = "train") -> list[dict]:
    raise NotImplementedError(
        "TODO: load_dataset('PKU-Alignment/BeaverTails') then unify_to_taxonomy(...)"
    )


def load_aegis(split: str = "train") -> list[dict]:
    raise NotImplementedError(
        "TODO: load_dataset('nvidia/Aegis-AI-Content-Safety-Dataset-2.0') then unify_to_taxonomy(...)"
    )
