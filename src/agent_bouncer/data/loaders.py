"""Dataset loading and taxonomy unification.

Each source dataset labels harm in its own scheme; we map them all onto one
`Hazard` taxonomy so training and evaluation are comparable across sources.

Design:
- *Normalizers* are pure functions ``raw_row -> record dict`` — unit-tested with
  no network (see tests/test_data.py).
- *Loaders* add the Hugging Face download on top; the heavy ``datasets`` import is
  lazy, so the core package stays dependency-light.

A record is ``{"text": str, "label": "safe"|"unsafe", "hazard": str, "source": str}``
— exactly what the eval harness and JSONL files consume.

NOTE: dataset field/category names occasionally change between releases. The maps
below reflect the dataset cards as of early 2026. If a load returns zero unsafe
rows, verify field names against the current card and adjust the constants.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable

from agent_bouncer.core.schema import Decision
from agent_bouncer.core.taxonomy import Hazard

#: Unmapped-but-unsafe rows fall back to this hazard.
DEFAULT_UNSAFE_HAZARD = Hazard.NON_VIOLENT_CRIMES


# --------------------------------------------------------------------------- maps
BEAVERTAILS_MAP: dict[str, Hazard] = {
    "animal_abuse": Hazard.NON_VIOLENT_CRIMES,
    "child_abuse": Hazard.CHILD_EXPLOITATION,
    "controversial_topics,politics": Hazard.NON_VIOLENT_CRIMES,
    "discrimination,stereotype,injustice": Hazard.HATE,
    "drug_abuse,weapons,banned_substance": Hazard.WEAPONS_CBRNE,
    "financial_crime,property_crime,theft": Hazard.NON_VIOLENT_CRIMES,
    "hate_speech,offensive_language": Hazard.HATE,
    "misinformation_regarding_ethics,laws_and_safety": Hazard.NON_VIOLENT_CRIMES,
    "non_violent_unethical_behavior": Hazard.NON_VIOLENT_CRIMES,
    "privacy_violation": Hazard.PRIVACY,
    "self_harm": Hazard.SUICIDE_SELF_HARM,
    "sexually_explicit,adult_content": Hazard.SEXUAL_CONTENT,
    "terrorism,organized_crime": Hazard.VIOLENT_CRIMES,
    "violence,aiding_and_abetting,incitement": Hazard.VIOLENT_CRIMES,
}

# Only WildGuard subcategories that differ from DEFAULT_UNSAFE_HAZARD are listed.
WILDGUARD_SUBCATEGORY_MAP: dict[str, Hazard] = {
    "violence_and_physical_harm": Hazard.VIOLENT_CRIMES,
    "toxic_language_hate_speech": Hazard.HATE,
    "social_stereotypes_and_unfair_discrimination": Hazard.HATE,
    "sexual_content": Hazard.SEXUAL_CONTENT,
    "mental_health_over-reliance_crisis": Hazard.SUICIDE_SELF_HARM,
    "private_information_individual": Hazard.PRIVACY,
    "sensitive_information_organization_government": Hazard.PRIVACY,
    "cyberattacks": Hazard.MALICIOUS_CODE,
}

# Substring-matched (case-insensitive) against Aegis's violated-category text.
AEGIS_MAP: dict[str, Hazard] = {
    "sexual (minor)": Hazard.CHILD_EXPLOITATION,
    "suicide and self harm": Hazard.SUICIDE_SELF_HARM,
    "guns and illegal weapons": Hazard.WEAPONS_CBRNE,
    "hate/identity hate": Hazard.HATE,
    "harassment": Hazard.HATE,
    "profanity": Hazard.HATE,
    "pii/privacy": Hazard.PRIVACY,
    "malware": Hazard.MALICIOUS_CODE,
    "sexual": Hazard.SEXUAL_CONTENT,
    "threat": Hazard.VIOLENT_CRIMES,
    "violence": Hazard.VIOLENT_CRIMES,
    "criminal planning/confessions": Hazard.NON_VIOLENT_CRIMES,
}


# OpenAI moderation eval flag columns -> our taxonomy (checked in this order).
OPENAI_MOD_EVAL_MAP: dict[str, Hazard] = {
    "S": Hazard.SEXUAL_CONTENT,  # sexual
    "H": Hazard.HATE,  # hate
    "V": Hazard.VIOLENT_CRIMES,  # violence
    "HR": Hazard.HATE,  # harassment
    "SH": Hazard.SUICIDE_SELF_HARM,  # self-harm
    "S3": Hazard.CHILD_EXPLOITATION,  # sexual/minors
    "H2": Hazard.HATE,  # hate/threatening
    "V2": Hazard.VIOLENT_CRIMES,  # violence/graphic
}


# ---------------------------------------------------------------------- helpers
def _record(text: str, label: Decision, hazard: Hazard, source: str) -> dict:
    return {"text": text, "label": label.value, "hazard": hazard.value, "source": source}


def _text(row: dict, field: str) -> str:
    return (row.get(field) or row.get("text") or "").strip()


def _is_true(value: object) -> bool:
    """Coerce a dataset flag (bool / int / '0'|'1' / 'true'|'yes') to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "unsafe", "harmful"}
    return bool(value)


# ------------------------------------------------------------------ normalizers
def normalize_beavertails(row: dict, text_field: str = "prompt", source: str = "beavertails") -> dict | None:
    text = _text(row, text_field)
    if not text:
        return None
    if bool(row.get("is_safe", False)):
        return _record(text, Decision.SAFE, Hazard.NONE, source)
    hazard = DEFAULT_UNSAFE_HAZARD
    for key, flagged in (row.get("category") or {}).items():
        if flagged and key in BEAVERTAILS_MAP:
            hazard = BEAVERTAILS_MAP[key]
            break
    return _record(text, Decision.UNSAFE, hazard, source)


def normalize_wildguard(row: dict, text_field: str = "prompt", source: str = "wildguardmix") -> dict | None:
    text = _text(row, text_field)
    if not text:
        return None
    label = (row.get("prompt_harm_label") or "").strip().lower()
    if label == "unharmful":
        return _record(text, Decision.SAFE, Hazard.NONE, source)
    if label != "harmful":
        return None  # unlabeled — skip
    sub = (row.get("subcategory") or "").strip().lower()
    hazard = WILDGUARD_SUBCATEGORY_MAP.get(sub, DEFAULT_UNSAFE_HAZARD)
    return _record(text, Decision.UNSAFE, hazard, source)


def normalize_aegis(row: dict, text_field: str = "prompt", source: str = "aegis") -> dict | None:
    text = _text(row, text_field)
    if not text:
        return None
    label = (row.get("prompt_label") or "").strip().lower()
    if label == "safe":
        return _record(text, Decision.SAFE, Hazard.NONE, source)
    if label != "unsafe":
        return None
    cats = row.get("violated_categories") or row.get("categories") or ""
    if isinstance(cats, (list, tuple)):
        cats = ",".join(map(str, cats))
    cats = str(cats).lower()
    hazard = next((hz for key, hz in AEGIS_MAP.items() if key in cats), DEFAULT_UNSAFE_HAZARD)
    return _record(text, Decision.UNSAFE, hazard, source)


def normalize_xstest(row: dict, text_field: str = "prompt", source: str = "xstest") -> dict | None:
    """XSTest mixes safe prompts with unsafe 'contrast' prompts; keep both, labeled."""
    text = _text(row, text_field)
    if not text:
        return None
    # Real XSTest encodes unsafe prompts via a `contrast_`-prefixed `type`
    # (e.g. "contrast_homonyms"); fall back to an explicit label if present.
    kind = (row.get("type") or row.get("label") or "safe").strip().lower()
    unsafe = kind.startswith("contrast") or kind in {"unsafe", "harmful"}
    return _record(
        text,
        Decision.UNSAFE if unsafe else Decision.SAFE,
        DEFAULT_UNSAFE_HAZARD if unsafe else Hazard.NONE,
        source,
    )


def normalize_prompt_injection(
    row: dict, text_field: str = "text", source: str = "prompt_injections"
) -> dict | None:
    """deepset/prompt-injections: ``label`` 1 = injection attempt (unsafe), 0 = benign."""
    text = _text(row, text_field)
    if not text:
        return None
    unsafe = _is_true(row.get("label"))
    return _record(
        text,
        Decision.UNSAFE if unsafe else Decision.SAFE,
        Hazard.PROMPT_INJECTION if unsafe else Hazard.NONE,
        source,
    )


def normalize_jailbreak_classification(
    row: dict, text_field: str = "prompt", source: str = "jailbreak_classification"
) -> dict | None:
    """jackhhao/jailbreak-classification: ``type`` in {benign, jailbreak}."""
    text = _text(row, text_field)
    if not text:
        return None
    unsafe = (row.get("type") or "").strip().lower() == "jailbreak"
    return _record(
        text,
        Decision.UNSAFE if unsafe else Decision.SAFE,
        Hazard.JAILBREAK if unsafe else Hazard.NONE,
        source,
    )


def normalize_openai_moderation(
    row: dict, text_field: str = "prompt", source: str = "openai_moderation"
) -> dict | None:
    """mmathys/openai-moderation-api-evaluation: 8 category flag columns; a row is
    unsafe iff any category is set. The first set category picks the hazard."""
    text = _text(row, text_field)
    if not text:
        return None
    hazard = Hazard.NONE
    for code, hz in OPENAI_MOD_EVAL_MAP.items():
        if _is_true(row.get(code)):
            hazard = hz
            break
    unsafe = hazard is not Hazard.NONE
    return _record(text, Decision.UNSAFE if unsafe else Decision.SAFE, hazard, source)


def normalize_toxicchat(row: dict, text_field: str = "user_input", source: str = "toxicchat") -> dict | None:
    """lmsys/toxic-chat: ``toxicity`` 0/1 is the headline label (a separate
    ``jailbreaking`` column exists; we key on toxicity to match the standard task)."""
    text = _text(row, text_field)
    if not text:
        return None
    unsafe = _is_true(row.get("toxicity"))
    return _record(
        text,
        Decision.UNSAFE if unsafe else Decision.SAFE,
        DEFAULT_UNSAFE_HAZARD if unsafe else Hazard.NONE,
        source,
    )


def normalize_jailbreakbench(
    row: dict, *, unsafe: bool, text_field: str = "Goal", source: str = "jailbreakbench"
) -> dict | None:
    """JailbreakBench/JBB-Behaviors: the label comes from the split (harmful/benign),
    not the row, so the loader passes ``unsafe`` explicitly. ``Goal`` holds the prompt."""
    text = _text(row, text_field)
    if not text:
        return None
    return _record(
        text,
        Decision.UNSAFE if unsafe else Decision.SAFE,
        DEFAULT_UNSAFE_HAZARD if unsafe else Hazard.NONE,
        source,
    )


_NORMALIZERS = {
    "beavertails": normalize_beavertails,
    "wildguardmix": normalize_wildguard,
    "aegis": normalize_aegis,
    "xstest": normalize_xstest,
    "prompt_injections": normalize_prompt_injection,
    "jailbreak_classification": normalize_jailbreak_classification,
    "openai_moderation": normalize_openai_moderation,
    "toxicchat": normalize_toxicchat,
}


def unify_to_taxonomy(records: Iterable[dict], source: str, text_field: str = "prompt") -> list[dict]:
    """Normalize raw rows from `source` into unified records (dropping empties)."""
    normalize = _NORMALIZERS.get(source)
    if normalize is None:
        raise ValueError(f"unknown source {source!r}; known: {sorted(_NORMALIZERS)}")
    out = []
    for row in records:
        rec = normalize(row, text_field=text_field)
        if rec is not None:
            out.append(rec)
    return out


# ----------------------------------------------------------------- split / io
def train_val_split(
    records: list[dict], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle + split. Returns (train, validation)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(records: Iterable[dict], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------- loaders
def _load_hf(hf_id: str, split: str, config: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the train extra: pip install -e '.[train]'") from exc
    return load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)


def load_wildguardmix(split: str = "train", text_field: str = "prompt") -> list[dict]:
    # gated dataset — set HF_TOKEN. Config selects the labeled train split.
    ds = _load_hf("allenai/wildguardmix", split, config="wildguardtrain")
    return unify_to_taxonomy(ds, "wildguardmix", text_field=text_field)


def load_beavertails(split: str = "30k_train", text_field: str = "prompt") -> list[dict]:
    ds = _load_hf("PKU-Alignment/BeaverTails", split)
    return unify_to_taxonomy(ds, "beavertails", text_field=text_field)


def load_aegis(split: str = "train", text_field: str = "prompt") -> list[dict]:
    ds = _load_hf("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split)
    return unify_to_taxonomy(ds, "aegis", text_field=text_field)


def load_xstest(split: str = "prompts", text_field: str = "prompt") -> list[dict]:
    ds = _load_hf("natolambert/xstest-v2-copy", split)
    return unify_to_taxonomy(ds, "xstest", text_field=text_field)


def load_prompt_injections(split: str = "test", text_field: str = "text") -> list[dict]:
    # deepset/prompt-injections — ungated prompt-injection detection benchmark.
    ds = _load_hf("deepset/prompt-injections", split)
    return unify_to_taxonomy(ds, "prompt_injections", text_field=text_field)


def load_jailbreak_classification(split: str = "test", text_field: str = "prompt") -> list[dict]:
    # jackhhao/jailbreak-classification — ungated jailbreak detection benchmark.
    ds = _load_hf("jackhhao/jailbreak-classification", split)
    return unify_to_taxonomy(ds, "jailbreak_classification", text_field=text_field)


def load_openai_moderation(split: str = "train", text_field: str = "prompt") -> list[dict]:
    # mmathys/openai-moderation-api-evaluation — content-moderation eval set.
    ds = _load_hf("mmathys/openai-moderation-api-evaluation", split)
    return unify_to_taxonomy(ds, "openai_moderation", text_field=text_field)


def load_toxicchat(split: str = "test", text_field: str = "user_input") -> list[dict]:
    # lmsys/toxic-chat (toxicchat0124) — toxicity detection. Non-commercial license:
    # use for internal evaluation only, not for training a released model.
    ds = _load_hf("lmsys/toxic-chat", split, config="toxicchat0124")
    return unify_to_taxonomy(ds, "toxicchat", text_field=text_field)


def load_jailbreakbench(text_field: str = "Goal") -> list[dict]:
    # JailbreakBench/JBB-Behaviors — 100 harmful + 100 benign red-teaming behaviors.
    # The label lives in the split name, so we build records per split directly.
    out: list[dict] = []
    for split, unsafe in (("harmful", True), ("benign", False)):
        ds = _load_hf("JailbreakBench/JBB-Behaviors", split, config="behaviors")
        for row in ds:
            rec = normalize_jailbreakbench(row, unsafe=unsafe, text_field=text_field)
            if rec is not None:
                out.append(rec)
    return out


LOADERS = {
    "wildguardmix": load_wildguardmix,
    "beavertails": load_beavertails,
    "aegis": load_aegis,
    "xstest": load_xstest,
    "prompt_injections": load_prompt_injections,
    "jailbreak_classification": load_jailbreak_classification,
    "openai_moderation": load_openai_moderation,
    "toxicchat": load_toxicchat,
    "jailbreakbench": load_jailbreakbench,
}
