"""Catalog of **base models** the training workflow can fine-tune or RL-tune.

Every entry is a Hugging Face id + metadata. The training code (`train/sft.py`,
`train/grpo.py`, `train/dpo.py`) is model-agnostic — it takes a `base_model` string —
so adding a model here immediately makes it trainable and selectable in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseModel:
    key: str
    hf_id: str
    family: str
    params: str
    arch: str  # "encoder" | "decoder"
    gated: bool = False  # needs HF_TOKEN + license acceptance
    techniques: tuple[str, ...] = ("sft", "grpo", "dpo")
    note: str = ""


#: The registry. Encoders classify; decoders emit a JSON verdict (and can be RL-tuned).
BASE_MODELS: dict[str, BaseModel] = {
    "distilbert": BaseModel(
        "distilbert", "distilbert-base-uncased", "BERT", "66M", "encoder", techniques=("sft",),
        note="Latency hero — the fast default encoder guard."),
    "modernbert-large": BaseModel(
        "modernbert-large", "answerdotai/ModernBERT-large", "BERT", "395M", "encoder",
        techniques=("sft",), note="Higher-capacity encoder."),
    "qwen3-0.6b": BaseModel(
        "qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen3", "0.6B", "decoder"),
    "qwen3-1.7b": BaseModel(
        "qwen3-1.7b", "Qwen/Qwen3-1.7B", "Qwen3", "1.7B", "decoder"),
    # --- focus models: the promising small guards we optimize for the paper ---
    "qwen2.5-1.5b": BaseModel(
        "qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5", "1.5B", "decoder",
        note="True 1.5B instruct base — the paper's canonical '1.5B guard'."),
    "smollm3-3b": BaseModel(
        "smollm3-3b", "HuggingFaceTB/SmolLM3-3B", "SmolLM3", "3B", "decoder",
        note="Strong small model (3B — above the strict 1.5B tier; note in the claim)."),
    # --- newly supported SLMs ---
    "deepseek-r1-1.5b": BaseModel(
        "deepseek-r1-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1", "1.5B",
        "decoder", note="Reasoning-distilled — a natural fit for the GRPO reasoning guard."),
    "smollm2-1.7b": BaseModel(
        "smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2", "1.7B", "decoder"),
    "gemma-1b": BaseModel(
        "gemma-1b", "google/gemma-3-1b-it", "Gemma", "1B", "decoder", gated=True,
        note="Gated — needs HF_TOKEN + accepting Google's license."),
}

#: Techniques the training UI/CLI exposes.
TECHNIQUES: dict[str, str] = {
    "sft": "Supervised fine-tuning",
    "grpo": "GRPO · reinforcement learning (verifiable reward)",
    "dpo": "DPO · preference tuning (over-refusal)",
}


def get_base_model(key: str) -> BaseModel:
    if key not in BASE_MODELS:
        raise ValueError(f"unknown base model {key!r}; known: {sorted(BASE_MODELS)}")
    return BASE_MODELS[key]


def valid_techniques(key: str) -> tuple[str, ...]:
    """Techniques applicable to a model (encoders: SFT only; decoders: SFT/GRPO/DPO)."""
    return get_base_model(key).techniques


def is_valid_combo(key: str, technique: str) -> bool:
    """Whether ``technique`` may be applied to base model ``key`` — the UI/API gate."""
    return technique in get_base_model(key).techniques


def assert_valid_combo(key: str, technique: str) -> None:
    """Raise a clear error for an unsupported model×technique pairing."""
    if technique not in TECHNIQUES:
        raise ValueError(f"unknown technique {technique!r}; known: {sorted(TECHNIQUES)}")
    if not is_valid_combo(key, technique):
        raise ValueError(
            f"technique {technique!r} is not supported by {key!r}; "
            f"valid for this model: {list(valid_techniques(key))}"
        )


def technique_matrix() -> dict[str, list[str]]:
    """``{model_key: [valid techniques]}`` — lets the UI grey out invalid combinations."""
    return {key: list(m.techniques) for key, m in BASE_MODELS.items()}


def catalog() -> list[dict]:
    """Registry as plain dicts for the API/UI."""
    return [
        {"key": m.key, "hf_id": m.hf_id, "family": m.family, "params": m.params,
         "arch": m.arch, "gated": m.gated, "techniques": list(m.techniques), "note": m.note}
        for m in BASE_MODELS.values()
    ]
