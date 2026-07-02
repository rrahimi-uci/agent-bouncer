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


def catalog() -> list[dict]:
    """Registry as plain dicts for the API/UI."""
    return [
        {"key": m.key, "hf_id": m.hf_id, "family": m.family, "params": m.params,
         "arch": m.arch, "gated": m.gated, "techniques": list(m.techniques), "note": m.note}
        for m in BASE_MODELS.values()
    ]
