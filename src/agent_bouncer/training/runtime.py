"""Shared runtime helpers for decoder training.

TRL 1.7 loads string model ids through a helper that defaults `device_map="auto"`.
On single-device Mac runs this can shard/offload the model, leaving meta tensors
that explode during backward once LoRA adapters live on MPS. We avoid that by
instantiating decoder models ourselves with an explicit single-device policy.
"""

from __future__ import annotations

from typing import Any


def training_device() -> str:
    """Best-effort single-device target for local training."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decoder_bf16_enabled(train_cfg: dict[str, Any] | None, device: str) -> bool:
    """Default to bf16 on Apple Silicon unless the config explicitly sets bf16."""
    train_cfg = train_cfg or {}
    if "bf16" in train_cfg:
        return bool(train_cfg["bf16"])
    return device == "mps"


def decoder_model_load_kwargs(
    train_cfg: dict[str, Any] | None,
    *,
    device: str,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    import torch

    use_bf16 = decoder_bf16_enabled(train_cfg, device)
    return {
        "dtype": torch.bfloat16 if use_bf16 else torch.float32,
        "device_map": None,
        "low_cpu_mem_usage": False,
        "trust_remote_code": trust_remote_code,
    }


def load_decoder_training_assets(
    model_id: str,
    *,
    train_cfg: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
):
    """Instantiate a decoder model/tokenizer for local training on one device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = training_device()
    kwargs = decoder_model_load_kwargs(train_cfg, device=device, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model, tokenizer, device, decoder_bf16_enabled(train_cfg, device)
