"""Single source of truth for training hyperparameters.

For each ``(arch, technique)`` combination this declares exactly which parameters
apply, their **recommended** (best-known) value, and the **accepted** values. It
drives the Studio's settings panel (so the UI only offers valid choices, pre-filled
with the recommended value) and backs server-side validation, so a run can never be
launched with an out-of-range value.

Field ``kind``:
- ``"select"`` → a closed set (``options``); the UI renders a dropdown.
- ``"int"`` / ``"float"`` → a bounded number (``min``/``max``/``step``); may be
  ``optional`` (empty is allowed, e.g. an uncapped "max steps").
"""

from __future__ import annotations


def _p(name, label, kind, default, *, options=None, min=None, max=None, step=None,
       optional=False, help=""):
    return {"name": name, "label": label, "kind": kind, "default": default,
            "options": options, "min": min, "max": max, "step": step,
            "optional": optional, "help": help}


# LoRA knobs shared by every decoder technique.
def _lora() -> list[dict]:
    return [
        _p("lora_r", "LoRA rank (r)", "select", 16, options=[8, 16, 32, 64],
           help="Adapter capacity. 16 is a strong default; 32–64 for harder or larger data."),
        _p("lora_alpha", "LoRA alpha", "select", 32, options=[16, 32, 64],
           help="Scaling factor — keep roughly 2× the rank."),
        _p("lora_dropout", "LoRA dropout", "select", 0.05, options=[0.0, 0.05, 0.1],
           help="Regularization; 0.05 is a safe default."),
    ]


def _max_steps(default=None) -> dict:
    return _p("max_steps", "Max steps (cap)", "int", default, min=1, max=100000, step=1,
              optional=True, help="Empty = train the full epochs. Set a number to time-box a run.")


def param_spec(arch: str, technique: str) -> list[dict]:
    """Ordered list of parameter specs that apply to this arch × technique."""
    arch = (arch or "").lower()
    technique = (technique or "sft").lower()

    if arch == "encoder":  # classifier fine-tune (SFT only)
        return [
            _p("epochs", "Epochs", "select", 3, options=[1, 2, 3, 4, 5],
               help="2–3 is the sweet spot for a small classifier on a few-thousand rows."),
            _p("lr", "Learning rate", "select", 2e-5, options=[1e-5, 2e-5, 3e-5, 5e-5],
               help="2e-5 is standard for BERT-family fine-tuning."),
            _p("batch_size", "Batch size", "select", 16, options=[8, 16, 32],
               help="16 fits most machines; lower if you hit memory limits."),
            _p("max_length", "Max input tokens", "select", 256, options=[128, 256, 512],
               help="256 covers ~97% of typical guardrail prompts; 512 is DistilBERT's ceiling."),
            _max_steps(),
        ]

    # decoders (Qwen3 / DeepSeek / SmolLM2 / Gemma)
    if technique == "grpo":
        return [
            _p("max_steps", "GRPO steps", "int", 200, min=1, max=100000, step=10,
               help="RL is step-based (not epochs). 150–300 is a good range; 60 is usually too "
                    "few. Slow — each step runs several rollouts."),
            _p("num_generations", "Rollouts per step", "select", 4, options=[2, 4, 6, 8],
               help="More rollouts = better reward signal, proportionally slower."),
            _p("max_completion_len", "Max completion tokens", "select", 96,
               options=[64, 96, 128, 200], help="Length budget for the generated reasoning + verdict."),
            _p("lr", "Learning rate", "select", 1e-6, options=[5e-7, 1e-6, 2e-6],
               help="Low LR keeps RL stable."),
            _p("grad_accum", "Grad accumulation", "select", 1, options=[1, 2, 4, 8],
               help="Effective batch = num_generations × accumulation."),
            *_lora(),
        ]
    if technique == "dpo":
        return [
            _p("epochs", "Epochs", "select", 1, options=[1, 2, 3],
               help="DPO is a light refinement of an SFT checkpoint; 1 epoch is standard."),
            _p("lr", "Learning rate", "select", 5e-6, options=[1e-6, 5e-6, 1e-5],
               help="Low LR for preference tuning."),
            _p("beta", "DPO beta", "select", 0.1, options=[0.05, 0.1, 0.3, 0.5],
               help="KL strength — higher keeps the model closer to the base."),
            _p("batch_size", "Batch size", "select", 2, options=[1, 2, 4],
               help="Small on purpose: DPO doubles the batch (chosen+rejected) and larger "
                    "values hit Apple MPS's tensor limit on big-vocab models."),
            _p("grad_accum", "Grad accumulation", "select", 4, options=[1, 2, 4, 8],
               help="Effective batch = batch × accumulation."),
            _p("max_seq_len", "Max sequence length", "select", 1024, options=[512, 768, 1024],
               help="Capped to stay under Apple MPS's INT_MAX tensor limit."),
            *_lora(),
            _max_steps(),
        ]
    # default: decoder SFT (LoRA)
    return [
        _p("epochs", "Epochs", "select", 1, options=[1, 2, 3],
           help="1 is a fine first pass; go to 2 only if the model underfits."),
        _p("lr", "Learning rate", "select", 2e-4, options=[5e-5, 1e-4, 2e-4, 3e-4],
           help="2e-4 is typical for LoRA SFT."),
        _p("batch_size", "Batch size", "select", 8, options=[1, 2, 4, 8, 16],
           help="Lower if you hit memory limits."),
        _p("grad_accum", "Grad accumulation", "select", 1, options=[1, 2, 4, 8],
           help="Effective batch = batch × accumulation."),
        _p("max_seq_len", "Max sequence length", "select", 512, options=[256, 512, 768, 1024],
           help="512 covers ~99% of typical guardrail prompts."),
        *_lora(),
        _max_steps(),
    ]


def recommended(arch: str, technique: str) -> dict:
    """The recommended (best-known) value for every parameter of this combo."""
    return {p["name"]: p["default"] for p in param_spec(arch, technique) if p["default"] is not None}


def validate_params(arch: str, technique: str, params: dict) -> dict:
    """Return a cleaned param dict, rejecting values outside the accepted set/range.

    Unknown keys are dropped (they don't apply to this combo). Raises ``ValueError``
    with an actionable message on an out-of-range value so a run is never launched
    with an invalid hyperparameter."""
    spec = {p["name"]: p for p in param_spec(arch, technique)}
    clean: dict = {}
    for name, value in (params or {}).items():
        p = spec.get(name)
        if p is None:
            continue  # not applicable to this arch/technique — ignore
        if value is None or value == "":
            if p.get("optional"):
                continue
            raise ValueError(f"{name} is required for {arch}/{technique}")
        if p["kind"] == "select":
            opts = p["options"] or []
            match = next((o for o in opts if float(o) == _num(value)), None)
            if match is None:
                raise ValueError(
                    f"{name}={value!r} is not allowed for {arch}/{technique}; "
                    f"accepted: {', '.join(map(str, opts))}"
                )
            clean[name] = match
        else:  # int / float
            n = _num(value)
            if n is None:
                raise ValueError(f"{name} must be a number, got {value!r}")
            if p.get("min") is not None and n < p["min"]:
                raise ValueError(f"{name}={value} below minimum {p['min']}")
            if p.get("max") is not None and n > p["max"]:
                raise ValueError(f"{name}={value} above maximum {p['max']}")
            clean[name] = int(n) if p["kind"] == "int" else n
    return clean


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
