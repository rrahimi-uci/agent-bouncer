"""Supervised fine-tuning (the workhorse).

Two interchangeable backends, selected in the config:

- ``mlx``      : local, Apple-Silicon-native (fast on an M-series Mac).
- ``unsloth``  : free Colab / CUDA, 4-bit QLoRA — the reproducibility path for
                 people without a beefy laptop.

Both produce a decoder guard that emits a schema-valid `Verdict`. An encoder
baseline (ModernBERT + classification head) lives behind ``arch: encoder``.
"""

from __future__ import annotations

from pathlib import Path


def run_sft(config_path: str | Path) -> None:
    """Fine-tune a guard from a YAML config (see configs/model/*.yaml)."""
    raise NotImplementedError(
        "TODO: load config; dispatch on backend {mlx, unsloth} and arch {encoder, decoder}. "
        "Log params/metrics/artifacts to MLflow. See docs/roadmap.md phase 3."
    )
