"""GRPO / RLVR — the reasoning guard (Agent Bouncer's headline experiment).

The model emits a short rationale then a verdict; the reward is *verifiable* from
the gold label via `agent_bouncer.rewards.composite_reward` — no reward model.
This directly optimizes the false-positive-on-benign metric (see rewards.py).

Backends: Unsloth GRPO (free Colab / small VRAM) or MLX-LM locally.
"""

from __future__ import annotations

from pathlib import Path

from ..rewards import RewardWeights, composite_reward

__all__ = ["run_grpo", "RewardWeights", "composite_reward"]


def run_grpo(config_path: str | Path) -> None:
    """RL-tune a reasoning guard from a YAML config (see configs/model/grpo_*.yaml)."""
    raise NotImplementedError(
        "TODO: build a TRL/Unsloth GRPOTrainer whose reward_funcs wrap "
        "rewards.composite_reward (parse completion -> Verdict, score vs gold). "
        "Log reward/KL/completion-length curves to MLflow. See docs/roadmap.md phase 4."
    )
