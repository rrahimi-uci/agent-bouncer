"""GRPO / RLVR — the reasoning guard (Agent Bouncer's headline experiment).

The model emits a rationale then a verdict; the reward is *verifiable* from the
gold label via `agent_bouncer.rewards.composite_reward` — no reward model.

`make_reward_fn` returns a TRL-compatible reward function (pure, unit-tested):
it receives the batch of `completions` plus the dataset columns `gold_decision`
and `gold_hazard`, parses each completion into a `Verdict`, and scores it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..models.decoder import parse_verdict
from ..rewards import RewardWeights, composite_reward
from ..schema import Decision, Verdict
from ..taxonomy import Hazard

__all__ = ["make_reward_fn", "run_grpo", "RewardWeights", "composite_reward"]


def make_reward_fn(weights: RewardWeights | None = None) -> Callable:
    """Build a TRL reward function that scores completions against gold labels."""
    weights = weights or RewardWeights()

    def reward_fn(
        completions: Sequence[str],
        gold_decision: Sequence[str],
        gold_hazard: Sequence[str] | None = None,
        **_: Any,
    ) -> list[float]:
        hazards = list(gold_hazard) if gold_hazard is not None else [None] * len(completions)
        rewards: list[float] = []
        for completion, decision, hazard in zip(completions, gold_decision, hazards, strict=True):
            pred = parse_verdict(completion)
            gold = Verdict(
                decision=Decision(decision),
                hazard=Hazard(hazard) if hazard else Hazard.NONE,
            )
            rewards.append(composite_reward(pred, gold, weights))
        return rewards

    return reward_fn


def _weights_from_cfg(cfg: dict[str, Any]) -> RewardWeights:
    r = cfg.get("reward", {})
    return RewardWeights(
        correctness=float(r.get("correctness", 1.0)),
        category=float(r.get("category", 0.5)),
        format=float(r.get("format", 0.2)),
        false_positive_penalty=float(r.get("false_positive_penalty", 1.0)),
        brevity=float(r.get("brevity", 0.1)),
        rationale_budget=int(r.get("rationale_budget", 256)),
    )


def run_grpo(config_path: str | Path) -> str:
    """RL-tune a reasoning guard with verifiable rewards."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from ..data import read_jsonl
    from ..models.decoder import build_prompt

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("data", {"train": "data/train.jsonl"})

    records = read_jsonl(cfg["data"]["train"])
    dataset = Dataset.from_list(
        [
            {
                "prompt": build_prompt(r["text"], reasoning=True),
                "gold_decision": r["label"],
                "gold_hazard": r.get("hazard", "none"),
            }
            for r in records
        ]
    )

    from peft import LoraConfig

    g = cfg.get("grpo", {})
    out_dir = cfg.get("output_dir", "outputs/grpo")
    num_gen = int(g.get("num_generations", 8))
    grpo_config = GRPOConfig(
        output_dir=out_dir,
        num_generations=num_gen,
        # generation_batch_size (= per_device batch x grad_accum) must be a
        # multiple of num_generations, so default the batch to num_generations.
        per_device_train_batch_size=int(g.get("batch_size", num_gen)),
        gradient_accumulation_steps=int(g.get("grad_accum", 1)),
        max_completion_length=int(g.get("max_completion_len", 256)),
        learning_rate=float(g.get("lr", 1e-6)),
        beta=float(g.get("kl_coef", 0.04)),
        max_steps=int(g.get("steps", 1000)),
        logging_steps=int(g.get("log_steps", 10)),
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    lora = cfg.get("lora", {})
    peft_config = (
        LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            task_type="CAUSAL_LM",
        )
        if lora
        else None
    )
    trainer = GRPOTrainer(
        model=cfg["base_model"],
        reward_funcs=[make_reward_fn(_weights_from_cfg(cfg))],
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[grpo] saved to {out_dir}")
    return out_dir
