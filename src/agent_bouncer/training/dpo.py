"""DPO — preference-tune to crush false positives (roadmap phase 5).

We build preference pairs where the *correct* verdict is chosen over the flipped
one. Feeding benign-but-scary prompts (over-refusal data) teaches the model to
prefer "safe" on them, directly targeting `fpr_on_benign`.

`build_preference_pairs` is pure and unit-tested; `run_dpo` wires TRL's DPOTrainer.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_bouncer.core.schema import Decision, Verdict
from agent_bouncer.core.taxonomy import Hazard
from agent_bouncer.models.decoder import build_prompt, format_target
from agent_bouncer.training.runtime import load_decoder_training_assets


def _flip(gold: Verdict) -> Verdict:
    if gold.decision == Decision.UNSAFE:
        return Verdict(decision=Decision.SAFE, hazard=Hazard.NONE)
    return Verdict(decision=Decision.UNSAFE, hazard=Hazard.NON_VIOLENT_CRIMES)


def build_preference_pairs(records: list[dict], *, reasoning: bool = False) -> list[dict]:
    """Each record -> {prompt, chosen (correct verdict), rejected (flipped verdict)}."""
    pairs = []
    for r in records:
        gold = Verdict(decision=Decision(r["label"]), hazard=Hazard(r.get("hazard", "none")))
        # Leading space on the completion matches the SFT/inference boundary
        # ("Verdict: {json}"), so DPO optimizes the tokens the model actually emits.
        pairs.append(
            {
                "prompt": build_prompt(r["text"], reasoning=reasoning),
                "chosen": " " + format_target(gold, reasoning=reasoning),
                "rejected": " " + format_target(_flip(gold), reasoning=reasoning),
            }
        )
    return pairs


def run_dpo(config_path: str | Path) -> str:  # pragma: no cover - runs a real TRL DPOTrainer
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    from agent_bouncer.data import read_jsonl

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("data", {"train": "data/train.jsonl"})

    pairs = build_preference_pairs(read_jsonl(cfg["data"]["train"]))
    d = cfg.get("dpo", {})
    out_dir = cfg.get("output_dir", "outputs/dpo")
    # Keep the per-token logits tensor small enough for Apple MPS: DPO concatenates
    # chosen+rejected (2x batch), so on a ~150k-vocab model batch*seq*vocab overflows
    # MPSGraph's INT_MAX element limit at TRL's default batch=8 / max_length=1024. A small
    # batch (with grad-accum for effective batch) keeps 2*batch*max_length*vocab < 2^31.
    dpo_config = DPOConfig(
        output_dir=out_dir,
        beta=float(d.get("beta", 0.1)),
        learning_rate=float(d.get("lr", 5e-6)),
        num_train_epochs=float(d.get("epochs", 1)),
        max_steps=int(d.get("max_steps", -1)),  # -1 = no cap (train by epochs)
        per_device_train_batch_size=int(d.get("batch_size", 2)),
        gradient_accumulation_steps=int(d.get("grad_accum", 4)),
        max_length=int(d.get("max_length", 1024)),
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    lora = cfg.get("lora", {})
    model, tokenizer, device, bf16 = load_decoder_training_assets(
        cfg["base_model"],
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )
    from agent_bouncer.training.progress import progress_callback
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)), task_type="CAUSAL_LM"
        ),
        callbacks=[progress_callback()],
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[dpo] saved to {out_dir} ({device}, {'bf16' if bf16 else 'fp32'})")
    return out_dir
