"""Supervised fine-tuning (the workhorse).

- ``arch: encoder`` : ModernBERT (or any BERT-family) sequence classifier via the
  Transformers ``Trainer`` — fast, runs on CPU/MPS, no bitsandbytes needed.
- ``arch: decoder`` : small instruct model + LoRA (peft) via TRL's ``SFTTrainer``,
  trained to emit the canonical JSON verdict from `models.decoder`.

`run_sft(config_path)` dispatches on ``arch``. The heavy imports live inside the
functions so importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..data import read_jsonl

BINARY_LABEL2ID = {"safe": 0, "unsafe": 1}
BINARY_ID2LABEL = {0: "safe", 1: "unsafe"}


def _load_cfg(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("data", {"train": "data/train.jsonl", "validation": "data/validation.jsonl"})
    return cfg


def _binary_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    benign = int((labels == 0).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": float((preds == labels).mean()),
        "fpr_on_benign": fp / benign if benign else 0.0,
    }


def train_encoder(cfg: dict[str, Any]) -> str:
    """Fine-tune a binary (safe/unsafe) sequence classifier. Returns output dir."""
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    t = cfg.get("train", {})
    set_seed(int(cfg.get("seed", 42)))
    base = cfg["base_model"]
    out_dir = cfg.get("output_dir", "outputs/encoder")
    max_len = int(t.get("max_length", 256))

    tokenizer = AutoTokenizer.from_pretrained(base)

    def to_dataset(records: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {
                "text": [r["text"] for r in records],
                "labels": [BINARY_LABEL2ID[r["label"]] for r in records],
            }
        )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_ds = to_dataset(read_jsonl(cfg["data"]["train"])).map(tokenize, batched=True)
    val_ds = to_dataset(read_jsonl(cfg["data"]["validation"])).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label=BINARY_ID2LABEL, label2id=BINARY_LABEL2ID
    )

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=float(t.get("epochs", 3)),
        per_device_train_batch_size=int(t.get("batch_size", 16)),
        per_device_eval_batch_size=64,
        learning_rate=float(t.get("lr", 2e-5)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_binary_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[encoder] saved to {out_dir} | eval: {metrics}")
    return out_dir


def train_decoder(cfg: dict[str, Any]) -> str:
    """LoRA fine-tune a small decoder to emit the canonical JSON verdict."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from ..models.decoder import build_prompt, format_target
    from ..schema import Decision, Verdict
    from ..taxonomy import Hazard

    reasoning = cfg.get("mode") == "reasoning"

    def to_text(records: list[dict]) -> Dataset:
        rows = []
        for r in records:
            gold = Verdict(decision=Decision(r["label"]), hazard=Hazard(r.get("hazard", "none")))
            prompt = build_prompt(r["text"], reasoning=reasoning)
            rows.append({"text": f"{prompt} {format_target(gold, reasoning=reasoning)}"})
        return Dataset.from_list(rows)

    lora = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        task_type="CAUSAL_LM",
    )
    t = cfg.get("train", {})
    out_dir = cfg.get("output_dir", "outputs/decoder")
    bf16 = bool(t.get("bf16", False))  # bf16 LoRA is the Mac path (no bitsandbytes/QLoRA)
    sft_config = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=float(t.get("epochs", 2)),
        per_device_train_batch_size=int(t.get("batch_size", 8)),
        gradient_accumulation_steps=int(t.get("grad_accum", 1)),
        learning_rate=float(t.get("lr", 2e-4)),
        max_length=int(t.get("max_seq_len", 1024)),
        bf16=bf16,
        model_init_kwargs={"dtype": "bfloat16"} if bf16 else None,
        report_to="none",
        seed=int(cfg.get("seed", 42)),
    )
    trainer = SFTTrainer(
        model=cfg["base_model"],
        train_dataset=to_text(read_jsonl(cfg["data"]["train"])),
        args=sft_config,
        peft_config=peft_config,
    )
    trainer.train()
    # Merge the LoRA adapter into the base weights so the guard loads as a
    # standalone model (otherwise the output dir holds only the adapter).
    from transformers import AutoTokenizer

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cfg["base_model"]).save_pretrained(out_dir)
    print(f"[decoder] merged + saved to {out_dir}")
    return out_dir


def run_sft(config_path: str | Path) -> str:
    cfg = _load_cfg(config_path)
    if cfg.get("arch") == "encoder":
        return train_encoder(cfg)
    return train_decoder(cfg)
