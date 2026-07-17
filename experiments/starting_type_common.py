#!/usr/bin/env python
"""Generalized adaptation harness for the starting-type study (papers/unified-report/proposal.md).

The 2x3 design applies {unmodified, +SFT, +KL-SFT} to both general instruction checkpoints AND
released purpose-built guards, each treated as a controlled STARTING CHECKPOINT. This module is the
shared, model-agnostic core:

  * a model registry loaded from configs/starting_type_adaptation_v1.yaml (general + purpose-built);
  * EXPLICIT adaptation conditions {unmodified, sft, kl_sft} (never `condition=sft` + a nullable beta);
  * a pluggable VerdictContract so each checkpoint keeps its NATIVE top-level verdict interface while
    sharing the frozen binary manifest, seeds, optimizer, and LoRA policy; and
  * a generalized adapt() trainer whose KL reference is the SAME unmodified starting checkpoint,
    recovered via PEFT disable_adapter() (no second model in memory). beta==0 reproduces ordinary SFT
    exactly (CE delegated to the base Trainer so grad-accum normalization matches).

Guard-native contracts (ShieldGemma/Qwen3Guard/Granite/Llama-Guard/WildGuard) are registered but raise
until their locked native renderer/verdict serialization is added after the Phase-0 preflight; the
general `paper_a_safe_unsafe` contract is fully implemented and reuses guard_research.prompts.

This module deliberately does NOT mutate the Paper A files or the running KL-SFT sweep. It is the new
namespace the proposal specifies (Section 12/13).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402  (dtype/budgeted_prompt/utcnow/sha helpers)

CONDITIONS = ("unmodified", "sft", "kl_sft")


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def condition_id(starting_key: str, condition: str, seed, beta) -> str:
    """Unambiguous per-cell id. Unmodified has no seed/beta; kl_sft records beta."""
    assert condition in CONDITIONS, condition
    if condition == "unmodified":
        return f"{starting_key}:unmodified"
    if condition == "sft":
        return f"{starting_key}:sft:seed{seed}"
    return f"{starting_key}:kl_sft:seed{seed}:beta{('%g' % float(beta)).replace('.', 'p')}"


def load_registry(path: str) -> dict:
    """Load the study registry YAML. Returns {recipe, conditions, general_checkpoints,
    purpose_built_checkpoints, ...}. Flattens a `checkpoints` view keyed by starting_key with
    a `starting_type` tag on each entry."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    ckpts = {}
    for k, v in (reg.get("general_checkpoints") or {}).items():
        ckpts[k] = {**v, "starting_type": "general", "starting_key": k}
    for k, v in (reg.get("purpose_built_checkpoints") or {}).items():
        ckpts[k] = {**v, "starting_type": "purpose_built", "starting_key": k}
    reg["checkpoints"] = ckpts
    return reg


# --------------------------------------------------------------------------------------
# Verdict contracts: each checkpoint keeps its native top-level verdict interface.
# A contract turns (user_text, gold) into (rendered_prompt, completion_token_ids) and exposes
# the two decision-token ids used for the unsafe margin at scoring time.
# --------------------------------------------------------------------------------------
class VerdictContract:
    """Base contract. Subclasses/registered builders implement render() + completion_ids()."""
    name = "abstract"

    def render(self, tok, user_text: str, max_len: int, reserved: int):
        raise NotImplementedError

    def completion_ids(self, tok, gold: int) -> list[int]:
        """Full supervised verdict token sequence for this gold label (contract owns EOS policy).
        Must be the COMPLETE verdict sequence (never just the first sub-token)."""
        raise NotImplementedError

    def decision_ids(self, tok) -> dict:
        """{'safe_id': int, 'unsafe_id': int} at the decision position for the scoring margin."""
        raise NotImplementedError


class PaperASafeUnsafe(VerdictContract):
    """The report's general safe/unsafe head (guard_research.prompts). Verdict is a single token
    (guaranteed by select_decision_tokens) followed by EOS."""
    name = "paper_a_safe_unsafe"

    def __init__(self, tok):
        self._build_prompt, _ = C.require_prompts()
        self._dt = C.resolve_decision_tokens(tok)
        self._verdict = {0: tok.encode(self._dt["safe_str"], add_special_tokens=False),
                         1: tok.encode(self._dt["unsafe_str"], add_special_tokens=False)}
        self._eos = tok.eos_token_id

    def render(self, tok, user_text, max_len, reserved):
        return C.budgeted_prompt(tok, self._build_prompt, user_text, max_len, reserved_tokens=reserved)

    def completion_ids(self, tok, gold):
        return list(self._verdict[gold]) + [self._eos]

    def decision_ids(self, tok):
        return {"safe_id": self._dt["safe_id"], "unsafe_id": self._dt["unsafe_id"]}


# Guard-native contracts are added after the Phase-0 preflight pins revisions + locks each native
# renderer/verdict serialization (proposal Sec 5.2, 8.2). Registered here so the panel is explicit.
_GUARD_CONTRACTS = {
    "shieldgemma_yes_no", "qwen3guard_toplevel", "granite_yes_no",
    "llama_guard_toplevel", "wildguard_prompt_harm",
}


def get_contract(name: str, tok) -> VerdictContract:
    if name == "paper_a_safe_unsafe":
        return PaperASafeUnsafe(tok)
    # Guard-native contracts (experiments/guard_contracts.py) are implemented per each model's
    # documented schema; they still require a Phase-0 generation-vs-likelihood fidelity gate against
    # the real (revision-pinned) model before their AP is claim-bearing.
    try:
        from guard_contracts import GUARD_CONTRACTS
    except Exception:
        GUARD_CONTRACTS = {}
    if name in GUARD_CONTRACTS:
        return GUARD_CONTRACTS[name](tok)
    if name in _GUARD_CONTRACTS:
        raise NotImplementedError(
            f"native verdict contract '{name}' is registered but its guard_contracts builder is "
            f"unavailable; it also needs the Phase-0 real-model fidelity gate before claim-bearing use.")
    raise KeyError(f"unknown verdict contract: {name}")


def load_study_tokenizer(ckpt: dict):
    """Load a checkpoint's tokenizer at its pinned revision with the study's pad/side setup.

    Robust fast->slow fallback: some released guards ship a SentencePiece `tokenizer.model` that
    transformers' fast path mis-routes to the tiktoken converter (e.g. allenai/wildguard, a Mistral
    tokenizer) -> retry with use_fast=False. Both paths require `protobuf` + `sentencepiece` at
    runtime. Centralizing here keeps train / eval / preflight byte-identical for a given checkpoint.
    """
    from transformers import AutoTokenizer
    kw = dict(revision=ckpt.get("tokenizer_revision"),
              trust_remote_code=bool(ckpt.get("trust_remote_code", False)))
    try:
        tok = AutoTokenizer.from_pretrained(ckpt["model_id"], use_fast=True, **kw)
    except Exception:
        tok = AutoTokenizer.from_pretrained(ckpt["model_id"], use_fast=False, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"
    return tok


def build_dataset(rows, tok, contract: VerdictContract, max_len: int):
    """Completion-only supervised dataset under `contract`. Supervises the COMPLETE verdict sequence
    (not the first sub-token); prompt tokens masked to -100."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self, rws):
            self.ex, self.total_tokens, self.truncated = [], 0, 0
            self.wrapper_ok = True
            for r in rws:
                c = contract.completion_ids(tok, int(r["gold"]))
                rendered, trunc = contract.render(tok, r["text"], max_len, reserved=len(c))
                p = tok(rendered, add_special_tokens=False, truncation=False)["input_ids"]
                if len(p) + len(c) > max_len or not trunc["wrapper_preserved"]:
                    raise C.ArtifactContractError("training prompt budget violated / wrapper lost")
                self.ex.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
                self.total_tokens += len(p) + len(c)
                self.truncated += int(trunc["truncated"])
                self.wrapper_ok = self.wrapper_ok and bool(trunc["wrapper_preserved"])

        def __len__(self):
            return len(self.ex)

        def __getitem__(self, i):
            return self.ex[i]

    return _DS(rows)


def _collate(tok):
    import torch

    def collate(b):
        m = max(len(x["input_ids"]) for x in b)
        pad = tok.pad_token_id
        ids, lab, att = [], [], []
        for x in b:
            g = m - len(x["input_ids"])
            ids.append(x["input_ids"] + [pad] * g)
            lab.append(x["labels"] + [-100] * g)
            att.append([1] * len(x["input_ids"]) + [0] * g)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
                "labels": torch.tensor(lab)}
    return collate


def adapt(*, ckpt: dict, contract_name: str, train_rows, method: str, beta: float,
          seed: int, data_order_seed: int, recipe: dict, out_dir: str, device: str | None = None) -> dict:
    """Adapt one starting checkpoint under one condition. method in {'sft','kl_sft'} (use 'unmodified'
    only for scoring, no training). KL reference is the SAME starting checkpoint via disable_adapter().
    Returns a run-meta dict with the EXPLICIT condition. Never mutates Paper A artifacts."""
    assert method in ("sft", "kl_sft"), method
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "starting_key": ckpt["starting_key"], "starting_type": ckpt["starting_type"],
        "model_id": ckpt["model_id"], "model_revision": ckpt.get("model_revision"),
        "tokenizer_revision": ckpt.get("tokenizer_revision"),
        "contract": contract_name, "condition": method,
        "kl_beta": float(beta) if method == "kl_sft" else 0.0,
        "seed": int(seed), "data_order_seed": int(data_order_seed),
        "condition_id": condition_id(ckpt["starting_key"], method, seed, beta),
        "out_dir": out_dir, "device": None, "start_utc": C.utcnow(),
    }
    t0 = time.time()
    try:
        import numpy as np
        import random
        import torch
        import torch.nn.functional as F
        from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                                   TrainingArguments)
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import RandomSampler

        dev = device or _default_device()
        meta["device"] = dev
        lora = recipe["lora"]
        max_len = int(recipe.get("max_length", 1024))
        max_steps = int(recipe.get("max_steps", 300))
        lr = float(recipe.get("learning_rate", 2e-4))
        warmup = float(recipe.get("warmup_ratio", 0.03))
        per_dev = int(recipe.get("per_device_batch", 1))
        accum = int(recipe.get("gradient_accumulation", 4))
        dtype_name = str(ckpt.get("dtype", "bfloat16"))

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        tok = load_study_tokenizer(ckpt)

        contract = get_contract(contract_name, tok)
        ds = build_dataset(train_rows, tok, contract, max_len)
        meta["dataset_rows"] = len(ds)
        meta["truncation"] = {"n_truncated": ds.truncated, "wrapper_preserved": bool(ds.wrapper_ok)}

        model = AutoModelForCausalLM.from_pretrained(
            ckpt["model_id"], revision=ckpt.get("model_revision"),
            dtype=C.torch_dtype_from_name(torch, dtype_name),
            trust_remote_code=bool(ckpt.get("trust_remote_code", False)))
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
            task_type="CAUSAL_LM", target_modules=list(lora["target_modules"])))
        model.enable_input_require_grads(); model.to(dev)

        class FixedOrderTrainer(Trainer):
            def _get_train_sampler(self, *a, **k):
                gen = torch.Generator(); gen.manual_seed(data_order_seed)
                return RandomSampler(self.train_dataset, generator=gen)

        class KLTrainer(FixedOrderTrainer):
            # CE delegated to parent (num_items_in_batch grad-accum normalization) so beta==0 == SFT
            # exactly; add beta * KL(pi_theta || pi_reference) on supervised verdict positions, with
            # the reference = the same starting checkpoint via disable_adapter().
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                ce_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kw)
                shift_logits = outputs.logits[:, :-1, :]
                mask = inputs["labels"][:, 1:] != -100
                with torch.no_grad():
                    with model.disable_adapter():
                        ref = model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"]).logits
                logp = F.log_softmax(shift_logits[mask].float(), dim=-1)
                logp_ref = F.log_softmax(ref[:, :-1, :][mask].float(), dim=-1)
                kl = (logp.exp() * (logp - logp_ref)).sum(-1).mean()
                self._kl_running = float(kl.detach())
                loss = ce_loss + float(beta) * kl
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=per_dev,
            gradient_accumulation_steps=accum, max_steps=max_steps, num_train_epochs=1,
            learning_rate=lr, lr_scheduler_type=recipe.get("scheduler", "cosine"),
            warmup_ratio=warmup,
            bf16=(dev == "cuda" and dtype_name in ("bfloat16", "bf16")),
            fp16=(dev == "cuda" and dtype_name in ("float16", "fp16", "half")),
            gradient_checkpointing=(dev == "cuda"), logging_steps=25,
            save_strategy="no", remove_unused_columns=False, report_to=[], seed=seed)
        cls = KLTrainer if method == "kl_sft" else FixedOrderTrainer
        trainer = cls(model=model, args=args, train_dataset=ds, data_collator=_collate(tok))
        trainer.train()
        if method == "kl_sft":
            meta["final_kl"] = getattr(trainer, "_kl_running", None)

        adir = os.path.join(out_dir, "adapter")
        model.save_pretrained(adir)
        meta["adapter_sha256"] = C.sha256_dir(adir)
        meta["status"] = "completed"
    except Exception as e:  # keep failed runs, like the Paper A trainer
        meta["status"] = "failed"
        meta["failure_reason"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc()
    finally:
        meta["wall_time_s"] = round(time.time() - t0, 3)
        meta["completion_utc"] = C.utcnow()
        C.write_json(os.path.join(out_dir, "run_meta.json"), meta)
    return meta


# --------------------------------------------------------------------------------------
# Preflight helpers (proposal Sec 4.3). CPU-runnable structural checks used by the preflight CLI
# and the tests; guards' full native-contract preflight is added with their contracts.
# --------------------------------------------------------------------------------------
def lora_targets_present(model, targets) -> dict:
    """Which of the requested LoRA target module suffixes actually exist in the model."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    present = {t: (t in names) for t in targets}
    return {"present": present, "all_present": all(present.values())}


def adapter_disable_recovers(peft_model, ref_model, enc, tol: float = 1e-4) -> dict:
    """At step zero a freshly-attached LoRA adapter is ~identity: disable_adapter() logits must match
    the separately-loaded reference within tol, and the KL(theta||ref) at init must be ~0."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        with peft_model.disable_adapter():
            dis = peft_model(**enc).logits
        base = ref_model(**enc).logits
        max_abs = float((dis - base).abs().max())
        # initial KL of the (identity) adapted model vs disabled path
        adapted = peft_model(**enc).logits
        lp = F.log_softmax(adapted.float(), -1)
        lpr = F.log_softmax(dis.float(), -1)
        init_kl = float((lp.exp() * (lp - lpr)).sum(-1).mean())
    return {"max_abs_logit_diff": max_abs, "recovers": max_abs < tol, "initial_kl": init_kl}


if __name__ == "__main__":
    reg = load_registry(os.path.join(ROOT, "configs", "starting_type_adaptation_v1.yaml"))
    print(f"[registry] {len(reg['checkpoints'])} checkpoints "
          f"({sum(1 for c in reg['checkpoints'].values() if c['starting_type']=='general')} general, "
          f"{sum(1 for c in reg['checkpoints'].values() if c['starting_type']=='purpose_built')} purpose-built)")
    print(f"[conditions] {reg['conditions']} | primary beta={reg['recipe']['kl']['primary_beta']}")
