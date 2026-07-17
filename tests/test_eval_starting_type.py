#!/usr/bin/env python
"""CPU-only smoke tests for the starting-type adaptation scorer (eval_starting_type_adaptation).

Uses the tiny cached SmolLM2-135M-Instruct under the paper_a_safe_unsafe contract on ~12 synthetic
rows (both gold classes) to validate the SCORING plumbing (not the science):

  * unmodified scoring emits complete Sec-12 records with explicit adaptation/condition_id/kl_beta;
  * RAW margins (score_raw == unsafe_logit - safe_logit) are finite; probability_raw == sigmoid;
  * a tiny SFT adapter can be scored via PeftModel and carries its adapter_sha256;
  * the unmodified sentinel is seed=-1 / kl_beta=None, sft is seed>=0 / kl_beta=0.0.

Run directly: python tests/test_eval_starting_type.py   (or via pytest)
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import starting_type_common as S  # noqa: E402
import eval_starting_type_adaptation as E  # noqa: E402

TINY = {
    "starting_key": "tiny_test", "starting_type": "general",
    "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "model_revision": None, "tokenizer_revision": None,
    "trust_remote_code": False, "dtype": "float32", "max_length": 512,
}
RECIPE = {
    "max_length": 512, "max_steps": 3, "learning_rate": 2e-4, "warmup_ratio": 0.0,
    "per_device_batch": 1, "gradient_accumulation": 2, "scheduler": "constant",
    "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}


def _assert_records(recs, condition):
    assert recs, "no records emitted"
    assert set(E.SCORE_COLUMNS).issubset(set(recs[0])), "record missing schema columns"
    golds = {r["gold"] for r in recs}
    assert golds == {0, 1}, f"expected both gold classes, got {golds}"
    for r in recs:
        assert r["adaptation"] == condition
        assert r["contract"] == "paper_a_safe_unsafe"
        assert isinstance(r["condition_id"], str) and condition in r["condition_id"]
        assert np.isfinite(r["score_raw"]), "score_raw not finite"
        assert abs(r["score_raw"] - (r["unsafe_logit"] - r["safe_logit"])) < 1e-9
        assert abs(r["probability_raw"] - 1.0 / (1.0 + np.exp(-r["score_raw"]))) < 1e-9
        assert 0.0 <= r["probability_calibrated"] <= 1.0
        assert r["parse_status"] in ("safe", "unsafe", "off_contract")
        assert r["contract_hash"]


def test_score_unmodified():
    rows = E.synthetic_rows(12)
    recs = E.score_condition(
        ckpt=TINY, contract="paper_a_safe_unsafe", adapter_dir=None,
        condition="unmodified", seed=None, beta=None, rows=rows,
        device="cpu", batch_size=4, dtype="float32")
    _assert_records(recs, "unmodified")
    assert all(r["seed"] == -1 for r in recs), "unmodified must use seed sentinel -1"
    assert all(r["kl_beta"] is None for r in recs), "unmodified kl_beta must be null"
    assert all(r["adapter_sha256"] is None for r in recs)
    assert len(recs) == len(rows)
    print(f"  [ok] unmodified: {len(recs)} records, seed=-1/kl_beta=None, RAW margins finite")


def test_score_sft_adapter(tmp="/tmp/sta_eval_test"):
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    train_rows = [{"text": f"Please help me with task number {i} about safety.", "gold": i % 2}
                  for i in range(8)]
    meta = S.adapt(ckpt=TINY, contract_name="paper_a_safe_unsafe", train_rows=train_rows,
                   method="sft", beta=0.0, seed=42, data_order_seed=42, recipe=RECIPE,
                   out_dir=tmp, device="cpu")
    assert meta["status"] == "completed", meta.get("failure_reason")
    adapter_dir = os.path.join(tmp, "adapter")
    rows = E.synthetic_rows(12)
    recs = E.score_condition(
        ckpt=TINY, contract="paper_a_safe_unsafe", adapter_dir=adapter_dir,
        condition="sft", seed=42, beta=0.0, rows=rows,
        device="cpu", batch_size=4, dtype="float32")
    _assert_records(recs, "sft")
    assert all(r["seed"] == 42 for r in recs)
    assert all(r["kl_beta"] == 0.0 for r in recs)
    assert all(r["adapter_sha256"] == meta["adapter_sha256"] for r in recs)
    assert all(r["condition_id"] == "tiny_test:sft:seed42" for r in recs)
    print(f"  [ok] sft: {len(recs)} records, adapter_sha256 attached, condition_id={recs[0]['condition_id']}")


def test_guard_rejects_and_unmodified_adapter_guard():
    # unmodified must refuse an adapter; sft must require one
    try:
        E.score_condition(ckpt=TINY, contract="paper_a_safe_unsafe", adapter_dir="/x",
                          condition="unmodified", seed=None, beta=None,
                          rows=E.synthetic_rows(4), device="cpu")
        assert False, "unmodified with adapter should raise"
    except ValueError:
        pass
    try:
        E.score_condition(ckpt=TINY, contract="paper_a_safe_unsafe", adapter_dir=None,
                          condition="sft", seed=42, beta=0.0,
                          rows=E.synthetic_rows(4), device="cpu")
        assert False, "sft without adapter should raise"
    except ValueError:
        pass
    print("  [ok] adapter/condition invariants enforced")


if __name__ == "__main__":
    print("=== eval_starting_type_adaptation scorer tests (CPU, tiny model) ===")
    test_guard_rejects_and_unmodified_adapter_guard()
    test_score_unmodified()
    test_score_sft_adapter()
    print("ALL PASSED")
