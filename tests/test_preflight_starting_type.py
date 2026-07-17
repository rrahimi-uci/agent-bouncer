#!/usr/bin/env python
"""CPU-only tests for the Sec 4.3 eligibility preflight (preflight_starting_type_adaptation).

Uses the tiny cached SmolLM2-135M-Instruct under the general `paper_a_safe_unsafe` contract.
Asserts the CPU-checkable subset the task specifies: LoRA targets present, contract decision
tokens distinct + complete, adapter-disable recovers the base (small max logit diff) with an
initial KL ~ 0. Also exercises the full preflight (including the beta0 and 32-row smoke training
checks) with a tiny recipe, and confirms an unimplemented guard contract fails fail-closed.

Run directly: python tests/test_preflight_starting_type.py   (or via pytest)
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import preflight_starting_type_adaptation as P  # noqa: E402

TINY = {
    "starting_key": "tiny_test", "starting_type": "general",
    "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "model_revision": None, "tokenizer_revision": None,
    "trust_remote_code": False, "dtype": "float32", "access": "ungated_apache2",
}
RECIPE = {
    "max_length": 256, "max_steps": 2, "learning_rate": 2e-4, "warmup_ratio": 0.0,
    "per_device_batch": 1, "gradient_accumulation": 2, "scheduler": "constant",
    "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]},
}


def _by_name(report):
    return {r["name"]: r for r in report["checks"]}


def test_structural_subset_passes():
    """The CPU-cheap subset (task-specified self-test assertions)."""
    report = P.run_preflight(
        ckpt=TINY, contract_name="paper_a_safe_unsafe", recipe=RECIPE,
        fixture_rows=P.default_fixture_rows(6), device="cpu", dtype="float32",
        include_training=False)
    by = _by_name(report)

    lt = by["lora_targets_present"]
    assert lt["status"] == "pass", lt
    assert lt["detail"]["all_present"] is True, lt

    ct = by["contract_tokens_distinct"]
    assert ct["status"] == "pass", ct
    assert ct["detail"]["safe_id"] != ct["detail"]["unsafe_id"], ct
    assert ct["detail"]["completion_ids"]["safe"] != ct["detail"]["completion_ids"]["unsafe"], ct

    dr = by["adapter_disable_recovers"]
    assert dr["status"] == "pass", dr
    assert dr["detail"]["recovers"] is True, dr
    assert dr["detail"]["initial_kl_ok"] is True, dr
    print(f"  [ok] structural subset: lora all-present, tokens distinct, "
          f"disable recovers (max|dlogit|={dr['detail']['max_abs_logit_diff']:.2e}, "
          f"init_kl={dr['detail']['initial_kl']:.2e})")


def test_full_preflight_eligible():
    """The full preflight including beta0 and smoke training checks -> eligible on the tiny model."""
    report = P.run_preflight(
        ckpt=TINY, contract_name="paper_a_safe_unsafe", recipe=RECIPE,
        fixture_rows=P.default_fixture_rows(6), device="cpu", dtype="float32",
        smoke_rows=8, include_training=True)
    by = _by_name(report)

    b0 = by["beta0_reproduces_sft"]
    assert b0["status"] == "pass", b0
    assert b0["detail"]["identical"] is True, b0

    sm = by["smoke_adapter_only_finite"]
    assert sm["status"] == "pass", sm
    assert sm["detail"]["base_unchanged"] is True, sm
    assert sm["detail"]["margins_finite"] is True and sm["detail"]["margins_nonconstant"] is True, sm

    assert report["eligible"] is True, report["summary"]
    assert by["license_access_note"]["status"] == "note"
    print(f"  [ok] full preflight eligible=True; beta0 sha match; "
          f"smoke base_unchanged + finite nonconstant margins (std={sm['detail']['margin_std']:.3g})")


def test_unknown_contract_fails_closed():
    """An UNKNOWN contract name fails contract_tokens_distinct (get_contract KeyError -> not
    eligible). The five guard contracts are now implemented + validated on real tokenizers; the
    fail-closed guarantee still holds for a genuinely unregistered contract."""
    report = P.run_preflight(
        ckpt=TINY, contract_name="does_not_exist_contract_v0", recipe=RECIPE,
        fixture_rows=P.default_fixture_rows(4), device="cpu", dtype="float32",
        include_training=False)
    by = _by_name(report)
    assert by["contract_tokens_distinct"]["status"] == "fail", by["contract_tokens_distinct"]
    assert report["eligible"] is False, report["summary"]
    print("  [ok] unknown contract fails closed -> eligible=False")


def test_implemented_guard_contract_tokens_distinct():
    """An implemented guard contract (shieldgemma_yes_no) now PASSES contract_tokens_distinct even
    with a stand-in tokenizer: distinct, EOS-free (appends_eos=False) native verdict sequences."""
    report = P.run_preflight(
        ckpt=TINY, contract_name="shieldgemma_yes_no", recipe=RECIPE,
        fixture_rows=P.default_fixture_rows(4), device="cpu", dtype="float32",
        include_training=False)
    by = _by_name(report)
    ct = by["contract_tokens_distinct"]
    assert ct["status"] == "pass", ct
    assert ct["detail"]["safe_id"] != ct["detail"]["unsafe_id"], ct
    print("  [ok] implemented guard contract passes contract_tokens_distinct (appends_eos=False)")


if __name__ == "__main__":
    print("=== preflight_starting_type_adaptation tests (CPU, tiny model) ===")
    test_structural_subset_passes()
    test_unknown_contract_fails_closed()
    test_implemented_guard_contract_tokens_distinct()
    test_full_preflight_eligible()
    print("ALL PASSED")
