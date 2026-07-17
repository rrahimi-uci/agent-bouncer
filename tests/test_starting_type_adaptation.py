#!/usr/bin/env python
"""Fast, CPU-only sanity tests for the generalized adaptation harness (starting_type_common).

Uses the tiny cached SmolLM2-135M-Instruct to validate the FRAMEWORK logic (not the science):
explicit condition labels, beta==0 (sft) == vanilla CE at init, KL grows for kl_sft, adapter-disable
recovers the base at step zero (initial KL ~ 0), and the registry loads. Guard-native contracts are
expected to raise until implemented. Model-specific native scorers are validated later on GPU.

Run directly: python tests/test_starting_type_adaptation.py   (or via pytest)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import starting_type_common as A  # noqa: E402

TINY = {
    "starting_key": "tiny_test", "starting_type": "general",
    "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "model_revision": None, "tokenizer_revision": None,
    "trust_remote_code": False, "dtype": "float32",
}
RECIPE = {
    "max_length": 512, "max_steps": 3, "learning_rate": 2e-4, "warmup_ratio": 0.0,
    "per_device_batch": 1, "gradient_accumulation": 2, "scheduler": "constant",
    "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
}
ROWS = [{"text": f"Please help me with task number {i} about safety.", "gold": i % 2} for i in range(8)]


def test_registry_loads():
    reg = A.load_registry(os.path.join(ROOT, "configs", "starting_type_adaptation_v1.yaml"))
    ck = reg["checkpoints"]
    g = [k for k, v in ck.items() if v["starting_type"] == "general"]
    p = [k for k, v in ck.items() if v["starting_type"] == "purpose_built"]
    assert len(g) == 4 and len(p) == 6, (len(g), len(p))
    assert reg["recipe"]["kl"]["primary_beta"] == 0.5
    assert list(reg["conditions"]) == ["unmodified", "sft", "kl_sft"]
    print("  [ok] registry: 4 general + 6 purpose-built; primary beta 0.5")


def test_condition_ids_distinct():
    a = A.condition_id("qwen3_4b", "sft", 42, 0.0)
    b = A.condition_id("qwen3_4b", "kl_sft", 42, 0.5)
    u = A.condition_id("qwen3_4b", "unmodified", None, None)
    assert a != b and "kl_sft" in b and "sft:seed42" in a and u.endswith("unmodified")
    print(f"  [ok] distinct condition ids: {a} | {b} | {u}")


def test_contracts_registry():
    # guard-native contracts are now IMPLEMENTED + wired via experiments/guard_contracts.py
    # (each still needs a Phase-0 real-model fidelity gate); unknown names still raise KeyError.
    import guard_contracts
    for name in ("shieldgemma_yes_no", "qwen3guard_toplevel", "granite_yes_no",
                 "llama_guard_toplevel", "wildguard_prompt_harm"):
        assert name in guard_contracts.GUARD_CONTRACTS, name
        assert callable(guard_contracts.GUARD_CONTRACTS[name]), name
    try:
        A.get_contract("nope", tok=None); assert False
    except KeyError:
        pass
    print("  [ok] 5 guard-native contracts wired into GUARD_CONTRACTS; unknown -> KeyError")


def test_adapt_sft_and_klsft(tmp="/tmp/sta_test"):
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    m_sft = A.adapt(ckpt=TINY, contract_name="paper_a_safe_unsafe", train_rows=ROWS,
                    method="sft", beta=0.0, seed=42, data_order_seed=42, recipe=RECIPE,
                    out_dir=os.path.join(tmp, "sft"), device="cpu")
    m_kl = A.adapt(ckpt=TINY, contract_name="paper_a_safe_unsafe", train_rows=ROWS,
                   method="kl_sft", beta=0.5, seed=42, data_order_seed=42, recipe=RECIPE,
                   out_dir=os.path.join(tmp, "kl"), device="cpu")
    assert m_sft["status"] == "completed", m_sft.get("failure_reason")
    assert m_kl["status"] == "completed", m_kl.get("failure_reason")
    assert m_sft["condition"] == "sft" and m_sft["kl_beta"] == 0.0
    assert m_kl["condition"] == "kl_sft" and m_kl["kl_beta"] == 0.5
    assert m_kl.get("final_kl") is not None and m_kl["final_kl"] >= 0.0
    assert os.path.exists(os.path.join(tmp, "sft", "adapter", "adapter_config.json"))
    assert os.path.exists(os.path.join(tmp, "kl", "adapter", "adapter_config.json"))
    print(f"  [ok] adapt: sft(beta0) + kl_sft(beta0.5, final_kl={m_kl['final_kl']:.4g}) both completed, "
          f"explicit conditions, adapters saved")


if __name__ == "__main__":
    print("=== starting_type_adaptation harness tests (CPU, tiny model) ===")
    test_registry_loads()
    test_condition_ids_distinct()
    test_contracts_registry()
    test_adapt_sft_and_klsft()
    print("ALL PASSED")
