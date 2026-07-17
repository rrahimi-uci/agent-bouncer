#!/usr/bin/env python
"""CPU-only sanity tests for the starting-type grid orchestrator (run_starting_type_adaptation).

Validates the ORCHESTRATION logic (not the science): the full-registry 2x3 grid enumerates to the
expected cardinality (10 checkpoints -> 10 unmodified + 50 sft + 50 kl_sft primary), every
condition_id is unique, --dry-run loads no models, the unmodified cell is a single seed=-1 no-train
reference (resumable), and one real 2-step SFT cell on the tiny cached SmolLM2-135M confirms adapt()
is wired end-to-end. Guard checkpoints are enumerated but never trained here.

Run directly: python tests/test_run_starting_type.py   (or via pytest)
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import paper_a_common as C  # noqa: E402
import starting_type_common as S  # noqa: E402
import run_starting_type_adaptation as R  # noqa: E402

REGISTRY = os.path.join(ROOT, "configs", "starting_type_adaptation_v1.yaml")


def test_full_grid_cardinality():
    reg = S.load_registry(REGISTRY)
    keys = list(reg["checkpoints"].keys())
    seeds = list(reg["recipe"]["seeds"])
    betas = [float(reg["recipe"]["kl"]["primary_beta"])]
    assert len(keys) == 10, len(keys)
    assert len(seeds) == 5, seeds
    cells = R.enumerate_grid(reg, "/tmp/sta_grid", keys, S.CONDITIONS, seeds, betas)
    summary = R.validate_grid(cells, keys, S.CONDITIONS, seeds, betas)
    assert summary["counts"] == {"unmodified": 10, "sft": 50, "kl_sft": 50}, summary["counts"]
    assert summary["total_cells"] == 110, summary
    ids = [c["condition_id"] for c in cells]
    assert len(set(ids)) == len(ids) == 110
    # unmodified cells carry the locked seed=-1 sentinel and no beta
    u = [c for c in cells if c["condition"] == "unmodified"]
    assert all(c["seed"] == -1 and c["beta"] is None for c in u) and len(u) == 10
    print("  [ok] full grid: 10 U + 50 SFT + 50 KL-SFT = 110 cells, 110 unique condition_ids")


def test_sensitivity_beta_cardinality():
    reg = S.load_registry(REGISTRY)
    keys = ["qwen25_15b"]
    seeds = [42, 43, 44, 45, 46]
    betas = [0.5, 1.0]  # primary + one sensitivity beta -> +5 kl_sft
    cells = R.enumerate_grid(reg, "/tmp/sta_grid", keys, S.CONDITIONS, seeds, betas)
    summary = R.validate_grid(cells, keys, S.CONDITIONS, seeds, betas)
    assert summary["counts"] == {"unmodified": 1, "sft": 5, "kl_sft": 10}, summary["counts"]
    print("  [ok] sensitivity beta adds a separate 5-seed kl_sft block (1 U + 5 SFT + 10 KL-SFT)")


def test_dry_run_loads_no_model():
    rc = R.main(["--registry", REGISTRY, "--dry-run"])
    assert rc == 0
    print("  [ok] --dry-run over full registry validated cardinality without loading a model")


def _tiny_registry(path, out_root):
    import yaml
    reg = {
        "schema_version": 1, "study_id": "starting_type_adaptation_smoke",
        "conditions": list(S.CONDITIONS),
        "recipe": {
            "train_manifest_from": "paper_a_sft_v2", "seeds": [42, 43, 44, 45, 46],
            "data_order_seed": 42, "max_steps": 300, "max_length": 512,
            "per_device_batch": 1, "gradient_accumulation": 2, "learning_rate": 2e-4,
            "warmup_ratio": 0.0, "scheduler": "constant",
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "kl": {"primary_beta": 0.5, "sensitivity_beta": [1.0]},
        },
        "general_checkpoints": {
            "tiny_test": {
                "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "model_revision": None, "tokenizer_revision": None,
                "trust_remote_code": False, "dtype": "float32", "family": "smollm",
                "contract": "paper_a_safe_unsafe",
            }
        },
        "purpose_built_checkpoints": {},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(reg, f)


def test_real_sft_and_unmodified_cell():
    import shutil
    base = "/tmp/sta_cli_test"
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    yaml_path = os.path.join(base, "registry.yaml")
    out_root = os.path.join(base, "runs")
    _tiny_registry(yaml_path, out_root)

    # unmodified reference cell: single no-train seed=-1 cell, then resumable skip
    rc = R.main(["--registry", yaml_path, "--out-root", out_root,
                 "--checkpoints", "tiny_test", "--conditions", "unmodified"])
    assert rc == 0
    umeta_path = os.path.join(out_root, "tiny_test", "unmodified", "seed_-1", "run_meta.json")
    umeta = C.read_json(umeta_path)
    assert umeta["seed"] == -1 and umeta["kl_beta"] is None
    assert umeta["condition"] == "unmodified" and umeta["status"] == "unmodified_reference"
    assert umeta["train_manifest_sha256"]

    # one real 2-step SFT cell (device cpu, recipe overridden) confirms adapt() is wired
    rc = R.main(["--registry", yaml_path, "--out-root", out_root,
                 "--checkpoints", "tiny_test", "--conditions", "sft", "--seeds", "42",
                 "--max-steps", "2", "--limit-train", "8", "--device", "cpu"])
    assert rc == 0
    cell = os.path.join(out_root, "tiny_test", "sft", "seed_42")
    assert C.adapter_is_present(C.adapter_dir(cell)), "adapter not saved"
    meta = C.read_json(os.path.join(cell, "run_meta.json"))
    assert meta["status"] == "completed", meta.get("failure_reason")
    assert meta["condition"] == "sft" and meta["kl_beta"] == 0.0
    assert meta["condition_id"] == "tiny_test:sft:seed42"

    # resumability: rerun the same SFT slice -> skip (no retrain) unless --force
    rc = R.main(["--registry", yaml_path, "--out-root", out_root,
                 "--checkpoints", "tiny_test", "--conditions", "sft", "--seeds", "42",
                 "--max-steps", "2", "--limit-train", "8", "--device", "cpu"])
    assert rc == 0
    manifest = C.read_json(os.path.join(out_root, "run_manifest.json"))
    assert manifest["status_tally"].get("skip_present", 0) >= 1, manifest["status_tally"]
    assert manifest["finalization_status"] == "nonfinal"
    print("  [ok] unmodified reference (seed=-1) + real 2-step SFT cell wired; resumable skip works")


if __name__ == "__main__":
    test_full_grid_cardinality()
    test_sensitivity_beta_cardinality()
    test_dry_run_loads_no_model()
    test_real_sft_and_unmodified_cell()
    print("ALL PASS")
