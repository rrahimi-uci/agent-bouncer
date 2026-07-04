"""Tests for the training hyperparameter spec + validation."""

import pytest

from agent_bouncer.training.hyperparams import param_spec, recommended, validate_params


def _names(arch, tech):
    return {p["name"] for p in param_spec(arch, tech)}


def test_spec_lists_the_right_params_per_combo():
    assert _names("encoder", "sft") == {"epochs", "lr", "batch_size", "max_length", "max_steps"}
    assert {"lora_r", "grad_accum", "max_seq_len"} <= _names("decoder", "sft")
    assert "num_generations" in _names("decoder", "grpo")
    assert "beta" in _names("decoder", "dpo")
    # LoRA knobs only on decoders, never the encoder
    assert "lora_r" not in _names("encoder", "sft")


def test_recommended_values_are_the_best_known_defaults():
    assert recommended("encoder", "sft")["max_length"] == 256   # bumped from 128
    assert recommended("encoder", "sft")["epochs"] == 3
    assert recommended("decoder", "sft")["max_seq_len"] == 512
    assert recommended("decoder", "sft")["lora_r"] == 16
    assert recommended("decoder", "grpo")["max_steps"] == 200
    assert recommended("decoder", "dpo")["batch_size"] == 2   # MPS-safe


def test_every_recommended_value_is_itself_accepted():
    # the recommended default must always be a valid choice
    for arch, tech in [("encoder", "sft"), ("decoder", "sft"), ("decoder", "grpo"), ("decoder", "dpo")]:
        for p in param_spec(arch, tech):
            if p["kind"] == "select" and p["default"] is not None:
                assert p["default"] in p["options"], f"{arch}/{tech}:{p['name']}"


def test_validate_accepts_valid_and_coerces_to_option():
    clean = validate_params("decoder", "sft", {"lora_r": 32, "epochs": 2, "max_seq_len": 512})
    assert clean == {"lora_r": 32, "epochs": 2, "max_seq_len": 512}


def test_validate_rejects_out_of_set_value():
    with pytest.raises(ValueError, match="not allowed"):
        validate_params("decoder", "sft", {"lora_r": 7})           # 7 not in {8,16,32,64}
    with pytest.raises(ValueError, match="not allowed"):
        validate_params("encoder", "sft", {"max_length": 1024})    # >512 not offered


def test_validate_drops_inapplicable_keys():
    # beta is a DPO-only knob; passing it to encoder SFT is silently dropped
    assert "beta" not in validate_params("encoder", "sft", {"beta": 0.1, "epochs": 3})


def test_validate_optional_max_steps_empty_ok_but_bounded():
    assert validate_params("decoder", "sft", {"max_steps": ""}) == {}      # empty = full epochs
    assert validate_params("decoder", "sft", {"max_steps": 300}) == {"max_steps": 300}
    with pytest.raises(ValueError, match="above maximum"):
        validate_params("decoder", "sft", {"max_steps": 10_000_000})
