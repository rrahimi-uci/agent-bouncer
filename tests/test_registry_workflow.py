"""Model×technique validity + the OpenAI reference-judge catalog (GPT-5.2 tiers)."""

import pytest

from agent_bouncer.evaluation.openai_guards import (
    REASONING_EFFORTS,
    build_openai_guard,
    openai_guard_specs,
)
from agent_bouncer.models.registry import (
    BASE_MODELS,
    assert_valid_combo,
    is_valid_combo,
    technique_matrix,
    valid_techniques,
)

TARGET_DECODERS = ("qwen3-0.6b", "qwen3-1.7b", "deepseek-r1-1.5b", "smollm2-1.7b")


def test_target_models_present_with_full_technique_set():
    for k in TARGET_DECODERS:
        assert k in BASE_MODELS
        assert set(valid_techniques(k)) == {"sft", "grpo", "dpo"}
    assert valid_techniques("distilbert") == ("sft",)  # encoders: SFT only


def test_is_valid_combo_gates_correctly():
    assert is_valid_combo("qwen3-0.6b", "grpo")
    assert not is_valid_combo("distilbert", "grpo")  # can't RL a classifier here
    assert not is_valid_combo("distilbert", "dpo")


def test_technique_matrix_covers_every_model():
    m = technique_matrix()
    assert set(m) == set(BASE_MODELS)
    assert m["distilbert"] == ["sft"]
    assert set(m["deepseek-r1-1.5b"]) == {"sft", "grpo", "dpo"}


def test_assert_valid_combo_raises_helpfully():
    assert_valid_combo("qwen3-0.6b", "sft")  # valid -> no raise
    with pytest.raises(ValueError, match="not supported"):
        assert_valid_combo("distilbert", "grpo")
    with pytest.raises(ValueError, match="unknown technique"):
        assert_valid_combo("qwen3-0.6b", "bogus")
    with pytest.raises(ValueError, match="unknown base model"):
        valid_techniques("does-not-exist")


def test_openai_specs_include_all_reasoning_tiers():
    names = [s["name"] for s in openai_guard_specs()]
    assert "openai-moderation" in names
    assert "openai-gpt-4o-mini" in names
    for eff in REASONING_EFFORTS:
        assert f"openai-gpt-5.2-{eff}" in names
    assert set(REASONING_EFFORTS) == {"low", "medium", "high"}


def test_build_openai_guard_from_specs():
    by_name = {s["name"]: s for s in openai_guard_specs()}
    high = build_openai_guard(by_name["openai-gpt-5.2-high"])
    assert high.name == "openai-gpt-5.2-high" and high.reasoning_effort == "high"
    assert build_openai_guard(by_name["openai-moderation"]).name == "openai-moderation"
    assert build_openai_guard(by_name["openai-gpt-4o-mini"]).model == "gpt-4o-mini"


def test_build_openai_guard_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_openai_guard({"kind": "bogus"})
