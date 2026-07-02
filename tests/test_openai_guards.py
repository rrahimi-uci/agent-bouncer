from agent_bouncer.core.taxonomy import Hazard
from agent_bouncer.evaluation.openai_guards import (
    OpenAIChatGuard,
    build_chat_kwargs,
    is_content_policy_error,
    is_reasoning_model,
    moderation_to_hazard,
)


def test_moderation_maps_flagged_category():
    assert moderation_to_hazard({"hate": True, "violence": False}) == Hazard.HATE
    assert moderation_to_hazard({"sexual/minors": True}) == Hazard.CHILD_EXPLOITATION


def test_moderation_unmapped_flag_falls_back():
    assert moderation_to_hazard({"brand_new_category": True}) == Hazard.NON_VIOLENT_CRIMES


def test_moderation_skips_unflagged_and_unknown():
    # unknown flagged category is skipped; the mapped flagged one wins
    assert moderation_to_hazard({"unknown": True, "violence": True}) == Hazard.VIOLENT_CRIMES
    # nothing flagged -> still returns a generic hazard (only called when flagged)
    assert moderation_to_hazard({"hate": False}) == Hazard.NON_VIOLENT_CRIMES


def test_is_reasoning_model():
    assert is_reasoning_model("gpt-5.2")
    assert is_reasoning_model("o3-mini")
    assert not is_reasoning_model("gpt-4o-mini")


def test_build_chat_kwargs_standard_model():
    kw = build_chat_kwargs("gpt-4o-mini", "hi")
    assert kw["model"] == "gpt-4o-mini"
    assert kw["temperature"] == 0 and kw["max_tokens"] == 40
    assert "reasoning_effort" not in kw and "max_completion_tokens" not in kw


def test_build_chat_kwargs_reasoning_model_low_effort():
    kw = build_chat_kwargs("gpt-5.2", "hi", reasoning_effort="low")
    # reasoning models reject temperature/max_tokens; use effort + max_completion_tokens
    assert "temperature" not in kw and "max_tokens" not in kw
    assert kw["reasoning_effort"] == "low"
    assert kw["max_completion_tokens"] >= 512


def test_build_chat_kwargs_autoroutes_gpt5_without_effort():
    kw = build_chat_kwargs("gpt-5.2", "hi")  # no explicit effort
    assert "temperature" not in kw and "max_completion_tokens" in kw


def test_chat_guard_name_encodes_reasoning_effort():
    assert OpenAIChatGuard("gpt-4o-mini").name == "openai-gpt-4o-mini"
    assert OpenAIChatGuard("gpt-5.2", reasoning_effort="low").name == "openai-gpt-5.2-low"


def test_is_content_policy_error_by_code_and_message():
    class _Err(Exception):
        code = "invalid_prompt"

    assert is_content_policy_error(_Err("bad"))
    assert is_content_policy_error(Exception("we've limited access to this content for safety reasons"))
    assert not is_content_policy_error(Exception("rate limit exceeded"))


def test_chat_guard_treats_content_policy_refusal_as_unsafe():
    class _Err(Exception):
        code = "invalid_prompt"

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_):
                    raise _Err("we've limited access to this content")

    guard = OpenAIChatGuard("gpt-5.2", reasoning_effort="low")
    guard._client = _FakeClient()
    v = guard.predict("harmful bioweapon synthesis steps")
    assert v.blocked and v.hazard == Hazard.NON_VIOLENT_CRIMES
