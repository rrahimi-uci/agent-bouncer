from agent_bouncer.eval.baselines import parse_llama_guard_output, parse_yes_no_output
from agent_bouncer.schema import Decision
from agent_bouncer.taxonomy import Hazard


def test_llama_guard_safe():
    v = parse_llama_guard_output("safe")
    assert v.decision == Decision.SAFE
    assert v.hazard == Hazard.NONE


def test_llama_guard_unsafe_with_code():
    v = parse_llama_guard_output("unsafe\nS10")
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.HATE


def test_llama_guard_unsafe_multiple_codes_takes_first_known():
    v = parse_llama_guard_output("unsafe\nS9,S1")
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.WEAPONS_CBRNE


def test_llama_guard_unsafe_without_code_falls_back():
    v = parse_llama_guard_output("unsafe")
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.NON_VIOLENT_CRIMES


def test_shieldgemma_yes_is_unsafe():
    assert parse_yes_no_output("Yes").decision == Decision.UNSAFE
    assert parse_yes_no_output("No").decision == Decision.SAFE
    assert parse_yes_no_output(" yes, it violates").decision == Decision.UNSAFE
