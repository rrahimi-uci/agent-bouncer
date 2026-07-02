from agent_bouncer.models.decoder import build_prompt, format_target, parse_verdict
from agent_bouncer.schema import Decision, Surface, Verdict
from agent_bouncer.taxonomy import Hazard


def test_parse_json_verdict():
    v = parse_verdict('{"decision": "unsafe", "hazard": "hate"}')
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.HATE


def test_parse_safe_forces_none_hazard():
    v = parse_verdict('{"decision": "safe", "hazard": "hate"}')
    assert v.decision == Decision.SAFE
    assert v.hazard == Hazard.NONE


def test_parse_reasoning_block_extracted():
    v = parse_verdict('<think>looks like a jailbreak</think>\n{"decision":"unsafe","hazard":"jailbreak"}')
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.JAILBREAK
    assert "jailbreak" in v.rationale


def test_parse_regex_fallback_when_not_json():
    v = parse_verdict("decision: unsafe\nhazard: privacy")
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.PRIVACY


def test_parse_unknown_hazard_becomes_none_not_fabricated():
    # An invalid category must NOT be silently mapped to a specific crime type.
    v = parse_verdict('{"decision": "unsafe", "hazard": "made_up"}')
    assert v.decision == Decision.UNSAFE
    assert v.hazard == Hazard.NONE


def test_parse_malformed_returns_none():
    assert parse_verdict("I cannot decide") is None
    assert parse_verdict("") is None


def test_format_target_roundtrips():
    original = Verdict(decision=Decision.UNSAFE, hazard=Hazard.HATE)
    reparsed = parse_verdict(format_target(original))
    assert reparsed.decision == original.decision
    assert reparsed.hazard == original.hazard


def test_build_prompt_includes_surface_and_reasoning_hint():
    p = build_prompt("hi", Surface.TOOL_CALL, reasoning=True)
    assert "tool_call" in p
    assert "<think>" in p
