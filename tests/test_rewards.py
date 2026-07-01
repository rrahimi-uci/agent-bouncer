from agent_bouncer.rewards import (
    RewardWeights,
    brevity_reward,
    composite_reward,
    false_positive_penalty,
)
from agent_bouncer.schema import Decision, Verdict
from agent_bouncer.taxonomy import Hazard


def _v(decision, hazard=Hazard.NONE, rationale=None):
    return Verdict(decision=decision, hazard=hazard, rationale=rationale)


def test_correct_unsafe_beats_wrong():
    gold = _v(Decision.UNSAFE, Hazard.HATE)
    correct = composite_reward(_v(Decision.UNSAFE, Hazard.HATE), gold)
    wrong = composite_reward(_v(Decision.SAFE), gold)
    assert correct > wrong


def test_false_positive_is_penalized():
    gold = _v(Decision.SAFE)
    assert false_positive_penalty(_v(Decision.UNSAFE, Hazard.HATE), gold) == 1.0
    # blocking a benign input scores worse than correctly passing it
    blocked = composite_reward(_v(Decision.UNSAFE, Hazard.HATE), gold)
    passed = composite_reward(_v(Decision.SAFE), gold)
    assert blocked < passed


def test_unparseable_output_gets_zero_format_reward():
    assert composite_reward(None, _v(Decision.SAFE)) == 0.0


def test_brevity_decays_past_budget():
    assert brevity_reward("x" * 10, budget=256) == 1.0
    assert brevity_reward("x" * 512, budget=256) < 1.0


def test_weights_are_configurable():
    gold = _v(Decision.SAFE)
    strict = RewardWeights(false_positive_penalty=5.0)
    assert composite_reward(_v(Decision.UNSAFE, Hazard.HATE), gold, strict) < composite_reward(
        _v(Decision.UNSAFE, Hazard.HATE), gold
    )
