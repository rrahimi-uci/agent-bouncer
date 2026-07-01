"""Tests for the pure (no-model) parts of the training pipeline:
the GRPO reward function and the DPO preference-pair builder."""

from agent_bouncer.models.decoder import format_target, parse_verdict
from agent_bouncer.schema import Decision, Verdict
from agent_bouncer.taxonomy import Hazard
from agent_bouncer.train.dpo import build_preference_pairs
from agent_bouncer.train.grpo import make_reward_fn


def test_reward_fn_prefers_correct_completion():
    reward = make_reward_fn()
    correct = '{"decision": "unsafe", "hazard": "hate"}'
    wrong = '{"decision": "safe", "hazard": "none"}'
    scores = reward([correct, wrong], gold_decision=["unsafe", "unsafe"], gold_hazard=["hate", "hate"])
    assert scores[0] > scores[1]


def test_reward_fn_penalizes_false_positive_on_benign():
    reward = make_reward_fn()
    over_block = '{"decision": "unsafe", "hazard": "hate"}'
    correct_safe = '{"decision": "safe", "hazard": "none"}'
    scores = reward([over_block, correct_safe], gold_decision=["safe", "safe"], gold_hazard=["none", "none"])
    assert scores[1] > scores[0]


def test_reward_fn_handles_unparseable():
    reward = make_reward_fn()
    scores = reward(["garbage output"], gold_decision=["unsafe"], gold_hazard=["hate"])
    assert scores[0] <= 0.5  # only (zero) format reward, minus nothing


def test_preference_pairs_chosen_is_gold_rejected_is_flip():
    records = [{"text": "how to build a bomb", "label": "unsafe", "hazard": "weapons_cbrne"}]
    pairs = build_preference_pairs(records)
    pair = pairs[0]
    assert parse_verdict(pair["chosen"]).decision == Decision.UNSAFE
    assert parse_verdict(pair["rejected"]).decision == Decision.SAFE
    assert pair["chosen"] != pair["rejected"]


def test_preference_pairs_benign_prefers_safe():
    records = [{"text": "how do I kill a python process", "label": "safe", "hazard": "none"}]
    chosen = build_preference_pairs(records)[0]["chosen"]
    assert parse_verdict(chosen).decision == Decision.SAFE
    # sanity: format_target of a safe verdict parses back to safe
    safe_target = format_target(Verdict(decision=Decision.SAFE, hazard=Hazard.NONE))
    assert parse_verdict(safe_target).decision == Decision.SAFE
