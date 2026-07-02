from agent_bouncer.core.schema import Decision
from agent_bouncer.core.taxonomy import Hazard
from agent_bouncer.models.encoder import aggregate_unsafe


def test_binary_head_unsafe():
    id2label = {0: "safe", 1: "unsafe"}
    decision, hazard, score = aggregate_unsafe(id2label, [0.2, 0.8])
    assert decision == Decision.UNSAFE
    assert hazard == Hazard.NON_VIOLENT_CRIMES  # binary head has no hazard name
    assert round(score, 3) == 0.8


def test_binary_head_safe():
    decision, hazard, score = aggregate_unsafe({0: "safe", 1: "unsafe"}, [0.9, 0.1])
    assert decision == Decision.SAFE
    assert hazard == Hazard.NONE


def test_multiclass_head_picks_top_hazard():
    id2label = {0: "none", 1: "hate", 2: "privacy"}
    decision, hazard, score = aggregate_unsafe(id2label, [0.1, 0.7, 0.2])
    assert decision == Decision.UNSAFE
    assert hazard == Hazard.HATE
    assert round(score, 3) == 0.9  # 0.7 + 0.2 unsafe mass


def test_threshold_respected():
    id2label = {0: "safe", 1: "unsafe"}
    decision, _, _ = aggregate_unsafe(id2label, [0.6, 0.4], threshold=0.5)
    assert decision == Decision.SAFE
