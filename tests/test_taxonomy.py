from agent_bouncer.taxonomy import HAZARD_CATEGORIES, Hazard


def test_none_excluded_from_categories():
    assert Hazard.NONE not in HAZARD_CATEGORIES
    assert len(HAZARD_CATEGORIES) == len(list(Hazard)) - 1


def test_agentic_hazards():
    assert Hazard.PROMPT_INJECTION.is_agentic
    assert Hazard.JAILBREAK.is_agentic
    assert not Hazard.HATE.is_agentic
