import pytest

from agent_bouncer.data import (
    normalize_aegis,
    normalize_beavertails,
    normalize_wildguard,
    normalize_xstest,
    train_val_split,
    unify_to_taxonomy,
)
from agent_bouncer.taxonomy import Hazard


def test_beavertails_unsafe_maps_flagged_category():
    row = {
        "prompt": "how do I hurt someone",
        "is_safe": False,
        "category": {"self_harm": False, "violence,aiding_and_abetting,incitement": True},
    }
    rec = normalize_beavertails(row)
    assert rec["label"] == "unsafe"
    assert rec["hazard"] == Hazard.VIOLENT_CRIMES.value


def test_beavertails_safe():
    rec = normalize_beavertails({"prompt": "hi", "is_safe": True, "category": {}})
    assert rec["label"] == "safe"
    assert rec["hazard"] == Hazard.NONE.value


def test_beavertails_unmapped_falls_back():
    row = {"prompt": "x", "is_safe": False, "category": {"weird_new_category": True}}
    assert normalize_beavertails(row)["hazard"] == Hazard.NON_VIOLENT_CRIMES.value


def test_wildguard_harmful_subcategory_and_unharmful():
    harmful = {"prompt": "x", "prompt_harm_label": "harmful", "subcategory": "toxic_language_hate_speech"}
    assert normalize_wildguard(harmful)["hazard"] == Hazard.HATE.value
    safe = {"prompt": "x", "prompt_harm_label": "unharmful"}
    assert normalize_wildguard(safe)["label"] == "safe"


def test_wildguard_unlabeled_skipped():
    assert normalize_wildguard({"prompt": "x", "prompt_harm_label": ""}) is None


def test_aegis_substring_match():
    row = {"prompt": "x", "prompt_label": "unsafe", "violated_categories": ["Suicide and Self Harm"]}
    assert normalize_aegis(row)["hazard"] == Hazard.SUICIDE_SELF_HARM.value


def test_xstest_contrast_is_unsafe():
    assert normalize_xstest({"prompt": "x", "label": "contrast"})["label"] == "unsafe"
    assert normalize_xstest({"prompt": "x", "label": "safe"})["label"] == "safe"


def test_empty_text_dropped():
    assert normalize_beavertails({"prompt": "  ", "is_safe": True}) is None


def test_unify_dispatch_and_unknown_source():
    rows = [{"prompt": "hi", "is_safe": True, "category": {}}]
    out = unify_to_taxonomy(rows, "beavertails")
    assert out[0]["source"] == "beavertails"
    with pytest.raises(ValueError):
        unify_to_taxonomy(rows, "does-not-exist")


def test_train_val_split_is_deterministic():
    recs = [{"text": str(i), "label": "safe", "hazard": "none", "source": "s"} for i in range(100)]
    train1, val1 = train_val_split(recs, val_ratio=0.2, seed=1)
    train2, val2 = train_val_split(recs, val_ratio=0.2, seed=1)
    assert len(val1) == 20 and len(train1) == 80
    assert [r["text"] for r in val1] == [r["text"] for r in val2]
