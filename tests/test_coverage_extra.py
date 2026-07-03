"""Small targeted tests for remaining branches: normalizer empty-text guards + the
training-device selector."""

import pytest

import agent_bouncer.data.loaders as L


@pytest.mark.parametrize("fn,field", [
    (L.normalize_beavertails, "prompt"),
    (L.normalize_wildguard, "prompt"),
    (L.normalize_aegis, "prompt"),
    (L.normalize_xstest, "prompt"),
    (L.normalize_prompt_injection, "text"),
    (L.normalize_jailbreak_classification, "prompt"),
    (L.normalize_openai_moderation, "prompt"),
    (L.normalize_toxicchat, "user_input"),
])
def test_normalizers_drop_empty_text(fn, field):
    assert fn({field: "   "}) is None


def test_jailbreakbench_drops_empty():
    assert L.normalize_jailbreakbench({"Goal": ""}, unsafe=True) is None


def test_aegis_list_categories_join():
    row = {"prompt": "x", "prompt_label": "unsafe", "violated_categories": ["Suicide and Self Harm"]}
    from agent_bouncer.core.taxonomy import Hazard
    assert L.normalize_aegis(row)["hazard"] == Hazard.SUICIDE_SELF_HARM.value


def test_training_device_selects_accelerator(monkeypatch):
    torch = pytest.importorskip("torch")  # skip off the ML stack (e.g. light CI)

    import agent_bouncer.training.runtime as rt
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert rt.training_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert rt.training_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert rt.training_device() == "cpu"
