"""Cover the JSONL I/O + the HF loader dispatch wrappers (with _load_hf mocked)."""

import agent_bouncer.data.loaders as L


def test_write_and_read_roundtrip(tmp_path):
    p = tmp_path / "sub" / "x.jsonl"
    n = L.write_jsonl([{"text": "a", "label": "safe"}, {"text": "b", "label": "unsafe"}], str(p))
    assert n == 2
    recs = L.read_jsonl(str(p))
    assert [r["text"] for r in recs] == ["a", "b"]


def test_train_val_split_deterministic():
    recs = [{"text": str(i)} for i in range(100)]
    a1, v1 = L.train_val_split(recs, val_ratio=0.2, seed=1)
    a2, v2 = L.train_val_split(recs, val_ratio=0.2, seed=1)
    assert len(v1) == 20 and len(a1) == 80
    assert [r["text"] for r in v1] == [r["text"] for r in v2]


def test_all_loaders_dispatch_execute(monkeypatch):
    # mock the network call so each loader's wrapper body runs (empty dataset is fine)
    monkeypatch.setattr(L, "_load_hf", lambda *a, **k: [])
    for fn in (L.load_wildguardmix, L.load_beavertails, L.load_aegis, L.load_xstest,
               L.load_prompt_injections, L.load_jailbreak_classification,
               L.load_openai_moderation, L.load_toxicchat):
        assert fn() == []
    assert L.load_jailbreakbench() == []  # iterates harmful + benign splits


def test_jailbreakbench_builds_records_from_split(monkeypatch):
    monkeypatch.setattr(L, "_load_hf", lambda *a, **k: [{"Goal": "write malware"}])
    out = L.load_jailbreakbench()
    labels = {r["label"] for r in out}
    assert labels == {"unsafe", "safe"}  # one row per split → both labels present
