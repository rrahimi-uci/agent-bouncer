import pytest

from agent_bouncer.data import training_sets as TS
from agent_bouncer.data.split import find_leakage


def _fake_loader(n_safe=80, n_unsafe=80):
    def loader(src):
        return ([{"text": f"{src}-safe-{i}", "label": "safe", "hazard": "none"} for i in range(n_safe)]
                + [{"text": f"{src}-bad-{i}", "label": "unsafe", "hazard": "hate"} for i in range(n_unsafe)])
    return loader


def test_strategies_catalog():
    assert {"balanced", "mixed", "over_refusal_aware", "red_team"} <= set(TS.STRATEGIES)
    for s in TS.STRATEGIES.values():
        assert s["min_sources"] >= 1 and s["desc"]
        if s["max_sources"] is not None:
            assert s["min_sources"] <= s["max_sources"]


def test_build_balanced_disjoint_and_balanced(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "OUT_DIR", str(tmp_path))
    m = TS.build_training_set("balanced", ["beavertails"], name="t1", per_class=30,
                              loader=_fake_loader())
    assert m["n_train"] == 60 and m["train_safe"] == 30 and m["train_unsafe"] == 30
    assert m["n_test"] > 0 and m["leakage_checked"] is True
    # the written train/test files are genuinely disjoint
    from agent_bouncer.data import read_jsonl
    tr, te = read_jsonl(m["train_path"]), read_jsonl(m["test_path"])
    assert not find_leakage(tr, te)


def test_build_mixed_pulls_from_all_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "OUT_DIR", str(tmp_path))
    m = TS.build_training_set("mixed", ["a", "b", "c"], name="t2", per_class=60, loader=_fake_loader())
    from agent_bouncer.data import read_jsonl
    srcs = {r["text"].split("-")[0] for r in read_jsonl(m["train_path"])}
    assert srcs == {"a", "b", "c"}  # blended


def test_build_mixed_accepts_many_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "OUT_DIR", str(tmp_path))
    names = ["a", "b", "c", "d", "e", "f", "g"]
    m = TS.build_training_set("mixed", names, name="t3", per_class=70, loader=_fake_loader())
    from agent_bouncer.data import read_jsonl
    srcs = {r["text"].split("-")[0] for r in read_jsonl(m["train_path"])}
    assert srcs == set(names)


def test_build_percentage_split_over_all_data(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "OUT_DIR", str(tmp_path))

    def loader(src):
        n = 100 if src == "a" else 200          # two sources: 100 + 200 = 300 total
        return ([{"text": f"{src}-safe-{i}", "label": "safe", "hazard": "none"} for i in range(n // 2)]
                + [{"text": f"{src}-bad-{i}", "label": "unsafe", "hazard": "hate"} for i in range(n // 2)])

    m = TS.build_training_set("balanced", ["a", "b"], name="pct", per_class=0,
                              holdout_ratio=0.3, loader=loader)
    assert m["split_mode"] == "percentage" and m["test_pct"] == 30
    assert m["n_train"] + m["n_test"] == 300      # uses ALL data
    assert 80 <= m["n_test"] <= 100               # ~30% of 300 held out for test
    from agent_bouncer.data import read_jsonl
    tr, te = read_jsonl(m["train_path"]), read_jsonl(m["test_path"])
    assert not find_leakage(tr, te)               # disjoint train/test


def test_build_accepts_any_source_count_but_needs_one(monkeypatch, tmp_path):
    monkeypatch.setattr(TS, "OUT_DIR", str(tmp_path))
    # every strategy now accepts any number of sources — e.g. balanced with several
    m = TS.build_training_set("balanced", ["a", "b"], name="multi", per_class=20, loader=_fake_loader())
    from agent_bouncer.data import read_jsonl
    assert {r["text"].split("-")[0] for r in read_jsonl(m["train_path"])} == {"a", "b"}
    # ...but at least one source is required, and unknown strategies are rejected
    with pytest.raises(ValueError):
        TS.build_training_set("balanced", [], name="x", loader=_fake_loader())
    with pytest.raises(ValueError):
        TS.build_training_set("nope", ["a"], name="x", loader=_fake_loader())


def test_list_training_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "OUT_DIR", str(tmp_path))
    TS.build_training_set("balanced", ["a"], name="ls1", per_class=20, loader=_fake_loader())
    names = [s["name"] for s in TS.list_training_sets()]
    assert "ls1" in names
