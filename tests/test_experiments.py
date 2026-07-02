import agent_bouncer.tracking.experiments as X


def test_ids_and_version_dir():
    assert X.make_id("smollm2-1.7b", "sft", "20260101-120000") == "smollm2-1.7b-sft-20260101-120000"
    assert X.version_dir("qwen3-0.6b", "v1").replace("\\", "/") == "outputs/models/qwen3-0.6b/v1"


def test_now_shapes():
    stamp, iso = X.now()
    assert len(stamp) == 15 and "-" in stamp  # YYYYMMDD-HHMMSS
    assert "T" in iso


def test_record_get_list_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "EXP_DIR", str(tmp_path))
    monkeypatch.setattr(X, "INDEX", str(tmp_path / "index.json"))
    e = X.Experiment(id="m-sft-1", kind="train", model_key="m", technique="sft",
                     version="1", created="2026-01-01T00:00:00")
    X.record(e)
    got = X.get("m-sft-1")
    assert got["kind"] == "train" and got["model_key"] == "m"
    idx = X.list_experiments()
    assert len(idx) == 1 and idx[0]["id"] == "m-sft-1"
    assert X.get("nope") is None


def test_versions_for_filters_train_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "EXP_DIR", str(tmp_path))
    monkeypatch.setattr(X, "INDEX", str(tmp_path / "index.json"))
    def rec(eid, kind, ver, created):
        X.record(X.Experiment(id=eid, kind=kind, model_key="m", version=ver, created=created))

    rec("m-sft-a", "train", "a", "2026-01-01T00:00:01")
    rec("m-sft-b", "train", "b", "2026-01-01T00:00:02")
    rec("m-eval-c", "eval", "b", "2026-01-01T00:00:03")
    vers = X.versions_for("m")
    assert [v["version"] for v in vers] == ["b", "a"]  # train-only, newest first
