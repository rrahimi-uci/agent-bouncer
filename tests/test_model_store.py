import pytest

from agent_bouncer.tracking.model_store import ModelRecord, ModelStore


def _rec(**kw):
    base = dict(base_model="qwen3-0.6b", arch="decoder", technique="sft",
                benchmarks=["beavertails"], sampling="stratified", split="ratio",
                test_ratio=0.3, n_train=140, n_test=60,
                metrics={"f1": 0.7}, per_benchmark={"beavertails": {"f1": 0.7}},
                path="outputs/models/qwen3-0.6b/v1")
    base.update(kw)
    return ModelRecord(**base)


@pytest.fixture(params=["sqlite", "fs"])
def store(request, tmp_path):
    return ModelStore(backend=request.param, root=str(tmp_path / "store"))


@pytest.mark.parametrize("backend", ["sqlite", "fs"])
def test_backend_closes_handles_no_resourcewarning(backend, tmp_path, recwarn):
    # AB-004: `with sqlite3.connect() as con` manages only the transaction (not the connection),
    # and `json.load(open(...))` leaks a file handle — both raise ResourceWarning. Every store op
    # must close its handle.
    import gc
    import warnings
    warnings.simplefilter("always", ResourceWarning)
    store = ModelStore(backend=backend, root=str(tmp_path / "s"))
    mid = store.save(_rec())
    store.get(mid)
    store.list()
    store.delete(mid)
    gc.collect()  # force finalizers so any leaked handle would warn now
    leaks = [w for w in recwarn.list if issubclass(w.category, ResourceWarning)
             and ("unclosed" in str(w.message))]
    assert not leaks, [str(w.message) for w in leaks]


def test_default_backend_saves_to_filesystem(tmp_path):
    import os
    root = tmp_path / "s"
    store = ModelStore(root=str(root))                 # no backend -> default
    assert store.backend == "fs"                        # all artifacts on disk, no DB
    mid = store.save(_rec())
    assert os.path.exists(root / f"{mid}.json")          # inspectable JSON per model
    assert not os.path.exists(root / "models.db")


def test_save_autofills_and_roundtrips(store):
    mid = store.save(_rec())
    assert mid  # id auto-generated
    got = store.get(mid)
    assert got is not None
    assert got.base_model == "qwen3-0.6b" and got.technique == "sft"
    assert got.version and got.created and got.name == mid
    assert got.metrics["f1"] == 0.7 and got.benchmarks == ["beavertails"]
    assert got.test_ratio == 0.3 and got.n_train == 140


def test_save_accepts_dict_and_preserves_explicit_id(store):
    mid = store.save({"id": "custom-1", "base_model": "smollm2-1.7b", "technique": "grpo"})
    assert mid == "custom-1"
    assert store.get("custom-1").base_model == "smollm2-1.7b"


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_list_and_filter(store):
    store.save(_rec(base_model="qwen3-0.6b", technique="sft"))
    store.save(_rec(base_model="qwen3-0.6b", technique="grpo"))
    store.save(_rec(base_model="smollm2-1.7b", technique="sft"))
    assert len(store.list()) == 3
    assert len(store.list(base_model="qwen3-0.6b")) == 2
    assert len(store.list(technique="sft")) == 2
    assert len(store.list(base_model="qwen3-0.6b", technique="grpo")) == 1


def test_list_newest_first(store):
    a = store.save(_rec(id="a", created="2026-01-01T00:00:00"))
    b = store.save(_rec(id="b", created="2026-06-01T00:00:00"))
    ids = [r.id for r in store.list()]
    assert ids.index(b) < ids.index(a)  # newer first


def test_delete(store):
    mid = store.save(_rec())
    assert store.delete(mid) is True
    assert store.get(mid) is None
    assert store.delete(mid) is False  # already gone


def test_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        ModelStore(backend="mongo", root=str(tmp_path))


def test_fs_backend_ignores_non_json_files(tmp_path):
    root = tmp_path / "s"
    fs = ModelStore(backend="fs", root=str(root))
    fs.save(_rec(id="x"))
    (root / "notes.txt").write_text("stray file")  # must be skipped by list()
    assert [r.id for r in fs.list()] == ["x"]
