"""Eval-only mode + the reusable scoring/aggregation helpers in training.runner.

All tests use a fake guard + injected loaders, so no torch/model weights are needed.
"""

import pytest

from agent_bouncer.core.schema import Decision, Surface, Verdict
from agent_bouncer.tracking.model_store import ModelRecord, ModelStore
from agent_bouncer.training import runner


class FakeGuard:
    name = "fake"

    def predict(self, text, *, surface=Surface.USER_PROMPT):
        unsafe = "bad" in text
        return Verdict(
            decision=Decision.UNSAFE if unsafe else Decision.SAFE,
            score=1.0 if unsafe else 0.0, surface=surface, latency_ms=1.0,
        )


def _bench_recs():
    return [
        {"text": "bad-1", "label": "unsafe"}, {"text": "bad-2", "label": "unsafe"},
        {"text": "ok-1", "label": "safe"}, {"text": "ok-2", "label": "safe"},
    ]


def test_macro_average_empty_and_nonempty():
    assert runner.macro_average({}) == {}
    metrics, _ = runner.score_guard(FakeGuard(), ["b1", "b2"], loader=lambda b: _bench_recs())
    macro = runner.macro_average(metrics)
    assert macro["f1"] == 1.0 and macro["fpr_on_benign"] == 0.0
    assert set(runner._MACRO_KEYS) <= set(macro)


def test_score_guard_streams_progress(capsys):
    # the per-record predict loop emits a PROGRESS marker (forced on the final item) so the
    # console shows movement during long benchmark evals
    metrics, _ = runner.score_guard(FakeGuard(), ["b1"], loader=lambda b: _bench_recs())
    out = capsys.readouterr().out
    assert "PROGRESS phase=test label=b1" in out and "step=4 total=4" in out
    assert metrics["b1"]["f1"] == 1.0


def test_score_guard_perfect_with_leakage_drop():
    metrics, leakage = runner.score_guard(
        FakeGuard(), ["b1"], loader=lambda b: _bench_recs(),
        train_recs=[{"text": "bad-1", "label": "unsafe"}],  # leaks into test → dropped
    )
    assert leakage["b1"]["dropped_leaked"] == 1 and leakage["b1"]["n"] == 4
    assert metrics["b1"]["f1"] == 1.0 and "roc_auc" in metrics["b1"]


def test_eval_only_with_path():
    res = runner.eval_only(
        benchmarks=["b1"], path="/models/x", model_key="qwen3-0.6b",
        guard_loader=lambda *a: FakeGuard(), bench_loader=lambda b: _bench_recs(),
        record_exp=False,
    )
    assert res["eval_only"] and res["model_key"] == "qwen3-0.6b"
    assert res["macro"]["f1"] == 1.0 and res["path"] == "/models/x"


def test_eval_only_with_saved_model_refreshes_store(tmp_path):
    store = ModelStore(backend="fs", root=str(tmp_path / "s"))
    mid = store.save(ModelRecord(base_model="qwen3-0.6b", arch="decoder",
                                 technique="sft", path="/m/x"))
    res = runner.eval_only(
        benchmarks=["b1"], model_id=mid, store=store,
        guard_loader=lambda *a: FakeGuard(), bench_loader=lambda b: _bench_recs(),
        record_exp=False, update_store=True,
    )
    assert res["macro"]["f1"] == 1.0
    refreshed = store.get(mid)
    assert refreshed.metrics["f1"] == 1.0
    assert refreshed.per_benchmark["b1"]["f1"] == 1.0


def test_eval_only_requires_a_source():
    with pytest.raises(ValueError, match="model_id or a path"):
        runner.eval_only(benchmarks=["b1"], guard_loader=lambda *a: FakeGuard(),
                         bench_loader=lambda b: _bench_recs())


def test_eval_only_unknown_saved_model(tmp_path):
    store = ModelStore(backend="fs", root=str(tmp_path / "s"))
    with pytest.raises(ValueError, match="unknown saved model"):
        runner.eval_only(benchmarks=["b1"], model_id="nope", store=store,
                         guard_loader=lambda *a: FakeGuard())


def test_save_trained_model_captures_workflow_metadata(tmp_path):
    store = ModelStore(backend="fs", root=str(tmp_path / "s"))
    exp = {
        "model_key": "smollm2-1.7b", "technique": "grpo", "version": "v9",
        "output_dir": "/m/y", "params": {"arch": "decoder", "name": "smollm2-1.7b-grpo-broad"},
        "data": {"n_train": 120, "dataset": "broad"}, "git_commit": "abc123", "notes": "run",
    }
    mid = runner.save_trained_model(
        exp, store=store, sampling="stratified", split="ratio",
        benchmarks=["beavertails", "toxicchat"], test_ratio=0.3, n_test=30,
        metrics={"f1": 0.66}, per_benchmark={"beavertails": {"f1": 0.66}},
    )
    rec = store.get(mid)
    assert rec.base_model == "smollm2-1.7b" and rec.arch == "decoder"
    assert rec.technique == "grpo" and rec.sampling == "stratified" and rec.split == "ratio"
    assert rec.benchmarks == ["beavertails", "toxicchat"] and rec.test_ratio == 0.3
    assert rec.n_train == 120 and rec.n_test == 30 and rec.metrics["f1"] == 0.66
    assert rec.path == "/m/y" and rec.git_commit == "abc123"
    # clear naming + training dataset captured
    assert rec.name == "smollm2-1.7b-grpo-broad" and rec.dataset == "broad"
