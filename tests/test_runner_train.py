"""Cover build_config, _merge_scoreboard, _load_guard, train_and_record, evaluate_and_record
with the heavy training/model bits mocked out (no torch, no downloads)."""

import json
import os

import pytest

from agent_bouncer.training import runner


def test_build_config_encoder():
    cfg = runner.build_config("distilbert", "sft", "tr.jsonl", "/o", {}, 42)
    # recommended defaults: 3 epochs, 256-token window (see training.hyperparams)
    assert cfg["arch"] == "encoder" and cfg["train"]["epochs"] == 3
    assert cfg["train"]["max_length"] == 256 and cfg["data"]["train"] == "tr.jsonl"


def test_build_config_decoder_sft_and_grpo():
    dec = runner.build_config("qwen3-0.6b", "sft", "tr.jsonl", "/o",
                              {"max_steps": 10, "bf16": True, "lora_r": 8}, 42)
    assert dec["arch"] == "decoder" and dec["train"]["max_steps"] == 10
    assert dec["train"]["bf16"] is True and dec["lora"]["r"] == 8
    grpo = runner.build_config("qwen3-0.6b", "grpo", "tr.jsonl", "/o", {}, 42)
    assert grpo["mode"] == "reasoning" and "grpo" in grpo


def test_merge_scoreboard_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_JSON", str(tmp_path / "r.json"))
    runner._merge_scoreboard("myguard", "0.6B", {"beavertails": {"f1": 0.7}})
    blob = json.load(open(tmp_path / "r.json"))
    assert blob["results"]["beavertails"]["myguard"]["f1"] == 0.7


def test_load_guard_dispatch(monkeypatch):
    import agent_bouncer.models.decoder as D
    import agent_bouncer.models.encoder as E
    monkeypatch.setattr(E, "EncoderGuard", lambda p, name: ("enc", p, name))
    monkeypatch.setattr(D, "DecoderGuard", lambda p, mode, name, device: ("dec", mode, device))
    assert runner._load_guard("distilbert", "encoder", "/o", "sft", "cpu")[0] == "enc"
    assert runner._load_guard("qwen3-0.6b", "decoder", "/o", "grpo", "mps")[1] == "reasoning"


def test_train_and_record(tmp_path, monkeypatch, capsys):
    import agent_bouncer.training.sft as sft
    monkeypatch.setattr(sft, "run_sft", lambda cfg_path: None)
    recorded = {}
    monkeypatch.setattr(runner.X, "record",
                        lambda e: recorded.update(e if isinstance(e, dict) else e.to_dict()))
    tr = tmp_path / "train.jsonl"
    tr.write_text('{"text": "a", "label": "safe"}\n{"text": "b", "label": "unsafe"}\n')
    exp = runner.train_and_record("distilbert", "sft", train_data=str(tr), params={"epochs": 1})
    assert exp["model_key"] == "distilbert" and exp["kind"] == "train"
    assert exp["data"]["n_train"] == 2 and recorded["id"] == exp["id"]
    # naming: <model>-<params>-<technique>-<dataset>-<stamp> (params added since not in key)
    assert exp["id"].startswith("distilbert-66M-sft-")
    assert exp["params"]["name"].startswith("distilbert-66M-sft-") and exp["data"]["dataset"]
    # beautiful, informative console banner (start header + done footer)
    out = capsys.readouterr().out
    assert "🚀 Training distilbert" in out and "1 safe / 1 unsafe" in out
    assert "⏱️ Estimated:" in out and "🏷️ Saves as:" in out
    assert "✅ Trained distilbert" in out


def test_train_and_record_cleans_temp_config_when_trainer_fails(tmp_path, monkeypatch):
    import agent_bouncer.training.sft as sft

    seen = {}

    def fail(cfg_path):
        seen["cfg"] = cfg_path
        assert os.path.exists(cfg_path)
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr(sft, "run_sft", fail)
    real_unlink = os.unlink
    unlinked = []

    def tracking_unlink(path):
        unlinked.append(path)
        real_unlink(path)

    monkeypatch.setattr(runner.os, "unlink", tracking_unlink)
    tr = tmp_path / "train.jsonl"
    tr.write_text('{"text": "a", "label": "safe"}\n')

    with pytest.raises(RuntimeError, match="trainer exploded"):
        runner.train_and_record("distilbert", "sft", train_data=str(tr), params={"epochs": 1})

    assert unlinked == [seen["cfg"]]
    assert not os.path.exists(seen["cfg"])


def test_training_console_helpers():
    assert runner.fmt_duration(45) == "45s"
    assert runner.fmt_duration(344) == "5m 44s"
    assert runner.fmt_duration(3900) == "1h 05m"
    assert runner._params_billions("0.6B") == 0.6 and runner._params_billions("66M") == 0.066
    assert runner._params_billions("weird") == 1.0
    assert runner._class_balance([{"label": "unsafe"}, {"label": "safe"}, {"label": "1"}]) == (1, 2)
    # steps: encoder ceil(n/batch)*epochs; grpo uses configured steps; max_steps wins
    enc = runner.build_config("distilbert", "sft", "t", "/o", {"batch_size": 8, "epochs": 2}, 0)
    assert runner._plan_steps(enc, "sft", 32) == 8            # ceil(32/8)*2
    grpo = runner.build_config("qwen3-0.6b", "grpo", "t", "/o", {}, 0)
    assert runner._plan_steps(grpo, "grpo", 999) == 60        # configured grpo steps
    # GRPO wall-clock estimate exceeds a plain SFT step count (rollouts dominate)
    assert runner._eta_seconds("decoder", 0.6, "mps", 60, "grpo", 4) > \
        runner._eta_seconds("decoder", 0.6, "mps", 60, "sft")
    assert runner._device_from_hw({"gpu": "mps"}) == "mps" and runner._device_from_hw({}) == "cpu"


def test_naming_helpers():
    assert runner.dataset_name("data/train_sets/bt-balanced/train.jsonl") == "bt-balanced"
    assert runner.dataset_name("data/demo/train.jsonl") == "demo"
    # params folded in only when the key doesn't already carry them
    assert runner.descriptive_name("qwen3-0.6b", "grpo", "bt") == "qwen3-0.6b-grpo-bt"
    assert runner.descriptive_name("distilbert", "sft", "bt") == "distilbert-66M-sft-bt"


def test_train_and_record_rejects_bad_technique():
    with pytest.raises(ValueError, match="supports"):
        runner.train_and_record("distilbert", "grpo", train_data="x")  # encoder can't GRPO


def test_evaluate_and_record_mocked(tmp_path, monkeypatch):
    import agent_bouncer.evaluation.benchmarks as B
    from agent_bouncer.core.schema import Decision, Surface, Verdict

    class FakeGuard:
        name = "fake"

        def predict(self, text, *, surface=Surface.USER_PROMPT):
            return Verdict(decision=Decision.UNSAFE if "bad" in text else Decision.SAFE,
                           score=0.0, surface=surface, latency_ms=1.0)

    train_exp = {"model_key": "qwen3-0.6b", "technique": "sft", "version": "v1",
                 "base_hf_id": "hf", "output_dir": "/o", "params": {"arch": "decoder"},
                 "data": {"train": str(tmp_path / "none.jsonl")}}
    monkeypatch.setattr(runner.X, "get", lambda i: train_exp)
    recorded = {}
    monkeypatch.setattr(runner.X, "record", lambda e: recorded.update(e.to_dict()))
    monkeypatch.setattr(runner, "_load_guard", lambda *a: FakeGuard())
    monkeypatch.setattr(B, "load_benchmark",
                        lambda b, balanced=True, per_class=40: [{"text": "bad", "label": "unsafe"},
                                                                {"text": "ok", "label": "safe"}])
    monkeypatch.setattr(B, "BENCHMARKS", {"beavertails": object()})
    exp = runner.evaluate_and_record("t1", benchmarks=["beavertails"], per_class=2)
    assert exp["kind"] == "eval" and exp["metrics_summary"]["f1"] == 1.0
    assert recorded["metrics"]["beavertails"]["f1"] == 1.0


def test_evaluate_on_created_test_set_drops_train_leakage(tmp_path, monkeypatch):
    from agent_bouncer.core.schema import Decision, Surface, Verdict

    class FakeGuard:
        name = "fake"

        def predict(self, text, *, surface=Surface.USER_PROMPT):
            return Verdict(decision=Decision.UNSAFE if "bad" in text else Decision.SAFE,
                           score=0.0, surface=surface, latency_ms=1.0)

    train = tmp_path / "train.jsonl"
    train.write_text('{"text": "bad-leak", "label": "unsafe"}\n')
    tdir = tmp_path / "set-1"
    tdir.mkdir()
    test = tdir / "test.jsonl"
    test.write_text('{"text": "bad-leak", "label": "unsafe"}\n'      # leaks from train → dropped
                    '{"text": "bad-2", "label": "unsafe"}\n'
                    '{"text": "ok", "label": "safe"}\n')
    train_exp = {"model_key": "qwen3-0.6b", "technique": "sft", "version": "v1",
                 "base_hf_id": "hf", "output_dir": "/o", "params": {"arch": "decoder"},
                 "data": {"train": str(train)}}
    monkeypatch.setattr(runner.X, "get", lambda i: train_exp)
    recorded = {}
    monkeypatch.setattr(runner.X, "record", lambda e: recorded.update(e.to_dict()))
    monkeypatch.setattr(runner, "_load_guard", lambda *a: FakeGuard())

    exp = runner.evaluate_and_record("t1", test_set=str(test))
    assert exp["data"]["test_set"] == str(test)
    assert recorded["data"]["leakage"]["set-1"]["dropped_leaked"] == 1   # bad-leak removed
    assert recorded["metrics"]["set-1"]["f1"] == 1.0                     # clean rows scored perfectly


def test_evaluate_created_test_set_missing_path_raises(tmp_path, monkeypatch):
    train_exp = {"model_key": "qwen3-0.6b", "technique": "sft", "version": "v1",
                 "base_hf_id": "hf", "output_dir": "/o", "params": {"arch": "decoder"},
                 "data": {"train": str(tmp_path / "train.jsonl")}}
    monkeypatch.setattr(runner.X, "get", lambda i: train_exp)
    loaded = False

    def load_guard(*a):
        nonlocal loaded
        loaded = True
        return object()

    monkeypatch.setattr(runner, "_load_guard", load_guard)

    with pytest.raises(ValueError, match="test set not found"):
        runner.evaluate_and_record("t1", test_set=str(tmp_path / "missing.jsonl"))
    assert loaded is False


def test_score_guard_raises_when_all_rows_leaked():
    recs = [{"text": "same prompt", "label": "unsafe"}]

    with pytest.raises(ValueError, match="no clean samples left"):
        runner.score_guard(object(), ["b1"], loader=lambda b: recs, train_recs=recs)
