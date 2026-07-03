"""Pure SFT/GRPO logic — config loading, the binary-metrics computation, and run_sft
dispatch (the real HF/TRL trainer bodies are covered by actual training runs)."""

import pytest

from agent_bouncer.training import grpo, sft

np = pytest.importorskip("numpy")


def test_load_cfg_defaults_data(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("base_model: distilbert-base-uncased\narch: encoder\n")
    cfg = sft._load_cfg(str(p))
    assert cfg["base_model"] == "distilbert-base-uncased"
    assert "train" in cfg["data"] and "validation" in cfg["data"]  # defaulted


def test_binary_metrics_perfect_and_fpr():
    logits = np.array([[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]])  # preds: safe, unsafe, safe
    labels = np.array([0, 1, 0])
    m = sft._binary_metrics((logits, labels))
    assert m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["fpr_on_benign"] == 0.0 and m["accuracy"] == 1.0


def test_binary_metrics_counts_false_positive():
    logits = np.array([[-1.0, 2.0], [2.0, -1.0]])  # preds: unsafe, safe
    labels = np.array([0, 0])                        # both benign → one FP
    m = sft._binary_metrics((logits, labels))
    assert m["fpr_on_benign"] == 0.5 and m["precision"] == 0.0


def test_run_sft_dispatches_on_arch(tmp_path, monkeypatch):
    monkeypatch.setattr(sft, "train_encoder", lambda cfg: "ENC")
    monkeypatch.setattr(sft, "train_decoder", lambda cfg: "DEC")
    enc = tmp_path / "e.yaml"
    enc.write_text("base_model: x\narch: encoder\n")
    dec = tmp_path / "d.yaml"
    dec.write_text("base_model: x\narch: decoder\n")
    assert sft.run_sft(str(enc)) == "ENC"
    assert sft.run_sft(str(dec)) == "DEC"  # default (no arch=encoder) → decoder


def test_grpo_weights_from_cfg():
    w = grpo._weights_from_cfg({"reward": {"correctness": 3.0, "false_positive_penalty": 2.0}})
    assert w.correctness == 3.0 and w.false_positive_penalty == 2.0
    default = grpo._weights_from_cfg({})
    assert default.correctness == 1.0  # falls back to defaults
