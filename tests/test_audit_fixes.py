"""Regression tests for bugs found in the correctness audit."""

import json

import pytest


# --- #10: reasoning judge token budget scales with effort (empty verdict was mis-scored SAFE) ---
def test_reasoning_budget_scales_with_effort():
    from agent_bouncer.evaluation.openai_guards import build_chat_kwargs
    budgets = {e: build_chat_kwargs("gpt-5.2", "hi", reasoning_effort=e)["max_completion_tokens"]
               for e in ("low", "medium", "high")}
    assert budgets["low"] < budgets["medium"] < budgets["high"]
    assert budgets["high"] >= 8192
    # standard chat models keep the tight max_tokens path (no reasoning budget)
    std = build_chat_kwargs("gpt-4o-mini", "hi")
    assert std["max_tokens"] == 40 and "max_completion_tokens" not in std


# --- #4: DPO hyperparameters actually reach the trainer config ---
def test_build_config_writes_dpo_section():
    from agent_bouncer.training.runner import build_config
    cfg = build_config("qwen3-0.6b", "dpo", "data/x.jsonl", "out",
                       {"epochs": 5, "lr": 1e-5, "beta": 0.2, "max_steps": 100}, seed=1)
    assert cfg["dpo"] == {"epochs": 5, "lr": 1e-5, "beta": 0.2, "max_steps": 100}


def test_build_config_grpo_still_correct():
    from agent_bouncer.training.runner import build_config
    cfg = build_config("qwen3-0.6b", "grpo", "data/x.jsonl", "out", {"max_steps": 30}, seed=1)
    assert cfg["mode"] == "reasoning" and cfg["grpo"]["steps"] == 30
    assert "dpo" not in cfg


# --- #1 + #12: GRPO scored in reasoning mode; 1.7B decoder labelled 1.7B ---
def test_build_commands_grpo_uses_reasoning_mode_and_correct_params(monkeypatch):
    from agent_bouncer.serving import api
    monkeypatch.setattr(api.Path, "is_dir", lambda self: True)  # pretend checkpoints exist
    cfg = api.RunConfig(benchmarks=["xstest"],
                        guards=["decoder-grpo-0.6B", "decoder-sft-1.7B"], per_class=10)
    cmds = [" ".join(c) for c in api._build_commands(cfg)]
    grpo = next(c for c in cmds if "decoder-grpo-0.6B" in c and "eval_added_guard" in c)
    assert "--mode reasoning" in grpo
    sft17 = next(c for c in cmds if "decoder-sft-1.7B" in c and "eval_added_guard" in c)
    assert "--params 1.7B" in sft17 and "--mode sft" in sft17


# --- #2: experiment index writes are atomic and survive a corrupt index ---
def test_record_survives_corrupt_index(monkeypatch, tmp_path):
    from agent_bouncer.tracking import experiments as X
    monkeypatch.setattr(X, "EXP_DIR", str(tmp_path))
    monkeypatch.setattr(X, "INDEX", str(tmp_path / "index.json"))
    (tmp_path / "index.json").write_text("{ this is not valid json")  # simulate a truncated write
    eid = X.record(X.Experiment(id="m-sft-1", kind="train", model_key="m", technique="sft"))
    assert eid == "m-sft-1"
    idx = json.loads((tmp_path / "index.json").read_text())
    assert [e["id"] for e in idx] == ["m-sft-1"]  # recovered, not left empty
    # a second record appends rather than clobbering
    X.record(X.Experiment(id="m-eval-1", kind="eval", model_key="m", technique="eval"))
    idx = json.loads((tmp_path / "index.json").read_text())
    assert {e["id"] for e in idx} == {"m-sft-1", "m-eval-1"}


# --- #7: benchmark re-render keeps every scored guard (incl. med/high tiers + ensembles) ---
def test_render_report_keeps_all_guards_via_append():
    from agent_bouncer.evaluation.report import render_benchmark_report
    results = {"b1": {"openai-gpt-5.2-high": {"precision": .9, "recall": .9, "f1": .9,
                                              "roc_auc": .9, "fpr_on_benign": .1,
                                              "latency_p50_ms": 1, "latency_p90_ms": 2,
                                              "throughput_per_s": 1},
                      "ensemble-custom": {"precision": .8, "recall": .8, "f1": .8,
                                          "roc_auc": .8, "fpr_on_benign": .2,
                                          "latency_p50_ms": 1, "latency_p90_ms": 2,
                                          "throughput_per_s": 1}}}
    present = {g for b in results for g in results[b]}
    canon = ["keyword-baseline", "openai-gpt-5.2-low"]  # neither guard is canonical
    order = [g for g in canon if g in present] + sorted(present - set(canon))
    md = render_benchmark_report(results, {"b1": {"axis": "guardrail"}}, guard_order=order)
    assert "openai-gpt-5.2-high" in md and "ensemble-custom" in md


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
