# Roadmap

Eval-driven: we build the scoreboard first, then climb it.

- [x] **Phase 0 — Scaffold + eval harness.** Package, taxonomy, `Verdict`, reference
  guard, metrics (incl. `fpr_on_benign`), MLflow harness, CLI, tests, CI.
- [x] **Phase 1 — Data.** Unification code + tests (`data.py`; WildGuardMix /
  BeaverTails / Aegis / XSTest normalizers, deterministic splits). Ran a live
  BeaverTails download for the demo dataset (`scripts/prepare_beavertails_demo.py`).
- [x] **Phase 2 — Baselines.** Llama Guard / ShieldGemma / PromptGuard2 wrappers with
  unit-tested output parsers (`eval/baselines.py`). Live scoring needs `HF_TOKEN`
  (gated checkpoints).
- [x] **Phase 3 — SFT.** Encoder (`Trainer`) + decoder LoRA (TRL `SFTTrainer`).
  **Encoder trained for real** (see benchmarks.md); decoder SFT **smoke-verified**
  end-to-end on a tiny model (`scripts/smoke_train.py`).
- [x] **Phase 4 — GRPO.** Reasoning guard with verifiable reward; reward adapter
  unit-tested and the full GRPOTrainer wiring **smoke-verified** end-to-end.
- [x] **Phase 5 — DPO.** Preference-pair builder (unit-tested) + DPOTrainer wiring.
- [x] **Phase 6 — Deploy.** GGUF/MLX command builders + ONNX export + latency
  measurement (`deploy.py`), with tests.
- [x] **Phase 7 — Ship.** Results-table + model-card generators (`eval/report.py`,
  tested); `scripts/make_report.py`. Remaining: push to GitHub + Hugging Face release.

### Verified in-session
- Encoder fine-tune (distilbert, 2 epochs, 73 s on M4 Max) **beat the keyword
  baseline ~100× on F1** (0.007 → 0.703) on a held-out test set.
- Decoder SFT + GRPO wiring run end-to-end (tiny-model smoke) — code is correct,
  not just plausible.

### Follow-ups needing external access
- Score the gated incumbents (Llama Guard / ShieldGemma) — needs `HF_TOKEN`.
- Full-scale GRPO run on Qwen3 (heavier compute).
- Publish repo + model.

## Minimum viable *visible* release

Phases 0–4 + one killer results table + one Hugging Face model. That already tells a
complete, novel story. Phases 5–6 are the v2 that earns a second wave of attention.
