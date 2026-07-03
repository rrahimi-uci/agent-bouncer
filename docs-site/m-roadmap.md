# Roadmap

Eval-driven: we build the scoreboard first, then climb it.

- [x] **Phase 0 — Scaffold + eval harness.** Package, taxonomy, `Verdict`, reference
  guard, metrics (incl. `fpr_on_benign`), eval harness, optional MLflow logging, CLI, tests, CI.
- [x] **Phase 1 — Data.** Unification code + tests (`data.py`; WildGuardMix /
  BeaverTails / Aegis / XSTest normalizers, deterministic splits). Ran a live
  BeaverTails download for the demo dataset (`scripts/data/prepare_beavertails_demo.py`).
- [x] **Phase 2 — Baselines.** Llama Guard / ShieldGemma / PromptGuard2 wrappers with
  unit-tested output parsers (`eval/baselines.py`). Live scoring needs `HF_TOKEN`
  (gated checkpoints).
- [x] **Phase 3 — SFT.** Encoder (`Trainer`) + decoder LoRA (TRL `SFTTrainer`).
  **Encoder trained for real** (see benchmarks.md); decoder SFT **smoke-verified**
  end-to-end on a tiny model (`scripts/train/smoke_train.py`).
- [x] **Phase 4 — GRPO.** Reasoning guard with verifiable reward; reward adapter
  unit-tested. **Ran real GRPO from the SFT checkpoint** (`configs/model/grpo_from_sft.yaml`):
  completions stay short/terminal (0% clipped), live reward, KL stable; the RL model
  merges + loads as a standalone guard and is **scored in the benchmark suite**.
- [x] **Phase 5 — DPO.** Preference-pair builder (unit-tested) + DPOTrainer wiring.
- [x] **Phase 6 — Deploy.** GGUF/MLX command builders + ONNX export + latency
  measurement (`deploy.py`), with tests.
- [x] **Phase 7 — Ship.** Results-table + model-card generators (`eval/report.py`,
  tested); `scripts/report/make_report.py`. Remaining: push to GitHub + Hugging Face release.
- [x] **Phase 8 — Standard benchmark suite.** Registry-driven multi-benchmark runner
  (`eval/benchmarks.py`) over **7 ungated standard benchmarks** across guardrail /
  red-teaming / over-refusal axes; **live comparison vs GPT-4o-mini and GPT-5.2 (low
  reasoning)** + OpenAI Moderation. `make bench` → `outputs/BENCHMARKS.md`.

### Verified in-session
- **7-benchmark suite, one harness:** GPT-5.2 (low) leads on macro-F1 *and* over-blocking
  (FPR 0.19); the **66M encoder ties OpenAI Moderation on macro-F1 at ~22× lower latency**;
  red-teaming (prompt-injection) is the hard axis for all guards. See benchmarks.md.
- Encoder fine-tune (distilbert, 2 epochs, 73 s) **beat the keyword baseline ~100× on F1**.
- Real decoder SFT size sweep on Qwen3 **0.6B and 1.7B**; real GRPO from the SFT checkpoint.

### Follow-ups needing external access
- Score the gated incumbents + benchmarks (Llama Guard / ShieldGemma / WildGuardMix /
  HarmBench / Lakera PINT) — needs `HF_TOKEN` + license acceptance.
- Full-scale **GPU** GRPO run + red-team training data to close the injection-recall gap.
- Publish repo + model.

## Minimum viable *visible* release

Phases 0–4 + one killer results table + one Hugging Face model. That already tells a
complete, novel story. Phases 5–6 are the v2 that earns a second wave of attention.
