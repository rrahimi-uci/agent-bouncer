# Roadmap

Eval-driven: we build the scoreboard first, then climb it.

- [x] **Phase 0 — Scaffold + eval harness.** Package, taxonomy, `Verdict`, reference
  guard, metrics (incl. `fpr_on_benign`), MLflow harness, CLI, tests, CI.
- [~] **Phase 1 — Data.** Unification code + tests landed (`agent_bouncer/data.py`,
  normalizers for WildGuardMix / BeaverTails / Aegis / XSTest, deterministic splits,
  `scripts/download_data.py`). Remaining: run the live download (needs `HF_TOKEN`
  for gated sets) and eyeball label balance.
- [ ] **Phase 2 — Baselines.** Wrap Llama Guard / ShieldGemma / PromptGuard2 and
  score them through our harness (`eval/baselines.py`).
- [ ] **Phase 3 — SFT.** Encoder (ModernBERT) + decoder QLoRA (Qwen3/Llama). MLX +
  Unsloth paths.
- [ ] **Phase 4 — GRPO.** Reasoning guard with verifiable reward (headline experiment).
- [ ] **Phase 5 — DPO.** Preference-tune on over-refusal pairs to crush false positives.
- [ ] **Phase 6 — Deploy.** Quantize (GGUF / ONNX / MLX), latency bench, browser demo.
- [ ] **Phase 7 — Ship.** Results table, model card, Hugging Face release, blog post.

## Minimum viable *visible* release

Phases 0–4 + one killer results table + one Hugging Face model. That already tells a
complete, novel story. Phases 5–6 are the v2 that earns a second wave of attention.
