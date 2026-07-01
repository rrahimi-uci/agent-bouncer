# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial repository scaffold: taxonomy, verdict schema, reference `KeywordGuard`,
  verifiable reward functions, metrics (incl. false-positive-on-benign), MLflow eval
  harness, CLI, and stubs for SFT/GRPO training and standard-benchmark adapters.
- Data unification (roadmap phase 1): offline-tested normalizers mapping WildGuardMix,
  BeaverTails, Aegis 2.0, and XSTest onto the unified taxonomy; deterministic
  train/validation split and JSONL I/O; wired `scripts/download_data.py`.
- Baselines (phase 2): Llama Guard / ShieldGemma / PromptGuard2 guard wrappers with
  unit-tested output parsers.
- Training (phases 3–5): encoder classifier (`Trainer`), decoder LoRA SFT, GRPO
  reasoning guard with a verifiable-reward adapter, and DPO with a preference-pair
  builder — pure cores unit-tested, decoder SFT + GRPO wiring smoke-verified.
- Deploy (phase 6): GGUF/MLX command builders, ONNX export, latency measurement.
- Ship (phase 7): Markdown results-table and Hugging Face model-card generators.
- Verified end-to-end: fine-tuned distilbert (Regime A) beats the keyword baseline
  ~100× on F1 (0.007 → 0.703) on a held-out BeaverTails test set — see
  `docs/benchmarks.md`. Reproduce with `make data-demo && make demo`.
