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
  train/validation split and JSONL I/O; wired `scripts/data/download_data.py`.
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
- **Standard benchmark suite (phase 8):** registry-driven multi-benchmark runner
  (`eval/benchmarks.py`) over 7 ungated benchmarks — BeaverTails, OpenAI-Moderation,
  ToxicChat (guardrail); deepset prompt-injections, jailbreak-classification,
  JailbreakBench (red-teaming); XSTest (over-refusal) — with loaders/normalizers,
  balanced subsampling, and a per-axis Markdown report generator. `scripts/eval/run_benchmarks.py`,
  `scripts/data/download_full_benchmarks.py`, `scripts/report/render_benchmarks.py`, `make bench`.
- **Live LLM-judge comparison:** `OpenAIChatGuard` now supports reasoning models
  (**GPT-5.2 with `reasoning_effort="low"`**) alongside GPT-4o-mini, and treats an
  OpenAI content-policy prompt refusal as an `unsafe` verdict (so red-team benchmarks
  never crash the run). Full 6-guard × 7-benchmark scoreboard in `outputs/BENCHMARKS.md`.
- **Real GRPO (RLVR) from the SFT checkpoint** (`configs/model/grpo_from_sft.yaml`):
  completions stay short/terminal (0% clipped), live verifiable reward, KL stable; the
  RL model merges + loads as a standalone guard and is scored in the suite.
- **Benchmark Studio dashboard** (`serve/api.py` + `serve/dashboard.html`): a FastAPI web
  UI to pick benchmarks / models + tuning technique / test-set size, launch the pipeline
  as a subprocess, and **stream each step over Server-Sent Events** — with live
  Precision / Recall / F1, over-blocking (FPR), latency, and **ROC / AUC** charts
  (Chart.js, vendored). Runs merge into the scoreboard rather than clobbering it.
- **ROC / PR / AUC** (`eval/curves.py`, pure + unit-tested): tie-corrected Mann–Whitney
  AUC + curve points; `scripts/report/compute_curves.py` writes `outputs/curves.json`.
- **`start.sh` / `stop.sh`** — one-command background launch/stop of the Studio (PID file,
  readiness wait, auto-open, idempotent).
- **`notebooks/agent_bouncer_studio.ipynb`** — a single self-contained notebook to configure,
  (optionally) fine-tune/RL, run the benchmark suite, and plot P/R/F1 + ROC/AUC — no CLI.
- **Docs** — rewritten `README.md` and a detailed `docs/architecture.md` with **mermaid**
  diagrams (request path, `Verdict` contract, repo map, data unification, guards, GRPO loop,
  eval harness, serving sequence). `notebook` extra added to `pyproject.toml`.

- **Training subsystem + experiment lifecycle:** a model registry (`models_registry.py`)
  adding **DeepSeek-R1-1.5B, SmolLM2-1.7B, and Gemma-1B** alongside the Qwen3 SLMs (same
  SFT/GRPO/DPO); **model versioning** + **experiment tracking** (`experiments.py`, JSON store);
  **hardware capture** (`hardware.py`: CPU/GPU/memory/runtime); **train/test separation** with
  anti-leakage guards (`split.py`); and orchestration (`training_runner.py`,
  `scripts/train/run_training.py`, `scripts/eval/run_testing.py`) that records params, data, hardware,
  git, and metrics for every run. Leakage-checked testing drops+reports train∩test overlap.
- **Metrics:** added **P90 latency** and **throughput** to `GuardMetrics`.
- **Benchmark Studio lifecycle UI:** new **Train & Test** and **Experiments** tabs — model +
  version + param selection, live streamed training/testing, **P90 & throughput graphs**,
  a **hardware panel**, model comparison, and experiment history. New API endpoints
  (`/api/models`, `/api/experiments`, `/api/train`, `/api/test`, `/api/hardware`).
- **GitHub Pages** landing page (`docs/index.md` + `_config.yml`).
- **Studio redesign → AI-engineering studio:** a **Benchmarks browser** (toolbar of
  benchmarks → view contents: searchable, safe/unsafe filter, hazard tags + per-model
  results); a **Datasets** tab with training-set **strategies** (`training_sets.py`:
  balanced / mixed / over-refusal-aware / red-team, split leakage-safe); the "Run pipeline"
  flow removed in favor of a train/test/experiment-centered nav. New endpoints
  `/api/benchmark/{name}`, `/api/datasets`, `/api/dataset/build`, `/api/train_sets`
  (`scripts/data/build_dataset.py`).
- **`.env` auto-load** (`envfile.py`): OPENAI_API_KEY / HF_TOKEN are picked up by the CLI,
  scripts, notebook, and server on import (`setdefault` — real env wins).

### Fixed

- `DecoderGuard`: truncate inputs (`max_input_tokens`) and select device (MPS/CPU) to
  bound latency; disable HF tokenizer parallelism in the runners to avoid a rayon
  deadlock when a decoder is scored in a loop.
- `run_grpo` now merges the LoRA adapter (like SFT) so the RL model loads standalone.
