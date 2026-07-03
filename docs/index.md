---
title: "Agent Bouncer — a tiny, fast LLM safety guardrail (SLM)"
description: "Open-source small language model (SLM) guardrail for LLMs and AI agents: detects prompt injection, jailbreaks, and unsafe content on the request path. Fine-tuned + RL-tuned (GRPO), benchmarked vs GPT-4o-mini, GPT-5.2, and OpenAI Moderation."
---

![Agent Bouncer Workbench](media/agent-bouncer.png)

# Agent Bouncer

**A tiny, fast safety bouncer for LLMs and agents** — screens prompts, tool calls, and
outputs *before* they reach your model. SLM guardrails, trained with **fine-tuning + RL**,
**benchmarked** on a standard suite, and shipped with an honest scoreboard and a web Workbench.

![Agent Bouncer Workbench](media/benchmark-studio.png)

## Highlights

- **7-benchmark standard suite** across guardrail, red-teaming, and over-refusal axes.
- **Compared live vs GPT-4o-mini and GPT-5.2 at low / medium / high reasoning** + OpenAI Moderation, through one harness.
- **Precision / Recall / F1 / ROC-AUC / latency / throughput / P90** — over-blocking (`fpr_on_benign`) front and center.
- **Full training lifecycle**: model registry (Qwen3, DeepSeek-R1-1.5B, SmolLM2-1.7B, Gemma-1B),
  configurable SFT/GRPO/DPO, **versioning**, **experiment tracking**, **hardware capture**, and
  **train/test leakage guards**.
- **Agent Bouncer Workbench** — configure, train, test, and compare from a polished web UI, with a
  **Leaderboard** (results table + ROC/PR/AUC curves), a **PDF report** export, and an
  **interactive ensemble builder**.

## Headline result (7 benchmarks, one harness)

> **Illustrative figures from a reference run** — the repo ships with an empty results directory.
> Run `make bench` (or use the Workbench) to generate your own; numbers vary with hardware, sampling,
> and model versions.

| Guard | Params | macro-F1 | macro-FPR@benign ↓ | p50 ms ↓ |
|-------|-------:|---------:|-------------------:|---------:|
| encoder (distilbert)     | 66M  | 0.579 | 0.288 | **8** |
| decoder-SFT (Qwen3)      | 0.6B | 0.672 | 0.355 | 523 |
| openai-gpt-4o-mini       | api  | **0.788** | 0.266 | 687 |
| **openai-gpt-5.2 (low)** | api  | **0.788** | **0.187** | 1117 |

The **66M encoder ties OpenAI Moderation on macro-F1 at ~22× lower latency**; **GPT-5.2 (low)**
leads on quality *and* over-blocking but is ~140× slower — not a per-call gate.

## Learn (course-style, with diagrams)

A self-contained mini-course on the models and methods behind SLM guardrails — written as
student training material:

- **[SLM Architectures — a visual guide](slm-architectures.md)** — encoder vs decoder, the modern
  decoder block (RMSNorm · RoPE · GQA · SwiGLU), and a deep dive into each model
  (DistilBERT, Qwen3-0.6B/1.7B, DeepSeek-R1-Distill-1.5B, SmolLM2-1.7B).
- **[Fine-tuning techniques](fine-tuning.md)** — SFT · LoRA · GRPO (RLVR) · DPO explained with
  diagrams, and which technique applies to which model.
- **[The guided workflow](workflow.md)** — benchmark → sampling/split → model+technique → train →
  test → save → evaluate, end to end.

## Reference docs

- [Architecture](architecture.md) — how it all fits together (with mermaid diagrams)
- [Benchmarks & results](benchmarks.md) — the full scoreboard + analysis
- [Ensembles](ensembles.md) — can combining SLMs match GPT-5.2 Low?
- [Datasets](datasets.md) · [Taxonomy](taxonomy.md) · [Roadmap](roadmap.md)
- Auto-generated scoreboard → `outputs/BENCHMARKS.md` (created by `make bench`)

## Run it

```bash
git clone <your-repo-url> agent-bouncer && cd agent-bouncer
make setup && ./start.sh          # → http://127.0.0.1:8000  (Agent Bouncer Workbench)
```

Or drive everything from the CLI / `make` targets — see the [README](https://github.com/rrahimi-uci/agent-bouncer#readme).

---
<sub>Apache-2.0 · a **defensive** security tool · no guardrail catches everything — pair with alignment + human review.</sub>
