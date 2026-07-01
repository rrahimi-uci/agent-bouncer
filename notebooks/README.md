# Notebooks

Reproducible, one-click training paths — the way most people (without an M-series
Mac) will run this.

Planned:

- `01_sft_colab.ipynb` — QLoRA SFT of Qwen3-0.6B with Unsloth on a free Colab T4.
- `02_grpo_colab.ipynb` — GRPO reasoning guard with Unsloth (verifiable rewards).
- `03_eval_and_plots.ipynb` — run the harness, pull MLflow metrics, render the
  results table + reward curves.

> Keep notebooks thin: they should call into `agent_bouncer`, not reimplement it.
