# Benchmarks & results

The research question this repo answers:

> **Does a reasoning + RL (GRPO) guard beat plain SFT classification for
> guardrails — and is it worth the extra latency?**

We compare three regimes against the incumbents, all through the *same* harness so
numbers are apples-to-apples.

## The table (populated by `make bench`)

| Guard | Params | GuardBench F1 | PINT (injection) | FPR on benign ↓ | p50 latency (CPU) ↓ |
|-------|-------:|--------------:|-----------------:|----------------:|--------------------:|
| Llama-Guard-3-1B (baseline) | 1B | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ShieldGemma-2B (baseline) | 2B | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| PromptGuard2 (baseline) | 86M | — | _TBD_ | _TBD_ | _TBD_ |
| **A — Encoder (ModernBERT)** | 395M | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **B — SFT decoder (Qwen3-0.6B)** | 0.6B | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **C — GRPO reasoning (Qwen3-1.7B)** | 1.7B | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> `fpr_on_benign` (over-blocking) is the headline metric — the number incumbents
> underreport and the one that decides real-world usability.

## Reproducing

```bash
make baselines      # score incumbents through our harness
make train-sft      # regime B
make train-grpo     # regime C
make bench          # GuardBench + PINT + XSTest -> the table above
```

All runs log to MLflow (`mlruns/` by default).
