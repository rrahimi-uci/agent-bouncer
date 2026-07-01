# Benchmarks & results

The research question this repo answers:

> **Does a reasoning + RL (GRPO) guard beat plain SFT classification for
> guardrails — and is it worth the extra latency?**

We compare three regimes against the incumbents, all through the *same* harness so
numbers are apples-to-apples.

## In-session demo result (Regime A)

Fine-tuned `distilbert-base-uncased` (Regime A, binary safe/unsafe) for 2 epochs
(**73 s on an M4 Max**) on a balanced BeaverTails subset, evaluated on a held-out
621-prompt test set through the harness — vs. the reference keyword baseline:

| Guard | Params | F1 | Recall | FPR@benign | p50 latency |
|-------|-------:|---:|-------:|-----------:|------------:|
| keyword-baseline | 0 | 0.007 | 0.003 | 0.00 | 0.004 ms |
| **agent-bouncer-encoder** | 66M | **0.703** | **0.708** | 0.27 | 6.7 ms |

**The learned approach moved the baseline ~100× on F1** (0.007 → 0.703, recall
0.003 → 0.708). Reproduce with `make data-demo && make demo`.

Honest caveats: (1) the baseline here is our *naive keyword matcher* — beating the
**incumbents** (Llama Guard / ShieldGemma) needs their gated checkpoints + `HF_TOKEN`
(wrappers in `eval/baselines.py`). (2) BeaverTails labels *response* safety, so the
prompt-level labels are noisy — hence the modest absolute F1. (3) The 0.27
false-positive-on-benign rate is exactly the over-blocking problem DPO (phase 5)
and reward shaping (phase 4) are designed to reduce.

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
