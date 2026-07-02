# Benchmarks & results

The research question this repo answers:

> **Does a reasoning + RL (GRPO) guard beat plain SFT classification for
> guardrails — and is it worth the extra latency?**

We compare three regimes against the incumbents, all through the *same* harness so
numbers are apples-to-apples.

## Results (in-session, on an M4 Max)

**Headline (full 621-prompt held-out test):** a fine-tuned `distilbert` encoder
(Regime A, 2 epochs, **73 s**) scores **F1 0.703 / recall 0.708** vs. the keyword
baseline's **F1 0.007** — the learned approach moves the baseline ~100×.
Reproduce: `make data-demo && make demo`.

### Regime + size sweep (balanced 250-prompt subset, identical harness)

| Guard | Params | F1 | Recall | Prec. | FPR@benign ↓ | p50 latency ↓ |
|-------|-------:|---:|-------:|------:|-------------:|--------------:|
| keyword-baseline         | 0    | 0.016 | 0.008 | 1.00 | 0.000 | 0.005 ms |
| **encoder (distilbert)** | 66M  | 0.683 | 0.672 | 0.69 | 0.296 | **7 ms** |
| decoder-SFT (Qwen3)      | 0.6B | **0.694** | 0.680 | 0.71 | 0.280 | 472 ms |
| decoder-SFT (Qwen3)      | 1.7B | 0.649 | 0.592 | 0.72 | **0.232** | 1093 ms |

Takeaways (reported straight):
1. **Every learned guard beats the keyword baseline ~40×** on F1.
2. **The 66M encoder is the best tradeoff** — ~99% of the top F1 at **~1/70th–1/160th
   the latency** of the decoders, which is decisive on the request path.
3. **Bigger ≠ better here.** The 1.7B decoder got more conservative (highest
   precision, lowest over-blocking at FPR 0.23) but *lower* recall/F1 than 0.6B, at
   2.3× the latency. With 1-epoch LoRA on noisy labels, scaling didn't buy accuracy —
   revisit with more epochs / higher LoRA rank / cleaner labels.
4. **Over-blocking is real** (FPR 0.23–0.30) — the target of DPO (phase 5) and the
   GRPO false-positive reward term.

### GRPO (RLVR) on a real model

Bounded GRPO on Qwen3-0.6B (LoRA, reasoning mode, 20 steps, ~40 s) runs end-to-end:
the verifiable reward is live (per-step mean reward swung ~−0.35 → +0.33, std > 0),
KL stayed ~0.001. This **validates the RLVR loop on a real SLM**, but 20 steps is a
pipeline smoke, not convergence — completions clip at max length (the base model
doesn't yet emit terminal JSON), so a real run needs many more steps, a GPU, and
ideally GRPO *from the SFT checkpoint*. See `configs/model/grpo_qwen3_1_7b.yaml`.

### Not runnable here

The gated incumbents — Llama Guard 3, ShieldGemma, PromptGuard2 — require `HF_TOKEN`
(confirmed 401 in this environment). Their `Guard` wrappers are implemented and their
output parsers unit-tested; run `make baselines` with a token to add their rows.

Caveats bound the *absolute* numbers (not the relative comparison, which uses one
test set + harness for all guards): BeaverTails labels *response* safety, so
prompt-level labels are noisy; the keyword baseline is deliberately naive.

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
