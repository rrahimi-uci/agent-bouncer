### Operating-point fairness: guard rankings flip with the threshold (in-house pooled, n=2018)

| Model | Native-threshold F1 (FPR) | Threshold-free AUPRC | Matched-FPR@0.10 F1 (recall) |
|---|---|---|---|
| Guard (ours) | 0.794 (0.306) | 0.844 | 0.581 (0.431) |
| gpt-5.4-mini | 0.784 (0.321) | — (fixed native point) | — |
| Llama-Guard-3-1B | 0.673 (0.178) | 0.639 | 0.360 (0.242) |
| ShieldGemma-2b | 0.424 (0.090) | 0.712 | 0.464 (0.341) |

All matched-FPR@0.10 thresholds are set on the in-distribution dev split and applied to the
pooled test set (guard clean calibration T=2.10, τ=0.59); numbers trace to
`summary_corrected.json`.

**Ranking flip.** At each model's *native* threshold, Llama-Guard-3-1B (F1 0.673) markedly
outranks ShieldGemma-2b (0.424). Under threshold-free AUPRC the order **reverses**
(ShieldGemma 0.712 > Llama-Guard 0.639), and matched-FPR@0.10 agrees (0.464 vs 0.360).
ShieldGemma only *looks* weak because it operates at a very conservative point (FPR 0.09,
low recall); its underlying discrimination is stronger. Naive F1-at-native-threshold guard
comparisons are therefore operating-point artifacts; threshold-free AUPRC + matched-FPR +
paired CIs are required for a fair ranking.
