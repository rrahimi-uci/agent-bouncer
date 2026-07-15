# Composition analysis — Compose, Don't Tune (clean v2 execution artifact; retrospective cohort)

Scores: `b941ddbaea7057ab…`  ·  seeds [42, 43, 44, 45, 46]  ·  bootstrap reps 4000 (rng 20260712).
Lock: `cabc8dee9b158773…`  ·  analysis status: **clean_v2_retrospective_estimation**.

## Panel macro-AP by combiner (represented / transfer)

| Combiner | represented | transfer |
|---|---:|---:|
| base | 0.658 | 0.866 |
| sft | 0.982 | 0.807 |
| calibrated_avg | 0.962 | 0.883 |
| raw_avg | 0.943 | 0.863 |
| logit_avg | 0.943 | 0.891 |
| max_cal | 0.961 | 0.826 |
| pit_avg | 0.904 | 0.885 |
| convex_blind | 0.982 | 0.840 |

## Per-model transfer macro-AP (base / SFT / composed calibrated_avg)

| Model | base | SFT | composed |
|---|---:|---:|---:|
| qwen25_15b | 0.819 | 0.780 | 0.855 |
| smollm2_17b | 0.790 | 0.830 | 0.857 |
| smollm3_3b | 0.910 | 0.823 | 0.907 |
| qwen3_4b | 0.944 | 0.794 | 0.914 |

## Bootstrap CIs — composed(calibrated_avg) advantage (panel)

| Regime | ens − SFT [95% CI] | ens − base [95% CI] |
|---|---|---|
| represented | -0.019 [-0.031, -0.010] | +0.297 [+0.250, +0.346] |
| transfer | +0.075 [+0.058, +0.093] | +0.017 [+0.005, +0.030] |

### Per-model transfer ens − base [95% CI]

- qwen25_15b: +0.036 [+0.008, +0.066]
- smollm2_17b: +0.066 [+0.038, +0.095]
- smollm3_3b: -0.003 [-0.012, +0.005]
- qwen3_4b: -0.029 [-0.043, -0.018]

### Per-benchmark transfer advantages [95% CI]

| Benchmark | ensemble − SFT | ensemble − base |
|---|---:|---:|
| jailbreakbench | +0.070 [+0.026, +0.126] | -0.002 [-0.030, +0.030] |
| xstest | +0.028 [+0.018, +0.041] | +0.016 [-0.004, +0.036] |
| wildguardtest | +0.083 [+0.060, +0.108] | +0.016 [-0.008, +0.037] |
| wildjailbreak | +0.118 [+0.083, +0.154] | +0.040 [+0.015, +0.065] |

## Matched-FPR operating point (target 5%) — realized rates

| Guard | regime | macro TPR | macro FPR | pooled FPR |
|---|---|---:|---:|---:|
| base | represented | 0.130 | 0.017 | 0.021 |
| base | transfer | 0.517 | 0.081 | 0.043 |
| sft | represented | 0.769 | 0.011 | 0.019 |
| sft | transfer | 0.581 | 0.155 | 0.170 |
| calibrated_avg | represented | 0.614 | 0.018 | 0.029 |
| calibrated_avg | transfer | 0.639 | 0.114 | 0.091 |

## Single-permutation shuffle diagnostics (panel ens − base)

| Regime | real | label-alignment shuffle | within-class row-pairing shuffle |
|---|---:|---:|---:|
| represented | +0.304 | -0.076 | +0.322 |
| transfer | +0.017 | -0.154 | +0.037 |

*This is retrospective, precision-focused evidence, not a prospective confirmatory result. Clean v2 execution repairs provenance but does not erase prior exposure to part of the transfer cohort. WiSE-FT weight interpolation is out of scope for this output-space analyzer.*
