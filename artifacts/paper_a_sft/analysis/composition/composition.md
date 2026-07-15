# Composition analysis — Compose, Don't Tune (legacy execution artifact)

Scores: `d4179244163f1d99…`  ·  seeds [42, 43, 44, 45, 46]  ·  bootstrap reps 4000 (rng 20260712).
Lock: `08c47c0459adbcdf…`  ·  analysis status: **legacy_retrospective_estimation**.

## Panel macro-AP by combiner (represented / transfer)

| Combiner | represented | transfer |
|---|---:|---:|
| base | 0.651 | 0.867 |
| sft | 0.983 | 0.817 |
| calibrated_avg | 0.965 | 0.890 |
| raw_avg | 0.946 | 0.873 |
| logit_avg | 0.942 | 0.897 |
| max_cal | 0.964 | 0.835 |
| pit_avg | 0.904 | 0.889 |
| convex_blind | 0.984 | 0.847 |

## Per-model transfer macro-AP (base / SFT / composed calibrated_avg)

| Model | base | SFT | composed |
|---|---:|---:|---:|
| qwen25_15b | 0.822 | 0.791 | 0.863 |
| smollm2_17b | 0.787 | 0.838 | 0.864 |
| smollm3_3b | 0.914 | 0.794 | 0.909 |
| qwen3_4b | 0.945 | 0.843 | 0.926 |

## Bootstrap CIs — composed(calibrated_avg) advantage (panel)

| Regime | ens − SFT [95% CI] | ens − base [95% CI] |
|---|---|---|
| represented | -0.018 [-0.030, -0.009] | +0.307 [+0.257, +0.357] |
| transfer | +0.073 [+0.056, +0.091] | +0.023 [+0.011, +0.036] |

### Per-model transfer ens − base [95% CI]

- qwen25_15b: +0.040 [+0.012, +0.072]
- smollm2_17b: +0.076 [+0.051, +0.102]
- smollm3_3b: -0.005 [-0.014, +0.003]
- qwen3_4b: -0.019 [-0.030, -0.009]

### Per-benchmark transfer advantages [95% CI]

| Benchmark | ensemble − SFT | ensemble − base |
|---|---:|---:|
| jailbreakbench | +0.074 [+0.027, +0.133] | +0.003 [-0.024, +0.033] |
| xstest | +0.028 [+0.016, +0.043] | +0.017 [-0.006, +0.041] |
| wildguardtest | +0.070 [+0.048, +0.095] | +0.031 [+0.011, +0.052] |
| wildjailbreak | +0.119 [+0.087, +0.151] | +0.041 [+0.017, +0.065] |

## Matched-FPR operating point (target 5%) — realized rates

| Guard | regime | macro TPR | macro FPR | pooled FPR |
|---|---|---:|---:|---:|
| base | represented | 0.134 | 0.017 | 0.022 |
| base | transfer | 0.526 | 0.083 | 0.044 |
| sft | represented | 0.766 | 0.010 | 0.020 |
| sft | transfer | 0.554 | 0.137 | 0.146 |
| calibrated_avg | represented | 0.618 | 0.018 | 0.030 |
| calibrated_avg | transfer | 0.648 | 0.120 | 0.095 |

## Single-permutation shuffle diagnostics (panel ens − base)

| Regime | real | label-alignment shuffle | within-class row-pairing shuffle |
|---|---:|---:|---:|
| represented | +0.314 | -0.071 | +0.329 |
| transfer | +0.024 | -0.155 | +0.038 |

*This is retrospective, precision-focused evidence, not a prospective confirmatory result. Clean v2 execution repairs provenance but does not erase prior exposure to part of the transfer cohort. WiSE-FT weight interpolation is out of scope for this output-space analyzer.*
