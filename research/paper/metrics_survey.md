# Evaluation-Metric Landscape vs. Our Protocol

Grounded strictly in the 15 extracted profiles. I separate **guard-classifier papers** (11) from **benchmark/eval-harness or fine-tuning studies** (4: `mazeika2024harmbench`, `xie2024sorrybench`, `cui2024orbench`, `hsiung2025guardrails`), because the latter use LLM-judge / ASR / refusal-rate conventions and would distort norm counts if pooled with classifier metrics.

## 1. Comparison Table

| Paper [key] | Primary metric(s) | Operating point | AUPRC / AUROC | CIs | Sig test | Calib | Latency/cost | Held-out/OOD |
|---|---|---|---|---|---|---|---|---|
| `inan2023llamaguard` | AUPRC (headline); P/R/F1; per-category | Threshold-free AUPRC; native 0.5 for P/R/F1 | AUPRC Y / AUROC N | N | N | N | N | Y |
| `fedorov2024llamaguard3_1b` | F1, FPR; per-language | Native/default | N / N | N | N | N | **Y** (tok/s, TTFT) | Y |
| `meta2024llamaguard2_3` | F1, FPR; VR/FRR (system) | Native/default | N / N | N | N | N | N | Y *(tables not read verbatim — flagged)* |
| `zeng2024shieldgemma` | **Optimal F1**, AU-PRC | **Tuned** (F1-max); AUPRC threshold-free | AUPRC Y / AUROC N | N | N | N | N | Y |
| `ghosh2024aegis` | AUPRC, F1, Acc, ASR | Native/default | AUPRC Y / AUROC N | N | N | N | N | Y |
| `padhi2024graniteguardian` | AUPRC, **AUC(AUROC)**, F1, R, P, **Recall/AUC @ fixed FPR (0.1/0.01/0.001)** | Native 0.5 + **fixed-FPR operating points** | **AUPRC Y / AUROC Y** | N | N | N | N | Y |
| `han2024wildguard` | binary F1, ASR, refusal rate | Native/default | N / N | N | N | N | N | Y |
| `kang2025r2guard` | AUPRC, UDR@0.5 | Mixed: threshold-free AUPRC + native 0.5 UDR | AUPRC Y / AUROC N | N | N | N | **Y** (runtime/instance) | Y |
| `deng2025duoguard` | F1 (per-lang + avg) | Default (unspecified) | N / N | N | N | N | **Y** (ms/input) | Y |
| `majhi2025gpuguard` | F1(pos), Acc, P, R, latency, cost | **Tuned** (F1-max validation sweep) | N / N | **Y** (bootstrap) | **Y** (McNemar) | **Y** (isotonic) | **Y** ($/1M) | Y (D1/D2/D3) |
| `markov2023holistic` | per-category AUPRC; F1 (QC only) | Threshold-free | AUPRC Y / AUROC N | ~ (error bars over iters/seeds, not paired CIs) | N | N | N | Y |
| `hsiung2025guardrails` | Harmfulness Score, utility | Native (Beaver-Dam-7B default) | N / N | N | N | N | N | Y (task-shift) |
| `mazeika2024harmbench` | ASR, classifier agreement | Threshold-free / native binary | N / N | N | N | N | N | Y |
| `xie2024sorrybench` | fulfillment rate, judge acc, Cohen's κ | Native binary | N / N | N | N | N | **Y** (time/pass) | Y |
| `cui2024orbench` | rejection/acceptance rate, judge acc, Spearman | Native (greedy) | N / N | N (±SD only) | N | N | N | **N** |

## 2. Field Norms

**Near-universal (report these or reviewers will notice their absence):**
- **F1** — reported by ~9–10 of 11 classifier papers (`inan`, `fedorov`, `meta`, `zeng`, `ghosh`, `padhi`, `han`, `deng`, `majhi`, `markov`-QC). The de-facto lingua franca.
- **Native/default operating point** — dominant. Most report F1 at the model's default decision (`fedorov`, `meta`, `ghosh`, `han`, `deng`, `inan`-for-P/R/F1, `padhi`-for-F1). Only 3 tune (`zeng`, `majhi`, `markov`-partial).
- **Held-out / OOD evaluation** — 10/11 classifier papers; only `cui2024orbench` lacks it (and it's a benchmark, not a classifier).

**Common but not universal:**
- **AUPRC** — 6/11 classifier papers (`inan`, `zeng`, `ghosh`, `padhi`, `kang`, `markov`), tracing to the `markov2023holistic` lineage. This is the field's *established threshold-free metric*.
- **Latency/cost** — 4/11 classifier papers (`fedorov`, `kang`, `deng`, `majhi`), but usually throughput/mean/TTFT — **not** batch=1 p50/p90.
- **Per-category / per-language breakdowns** — `inan`, `fedorov`, `meta`, `deng`, `markov`.
- **ASR / jailbreak-robustness metric** — `ghosh`, `han`, `kang`(UDR), plus the benchmark papers `mazeika`, — expected in the *robustness* sub-community.

**Rare (≈1 paper each):**
- **AUROC** — only `padhi2024graniteguardian` (1/11).
- **Confidence intervals** — only `majhi2025gpuguard` (bootstrap; not stated as paired).
- **Significance tests** — only `majhi2025gpuguard` (McNemar).
- **Calibration** — only `majhi2025gpuguard` (isotonic).
- **Matched/fixed-FPR operating point** — only `padhi2024graniteguardian` (Recall/AUC @ FPR 0.1/0.01/0.001).

## 3. Gaps We Exploit (genuine rigor delta)

Ranked by how absent they are in prior work:

1. **Paired-bootstrap 95% CIs on comparisons** — *absent everywhere except* `majhi2025gpuguard` (and even there, not described as paired). No flagship guard paper (`inan`, `zeng`, `ghosh`, `padhi`, `han`) reports CIs. This is our strongest differentiator.
2. **McNemar significance tests** — only `majhi2025gpuguard`. Everyone else asserts "outperforms by X%" with no test (e.g. `kang` "+30.2% on ToxicChat", `deng` "by over 10%", `zeng` "+10.8% AU-PRC" — all point estimates, no significance).
3. **Calibration** — only `majhi2025gpuguard` (isotonic). Absent in all flagship guards.
4. **Matched-FPR@0.10 operating point** — only `padhi2024graniteguardian` does anything in this family (fixed-FPR points). Our matched-FPR *across systems* for fair operating-point comparison is essentially novel to the guard literature.
5. **AUROC** — only `padhi`. Our AUPRC+AUROC pairing is rare.
6. **True batch=1 p50/p90 latency** — latency papers (`fedorov`, `kang`, `deng`, `majhi`) report throughput/mean/cost, not tail percentiles at batch=1.
7. **Family-level decontamination** — no profile mentions decontamination methodology; OOD is handled by dataset choice (`majhi`'s D2 source-partitioned split is the closest). Our explicit decontamination is stronger.

**Honesty check on our thesis.** The claim "most guard papers report F1 at native threshold **without threshold-free metrics**" is only *half* right and should be softened: **6/11 classifier papers DO report threshold-free AUPRC**, so threshold-free evaluation is an existing norm, not a gap. The defensible, evidence-backed version of our thesis is: *"guard papers report F1 at a native threshold **without CIs, significance tests, calibration, or matched-FPR operating points**, and comparisons are point-estimate-only."* The pure "native-threshold F1, no threshold-free metric at all" critique lands cleanly only against `fedorov`, `meta`, `han`, `deng`.

## 4. Risks / Alignment (also report these, or reviewers can't place us)

- **AUPRC must be the headline anchor, not just AUROC.** 6 papers use AUPRC and it is the field's comparability currency (Markov lineage). If we lead with AUROC (which only `padhi` reports), reviewers can't line us up against `inan`/`zeng`/`ghosh`/`padhi`/`kang`. Report **both**, AUPRC first.
- **Report "Optimal-F1" alongside calibrated-threshold F1.** `zeng2024shieldgemma` and `majhi2025gpuguard` quote *best-achievable* F1. A single calibrated-threshold F1 will look artificially low next to their tuned numbers; reporting Optimal-F1 too makes the comparison apples-to-apples and preempts the "you cherry-picked a bad threshold" critique.
- **Per-category (and per-language, if multilingual) breakdown.** Norm in `inan`, `fedorov`, `meta`, `deng`, `markov`. A single pooled number will read as coarse.
- **ASR / jailbreak-robustness metric — only if we test adversarial prompts.** `ghosh`, `han`, `kang`, `mazeika` expect it. If our benchmark includes jailbreaks/adversarial splits and we omit ASR/UDR, the robustness community discounts us. If we don't test adversarial prompts, state that scope limit explicitly. *(Flag: cannot tell from the prompt whether our benchmark has an adversarial split.)*
- **Latency comparability.** Since `fedorov`/`deng`/`majhi` report throughput/cost, add a mean-throughput or cost figure next to our p50/p90 so we're comparable, not just more rigorous.

## 5. Recommended Metric Set

To be **both rigorous and comparable**, report:

**A. Threshold-free (primary, fair) — comparability + rigor**
- **AUPRC** (headline anchor; matches `inan`, `zeng`, `ghosh`, `padhi`, `kang`, `markov`)
- **AUROC** (rigor add; matches only `padhi`)
- **Paired-bootstrap 95% CIs on both, and on all pairwise deltas** (our #1 differentiator; else only `majhi`)

**B. Operating points**
- **Matched-FPR@0.10** across all systems (our fairness contribution; nearest prior art `padhi`'s Recall@FPR=0.1)
- **Calibrated-threshold** F1 / precision / recall / FPR
- **Optimal-F1** (best-threshold) reported too — comparability to `zeng`, `majhi`

**C. Statistical rigor**
- **McNemar** between systems at the matched operating point (matches `majhi`)
- **Calibration** report/curve (matches `majhi`; rare elsewhere)

**D. Systems / efficiency**
- **True batch=1 latency p50/p90** + one throughput/cost figure for comparability (`fedorov`, `deng`, `majhi`)

**E. Breakdowns & splits**
- **Per-category** (and per-language if applicable) F1 + AUPRC (`inan`, `fedorov`, `deng`, `markov`)
- **In-distribution vs held-out** split with **family-level decontamination** (OOD is near-universal; decontamination is our add)
- **ASR/UDR** *only if* an adversarial split exists (`ghosh`, `han`, `kang`)

**One-line positioning for the paper:** *"We keep the field's AUPRC anchor and per-category F1 for comparability, and add what the literature lacks — AUROC, paired-bootstrap CIs, McNemar tests, calibration, and a matched-FPR@0.10 operating point — making ours the only guard evaluation besides `majhi2025gpuguard` to report CIs/significance/calibration, and the only one to combine those with threshold-free AUPRC+AUROC and matched-FPR fairness."*

**Uncertainties flagged:** `meta2024llamaguard2_3` metric detail is from secondary summaries (HTML 404 / binary PDF), not read verbatim. `markov2023holistic` shows error bars over sampling iterations/seeds but not paired CIs on head-to-head comparisons. Whether our own benchmark contains an adversarial/jailbreak split (which governs the ASR recommendation) is not determinable from the inputs.