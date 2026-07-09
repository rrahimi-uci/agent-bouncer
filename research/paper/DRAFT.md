# Agent-Bouncer: An Operating-Point-Fair Measurement Study of a Laptop-Trained SmolLM3-3B Safety Guard

> **Note:** `safety_guard_operating_point_artifacts.tex` (compiled to `safety_guard_operating_point_artifacts.pdf`) is the authoritative paper. This prose draft
> predates the grounding audit and its base-vs-tuned / calibration numbers (e.g. base 0.711,
> tuned 0.791, matched-FPR figures) were superseded by the clean per-model-calibrated recompute
> now reflected in `safety_guard_operating_point_artifacts.tex`: base **0.713** / tuned **0.794** (Δ+0.081 [0.062,0.100]), base
> in-house AUPRC **0.696**. See `README.md` "Provenance" for the producing scripts.

## Abstract

We report a measurement-and-controlled-study on a small, local LLM safety guard rather than a state-of-the-art chase. We fine-tune SmolLM3-3B into a single-token binary guard (one forward pass, softmax over the {safe, unsafe} logits) using LoRA (60.5M trainable parameters, 1.93% of 3.14B) on a consumer Apple M4 Max laptop in ~71 minutes. Our first contribution is an operating-point-fair protocol: guard rankings depend heavily on the decision threshold, so naive per-model-threshold comparisons mislead. We report threshold-free AUPRC/AUROC, a matched-FPR@0.10 operating point, paired-bootstrap 95% confidence intervals (CIs), and McNemar tests, and we document real evaluation bugs we found and fixed in our own first pipeline (calibration leakage, batch-amortized latency, an asymmetric per-model threshold advantage, and a GPT error-handling bias).

On pooled in-house AUPRC the guard reaches 0.844 [0.825, 0.866] versus ShieldGemma-2b 0.712 and Llama-Guard-3-1B 0.639 (baseline figures are point estimates without CIs). On four novel benchmarks the guard never trained on, its aggregate AUPRC is 0.781 [0.751, 0.811] versus Llama-Guard-3-1B 0.701 [0.673, 0.733], with non-overlapping CIs; ShieldGemma's novel-benchmark runs are pending. Against gpt-5.4-mini the guard is a statistical tie on F1 (ΔF1 +0.007, 95% CI [-0.008, 0.023], McNemar p=0.78) at roughly 4x lower batch=1 latency (124 ms vs ~512 ms per request). A base-vs-tuned decomposition shows the instruction-tuned base is already a competent classifier by F1 (0.711, above both open guards' zero-shot F1); LoRA (+0.080 [0.063, 0.096], McNemar p<1e-4) specializes in-distribution and, on the two in-house held-out benchmarks we can measure, does not improve generalization. Measuring the base on the same four novel benchmarks resolves the attribution: the *untuned* base outranks the tuned guard out-of-distribution (aggregate AUPRC 0.886 [0.870, 0.900] vs 0.781, non-overlapping CIs) at comparable best-threshold F1 (0.792 vs 0.794). The guard's novel-set lead over Llama-Guard is therefore inherited base competence, not LoRA; fine-tuning measurably degrades OOD score ranking (operating-point robustness) while leaving peak achievable F1 unchanged.

---

## 1. Introduction

Large language models are now deployed as chat assistants and autonomous agents, where a single unsafe completion, a successful jailbreak, or an over-eager refusal all carry real cost. The dominant mitigation is a *safety guard*: a classifier that screens prompts (and responses) as `safe` or `unsafe` before they reach the model or the outside world. In practice, teams face an unattractive choice. Purpose-built open guards (Llama Guard 1/2/3 and Prompt Guard, ShieldGemma, Aegis, Granite Guardian, WildGuard) are convenient but are often reported on their own favorable operating points; proprietary frontier classifiers are strong but add per-request cost, latency, and a data-sharing dependency. Meanwhile, published guard comparisons frequently tune a threshold for the proposed system while holding baselines at fixed points, report latency as batch-amortized throughput, and decontaminate with exact string matching — choices that quietly inflate the reported advantage.

This paper is a **measurement and controlled-study** contribution, not a state-of-the-art chase. We fine-tune SmolLM3-3B into a binary guard that emits a single-token verdict — a softmax over the `{safe, unsafe}` token logits at the last position, i.e. one forward pass — with temperature and decision threshold calibrated on an *in-distribution* dev split only (in-house held-out and novel benchmarks contribute zero rows to calibration). Training is parameter-efficient LoRA (r=32, all seven projections; 60.5M trainable params, 1.93% of 3.14B) run in ~71 minutes / 300 steps on a single consumer Apple M4 Max laptop (36 GB unified memory, MPS backend, no CUDA). Everything — training, calibration, and evaluation — fits on a laptop, so the protocol is cheap to reproduce and audit.

Our emphasis is on *how* we measure, and we are careful with the word "held-out," which we split into two distinct pools throughout the paper:

- **In-house held-out** — two of our six in-house benchmarks (JailbreakBench, XSTest) are held out from training and calibration but drawn from the same overall pool.
- **Novel held-out (OOD)** — four benchmarks from source families the guard never saw during training (WildGuardTest, WildJailbreak, OR-Bench-Hard, HarmBench).

We evaluate on 6 in-house benchmarks (2,018 balanced, leakage-filtered test rows across guardrail, red-team, and over-refusal axes) and on the 4 novel held-out benchmarks (2,020 balanced rows across three sets, plus 200 all-harmful HarmBench rows). We compare against gpt-5.4-mini, Llama-Guard-3-1B, ShieldGemma-2b, and a keyword matcher, using paired-bootstrap 95% CIs, McNemar tests, threshold-free AUPRC/AUROC, and a matched-FPR@0.10 operating point, with latency measured at true batch=1. Building this pipeline surfaced real bugs in our own first evaluation — a calibration leak, batch-amortized latency, an asymmetric threshold advantage for our own model, and an API-error default that biased the GPT baseline's FPR — which we document and fix.

The headline findings are deliberately unembellished. On threshold-free AUPRC the guard leads the open guards on the in-house pool: pooled 0.844 [0.825, 0.866] vs ShieldGemma-2b 0.712 and Llama-Guard-3-1B 0.639 (baseline point estimates). On the four novel held-out benchmarks the separation from Llama-Guard-3-1B is decisive — the aggregate CIs do not overlap (0.781 [0.751, 0.811] vs 0.701 [0.673, 0.733]); ShieldGemma's novel runs are pending, so the novel-set win is claimed *only against Llama-Guard*. Against gpt-5.4-mini the guard is a statistical **tie** on F1 (paired ΔF1 +0.007, 95% CI [-0.008, 0.023], McNemar p=0.78), not a beat — but an efficiency win in deployment (local, ~4x lower batch=1 latency, 124 ms vs ~512 ms, no API cost). A base-vs-tuned decomposition shows the instruction-tuned 3B base is *already* a competent safety classifier zero-shot (F1 0.711, above zero-shot Llama-Guard-3-1B 0.673 and ShieldGemma-2b 0.424; see the operating-point caveat below), and that LoRA (+0.080 [0.063, 0.096], McNemar p<1e-4) specializes almost entirely in-distribution while staying flat-to-slightly-negative on the two in-house held-out benchmarks we can measure.

| Metric (in-house pooled unless noted) | Guard | Llama-Guard-3-1B | ShieldGemma-2b | gpt-5.4-mini |
|---|---|---|---|---|
| AUPRC | 0.844 [0.825, 0.866] | 0.639 | 0.712 | — |
| AUPRC, 4 novel held-out (aggregate) | 0.781 [0.751, 0.811] | 0.701 [0.673, 0.733] | pending | — |
| Matched-FPR@0.10 F1 | 0.581 | 0.360 | 0.464 | — |
| F1 (paired vs guard) | tie (ΔF1 +0.007, p=0.78) | — | — | native point 0.784 |
| Latency, batch=1 p50 | 124 ms | not measured | not measured | ~512 ms (remote API) |

Only the guard's pooled/held-out AUPRC, the guard-vs-Llama novel aggregate, the base-vs-tuned delta, and the GPT ΔF1 carry CIs; baseline AUPRCs are point estimates (see §5, §6).

We are explicit about limits. Matched-FPR@0.10 makes every system conservative (guard recall 0.431 there), so AUPRC is our fair primary summary; at the *deployed* operating point (T=2.10, τ=0.59) the guard's overall FPR is 0.306, i.e. it over-blocks benign prompts, which we treat as a real deployment concern. ShieldGemma is run somewhat outside its designed content-policy regime and its novel-benchmark numbers are still pending; a single global threshold is structurally in-distribution-optimized. Positioned against the two closest works, we complement CPU-class efficient-guard studies ("Do You Really Need a GPU to Guard Your LLM?") by adding rigorous operating-point-fair comparison and a base-vs-tuned decomposition on a 3B generative guard, and we serve as the constructive counterpart to "Why LLM Safety Guardrails Collapse After Fine-tuning" (ICML 2025): where they show tuning can destroy alignment, we show that even when tuning *works*, its measurable gains are memorized in-distribution and do not transfer to the in-house held-out sets we can test.

### 1.1 Contributions

- **Operating-point-fair evaluation.** We show guard rankings depend heavily on the decision threshold, so naive per-model-threshold comparisons are misleading; we instead report matched-FPR@0.10, threshold-free AUPRC/AUROC, and paired-bootstrap CIs with McNemar tests.
- **A base-vs-tuned decomposition.** An instruction-tuned 3B base already carries most of the safety-classification competence measurable on our in-house benchmarks (its zero-shot F1 beats both open guards' zero-shot F1); parameter-efficient tuning specializes in-distribution and does *not* improve the two in-house held-out benchmarks we can measure, and slightly regresses over-refusal.
- **A small, local, laptop-trained guard.** On threshold-free AUPRC it surpasses purpose-built open guards — Llama-Guard-3-1B decisively (non-overlapping CIs), *including* on the novel held-out benchmarks, and ShieldGemma-2b on the in-house pooled / in-dist / in-house-held-out sets (ShieldGemma's novel runs are pending) — and it matches frontier gpt-5.4-mini on F1 at ~4x lower latency.
- **A rigorous, reproducible, laptop-scale protocol.** Paired CIs, McNemar, matched-FPR, and family-level decontamination — a pipeline that exposed real evaluation bugs in our own first attempt — released as an open notebook.

---

## 2. Related Work

**LLM safety guards and content moderation.** A growing family of purpose-built classifiers screens prompts and responses for LLM and agent pipelines. Meta's Llama Guard line (1/2/3) [inan2023llamaguard, meta2024llamaguard2_3, fedorov2024llamaguard3_1b] and Prompt Guard [meta2024promptguard], Google's ShieldGemma [zeng2024shieldgemma], NVIDIA's Aegis [ghosh2024aegis], IBM's Granite Guardian [padhi2024graniteguardian], and AI2's WildGuard [han2024wildguard] all frame moderation as a supervised classification task over a safety taxonomy, typically emitting a verdict token or category label. Recent work pushes on the *reasoning* axis (R²-Guard's knowledge-enhanced logical-reasoning inference [kang2025r2guard] at ICLR 2025, DuoGuard's two-player RL/multilingual training [deng2025duoguard], and other reasoning-guards). Our guard is deliberately minimal by comparison: a single-token verdict read from the {safe, unsafe} logits in one forward pass, LoRA-tuned (60.5M trainable params, 1.93% of 3.14B) on a consumer laptop. We compare directly against two open guards under a fair protocol; on threshold-free AUPRC our guard leads both on the in-house pool (baseline values are point estimates without CIs):

| System | AUPRC (in-house pooled) | AUPRC (in-dist) | AUPRC (in-house held-out) |
|---|---|---|---|
| Guard (ours) | 0.844 [0.825, 0.866] | 0.846 | 0.860 [0.803, 0.909] |
| ShieldGemma-2b | 0.712 | 0.702 | 0.785 |
| Llama-Guard-3-1B | 0.639 | 0.610 | 0.762 |

**Benchmarks.** We evaluate on established suites spanning guardrail, red-team, and over-refusal axes: BeaverTails [ji2023beavertails], ToxicChat [lin2023toxicchat], JailbreakBench [chao2024jailbreakbench], XSTest [rottger2024xstest], OR-Bench [cui2024orbench], HarmBench [mazeika2024harmbench], and prompt-injection/jailbreak-classification sets, plus OpenAI's moderation evaluation set [markov2023holistic]. To probe generalization rather than memorization, we additionally reserve four novel benchmarks the guard never trained on — WildGuardTest [han2024wildguard], WildJailbreak [jiang2024wildteaming], OR-Bench-Hard [cui2024orbench], and HarmBench [mazeika2024harmbench] — held out at the source-family level, and report per-benchmark and aggregate results with paired-bootstrap 95% CIs and McNemar tests. On these four novel held-out benchmarks the guard's aggregate AUPRC is 0.781 [0.751, 0.811] versus Llama-Guard-3-1B 0.701 [0.673, 0.733], the only cross-system AUPRC comparison in which both sides carry CIs and they do not overlap.

**Measurement rigor.** A recurring hazard in guard evaluation is that rankings are sensitive to the decision threshold, and that a model tuned on its own dev split is compared against baselines evaluated at fixed operating points. We treat this as a first-class methodological concern rather than an implementation detail: we report threshold-free AUPRC/AUROC, matched-FPR@0.10 operating points, true batch=1 latency (p50 124 ms / p90 188 ms), paired CIs, and family-level holdout. This positions our work as a *measurement and controlled study* rather than a leaderboard entry.

### 2.1 Differentiation from the two closest works

**"Do You Really Need a GPU to Guard Your LLM?" [majhi2025gpuguard] (CPU-class classifiers / efficient pipelines).** That line argues small, cheap classifiers can serve as practical guards. We share the efficiency motivation — our guard is laptop-trained (~71 min / 300 steps, bf16, Apple M4 Max, MPS, no CUDA) and runs locally at ~4x lower latency than gpt-5.4-mini (124 ms vs ~512 ms per request; see the latency caveat in §5.3). Our distinct contribution is not a smaller classifier but a *rigorous fair comparison* on a 3B generative guard: operating-point-fair evaluation (matched-FPR + threshold-free AUPRC + paired CIs) and a base-vs-tuned decomposition that prior efficient-guard work does not report.

**"Why LLM Safety Guardrails Collapse After Fine-tuning" [hsiung2025guardrails] (ICML 2025 DIG-BUGS).** That work shows fine-tuning can *destroy* a model's alignment. We are the constructive complement. Even when fine-tuning "works," our base-vs-tuned ablation (identical paired rows; base = zero-shot SmolLM3-3B with the same logprob head) shows the measurable gains are almost entirely in-distribution and do not improve the two in-house held-out benchmarks we can test:

| Benchmark | Split | Base F1 | Tuned F1 | Delta |
|---|---|---|---|---|
| jailbreak_classification | in-dist | 0.328 | 0.966 | +0.639 |
| toxicchat | in-dist | 0.765 | 0.916 | +0.151 |
| prompt_injections | in-dist | 0.690 | 0.814 | +0.124 |
| beavertails | in-dist | 0.688 | 0.716 | +0.028 |
| jailbreakbench | in-house held-out | 0.809 | 0.803 | -0.006 |
| xstest | in-house held-out | 0.862 | 0.828 | -0.034 |
| **Aggregate (paired rows)** | | **0.711** | **0.791** | **+0.080 [0.063, 0.096]** |

The zero-shot base (F1 0.711) exceeds zero-shot Llama-Guard-3-1B (0.673) and ShieldGemma-2b (0.424) *by F1*, suggesting an instruction-tuned 3B base already carries much of the safety-classification competence; we did not compute base AUPRC, so by our own threshold-free standard this ranking is suggestive rather than decisive (§6.1). Where the ICML work demonstrates that tuning can break alignment, we show that tuning which appears successful still only measurably specializes in-distribution, and we decompose how much competence is base versus tuned. We have since run the base on the four novel benchmarks (§6.1): the base *outranks* the tuned guard out-of-distribution (aggregate AUPRC 0.886 vs 0.781, disjoint CIs) at tied best-threshold F1, so the guard's novel-set lead reflects inherited base competence, and tuning degrades OOD score ranking rather than improving generalization.

### 2.2 What is novel here

We are explicitly a *measurement and controlled study*, not a new architecture or training objective. Two findings are, to our knowledge, genuinely new; the other two are engineering/measurement contributions.

- **(Novel) Operating-point-fair guard comparison.** Threshold-free AUPRC is *already* an established guard metric (originating with OpenAI's moderation work [markov2023holistic] and reported by Llama Guard [inan2023llamaguard], ShieldGemma [zeng2024shieldgemma], Aegis [ghosh2024aegis], Granite Guardian [padhi2024graniteguardian], and R²-Guard [kang2025r2guard]); we therefore do **not** claim AUPRC as our innovation. What is missing in prior work is (i) *head-to-head* F1 typically reported at each model's *native* threshold as point estimates without significance tests, and (ii) no combination of a matched operating point with paired uncertainty. We show native-threshold F1 is not merely imprecise but rank-*inverting*: ShieldGemma-2b [zeng2024shieldgemma] appears worse than Llama-Guard-3-1B [fedorov2024llamaguard3_1b] at native thresholds yet better under AUPRC (§6.2). Our contribution is the *combination* the field lacks — matched-FPR@0.10 across systems (the nearest prior art being Granite Guardian's fixed-FPR points [padhi2024graniteguardian]), AUPRC **and** AUROC, paired-bootstrap CIs and McNemar tests on a shared calibration set [guo2017calibration], and calibration reporting — a rigor set otherwise present only piecemeal (CIs/McNemar/calibration in the CPU-classifier study [majhi2025gpuguard]; fixed-FPR in Granite Guardian [padhi2024graniteguardian]).
- **(Novel) Base-vs-tuned decomposition.** We separate what an instruction-tuned base [bakouch2025smollm3, ouyang2022instructgpt] already knows zero-shot from what LoRA [hu2021lora] adds, and find that tuning specializes in-distribution without lifting the in-house held-out F1. This is the constructive complement to [hsiung2025guardrails] (tuning can *collapse* alignment): we isolate the guard-classification case and characterize it as in-distribution *specialization* rather than collapse.
- **(Engineering) A local, laptop-trained 3B guard** that surpasses the open guards on threshold-free AUPRC — decisively over Llama-Guard-3-1B [fedorov2024llamaguard3_1b], including on four novel held-out benchmarks — and ties gpt-5.4-mini on F1 at ~4× lower latency. The nearest efficiency-focused work [majhi2025gpuguard] reaches CPU-class cost with TF-IDF classifiers matched in-distribution; we instead evaluate a 3B LLM guard for *out-of-distribution* generalization under a threshold-free metric.
- **(Engineering) A reproducible laptop-scale protocol** — family-level decontamination and paired CIs across the benchmark suite [mazeika2024harmbench, han2024wildguard, jiang2024wildteaming, xie2024sorrybench, rottger2024xstest, cui2024orbench, chao2024jailbreakbench, lin2023toxicchat, ji2023beavertails, markov2023holistic] — released as an open notebook. We scope claims to detection-based content classification and note that adversarial evasion remains open [hackett2025bypassing].

---

## 3. Method

### 3.1 Guard formulation: a single-token logprob head

The guard treats safety classification as a single next-token prediction. A prompt to be screened is wrapped in a fixed instruction template and the model is asked to emit a one-word verdict, `safe` or `unsafe`. Rather than sampling text, we read the logits at the final position and restrict attention to the two verdict tokens. Let $z_{\text{safe}}$ and $z_{\text{unsafe}}$ be those logits; the guard score is a temperature-scaled two-way softmax

$$
P(\text{unsafe}) = \frac{\exp(z_{\text{unsafe}}/T)}{\exp(z_{\text{safe}}/T) + \exp(z_{\text{unsafe}}/T)}.
$$

A single forward pass yields the score — no autoregressive decoding, no multi-token or chain-of-thought generation. This keeps inference cheap and makes the guard a proper threshold-free scorer, so it can be summarized by AUPRC/AUROC in addition to a hard verdict. A prompt is flagged `unsafe` when $P(\text{unsafe}) \ge \tau$ for a fixed decision threshold $\tau$.

### 3.2 Calibration

The temperature $T$ (temperature scaling [guo2017calibration]) and threshold $\tau$ are the only post-training free parameters, and both are fit exclusively on an **in-distribution dev split** drawn from the four in-distribution benchmarks (BeaverTails, ToxicChat, Prompt Injections, Jailbreak Classification). The two in-house held-out benchmarks (JailbreakBench, XSTest) and the four novel benchmarks contribute zero rows to calibration; this is a deliberate correction of an earlier pipeline in which held-out dev rows had leaked into temperature/threshold fitting. The clean, leak-free calibration yields:

| Parameter | Value |
|---|---|
| Temperature $T$ | 2.10 |
| Decision threshold $\tau$ | 0.59 |

Because $\tau$ is tuned on in-distribution dev data, a single global threshold is structurally in-distribution-optimized; we treat this as a limitation and, for cross-model comparison, additionally report matched-FPR and threshold-free operating points rather than relying on the tuned point alone.

### 3.3 LoRA training recipe

We fine-tune SmolLM3-3B [bakouch2025smollm3] with LoRA adapters [hu2021lora] and a completion-only objective: the loss is computed on the verdict token only, so the model is trained to place probability mass on `safe`/`unsafe` at the decision position rather than to reproduce the prompt. Adapters are applied to all seven projection matrices (q, k, v, o, gate, up, down).

| Setting | Value |
|---|---|
| Base model | SmolLM3-3B (instruction-tuned) |
| Adapter | LoRA, rank $r=32$, $\alpha=64$, dropout $0.05$ |
| Target modules | q, k, v, o, gate, up, down (all 7 projections) |
| Loss | Completion-only, on the verdict token |
| Trainable params | 60.5M (1.93% of 3.14B) |
| Precision | bf16 |
| Optimizer steps | 300 |
| Wall-clock | ~71 min |
| Seed | 42 |

Training ran entirely on a consumer Apple M4 Max laptop (36 GB unified memory, MPS backend, no CUDA), demonstrating that the recipe is reproducible at laptop scale. Because ToxicChat is non-commercial, it is used for evaluation only and is excluded from any released model's training data.

---

## 4. Experimental Setup

### 4.1 Task and label mapping

The guard is a binary classifier that emits a single-token verdict, `safe` or `unsafe`. We read the logits at the last position, take a softmax over the `{safe, unsafe}` token pair, and treat the resulting P(unsafe) as the decision score. A prompt is labeled `unsafe` if it should be blocked (harmful content, successful jailbreak/injection, policy-violating request) and `safe` otherwise; the over-refusal axis contributes benign-but-superficially-risky prompts whose gold label is `safe`. All evaluations reduce to this one score per row, obtained in a single forward pass.

### 4.2 Benchmarks and terminology

We use three explicitly distinguished pools, and we reserve the word "held-out" for two of them:

- **In-distribution (in-dist):** BeaverTails, ToxicChat, Prompt Injections, Jailbreak Classification. Contribute dev rows to calibration plus balanced test rows.
- **In-house held-out:** JailbreakBench, XSTest. Same in-house pool, but test-only — zero dev/calibration rows.
- **Novel held-out (OOD):** WildGuardTest, WildJailbreak, OR-Bench-Hard, HarmBench. Held out at the source-family level; the guard never trained on them.

All splits are class-balanced (matched-n). HarmBench is an all-harmful stress set (no negatives), so it is scored by mean P(unsafe) rather than F1/AUPRC and is **excluded** from any aggregate AUPRC.

| Benchmark | Axis | Pool | Role | Notes |
|---|---|---|---|---|
| BeaverTails | guardrail | in-house | in-dist (dev/test) | |
| ToxicChat | guardrail | in-house | in-dist (dev/test) | non-commercial license → eval-only for any released model |
| Prompt Injections | red_team | in-house | in-dist (dev/test) | |
| Jailbreak Classification | red_team | in-house | in-dist (dev/test) | |
| JailbreakBench | red_team | in-house | in-house held-out (test) | test-only, no dev rows |
| XSTest | over_refusal | in-house | in-house held-out (test) | benign-but-risky; gold label `safe`; test-only, no dev rows |
| WildGuardTest (prompt) | — | novel | novel held-out (test) | never trained on |
| WildJailbreak (eval) | — | novel | novel held-out (test) | never trained on |
| OR-Bench-Hard | — | novel | novel held-out (test) | never trained on |
| HarmBench | — | novel | novel held-out (test) | 200 rows, all-harmful; mean P(unsafe) only; excluded from aggregate AUPRC |

The in-house pool contributes 2,018 balanced test rows. The three balanced novel sets contribute 2,020 balanced rows; HarmBench adds 200 all-harmful rows scored separately.

### 4.3 Baselines

We compare against four systems, each run in its own native regime:

- **gpt-5.4-mini** — zero-shot classifier prompt at a fixed native decision point (no tunable threshold). On API errors we abstain rather than defaulting to a label (a corrected bug; see §4.6).
- **Llama-Guard-3-1B** — its native chat template, reading the first-token verdict.
- **ShieldGemma-2b** — its guideline prompt, scored by P(Yes). ShieldGemma is a content-policy / general-guideline guard and is used somewhat outside its designed regime here; it is not an injection detector.
- **Keyword matcher** — a simple substring/keyword baseline.

For a matched comparison we also evaluate an untuned **base** SmolLM3-3B: zero-shot, with the identical single-token logprob head and no training, on the identical paired rows as the tuned guard. We flag as a caveat that we did not separately calibrate a base-specific temperature/threshold; the base-vs-tuned comparison is therefore reported primarily as a paired F1 delta with McNemar on matched rows (robust to a shared decision rule), and the exact base operating-point provenance is an open item (§7).

### 4.4 Metrics and which comparisons carry CIs

We report F1 and false-positive rate (FPR) at the operating threshold, and the threshold-free summaries AUPRC and AUROC. Because only our guard receives a dev-tuned threshold while baselines are fixed operating points, we additionally compare all tunable systems at **matched-FPR@0.10** (threshold chosen so FPR = 0.10 on the in-distribution dev split) and rely on AUPRC as the fair threshold-free summary. Uncertainty is reported as paired-bootstrap 95% CIs; pairwise agreement/disagreement between systems is tested with McNemar. Latency is measured at true batch=1 (p50/p90), not batch-amortized.

We are explicit about statistical support: **only** the guard's pooled/in-house-held-out AUPRC, the guard-vs-Llama-Guard novel aggregate AUPRC, the base-vs-tuned F1 delta, and the guard-vs-GPT ΔF1 carry CIs. Baseline AUPRCs (ShieldGemma, Llama-Guard on the in-house pool) are point estimates. We therefore reserve the word "decisive" for the one comparison in which both sides carry CIs and they do not overlap (novel-set guard vs Llama-Guard); elsewhere we say "leads on point estimates."

We report several distinct F1 aggregations and define each to prevent conflation:

- **Pooled micro-F1 (0.794):** computed over all 2,018 in-house test rows pooled.
- **Macro-F1, in-dist (0.860):** unweighted mean of per-benchmark F1 over the four in-distribution benchmarks.
- **Macro-F1, in-house held-out (0.827):** unweighted mean of per-benchmark F1 over JailbreakBench and XSTest.
- **Tuned aggregate F1 (0.791):** F1 on the paired ablation row set shared between base and tuned models (§6.1).

### 4.5 Decontamination

Training data is leakage-filtered against the test splits. Our initial pipeline used exact-string-match decontamination. We recommend, and adopt as protocol, source-family holdout plus near-duplicate detection; source-family holdout is what defines and separates the four novel benchmarks, so the guard is evaluated on data from families it never saw during training. We describe near-duplicate detection as the recommended companion to family holdout rather than claiming an exhaustively executed near-duplicate sweep of the release pipeline.

### 4.6 Evaluation bugs found and fixed

Building the pipeline surfaced four real bugs in our own first evaluation, each of which we corrected before reporting: (1) a **calibration leak** in which in-house held-out dev rows had leaked into temperature/threshold fitting — fixed to in-dist-only calibration (T=2.10, τ=0.59); (2) **batch-amortized latency** (throughput/16) — replaced with true batch=1 p50/p90; (3) an **asymmetric threshold advantage** (only our model got a dev-tuned threshold while baselines were fixed points) — addressed via matched-FPR + threshold-free AUPRC; and (4) a **GPT error-handling bias** in which API errors defaulted to `unsafe` (inflating the GPT FPR) — fixed to abstain.

### 4.7 Hardware

Training and evaluation run on a single consumer Apple M4 Max laptop (36 GB unified memory, MPS backend, no CUDA). Fine-tuning uses LoRA (r=32, α=64, dropout 0.05, all seven projections) with completion-only loss on the verdict token: 60.5M trainable parameters (1.93% of 3.14B), bf16, 300 optimizer steps in ~71 minutes, seed 42. Guard inference latency at batch=1 is p50 124 ms / p90 188 ms.

---

## 5. Results

We report all systems at threshold-free operating points (AUPRC/AUROC) and at a common, conservative operating point (matched-FPR@0.10), with paired-bootstrap 95% CIs where available. Latency is measured at true batch size 1.

### 5.1 Main comparison (in-house pool)

Threshold-free AUPRC here is pooled over the **in-house test rows only** (2,018 rows) — not over all evaluation rows, since ShieldGemma's novel runs are pending and the novel sets are reported as a separate pool in §5.2. Matched-FPR@0.10 fixes every tunable system to the same false-positive rate using a threshold set on the in-distribution dev split. gpt-5.4-mini is reported at its fixed native decision point (no tunable threshold), so it is not directly comparable at matched FPR.

| System | AUPRC, in-house pooled [95% CI] | Recall @FPR 0.10 | F1 @FPR 0.10 | FPR @0.10 | Latency (batch=1) |
|---|---|---|---|---|---|
| Guard (SmolLM3-3B, ours) | **0.844** [0.825, 0.866] | 0.431 | 0.581 | 0.051 | p50 124 ms / p90 188 ms |
| ShieldGemma-2b | 0.712 | 0.341 | 0.464 | — | not measured |
| Llama-Guard-3-1B | 0.639 | 0.242 | 0.360 | — | not measured |
| gpt-5.4-mini (fixed native point) | — | 0.856 | 0.784 | 0.321 | ~512 ms (remote API) |
| keyword matcher | — | 0.096 | 0.168 | — | not measured |

On pooled in-house AUPRC the guard (0.844) leads ShieldGemma-2b (0.712) and Llama-Guard-3-1B (0.639) on point estimates (the baselines carry no CIs, so we do not assert statistical significance here). At matched FPR@0.10 all tunable systems are conservative (the guard operates at recall 0.431, FPR 0.051); the guard still leads the other tunable guards on both recall and F1 on point estimates. The guard's in-dist AUPRC is 0.846 and its in-house held-out AUPRC is 0.860 [0.803, 0.909]. At each system's best threshold (Optimal-F1), the guard reaches 0.794, ShieldGemma-2b 0.729, and Llama-Guard-3-1B 0.670 — the same guard > ShieldGemma > Llama-Guard order, which (like AUPRC) inverts the native-threshold F1 ordering of the two open guards (§6.2).

At the **deployed** operating point (T=2.10, τ=0.59), the guard's per-benchmark F1 is BeaverTails 0.716, ToxicChat 0.915, Prompt Injections 0.828, Jailbreak Classification 0.980, JailbreakBench 0.821, XSTest 0.833. This yields macro-F1 0.860 (in-dist) / 0.827 (in-house held-out) and pooled micro-F1 0.794. The deployed overall FPR is **0.306** — i.e. at τ=0.59 roughly 31% of benign in-house prompts are flagged. This is high and is a genuine deployment concern that would in practice motivate a higher, application-specific threshold; we return to it in §7. (Note: the macro-F1 in-dist value 0.860 coincides numerically with the *in-house held-out AUPRC* 0.860 but is a different metric on a different split.)

### 5.2 Novel held-out benchmarks (guard never trained on them)

On the three balanced benchmarks the guard never saw during training, we compare threshold-free AUPRC across the tuned guard, the *zero-shot base*, and Llama-Guard-3-1B (both guard-vs-Llama and base-vs-guard aggregate CIs are non-overlapping).

| Benchmark | Base (zero-shot) AUPRC | Guard (tuned) AUPRC | Llama-Guard-3-1B AUPRC |
|---|---|---|---|
| wildguardtest | 0.894 | 0.798 | 0.751 |
| wildjailbreak | 0.837 | 0.722 | 0.597 |
| orbench_hard | 0.940 | 0.845 | 0.750 |
| **Aggregate (2,020 rows)** | **0.886** [0.870, 0.900] | **0.781** [0.751, 0.811] | **0.701** [0.673, 0.733] |
| Aggregate Optimal-F1 | 0.792 | 0.794 | 0.691 |

The tuned guard's aggregate AUPRC (0.781 [0.751, 0.811]) exceeds Llama-Guard-3-1B's (0.701 [0.673, 0.733]) with non-overlapping CIs — a decisive cross-system result. But the *untuned base* ranks higher still (0.886 [0.870, 0.900], disjoint from the tuned guard), while best-threshold F1 is essentially tied across base and guard (0.792 vs 0.794, both above Llama-Guard's 0.691). Read together: out-of-distribution the base is the strongest *ranker* (most operating-point-robust), base and tuned guard reach the same *peak* F1, and both beat Llama-Guard. The guard's novel-set lead over Llama-Guard is thus inherited from the base, not produced by LoRA; fine-tuning degrades OOD score ranking (AUPRC) without changing peak achievable F1 (§6.1).

The aggregate 95% CIs ([0.751, 0.811] vs [0.673, 0.733]) do not overlap, so this is our one decisive cross-system AUPRC result. The aggregate is over the three balanced novel sets (2,020 rows); **HarmBench is excluded** from it. On HarmBench (200 all-harmful prompts), mean predicted P(unsafe) is 0.967 for the zero-shot base, 0.780 for the tuned guard, and 0.390 for Llama-Guard-3-1B; because this compares raw probabilities across differently temperature-scaled models it is *suggestive, not decisive* and is not a threshold-free ranking metric (though it echoes the AUPRC ordering base > guard > Llama-Guard). ShieldGemma-2b on the novel benchmarks is **pending** (Gemma-2 stalls on the throttled MPS laptop); on the in-house held-out set its AUPRC is 0.785 versus the guard's 0.860. We therefore make **no** ShieldGemma novel-set claim.

### 5.3 Frontier comparison: statistical tie and Pareto-style position

Against gpt-5.4-mini on paired rows, the guard shows no statistically significant F1 difference.

| Comparison (guard vs gpt-5.4-mini) | Value |
|---|---|
| ΔF1 (guard − gpt) | +0.007 |
| 95% CI | [−0.008, 0.023] |
| McNemar p | 0.78 |
| Guard latency (batch=1) | 124 ms (local M4 Max) |
| gpt-5.4-mini latency | ~512 ms (remote API) |

The F1 difference is not significant (McNemar p=0.78; CI spans zero, its lower bound admitting the guard being 0.008 *worse*), so this is a statistical **tie** with the guard nominally ahead, not an accuracy beat. Two caveats. First, this comparison places the guard at its in-dist-tuned threshold (τ=0.59) against gpt at its fixed native point — the same per-model-threshold asymmetry we criticize in §6.2; because gpt exposes no tunable threshold, a matched-FPR comparison is not possible, so the tie is threshold-advantaged in the guard's favor and should be read as such. Second, latency is cross-substrate: 124 ms is local M4 Max inference while ~512 ms is a remote API round-trip (network plus server-side batching), so "~4x lower latency" is deployment-realistic but *not* a controlled hardware comparison, and it is a **guard-vs-gpt claim only** — Llama-Guard and ShieldGemma latency were not measured. With those caveats, the guard achieves parity accuracy locally at roughly 4x lower batch=1 latency with no API cost: an efficiency win at parity accuracy rather than an accuracy win.

---

## 6. Analysis

### 6.1 Base-vs-tuned decomposition: PEFT specializes in-distribution and degrades OOD ranking

We isolate the contribution of parameter-efficient fine-tuning by evaluating the base SmolLM3-3B model zero-shot with the identical single-token logprob head (no training), on the identical paired rows.

Two findings stand out. First, the *untuned* base is already a competent safety classifier by F1: its zero-shot F1 of 0.711 exceeds the zero-shot F1 of both purpose-built open guards — Llama-Guard-3-1B (0.673) and ShieldGemma-2b (0.424). We flag an important methodological caveat, consistent with our own Contribution #1: this is an **F1-at-operating-point** comparison, and we did not compute base AUPRC. By our own threshold-free standard, an F1-at-operating-point ranking is exactly the kind of comparison we caution against; since ShieldGemma's AUPRC is 0.712, a base AUPRC in that range could change the ranking. We therefore keep the *in-house* claim cautious — the base is already competent *by F1*, and we did not compute in-house base AUPRC. On the four *novel* benchmarks, however, we did compute base AUPRC, and there the base beats the open guard we could measure under the threshold-free standard (base 0.886 vs Llama-Guard-3-1B 0.701; §5.2), so the threshold-free "base is a strong guard" claim holds where we could test it out-of-distribution.

Second, LoRA adds a real but *localized* gain: aggregate F1 on the paired rows rises from 0.711 to 0.791 (delta +0.080, 95% CI [0.063, 0.096], McNemar p < 1e-4). Decomposing this delta per benchmark shows it is concentrated almost entirely in the in-distribution axes, and is flat or slightly negative on the two in-house held-out benchmarks.

| Benchmark | Split | Base F1 | Tuned F1 | Delta |
|---|---|---|---|---|
| jailbreak_classification | in-dist | 0.328 | 0.966 | +0.639 |
| toxicchat | in-dist | 0.765 | 0.916 | +0.151 |
| prompt_injections | in-dist | 0.690 | 0.814 | +0.124 |
| beavertails | in-dist | 0.688 | 0.716 | +0.028 |
| jailbreakbench | in-house held-out | 0.809 | 0.803 | -0.006 |
| xstest | in-house held-out | 0.862 | 0.828 | -0.034 |
| **Aggregate (paired rows)** | | **0.711** | **0.791** | **+0.080 [0.063, 0.096]** |

The pattern is unambiguous within its scope: every large positive delta is in-distribution (the extreme case, jailbreak_classification, +0.639, is a task the base essentially could not do), while both in-house held-out benchmarks move slightly *down*, including a -0.034 regression on the over-refusal probe (XSTest). LoRA specializes the guard to the training distribution's label conventions and formatting; on the two in-house held-out benchmarks it does not improve — and can mildly degrade — generalization, including over-refusal behavior.

We have now measured the base on the four novel benchmarks, which **confirms and sharpens** this claim. Out-of-distribution, the *untuned* base ranks higher than the tuned guard (aggregate AUPRC 0.886 [0.870, 0.900] vs 0.781 [0.751, 0.811], non-overlapping CIs), while both reach essentially the same best-threshold F1 (0.792 vs 0.794). Fine-tuning therefore does not merely fail to help OOD — it *degrades the guard's OOD score ranking / operating-point robustness*: a lower AUPRC means a worse precision–recall trade-off across thresholds, even though a per-distribution re-tuned threshold would recover the same peak F1 (a re-tuning that is, by definition, unavailable on unseen data). This resolves the earlier tension between Contribution #2 (tuning does not improve — and on OOD ranking actively harms — generalization) and Contribution #3 (the tuned guard still beats Llama-Guard on novel data): the guard's novel-set lead is *inherited from the base*, not produced by LoRA, and the base in fact beats Llama-Guard by an even larger margin.

This is the constructive complement to prior reports that fine-tuning can *destroy* safety alignment: here tuning "works," yet the measurable gain is memorization of the in-distribution operating regime rather than demonstrated transferable safety competence.

### 6.2 Operating-point fairness: rankings depend on the threshold

Guard rankings shift substantially with the decision threshold, so any comparison that reads F1 at each model's *native* operating point conflates classifier quality with operating-point choice. Concretely, at native thresholds Llama-Guard-3-1B posts a higher F1 than ShieldGemma-2b (0.673 vs 0.424), but this ordering **reverses** under both threshold-free AUPRC (0.639 vs 0.712) and best-threshold Optimal-F1 (0.670 vs 0.729): a ranking flip that is purely an artifact of where each model's alarm is set. Only our model received a dev-tuned threshold; the baselines were fixed points. We therefore report both threshold-free AUPRC and a matched-FPR comparison, with paired-bootstrap CIs where available.

Threshold-free, the guard leads the open guards across every in-house pooling, on point estimates (baseline AUPRCs carry no CIs):

| System | AUPRC in-house pooled | AUPRC in-dist | AUPRC in-house held-out |
|---|---|---|---|
| Guard | 0.844 [0.825, 0.866] | 0.846 | 0.860 [0.803, 0.909] |
| ShieldGemma-2b | 0.712 | 0.702 | 0.785 |
| Llama-Guard-3-1B | 0.639 | 0.610 | 0.762 |

At a matched, conservative operating point (FPR@0.10, threshold taken from the in-dist dev split for all tunable systems), the ranking is preserved on point estimates but all systems are pushed to low recall:

| System | Recall | F1 | FPR |
|---|---|---|---|
| Guard | 0.431 | 0.581 | 0.051 |
| ShieldGemma-2b | 0.341 | 0.464 | — |
| Llama-Guard-3-1B | 0.242 | 0.360 | — |
| keyword matcher | 0.096 | 0.168 | — |
| gpt-5.4-mini (fixed native point) | 0.856 | 0.784 | 0.321 |

Why native-threshold F1 comparisons mislead: gpt-5.4-mini's native point trades a high FPR (0.321) for high recall (0.856), producing an F1 of 0.784 that is not comparable to a guard evaluated at FPR 0.051. Conversely, the matched-FPR@0.10 constraint makes *every* system conservative — the guard's recall there is only 0.431 — so matched-FPR alone understates absolute capability. The two views are complementary: matched-FPR fixes the confound of who got to tune a threshold, and AUPRC is the threshold-free summary that removes operating-point choice entirely. Read together they give a consistent picture on point estimates — guard > ShieldGemma-2b > Llama-Guard-3-1B — with the one formally supported (non-overlapping-CI) statement being the guard's novel-set lead over Llama-Guard (§5.2).

---

## 7. Limitations

We frame these results as a measurement and controlled study, not a state-of-the-art claim. Several caveats bound their interpretation.

**Deployed operating point over-blocks.** At the deployed point (T=2.10, τ=0.59) the guard's overall FPR is 0.306 — roughly 31% of benign in-house prompts are flagged `unsafe`. This is a real deployment concern: a global τ tuned for balanced F1 on in-dist dev data is too aggressive for many production settings, which would need a higher, application-specific threshold (trading recall for fewer false positives). We report the deployed point for transparency, but we do not recommend τ=0.59 as an out-of-the-box safe default.

**Operating point and matched-FPR conservatism.** A single global threshold is structurally in-distribution-optimized, and guard rankings depend heavily on the decision threshold. Our matched-FPR@0.10 comparison forces all systems into a conservative regime: at that operating point the guard's recall is only 0.431 (F1 0.581, FPR 0.051). We therefore treat threshold-free AUPRC as the fair primary summary and matched-FPR as a secondary, deliberately strict view.

**Statistical support is uneven.** Only four comparisons carry CIs (guard pooled/in-house-held-out AUPRC, guard-vs-Llama novel aggregate, base-vs-tuned F1 delta, guard-vs-GPT ΔF1). Baseline AUPRCs are point estimates, so the pooled/in-dist "leads" and the matched-FPR ranking are point-estimate claims, not significance tests. We reserve "decisive" for the single non-overlapping-CI result (novel guard vs Llama-Guard).

**GPT is a tie, not a beat, and threshold-advantaged.** Against gpt-5.4-mini the guard shows no significant F1 difference:

| Comparison | ΔF1 | 95% CI | McNemar |
|---|---|---|---|
| guard vs gpt-5.4-mini | +0.007 | [-0.008, 0.023] | p=0.78 |

The guard is nominally ahead but the difference is not significant, and the comparison uses the guard's dev-tuned threshold against gpt's fixed native point (gpt exposes no tunable threshold, so a matched-FPR comparison is impossible). The advantage over GPT is an efficiency one (local, ~4x lower batch=1 latency: 124 ms vs ~512 ms per request, no API cost), not an accuracy win, and the latency figure is cross-substrate (local inference vs remote API), not a controlled hardware benchmark.

**Novel-set win is inherited, not tuned.** We ran the base SmolLM3-3B on the four novel benchmarks: it *outranks* the tuned guard OOD (aggregate AUPRC 0.886 [0.870, 0.900] vs 0.781), so the guard's novel-set advantage over Llama-Guard reflects inherited base competence, and LoRA in fact degrades OOD ranking (§6.1). Because the base-vs-tuned comparison is now anchored on threshold-free AUPRC and best-threshold Optimal-F1 (both operating-point-independent), the earlier calibration-provenance concern for the base does not apply to this result. Relatedly, the base operating-point provenance (whether the base was scored under the tuned model's decision rule or a base-specific calibration) was not separately fit; the base-vs-tuned result is reported as a paired F1 delta on matched rows, and readers should treat the absolute base F1 as an operating-point-dependent estimate.

**Generalization claim rests on two benchmarks.** The "PEFT specializes but does not improve generalization" finding is evidenced by only the two in-house held-out benchmarks (JailbreakBench, XSTest); it is not a broad OOD-generalization result.

**ShieldGemma out-of-regime and pending on the novel set.** ShieldGemma-2b is designed for content-policy / general-guideline moderation and is not an injection detector; using it across our red-team axes is somewhat out of its designed regime, so its numbers may understate its intended performance. On the in-house held-out set its AUPRC is 0.785 vs the guard's 0.860. Its results on the four novel held-out benchmarks are **pending** (Gemma-2 stalls on the throttled MPS laptop) and are not reported here; we make no ShieldGemma novel-set claim.

**HarmBench metric is calibration-dependent.** The HarmBench comparison (mean P(unsafe) 0.780 vs 0.390) compares raw probabilities across differently temperature-scaled models; it is suggestive, not decisive, and HarmBench is excluded from the aggregate AUPRC.

**Dataset licensing.** ToxicChat is non-commercial; it is eval-only for any released model and cannot be used to train a distributed guard.

**Sample sizes and CIs.** Some benchmarks have small n and correspondingly wide CIs, so per-benchmark point estimates should be read with their intervals (e.g., in-house held-out AUPRC 0.860 [0.803, 0.909]).

**Decontamination scope.** We adopt source-family holdout to define the novel benchmarks and recommend near-duplicate detection as the companion protocol; we do not claim an exhaustive near-duplicate sweep of the entire release pipeline.

**Scope of the guard and evaluation.** The guard emits a single-token verdict from one forward pass; we do not evaluate a multi-token or reasoning-style guard, and adversarial-robustness evaluation (e.g., against adaptive attacks) is future work.

---

## 8. Conclusion

We presented agent-bouncer, a binary safety guard obtained by parameter-efficient fine-tuning of SmolLM3-3B on a single consumer laptop (Apple M4 Max, MPS, no CUDA; ~71 min / 300 steps; 60.5M trainable params, 1.93% of 3.14B). The guard emits a single-token verdict in one forward pass at batch=1 p50 124 ms / p90 188 ms. Our central finding is a measurement one: guard rankings depend heavily on the decision threshold, so we evaluate at matched FPR, with threshold-free AUPRC/AUROC, paired-bootstrap 95% CIs, and McNemar tests. Building the pipeline surfaced and let us fix real evaluation bugs (calibration leak, batch-amortized latency, per-model-only threshold tuning, GPT error-to-"unsafe" default, exact-string decontamination).

Under this protocol the results are modest and specific rather than sweeping:

| Comparison | Metric | Guard | Baseline |
|---|---|---|---|
| vs Llama-Guard-3-1B (novel held-out) | aggregate AUPRC | 0.781 [0.751, 0.811] | 0.701 [0.673, 0.733] |
| vs Llama-Guard-3-1B (in-house pooled) | AUPRC (point est.) | 0.844 [0.825, 0.866] | 0.639 |
| vs ShieldGemma-2b (in-house pooled) | AUPRC (point est.) | 0.844 | 0.712 |
| vs gpt-5.4-mini (paired) | ΔF1 | +0.007 [-0.008, 0.023], McNemar p=0.78 | — |
| vs gpt-5.4-mini | latency (batch=1) | 124 ms (local) | ~512 ms (remote API) |

On threshold-free AUPRC the guard surpasses Llama-Guard-3-1B decisively — the only non-overlapping-CI cross-system result, and it holds on the four novel held-out benchmarks — and it exceeds ShieldGemma-2b on the in-house pooled / in-dist / in-house-held-out sets on point estimates (ShieldGemma's novel runs are pending). It is a statistical tie with gpt-5.4-mini on F1 at ~4x lower batch=1 latency and no API cost: an efficiency win at parity accuracy, not an accuracy beat.

Our base-vs-tuned decomposition tempers the fine-tuning narrative. The zero-shot SmolLM3-3B base already scores F1 0.711, above zero-shot Llama-Guard-3-1B (0.673) and ShieldGemma-2b (0.424) — an F1-at-operating-point comparison (we did not compute base AUPRC), so read as "the base is already a competent classifier by F1," not a threshold-free ranking. LoRA lifts paired-row F1 to 0.791 (+0.080 [0.063, 0.096], McNemar p<1e-4), but the gains are almost entirely in-distribution (e.g. jailbreak_classification 0.328→0.966) and flat-to-slightly-negative on the two in-house held-out benchmarks (JailbreakBench -0.006, XSTest -0.034). On the four novel benchmarks the *untuned* base outranks the tuned guard (aggregate AUPRC 0.886 [0.870, 0.900] vs 0.781, disjoint CIs) at tied best-threshold F1 (0.792 vs 0.794), so the novel-set win over Llama-Guard is inherited base competence rather than a product of tuning — and LoRA in fact degrades the guard's out-of-distribution score ranking.

We are candid about scope. Matched-FPR@0.10 makes every system conservative (guard recall 0.431 there), so AUPRC is our fairest summary, and the *deployed* point over-blocks (FPR 0.306). A single global threshold is structurally in-distribution-optimized. ShieldGemma is exercised somewhat outside its designed content-policy regime and its novel-benchmark AUPRC is still pending. Some benchmark n are small (wide CIs). The guard is single-token with no reasoning or adversarial-robustness evaluation, and ToxicChat is eval-only under its non-commercial terms.

### 8.1 Future Work

- **Specializing without sacrificing OOD robustness.** Since fine-tuning specialized in-distribution but *degraded* the base's OOD score ranking (§6.1), a central direction is training recipes — broad, family-decontaminated data plus regularization — that add in-distribution competence *without* losing the untuned base's operating-point robustness on unseen distributions.
- **Broad-data training as the generalization lever.** Since tuning specialized in-distribution without improving the in-house held-out F1, the most promising direction is scaling and diversifying training data (source-family coverage, near-duplicate-filtered) to test whether generalization, not just in-distribution fit, can be trained in.
- **Controlled objective comparison.** A like-for-like SFT vs DPO vs KTO vs ORPO head-to-head under identical data, splits, and calibration, to isolate the effect of the learning objective on both in-distribution specialization and held-out transfer.
- **Adversarial robustness.** Evaluation and hardening against paraphrase, obfuscation, and prompt-level attacks, which our current single-token guard does not measure.
- **Agent surfaces.** Extending the guard beyond input prompts to tool-call arguments and tool/output content, the moderation surfaces that matter most for deployed agents.
- **Completing the baseline matrix.** Resolving the pending ShieldGemma-2b runs on the novel held-out benchmarks, and measuring open-guard latency, for a complete threshold-free and efficiency comparison.

Code, calibration, and the full evaluation protocol are released as an open notebook to support reproduction at laptop scale.

---

*Baseline note: `gpt-5.4-mini` is a hosted commercial API model with no public paper or model card; we cite it as an API baseline rather than a bibliographic reference.*

## References

1. **[inan2023llamaguard]** Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa. "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations." arXiv preprint, 2023. arXiv:2312.06674.
2. **[fedorov2024llamaguard3_1b]** Igor Fedorov, Kate Plawiak, Lemeng Wu, Tarek Elgamal, Naveen Suda, Eric Smith, Hongyuan Zhan, Jianfeng Chi, Yuriy Hulovatyy, Kimish Patel, Zechun Liu, Changsheng Zhao, Yangyang Shi, Tijmen Blankevoort, Mahesh Pasupuleti, Bilge Soran, Zacharie Delpierre Coudert, Rachad Alao, Raghuraman Krishnamoorthi, Vikas Chandra. "Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations." arXiv preprint, 2024. arXiv:2411.17713.
3. **[meta2024llamaguard2_3]** Meta (Llama Team); Grattafiori et al. "Llama Guard 2 / Llama Guard 3" (Meta model cards; described in "The Llama 3 Herd of Models"). arXiv preprint / Meta model cards, 2024. arXiv:2407.21783 (Llama 3 Herd); model cards: https://github.com/meta-llama/PurpleLlama.
4. **[meta2024promptguard]** Meta (Llama / PurpleLlama Team). "Prompt Guard (Llama Prompt Guard): mDeBERTa-based prompt-injection / jailbreak classifier." Meta model card (PurpleLlama), 2024. https://github.com/meta-llama/PurpleLlama/blob/main/Prompt-Guard/MODEL_CARD.md (model card only; no arXiv paper).
5. **[zeng2024shieldgemma]** Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Radharapu, Olivia Sturman, Oscar Wahltinez. "ShieldGemma: Generative AI Content Moderation Based on Gemma." arXiv preprint, 2024. arXiv:2407.21772.
6. **[ghosh2024aegis]** Shaona Ghosh, Prasoon Varshney, Erick Galinkin, Christopher Parisien. "AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts." arXiv preprint, 2024. arXiv:2404.05993.
7. **[padhi2024graniteguardian]** Inkit Padhi, Manish Nagireddy, Giandomenico Cornacchia, Subhajit Chaudhury, Tejaswini Pedapati, Pierre Dognin, Keerthiram Murugesan, Erik Miehling, Martín Santillán Cooper, Kieran Fraser, Giulio Zizzo, Muhammad Zaid Hameed, Mark Purcell, Michael Desmond, Qian Pan, Zahra Ashktorab, Inge Vejsbjerg, Elizabeth M. Daly, Michael Hind, Werner Geyer, Ambrish Rawat, Kush R. Varshney, Prasanna Sattigeri. "Granite Guardian." arXiv preprint, 2024. arXiv:2412.07724.
8. **[han2024wildguard]** Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri. "WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs." NeurIPS 2024 (Datasets & Benchmarks), 2024. arXiv:2406.18495. *(Merged entry: single bibkey `han2024wildguard`; the WildGuardMix training set and WildGuardTest eval set both cite this.)*
9. **[kang2025r2guard]** Mintong Kang, Bo Li. "$R^2$-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning." ICLR 2025. arXiv:2407.05557. *(Title corrected to R-squared per fact-check.)*
10. **[deng2025duoguard]** Yihe Deng, Yu Yang, Junkai Zhang, Wei Wang, Bo Li. "DuoGuard: A Two-Player RL-Driven Framework for Multilingual LLM Guardrails." arXiv preprint, 2025. arXiv:2502.05163.
11. **[mazeika2024harmbench]** Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." ICML 2024. arXiv:2402.04249.
12. **[jiang2024wildteaming]** Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, Nouha Dziri. "WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models." NeurIPS 2024. arXiv:2406.18510.
13. **[xie2024sorrybench]** Tinghao Xie, Xiangyu Qi, Yi Zeng, Yangsibo Huang, Udari Madhushani Sehwag, Kaixuan Huang, Luxi He, Boyi Wei, Dacheng Li, Ying Sheng, Ruoxi Jia, Bo Li, Kai Li, Danqi Chen, Peter Henderson, Prateek Mittal. "SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal." ICLR 2025. arXiv:2406.14598.
14. **[rottger2024xstest]** Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy. "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." NAACL 2024. arXiv:2308.01263.
15. **[cui2024orbench]** Justin Cui, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh. "OR-Bench: An Over-Refusal Benchmark for Large Language Models." ICML 2025. arXiv:2405.20947.
16. **[chao2024jailbreakbench]** Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, Eric Wong. "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." NeurIPS 2024 (Datasets & Benchmarks). arXiv:2404.01318.
17. **[lin2023toxicchat]** Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, Jingbo Shang. "ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation." Findings of EMNLP 2023. arXiv:2310.17389.
18. **[ji2023beavertails]** Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Ruiyang Sun, Boyuan Chen, Yizhou Wang, Yaodong Yang. "BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset." NeurIPS 2023 (Datasets & Benchmarks). arXiv:2307.04657.
19. **[markov2023holistic]** Todor Markov, Chong Zhang, Sandhini Agarwal, Tyna Eloundou, Teddy Lee, Steven Adler, Angela Jiang, Lilian Weng. "A Holistic Approach to Undesired Content Detection in the Real World." AAAI 2023. arXiv:2208.03274.
20. **[hu2021lora]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022 (arXiv 2021). arXiv:2106.09685.
21. **[rafailov2023dpo]** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. arXiv:2305.18290.
22. **[azar2023ipo]** Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Rémi Munos. "A General Theoretical Paradigm to Understand Learning from Human Preferences" (Ψ-PO / IPO). AISTATS 2024 (arXiv 2023). arXiv:2310.12036.
23. **[ethayarajh2024kto]** Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela. "KTO: Model Alignment as Prospect Theoretic Optimization." ICML 2024. arXiv:2402.01306.
24. **[hong2024orpo]** Jiwoo Hong, Noah Lee, James Thorne. "ORPO: Monolithic Preference Optimization without Reference Model." EMNLP 2024. arXiv:2403.07691.
25. **[shao2024deepseekmath]** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (introduces GRPO). arXiv preprint, 2024. arXiv:2402.03300.
26. **[schulman2017ppo]** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. "Proximal Policy Optimization Algorithms." arXiv preprint, 2017. arXiv:1707.06347.
27. **[ouyang2022instructgpt]** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe. "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT). NeurIPS 2022. arXiv:2203.02155.
28. **[guo2017calibration]** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. "On Calibration of Modern Neural Networks." ICML 2017. arXiv:1706.04599.
29. **[bakouch2025smollm3]** Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Patiño, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallouédec, Kashif Rasul, Nathan Habib, Clémentine Fourrier, Hynek Kydlíček, Guilherme Penedo, Hugo Larcher, Mathieu Morlon, Vaibhav Srivastav, Joshua Lochner, Xuan-Son Nguyen, Colin Raffel, Leandro von Werra, Thomas Wolf (Hugging Face). "SmolLM3: smol, multilingual, long-context reasoner." Hugging Face blog + model card (HuggingFaceTB/SmolLM3-3B), 2025. https://huggingface.co/blog/smollm3 (no arXiv paper; cite blog + model card).
30. **[majhi2025gpuguard]** Vasudev Majhi, Dhruv Gupta, Advait Singh, Matthew Barker, Dhruv Kumar. "Do You Really Need a GPU to Guard Your LLM? CPU-Class Classifiers and Multi-Stage Pipelines for Safety Enforcement at Scale." arXiv preprint, 2025 (v3 revised 2026; under review). arXiv:2512.19011.
31. **[hsiung2025guardrails]** Lei Hsiung, Tianyu Pang, Yung-Chen Tang, Linyue Song, Tsung-Yi Ho, Pin-Yu Chen, Yaoqing Yang. "Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets." ICML 2025 DIG-BUGS workshop (also ACL Anthology 2026). arXiv:2506.05346.
32. **[hackett2025bypassing]** William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan. "Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems." First Workshop on LLM Security (LLMSEC 2025), ACL Anthology 2025.llmsec-1.8. arXiv:2504.11168 (v3 title; v1/v2 titled "Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails").

*Note: all 32 entries were confirmed by the fact-checker (arXiv IDs, titles, authors, years), so none carry an [unverified] tag. The only claim lacking a citable source is the GPT-5.4-mini frontier baseline — flagged [unverified — check before submission] in the citation map; supply an official model/system card or describe it as an API baseline before submission.*
