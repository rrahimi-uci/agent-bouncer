# Glossary — plain definitions

Quick reference for the terms used in the [plain-language edition](README.md). Ordered
roughly from "what is being built" to "how it's measured."

### Prompt-safety guard
A filter that reads a user's **prompt** and outputs **safe** or **unsafe**. Cheap to run in
front of a bigger model. Here "unsafe" = harmful content, jailbreak, or prompt injection.

### Jailbreak / prompt injection
- **Jailbreak** — a prompt that tries to trick the model into ignoring its safety rules.
- **Prompt injection** — hidden instructions inside the input that try to hijack the system.

### Base model / checkpoint
The original instruction-tuned chat model *before* any fine-tuning (e.g. Qwen2.5-1.5B). The
study uses 4 of them, 1.5B–4B parameters.

### LoRA
"Low-Rank Adaptation." A way to fine-tune by **freezing the whole base model** and training a
small bolt-on set of weights (an **adapter**). Cheap, and produces a small file that snaps onto
the frozen base. `r=32, alpha=64` are its size/scaling knobs.

### SFT (supervised fine-tuning)
Training on labeled examples — here, prompts paired with the correct verdict word.
**Completion-only loss on the verdict token** = the model is graded only on producing the one
answer token (`safe`/`unsafe`), not on echoing the prompt.

### Seed
The random number that initializes a training run. Running **5 seeds (42–46)** per model gives
5 slightly different fine-tunes, so you can see how much the result wobbles from randomness.

### Logit / logit-difference score
A **logit** is the model's raw, unnormalized preference number for a candidate next word. The
guard's score is the gap `s(x) = z_unsafe − z_safe` at the final token. Positive → leans unsafe.

### Softmax
Turns the two logits into two probabilities that add up to 1, so the score becomes a 0–1
"probability of unsafe."

### Represented sources
Held-out test rows from the **same datasets used in training** (same style, new examples).
Like this year's exam on a topic you crammed from last year's.

### Transfer (dataset-held-out) benchmarks
**Entirely different datasets** never used in training. A new-material exam — tests whether the
skill generalizes. (In this paper they were seen during development, so "held-out," not
"sealed/never-seen.")

### In-distribution / out-of-distribution (OOD, "off-distribution")
Common ML shorthand. **In-distribution** = data that looks like what the model trained on =
the **represented** / trained-on regime. **Out-of-distribution (OOD)** = data unlike the
training data = the **transfer** / new regime. This edition mostly says "trained-on / new" to
avoid jargon.

### Stress sets
One-sided diagnostic sets: **OR-Bench** (all benign → measures false alarms) and **HarmBench**
(all harmful → measures how many attacks are caught). No AP is computed on them.

### Precision / Recall (TPR) / FPR
- **Precision** — of the prompts you flagged unsafe, how many really were.
- **Recall / TPR (true-positive rate)** — of the truly-unsafe prompts, how many you caught.
- **FPR (false-positive rate)** — of the safe prompts, how many you wrongly flagged.

### Average Precision (AP)
A **threshold-free** score (0–1, higher better) = **area under the precision-recall curve**. It
measures how well the guard **ranks** unsafe prompts above safe ones, across all cutoffs at
once. Better than **accuracy** for a scorer, because accuracy can look great by labeling
everything "safe" when unsafe prompts are rare. **Tie-aware** = fair handling when two prompts
get the same score.

### Macro-AP / benchmark-macro then panel-mean
Average the AP **across benchmarks** so each dataset counts equally ("macro"), then **across the
4 models** ("panel"), and for fine-tuned guards **across the 5 seeds** too. The headline
"+0.33 / −0.05" numbers are these averages of averages.

### Calibration / temperature scaling
Rescaling the score into an honest probability by dividing by one tuned number (the
"temperature"), fit on a held-back calibration set. It **doesn't reorder** prompts — just makes
stated probabilities match reality better.

### Threshold / operating point
The cutoff score above which a prompt is flagged. Chosen here as a **conservative cap** so at
most ~5% of safe prompts are flagged (target FPR). The realized TPR/FPR are then **measured** on
test data.

### Bootstrap (paired, hierarchical)
A way to estimate uncertainty by **resampling your own results** (with replacement) thousands of
times and recomputing the number each time. **Paired** = compare base vs. fine-tuned to measure
the *change*. **Hierarchical** = resample at two levels: which seeds count, and — since the eval
sets contain many near-duplicate prompts — how much each group of those duplicates counts. The 4
models are held fixed.

### Confidence interval (CI) / one-sided bound (LCB, UCB)
- **Two-sided 95% CI** — the middle 95% of the bootstrap results; a plausible range for the true
  average change.
- **LCB** (lower confidence bound) — "at least this much." **UCB** (upper) — "no more than this."

### Precision-focused / descriptive (analysis mode)
The deliberate choice to report **intervals only — no p-value, no "statistically significant,"
no pass/fail** — because the scores are a legacy artifact analyzed after the fact. Directions are
**suggestive, not proven**; a clean pre-registered rerun would be needed to claim significance.

### Specialization plane
The core figure: x = represented AP change, y = transfer AP change, one dot per (model, seed).
Lower-right quadrant = "specialization" (better on familiar data, worse on new). 14 of 20 dots
land there.

### Specialization (the finding)
Fine-tuning making a guard **better on its training sources but not broadly better** (and often
worse on novel data) — as opposed to a general capability upgrade.

### Ensemble / composition
Using more than one model and combining their outputs into one decision. Here: run the untuned
base **and** the fine-tuned adapter on the same prompt and **average their unsafe-probabilities**.
A "second opinion" that keeps the specialist's in-domain skill and the generalist's out-of-domain
robustness — the "compose, don't tune" fix.
