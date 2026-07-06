# Benchmarks & results

The research questions this repo answers:

> 1. Can a **small guard model (SLM)** — trained with SFT, refined with RL — stand at
>    the request path of an LLM/agent, screen prompts fast enough to matter, and *not*
>    over-block benign traffic?
> 2. How does it compare, **apples-to-apples on standard benchmarks**, to frontier LLM
>    judges — **GPT-4o-mini** and **GPT-5.2 (low reasoning)** — and to a purpose-built
>    moderation API?

Everything below is scored through **one harness** (`agent_bouncer.evaluation`), so the
numbers are directly comparable. Metrics: **precision / recall / F1** (positive class =
`unsafe`), **`fpr_on_benign`** (over-blocking — the headline usability metric), and
**p50 latency**.

## The standard benchmark suite

We download and run **7 ungated standard benchmarks** across three axes. Each is
normalized to the unified taxonomy in [`data/loaders.py`](../src/agent_bouncer/data/loaders.py).

By default the headline scoreboard uses a **class-balanced subset (≤100/class)** so
precision/recall are comparable across benchmarks of very different sizes. You can also run
on the **full** benchmarks — `run_benchmarks.py --full` (or `--per-class 0`), and the Agent Bouncer
Workbench's *Explore Benchmarks* tab / test controls (**size 0 = full**) — using the sets fetched by
[`download_full_benchmarks.py`](../scripts/data/download_full_benchmarks.py) (e.g. BeaverTails
**3021**, ToxicChat **5083**, OpenAI-Moderation **1680**, XSTest **450**).

| Axis | Benchmark | HF dataset | Measures |
|------|-----------|------------|----------|
| Guardrail | **BeaverTails** (30k_test) | `PKU-Alignment/BeaverTails` | 14-category harmful-QA prompt safety |
| Guardrail | **OpenAI-Moderation** | `mmathys/openai-moderation-api-evaluation` | 8-category content-moderation gold set |
| Guardrail | **ToxicChat** | `lmsys/toxic-chat` | Real user-input toxicity detection |
| Red-team | **prompt-injections** | `deepset/prompt-injections` | Prompt-injection attack detection |
| Red-team | **jailbreak-classification** | `jackhhao/jailbreak-classification` | Jailbreak vs. benign prompts |
| Red-team | **JailbreakBench** | `JailbreakBench/JBB-Behaviors` | 100 harmful + 100 benign red-team behaviors |
| Over-refusal | **XSTest** | `natolambert/xstest-v2-copy` | Safe-but-scary prompts → over-blocking (FPR) |

Gated benchmarks (**WildGuardMix, HarmBench, AdvBench, Lakera PINT**) need `HF_TOKEN`
+ license acceptance; the pipeline reports them as *not run* rather than fabricating
numbers. BeaverTails uses the held-out `30k_test` split (disjoint from the demo
training data — no leakage).

Full per-benchmark tables are auto-generated in
[`outputs/BENCHMARKS.md`](../outputs/BENCHMARKS.md). Reproduce with `make bench`.

### Headline: macro-average across all 7 benchmarks

> **Illustrative figures from a reference run** (`per_class=100`, M-series Mac; local guards on
> CPU, decoders on Apple MPS, OpenAI over the API). They are **not** shipped in the repo — the
> results directory starts empty. Reproduce your own with `make bench` (then open the
> **Leaderboard** tab, or export a PDF from it). Your numbers will vary with hardware, sampling,
> and model versions.

The reasoning judge is scored at **three effort tiers** — `reasoning_effort` ∈ {`low`, `medium`,
`high`} — surfaced as `openai-gpt-5.2-low` / `-medium` / `-high`. The row below shows the `low`
tier; higher tiers trade more latency (and cost) for recall.

Latency is **device-dependent** (captured per run): encoder/keyword on **CPU**, decoders on
**Apple MPS**, OpenAI over the **API**.

| Guard | Params | Precision | Recall | **F1** | ROC-AUC | **FPR@benign ↓** | p50 ms ↓ | p90 ms ↓ |
|-------|-------:|----------:|-------:|-------:|--------:|-----------------:|---------:|---------:|
| keyword-baseline          | 0    | 0.571 | 0.076 | 0.113 | 0.538 | **0.000** | **0.01** | **0.05** |
| **encoder (distilbert)**  | 66M  | 0.670 | 0.564 | 0.579 | 0.703 | 0.288 | **9** | **14** |
| **decoder-SFT (Qwen3)**   | 0.6B | 0.677 | 0.700 | 0.672 | 0.672 | 0.355 | 292 | 419 |
| **decoder-SFT (Qwen3)**   | 1.7B | 0.708 | 0.631 | 0.636 | 0.673 | 0.285 | 343 | 593 |
| **decoder-GRPO (Qwen3, RL)** | 0.6B | 0.668 | 0.711 | 0.673 | 0.667 | 0.377 | 298 | 413 |
| openai-moderation         | api  | 0.766 | 0.525 | 0.577 | 0.678 | 0.170 | 203 | 298 |
| openai-gpt-4o-mini        | api  | 0.781 | **0.857** | 0.794 | 0.796 | 0.266 | 744 | 1069 |
| **openai-gpt-5.2 (low)**  | api  | **0.836** | 0.830 | **0.804** | **0.823** | **0.184** | 1196 | 2030 |

### Takeaways (reported straight)

1. **GPT-5.2 (low reasoning) is the quality + usability leader.** It leads on macro-F1 (0.804)
   *and* ROC-AUC (0.823), with the **highest precision (0.836)** and the **lowest over-blocking
   of any capable guard (FPR 0.184)** — it flags fewer benign prompts. The cost is tail latency:
   **p90 ~2.0 s/call**, and it's slowest overall.
2. **The 66M encoder ties OpenAI's Moderation API on macro-F1 (0.579 vs 0.577) at ~22× lower
   latency** (9 ms vs 203 ms) — and its **swept ROC-AUC (0.703) is the best of the local
   guards**. For a guard that runs on *every* call, it is the only option fast enough.
3. **Bigger SLM → more conservative, not clearly better.** The 1.7B decoder-SFT is more
   precise (0.708) and over-blocks less (FPR 0.285) than the 0.6B, but its recall/F1 are lower
   at 1-epoch LoRA and its p90 is highest of the decoders (593 ms) — scaling didn't buy accuracy
   here. The 0.6B SFT/GRPO stay the better decoder tradeoff.
4. **Red-teaming is the hard axis, and it exposes the SLMs' training gap.** On
   `prompt_injections`, recall collapses for everyone (encoder 0.09, moderation 0.11,
   even GPT-5.2 only 0.20) — injection is subtle. The SLMs were trained on *content
   safety* (BeaverTails), so they under-detect *adversarial* prompts. Clearest signal for
   what to train next.
5. **Over-blocking separates the field.** GPT-4o-mini flags **27% of benign traffic**;
   GPT-5.2 cuts that to **19%**; the encoder sits at 29%. `fpr_on_benign` is exactly what
   the DPO stage and the GRPO false-positive reward optimize against.

### Per-axis highlights

- **Content safety:** GPT-5.2 leads (OpenAI-Mod F1 0.891, ToxicChat 0.881). The 66M
  encoder is competitive *in-domain* (BeaverTails 0.718, ToxicChat 0.682 — it was trained
  on BeaverTails) but weaker on the OpenAI-Moderation distribution (0.596).
- **Red-teaming:** the LLM judges dominate `jailbreak_classification` (GPT-5.2 F1 0.925,
  GPT-4o-mini 0.917) and `jailbreakbench` (GPT-5.2 catches **100%** of harmful behaviors,
  R=1.0, but over-blocks benign at FPR 0.36). Both SLMs trail here.
- **Over-refusal (XSTest):** GPT-5.2 best (F1 0.920, FPR 0.150); the SLMs over-block the
  safe-but-scary prompts (encoder FPR 0.54, decoder 0.50) — the failure mode Agent
  Bouncer is designed to drive down.

### ROC-AUC (all guards)

`scripts/report/compute_curves.py` reports **ROC-AUC for every guard**, written into
`outputs/curves.json` and merged back into the results:

- **Continuous-score guard (the encoder)** → a **threshold-swept** ROC/PR curve and its true
  AUC. Notably the encoder's macro ROC-AUC (**≈0.70**) is *higher* than its fixed-0.5-threshold
  macro-F1 (0.58) — its score *ranking* is better than the default operating point shows, so
  threshold tuning (or DPO/GRPO) has real headroom.
- **Hard-decision guards** (keyword · decoders · OpenAI) emit a single operating point, so the
  ROC is `(0,0)→(FPR,TPR)→(1,1)` and **AUC = (recall + 1 − FPR) / 2** — derived exactly from the
  stored recall/FPR (no re-running). By this measure GPT-5.2 leads (macro AUC ≈0.81), then
  GPT-4o-mini (≈0.79); the encoder's swept AUC (≈0.70) beats OpenAI Moderation (≈0.68) and the
  decoders (≈0.67). The Agent Bouncer Workbench's **Leaderboard** tab plots all of these (ROC / PR curves
  and ROC-AUC by benchmark) above the macro-average results table.

### Model size: Qwen3-0.6B vs 1.7B (SFT)

The suite includes both decoder sizes so scaling is visible on the same axes. Consistent with
the earlier size sweep, the larger 1.7B decoder tends to be more conservative (higher precision,
lower over-blocking) but not dramatically more accurate at 1-epoch LoRA — and it costs ~2× the
latency. See the `decoder-sft-1.7B` row in [`outputs/BENCHMARKS.md`](../outputs/BENCHMARKS.md).

## RL: verifiable-reward GRPO (RLVR) on a real SLM

The guard has ground-truth labels, so the **label is the reward** — no reward model.
[`training/rewards.py`](../src/agent_bouncer/training/rewards.py) scores each rollout on correctness +
hazard category + parseable format − **false-positive penalty** + brevity;
[`training/grpo.py`](../src/agent_bouncer/training/grpo.py) wires it into TRL's `GRPOTrainer`
and (like SFT) merges the LoRA adapter so the RL model loads as a standalone guard.

`configs/model/grpo_from_sft.yaml` runs GRPO **from the SFT checkpoint** — the recommended
recipe. Because the base already emits terminal verdict JSON, completions stay short
(mean ~16 tokens, **0% clipped**, vs the earlier from-scratch smoke that clipped at max
length), the verifiable reward is live (per-step mean ~0.3–1.6, std > 0), and KL stays
~1e-5. This produces an **evaluable RL guard** scored in the suite above.

**What the bounded run shows (reported straight):** 60 CPU steps at lr 1e-6 from an
already-good SFT checkpoint **barely move the model** — macro-F1 0.672 → **0.673**, with
recall nudged up (0.700 → 0.711) at a small over-blocking cost (FPR 0.355 → 0.377). That
is the expected outcome of a *bounded* RLVR run on a converged-enough prior: it validates
the full pipeline end-to-end (train → merge → load → benchmark) but is **not** a converged
model. Real gains need a GPU (many more steps, higher LoRA rank) and, above all, **red-team
training data** to close the prompt-injection recall gap the suite exposes.

## Earlier in-session results (context)

- **Encoder fine-tune moved the baseline ~100×.** A `distilbert` encoder (2 epochs, 73 s)
  scored **F1 0.703 / recall 0.708** on a 621-prompt held-out BeaverTails test vs the
  keyword baseline's F1 0.007. Reproduce: `make data-demo && make demo`.
- **Size sweep (Qwen3 0.6B vs 1.7B):** the tiny encoder stayed the best accuracy/latency
  tradeoff; the 1.7B decoder got more conservative (lower FPR) but not more accurate at
  1-epoch LoRA. See [`outputs/size_sweep.json`](../outputs/size_sweep.json).

## Reproducing

```bash
make bench                                     # download + run the whole suite
python scripts/eval/run_benchmarks.py --per-class 100          # with knobs
python scripts/eval/run_benchmarks.py --no-openai              # local guards only (offline)
python scripts/eval/eval_added_guard.py --path outputs/grpo-qwen3-0.6b \
    --mode reasoning --name decoder-grpo-0.6B --params 0.6B   # add the RL guard
python scripts/data/download_full_benchmarks.py                # full-size datasets
```

Runs log to MLflow when installed (`mlruns/`). Caveats bound the *absolute* numbers
(BeaverTails labels *response* safety, so prompt labels are noisy; ToxicChat / jailbreak
sets carry label noise) — but the *relative* comparison is clean: one harness, one
balanced subset per benchmark, every guard scored on identical inputs.
