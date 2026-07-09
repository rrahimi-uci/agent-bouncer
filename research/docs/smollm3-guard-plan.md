# Optimizing SmolLM3-3B as a Safety Guard to Match the Mini Judges

> Focused, executable plan for a **single** model — SmolLM3-3B (`smollm3-3b` →
> HuggingFaceTB/SmolLM3-3B). Scope narrowed from the earlier two-model plan (Qwen2.5-1.5B dropped).
> Every config/command references a real module in this repo.

## 0. Objective & success criteria

Turn `smollm3-3b` into a safety guard that competes with GPT-4o-mini / GPT-5-mini.

- **Claim (honest framing):** *a compact **~3B open** guard (SmolLM3-3B), fine-tuned and calibrated,
  matches the closed mini judges on Precision/Recall/F1 at a fraction of their latency and cost, and the
  result is reproducible.* SmolLM3 is 3.08B, so this is a **small/sub-4B** claim — not the stronger 1.5B
  story. Single model ⇒ **no ensemble/cascade**; the cost lever is a single-forward verdict head + quantization.
- **Quality metric:** Precision / Recall / F1 only (positive class = `unsafe`), macro + per-axis. No ROC-AUC.
- **Efficiency (separate table):** p50/p90 latency (pinned reference device) + \$/1k requests.
- **Pre-registered parity rule** (`evaluation/stats.py`, §6): SmolLM3 is **non-inferior** to mini M iff
  the paired-bootstrap 95% CI lower bound of `F1(SmolLM3)−F1(M) ≥ −0.03`; **matches** iff also
  `FPR` CI upper ≤ 0 and `p50 ≤ p50(M)/3`; **beats** iff CI lower > 0; else **inconclusive**.
- **Targets** (re-freeze from the matched-n run — current values are mixed-n): GPT-5-mini F1 ≈ 0.762 /
  FPR ≈ 0.223; GPT-4o-mini ≈ 0.833 / 0.263.

**Why achievable:** the earlier decoders were **under-trained** (1 epoch, LoRA r=16, no completion-only
loss), not capacity-limited. At 3B, SmolLM3 has the most headroom of the small open models — expected
macro-F1 ≥ ~0.78 once the recipe is fixed and no-think is airtight.

---

## 1. Prerequisite code changes (blockers — mostly no-GPU)

| # | File | Change | Why |
|--:|---|---|---|
| P1 | `training/sft.py` | Add explicit `target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']`; **completion-only loss** (`completion_only_loss=True` / `DataCollatorForCompletionOnlyLM`); `gradient_checkpointing=True`, `lr_scheduler_type='cosine'`, `warmup_ratio=0.03` | **The under-training fix.** No `target_modules` today → PEFT may KeyError on SmolLM3 or adapt 2 modules; full-sequence loss wastes >95% of gradient on the prompt |
| P2 | `training/hyperparams.py`, `runner.build_config` | Decoder-SFT defaults epochs 1→3, `lora_r` 16→32, `lora_alpha` 32→64; **thread `trust_remote_code=True`** (SmolLM3 needs it on `transformers<4.53`) | Defaults encode the under-trained regime; SmolLM3 load path needs the flag |
| P3 | `data/training_sets.py::_TRAIN_SPLIT_LOADERS` | Add train-split loaders for `toxicchat`, `prompt_injections`, `jailbreak_classification` | **Leakage blocker:** they currently fall through to the *eval* split |
| P4 | `models/decoder.py` | Single-token verdict + logprob head with **no-think forcing** (§3) | Calibrated score + ~40× lower latency; prevents the think-mode fail-closed trap |
| P5 | `evaluation/stats.py` (NEW) | `paired_bootstrap_ci`, `mcnemar`, `holm` | No CIs/significance exist today — critical path |
| P6 | `evaluation/ensembles.py::_align_rows` | **RAISE** on a missing/mismatched `sample_key`; add per-axis equal-weight macro | Matched-n integrity; stop red-team dilution |
| P7 | `evaluation/openai_guards.py` | Capture `resp.usage` → \$/1k via dated price | Cost axis |
| P8 | `evaluation/calibration.py` + `eval_splits.py` (NEW) | temperature/Platt + ECE + iso-FPR threshold; dev/test split by `sample_key` grouped by benchmark | §5 |

*(The two-model cascade patches from the earlier plan are dropped — single model.)*

---

## 2. SmolLM3-3B training recipe (SFT backbone)

**Recipe:** LoRA **r=32 / α=64 / dropout=0.05** on the full linear stack; **3 epochs**; lr **2e-4** cosine,
warmup 0.03; **max_seq_len=1024** (→1536 only if §4 shows >2–3% of prompts exceed 1024); **bf16**;
gradient checkpointing; **completion-only loss**; effective batch = 8 (CUDA `--batch 8 --grad-accum 2`;
MPS `--batch 2 --grad-accum 8`, but MPS is impractical for full runs — see §8). Val each epoch
(grouped-by-benchmark); early-stop if epoch-3 < epoch-2.

**Critical — no-think double-lock.** SmolLM3 is a hybrid think/no-think model. A guard **must** run
no-think, or it emits a `<think>` trace that overruns the 48-token generation cap in
`DecoderGuard.predict`, `parse_verdict` returns None, and it **fails closed to `unsafe`** (silent
over-block that wrecks FPR). Enforce both:
1. `apply_chat_template(enable_thinking=False)` (or append `/no_think` to the SmolLM3 system prompt) in
   `build_prompt`, used **identically at train and inference**.
2. SFT targets stay verdict-only (no `<think>`), so training actively teaches single-shot verdicts.

Needs `transformers>=4.53` (native SmolLM3) or `trust_remote_code=True` (thread via P2).

```bash
python scripts/train/run_training.py --model smollm3-3b --technique sft \
  --train-data data/train_sets/guard-main/train.jsonl \
  --epochs 3 --lr 2e-4 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --max-seq-len 1024 --bf16 --seed 42 --batch 8 --grad-accum 2      # A100. MPS: smoke only (--max-steps 60)
```

**RL/preference methods are ablation-only** (GRPO/DPO collapsed to FPR 0.77–0.99; root cause: `rewards.py`
`false_positive_penalty == correctness`). If run, set `false_positive_penalty≈2.5`, start from the SFT
checkpoint, GPU only, and report FPR recovery as a negative-result row. (SmolLM3's ~128k vocab risks the
Apple MPS INT_MAX limit in DPO — do RL on A100 only.)

**Technique progression (for the ablations / future work).** Because the reward here is **verifiable**
(the gold label *is* the reward), extend in the reward-model-free order and keep reward-model RL for last:
**SFT → GRPO → DPO → RLOO → ORPO → (maybe) KTO → [RewardTrainer + PPO only if going reward-model-based].**
RLOO is the critic-free sibling of GRPO (leave-one-out baseline); ORPO fuses preference into SFT with no
reference model; KTO fits our *unpaired* binary safe/unsafe labels; `RewardTrainer + PPO` add a learned
reward + value model that buy little over an exact reward, so they come last and only on purpose. Only
SFT/GRPO/DPO are wired today; the rest are thin TRL-trainer wrappers to add. Rationale + trainer map:
[`docs/fine-tuning.md`](fine-tuning.md#extending-beyond-sft--grpo--dpo--the-right-order-for-this-repo).
The SmolLM3 **headline guard stays SFT**; all RL/preference methods are research ablations, not the
shipped guard.

---

## 3. Single-token verdict + logprob head (`models/decoder.py`)

One forward pass → a decision **and** a calibrated score, on the no-think path.

- **Tokenization gate (verify online before training):** need `' safe'` / `' unsafe'` (leading space) to
  be single tokens for SmolLM3's tokenizer (SmolLM2 proxy: 2991 / 20408 single, but bare `unsafe` is
  multi-token → the leading-space variant is mandatory). `resolve_label_ids(tokenizer)` picks the
  single-token variant at load, else raises (fail loud). SmolLM3's Llama-3-style 128k vocab should give
  single tokens; the first-differing-subtoken fallback covers it if not (record as a caveat).
- **`_predict_verdict`:** `logits[:, -1, :]` → gather the two label logits →
  `p_unsafe = softmax([l_unsafe, l_safe]/temperature)[0]` → `score = p_unsafe`,
  `decision = UNSAFE if p_unsafe ≥ threshold`. `DecoderGuard` gains `mode='verdict'`, `threshold=0.5`,
  `temperature=1.0`. Fail-closed on any exception / non-finite logits.
- Because we read constrained first-token logits (never `.generate`), the score is valid **regardless of
  think habit** — but still force no-think for clean calibration and low latency.
- Keep the JSON/`parse_verdict` path untouched (existing tests + GRPO reward unaffected).
- **Unlocks:** calibration, iso-FPR thresholds, and the p50/p90 win (1 pass vs 48-token generation).

---

## 4. Data curriculum

Train-split inventory (deduped): BeaverTails `30k_train` 7,754 (2,066 safe / 5,688 unsafe); ToxicChat
train 4,964 (~4,590 / 374, **non-commercial**); deepset prompt-injections 546 (343/203);
jailbreak-classification 1,044 (~510/521); OR-Bench (capped) 4,000 all-safe; **JailbreakBench 200 held
out** (LOBO). Over-refusal negatives from **OR-Bench**, XSTest eval-only (already wired).

```bash
# primary paper set (over_refusal_aware over-weights the small red-team sources; OR-Bench aug on train only)
python scripts/data/build_dataset.py --strategy over_refusal_aware --name guard-main \
  --sources beavertails toxicchat prompt_injections jailbreak_classification --per-class 4000 --holdout 0.2 --seed 42
# license-clean released variant (no ToxicChat)
python scripts/data/build_dataset.py --strategy over_refusal_aware --name guard-main-rel \
  --sources beavertails prompt_injections jailbreak_classification --per-class 6000 --holdout 0.2 --seed 42
# isolated red-team → JailbreakBench transfer (LOBO)
python scripts/data/build_dataset.py --strategy red_team --name rt-lobo \
  --sources prompt_injections jailbreak_classification --per-class 2000 --holdout 0.2 --seed 42
```
`guard-main` ≈ 4,738 train (guardrail 55% / red-team 24% / over-refusal 21%, ~61% safe). Verify
`meta.json`: `augmentation_added>0`, `augmentation_source=='or_bench'`; and after P3 eval `dropped_leaked`
is near-zero for the three added sources. Check tokenized prompt lengths on SmolLM3 (bump seq-len only if
>2–3% exceed 1024 — some long jailbreaks do).

---

## 5. HPO + calibration + threshold protocol

**Bounded search (~6 runs):** fix dropout 0.05, α=2·r, effective batch 8, seed 42, max_seq_len 512 for
search. Sweep `r∈{16,32}` at epochs=2 (2 runs) → add winning-r at epochs=3 (1) → 1 LR refine
`{1e-4 or 3e-4}` on the winner → 2 seeds on the overall winner. **Selection (Gate G1):** score → DEV →
calibrate → iso-FPR threshold → argmax DEV macro-F1 s.t. DEV macro-FPR ≤ mini budget, on the *same*
pipeline used for TEST.

**Calibration** (`evaluation/calibration.py`, pure Python): fit temperature (golden-section) & Platt (GD)
on DEV logits; pick lower-ECE; accept only if post-cal ECE ≤ 0.10 & monotone. **Threshold:**
`choose_threshold_iso_fpr` on DEV via grouped 5-fold CV, refit on full DEV, freeze, apply once to disjoint
TEST; also report τ=0.5 and unconstrained-max-F1. **DEV/TEST** = 40/60 by `sample_key`, stratified within
benchmark, seed 42, cached to `outputs/eval_splits.json`.

---

## 6. Evaluation & statistics (matched-n)

- **Frozen sample set:** drop the SmolLM3 training file's leaked keys from **all** guards (baselines/GPT
  have no training file → the current mixed-n bug), score everyone on identical rows. Publish **Table 1**
  = per (guard×benchmark) `n / n_safe / n_unsafe` after filtering.
- **Size policy:** full (`--per-class 0`) for small sets (XSTest 450, JailbreakBench 200,
  prompt_injections ~116, jailbreak_classification); cap the three big content sets at `per_class=800`.
- **Metrics:** P/R/F1 (macro + per-axis equal-weight); FPR@benign secondary.
- **Statistics:** `paired_bootstrap_ci` (B=10000, strata=benchmark) + `mcnemar` + Holm across (axis×pair);
  CI on every SmolLM3−mini difference; apply the pre-registered rule (§0).
- **Baselines:** keyword; Llama-Guard-3-1B, ShieldGemma-2b, PromptGuard-2-86M (`baselines.py`);
  WildGuard-7B (add wrapper or "not run"); OpenAI-Moderation, GPT-4o-mini, GPT-5-mini (**dated snapshots**).
  Llama-Guard-3-1B / ShieldGemma are the most honest *peers*; GPT minis are the aspirational bar.
- **Latency/\$:** measured **separately** on the pinned Mac over a fixed 300-sample micro-bench; the
  **single-token-head SmolLM3** carries the latency claim. \$/1k from token usage × dated price.

---

## 7. Ablation table (single model; every row matched-n, per-axis F1 + CI vs nearest mini)

| Row | System | Isolates | Gate |
|---|---|---|---|
| R0 | Field baselines (keyword, Llama-Guard-3-1B, ShieldGemma-2b, PromptGuard-2-86M, WildGuard?, moderation, gpt-4o-mini, gpt-5-mini) | the bar | — |
| R1 | base: 1 epoch, r16, content-only | under-trained floor (≈0.68) | — |
| R2 | +epochs/rank: 3 ep, r32/α64 | **the under-training fix** | macro-F1 0.68→≥0.78 |
| R3 | +red-team (LOBO, hold out JailbreakBench) | red-team lift + transfer | red-team F1 0.51→≥0.70; JBB up; no guardrail regression |
| R4 | +over-refusal (OR-Bench) — **headline, ×3 seeds** | FPR/precision | XSTest FPR −≥10 pts; macro-F1 not down > CI |
| R5 | +verdict head (no retrain) | latency/\$ | p50 −5–40×; F1 within CI of R4 |
| R6 | +calibration + iso-FPR threshold | operating point | dev-selected τ, reported on test |
| R7 | +distillation (independent teacher — GPT-5.x-high or open-guard ensemble, **never** the mini baselines; gold-only control) | distillation delta | marginal F1 vs R4 |
| R8/R9 | +DPO / +GRPO (from SFT, FP-penalty≈2.5×) | RL negative result | FPR ≤ 0.2 at held recall, else documented degeneracy |

*(No two-model cascade row — single model.)*

---

## 8. Run schedule, compute budget & critical path

**Phase 0 (infra, no GPU):** `stats.py` + tests; `_align_rows` raise; per-axis macro; token-usage/\$
capture; (optional) WildGuard wrapper. **Gate G0:** `make test` green + a hand-checked CI toy case.
**Phase 1 (first submittable number):** build `guard-main`; train **R4** on A100; matched-n full eval of
R4 + all R0 baselines + GPT judges with prediction dumps; Mac latency micro-bench; stats CIs + Table 1.
**Gate G1 = the honest, leakage-audited, CI-bearing P/R/F1-vs-cost table.**
**Phase 2 (efficiency):** verdict head (R5) → calibration (R6). **Gate G2:** SmolLM3 non-inferior to
GPT-5-mini on F1 at p50 ≤ mini/3.
**Phase 3 (sharpening, parallel):** red-team LOBO isolation (R3), 3-seed spread, distillation (R7), RL
negatives (R8/R9).

**Compute budget** (single model): SmolLM3-3B ≈ 10 s/step MPS (~7 h/run — **don't train on MPS**),
~3 s/step A100 (~60–100 min/run). Grid ≈ 6 search + 3 seeds + a couple ablations ≈ 8–12 runs ≈ **10–18
GPU-hours ≈ \$20–35**; eval + API < \$20 → **~\$40–55 total** (roughly half the two-model budget).
**Where:** training + quality/prediction eval on a rented A100 (~1 day); **latency p50/p90 on the pinned
Mac** (the "runs on a laptop" story needs the reference device); offline stats/analysis over cached
`outputs/predictions/*.json` anywhere.

**Critical path:** `G0 (stats.py + align-raise + per-axis macro) → train R4 (fixed recipe) on A100 →
matched-n eval of R4 + open-guard baselines + GPT judges (dump predictions) → Mac latency + $/1k →
bootstrap CIs + Table 1.`

---

## 9. Top risks & gates

- **Mixed-n comparison (FATAL, current bug):** `_align_rows` RAISES; Table 1 receipt; matched-n assertion.
- **F1 parity may not hold** (SmolLM3 ~0.78 vs gpt-4o-mini 0.83). → pre-registered non-inferiority CI; if
  it straddles, report "inconclusive" and fall back to the frontier/measurement framing (dominate
  moderation + match GPT-5-mini on over-blocking/latency/cost).
- **SmolLM3 think-leak** → 48-token truncation → fail-closed over-block. → no-think double-lock; DEV probe
  requires `<think>` in <1% of outputs before any eval; verify `trust_remote_code`/`transformers>=4.53`
  at first load.
- **Completion-only loss silently not applied** → F1 stays ~0.68. → log non-`-100` label-token count.
- **P3 loader fix omitted** → training on eval rows. → eval `dropped_leaked` near-zero for the three sources.
- **Verdict-mode F1 regresses vs JSON SFT** → target/masking/tokenization bug. → gate: verdict-mode
  macro-F1 ≥ JSON-mode SFT.
- **Latency claim rides the wrong config** → the F1 and latency claims must be the **same** SmolLM3
  configuration, measured on the pinned Mac.
- **Red-team may not lift** → invoke the 2-phase curriculum and/or add gated red-team data
  (WildJailbreak/StrongREJECT).
- **ToxicChat non-commercial** → any released checkpoint trains on `guard-main-rel`.
- **Re-freeze the mini FPR budget** from the matched-n run before the final iso-FPR TEST report.

## 10. Note on the dropped model

Qwen2.5-1.5B is out of scope per this decision but remains in the registry (`qwen2.5-1.5b`) as an option.
If you later want the stronger "**1.5B** competes with mini" headline or the diversity/cascade story back,
re-adding it is a one-line focus change — the recipe, data, and eval here transfer unchanged.
