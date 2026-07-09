# Hardened mortgage benchmark — results

**Dataset (what the paper analyzes):** `notebooks/data/benchmarks/full/guard_benchmark_hard.jsonl` — **334 trap-typed items** (195 unsafe / 139 safe; difficulty very-hard 284 / borderline 50; loan-officer and consumer personas). Fields: `id, text, persona, label_binary, label_category, label_subcategory, severity, difficulty, trap_type, rationale`. No train/dev split, minimal-pair ids, or protected-class tags.

Trap types: hard_negative 75, minimal_pair 60, euphemism 52, business_justified 45, buried_injection 39, over_refusal_bait 20, dual_use 15, multi_turn 15, coded_proxy 13. Categories: safe 139, security_misuse 89, fair_lending 65, compliance_bad_advice 41.

**Scorer:** `scripts/eval_mortgage_hard.py` (run from `research/`). Because the set has no dev split, it reports operating-point-**independent** AUPRC and Optimal-F1 as the fair primary, plus accuracy/recall/precision/FPR at each system's own best-F1 point (gpt-5.4-mini at its native point). These operating points are **not matched** across systems — see the caveat.

## Results

| system | AUPRC [95% CI] | Opt-F1 | accuracy | recall | precision | FPR |
|--------|---------------:|-------:|---------:|-------:|----------:|----:|
| base (zero-shot SmolLM3-3B) | 0.892 [0.854, 0.925] | 0.819 | 0.754 | 0.954 | 0.718 | 0.525 |
| mortgage-SFT (in-domain LoRA) | 0.924 [0.849, 0.946] | 0.885 | 0.856 | 0.949 | 0.830 | 0.273 |
| general-safety guard | 0.824 [0.774, 0.873] | 0.756 | 0.650 | 0.928 | 0.637 | 0.741 |
| gpt-5.4-mini (frontier ref) | — | — | 0.949 | 0.959 | 0.954 | 0.065 |

## Reading (matches paper §Case study / Table 13)
- **De-saturates:** the saturated legacy set left every system near-ceiling; here AUPRC spreads 0.82→0.92, accuracy 0.65→0.95.
- **Over-blocking, not misses:** all small guards keep high recall (~0.93–0.95) but run high FPR on the hard negatives (0.27–0.74); gpt-5.4-mini sits at FPR 0.065. This is at *unmatched* operating points (each local guard at its own best-F1, gpt at native), so it is *suggestive* of a capability gap, not a controlled matched-FPR comparison.
- **Benchmark-dependent winner (on threshold-free AUPRC):** in-domain mortgage-SFT is highest (0.924; gap to base within overlapping CIs), the general-safety guard clearly lowest (0.824) — the opposite of the general-OOD result where the untuned base leads.

## Caveats
- Seed-scale (334 items), single-annotator; the discriminating signal is AUPRC/FPR, not recall (recall is uniformly high). Magnitudes will move under broader adjudication. The paper frames this as an **exploratory** case study, not a headline result.
- Operating points are not matched across systems (see above).

## Superseded history
An earlier, **synthetic** hardened set (318 rows: 288 minimal-pair items across 144 pairs + 30 wrapper variants, jury-gated) was built by `scripts/wf_build_hard_benchmark_v2.mjs` → `scripts/build_hard_jsonl.mjs` and scored with a dev-split Recall@FPR / minimal-pair / guard-fairness protocol. It is **superseded** by the trap-typed 334-row set above and is **not** what the paper analyzes; those builder scripts do not regenerate the committed file.
