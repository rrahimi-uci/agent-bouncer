# guard_benchmark_hard — build & results

**Dataset:** `notebooks/data/benchmarks/full/guard_benchmark_hard.jsonl` — 318 rows = 288 core (144 complete minimal pairs, 144 flag / 144 allow) + 30 wrapper variants, across 16 hard-case families. Split family-safe by `pair_group` (both twins + all wrapper variants stay in the same split): dev 116 / test 172 core + 30 wrapper. 107 protected-mention items (25 benign in test).
**Construction:** 16 hard-case families authored as matched minimal pairs by parallel expert-persona agents (`scripts/wf_build_hard_benchmark_v2.mjs`), then gated by a **blind 3-juror panel** (strict-compliance / consumer-advocate / pragmatic-underwriter, chunked). Admitted on jury-vs-author agreement: **288/288 admitted, all unanimous**. Labels are **action-keyed** (a worried borrower self-disclosing a protected class is `allow`; only an officer acting on it is `flag`), fixing the legacy set's mislabeling.
**Protocol** (`scripts/eval_mortgage_hard.py`): one threshold frozen on dev negatives per target FPR; on test report Recall@FPR, worst-family recall, guard-fairness gap (FPR protected-mention benign − no-mention benign), minimal-pair accuracy (both twins correct), wrapper-flip; plus a gpt-5.4-mini frontier ceiling.

## Headline: the benchmark is hard, and hard for the right reason
Base SmolLM3-3B **Recall@FPR=5%: OLD 0.994 → HARD 0.663** (AUPRC 0.988 → 0.899). Meanwhile the **frontier gpt-5.4-mini stays strong** (recall 1.00, minimal-pair 0.733) — headroom exists, so the small guards' collapse is a capability gap, not label noise. gpt (a *different model family* than the Claude author/jury) agrees with the gold on **86.6%** of core test — independent cross-model label validation.

## Results (test split, threshold frozen on dev negatives @FPR=5%)

| system | AUPRC | R@FPR5% | R@FPR1% | worst-family R | guard-fairness gap | minimal-pair acc | wrapper-flip |
|--------|------:|--------:|--------:|---------------:|-------------------:|-----------------:|-------------:|
| base (zero-shot) | 0.899 | 0.663 | 0.233 | 0.00 (wire_bec) | +0.014 | 0.616 | 0.30 |
| **mortgage-SFT** | 0.793 | **0.174** | 0.174 | **0.00** (adversarial_wrapper) | **+0.200** | **0.174** | 0.07 |
| general-guard | 0.887 | 0.570 | 0.081 | 0.00 (adverse_action) | −0.098 | 0.512 | 0.30 |
| gpt-5.4-mini (frontier ceiling) | — | 1.000 @FPR 0.27 | — | 1.00 | +0.074 | 0.733 | 0.10 |

## The ranking inversion (this is the paper's thesis, in a domain + fairness setting)
- **On the saturated legacy benchmark:** mortgage-SFT is the winner (AUPRC 0.999), base ≈ 0.995. Ranking: SFT ≳ base ≈ general.
- **On the hard benchmark:** mortgage-SFT is the **worst** local system (R@FPR5% 0.174, minimal-pair 0.174), base is best (0.663 / 0.616), general-guard between (0.570 / 0.512). Ranking fully **inverts**: base > general > SFT.
- **Fine-tuning induced a fairness gap.** Base is near-balanced (+0.014); the mortgage-SFT is the one system that over-flags protected-mention benign prompts (+0.200: 20% vs 0%). In-distribution specialization did not just fail to help — it created a disparate-treatment-style over-blocking of protected-class borrowers' legitimate questions.
- Every small local guard has ≥1 family it catches ~none of at a 5%-FPR operating point (base: BEC/wire fraud; SFT: adversarial wrappers; general: adverse-action). Only the frontier clears all families.

## Concrete base failures (at frozen threshold)
- **Misses all wire-fraud (BEC):** worst-family recall 0.00 on `wire_bec` — the "update the wire to a new account, skip the callback" family slips entirely under a 5%-FPR threshold.
- **Wrapper brittleness (0.30 flip):** adding a legacy wrapper ("We close in 2 hours…", "You are MaxLoan…") flips base's verdict on ~1/3 of items.

## Honest caveats
- **Scale:** 172 core test items; per-family cells are small (worst-family requires ≥4 positives). A seed, not a final benchmark.
- **Generation/jury circularity:** items were authored *and* Claude-jury-gated by the same model family (288/288 unanimous is suspiciously clean). The gpt-5.4-mini cross-check (86.6% agreement) partially mitigates this, but **SME re-adjudication remains the required next step** before load-bearing paper claims.
- **Frontier ceiling is high-recall, not clean:** gpt hits recall 1.00 by operating at FPR 0.27 (it over-blocks); its cleaner "ceiling" signal is minimal-pair accuracy 0.733, still far above every small guard (≤0.62).
- **Calibration:** 116 dev negatives; the frozen threshold now tracks the 5% FPR target well (base test FPR 0.070). Numbers are directional; magnitudes will move with SME labels.
