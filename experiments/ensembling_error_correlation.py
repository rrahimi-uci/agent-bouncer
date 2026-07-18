#!/usr/bin/env python3
"""Direct measurement of the composition diversity mechanism (addresses the reviewer
point that it was asserted via an AP-midpoint heuristic, not measured).

For each checkpoint, on held-out transfer rows, we form a per-row standardized
"wrongness" signal  w = -(2*gold-1) * zscore(score_raw)  (high = ranked wrong),
and measure the Pearson correlation of wrongness between members:
  - base  vs its SFT adapter   (diverse pair: un-tuned vs tuned)
  - SFT seed_i vs SFT seed_j   (redundant pair: same recipe, different seed)

An ensemble cancels error only when members' errors are decorrelated, so the
base+SFT recovery should coincide with base-vs-SFT correlation being LOWER than
SFT-vs-SFT correlation. Equal-family mean over the same panel as the rest of the
ensembling appendix (excl. the degenerate Llama-Guard null cell).

Writes artifacts/starting_type_adaptation_v1/analysis/ensembling_error_correlation.json.
"""
import itertools, json
import numpy as np, pandas as pd

COMBINED = "artifacts/starting_type_adaptation_v1/scores/combined.parquet"
OUT = "artifacts/starting_type_adaptation_v1/analysis/ensembling_error_correlation.json"
FAMILIES = {
    "qwen": ["qwen25_15b", "qwen3_4b", "qwen3guard_gen_06b", "qwen3guard_gen_4b"],
    "smollm": ["smollm2_17b", "smollm3_3b"], "gemma": ["shieldgemma_2b"],
    "granite": ["granite_guardian_31_2b"], "llama": ["llama_guard_3_1b"],
    "mistral": ["wildguard_7b"]}
CK2FAM = {c: f for f, cs in FAMILIES.items() for c in cs}
NULLCK = "llama_guard_3_1b"

df = pd.read_parquet(COMBINED, columns=[
    "sample_id", "source", "split", "gold", "starting_model_key", "adaptation", "seed", "score_raw"])
CKS = [c for c in sorted(df.starting_model_key.unique()) if c != NULLCK]


def wrongness(frame):
    z = (frame.score_raw - frame.score_raw.mean()) / (frame.score_raw.std() + 1e-9)
    return (-(2 * frame.gold - 1) * z).values


rows = []
for ck in CKS:
    d = df[(df.starting_model_key == ck) & (df.split == "transfer_test")]
    base = d[d.adaptation == "unmodified"].set_index("sample_id").sort_index()
    seeds = sorted(d[d.adaptation == "sft"].seed.unique())
    sft = {s: d[(d.adaptation == "sft") & (d.seed == s)].set_index("sample_id").sort_index() for s in seeds}
    idx = base.index
    wb = wrongness(base)
    ws = {s: wrongness(sft[s].loc[idx]) for s in seeds}
    r_bs = float(np.mean([np.corrcoef(wb, ws[s])[0, 1] for s in seeds]))
    r_ss = float(np.mean([np.corrcoef(ws[i], ws[j])[0, 1] for i, j in itertools.combinations(seeds, 2)]))
    rows.append({"ck": ck, "family": CK2FAM[ck], "r_base_sft": r_bs, "r_sft_sft": r_ss})

R = pd.DataFrame(rows)
efm = lambda c: float(R.groupby("family")[c].mean().mean())
out = {
    "estimand": ("Pearson correlation of per-row standardized wrongness on transfer_test; "
                 "equal-family mean, excl. Llama-Guard null cell"),
    "source_scores": COMBINED,
    "r_base_sft": efm("r_base_sft"),   # diverse pair (un-tuned vs tuned)
    "r_sft_sft": efm("r_sft_sft"),     # redundant pair (seed vs seed)
    "per_checkpoint": rows,
}
json.dump(out, open(OUT, "w"), indent=2)
print(f"corr(base err, SFT err) = {out['r_base_sft']:.3f}")
print(f"corr(SFT err, SFT err)  = {out['r_sft_sft']:.3f}")
print(f"wrote {OUT}")
