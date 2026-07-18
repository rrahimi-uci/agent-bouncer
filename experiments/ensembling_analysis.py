#!/usr/bin/env python3
"""Retrospective ENSEMBLING analysis on committed guard scores.

Question: ensembling is a classic way to make a classifier generalize. Does it
help the *fine-tuned* guards (SFT, KL-SFT) recover the transfer they lose?

We reuse the confirmatory adaptation study's committed per-row margins
(score_raw = unsafe_logit - safe_logit), which the paper ranks on. All regimes
of a given checkpoint score the *same* rows, and within a checkpoint the base
and its adapters share one output head, so raw margins are scale-comparable and
directly averageable. Across checkpoints margins are NOT comparable, so the
cross-model committee uses rank-percentile normalization per (checkpoint, source).

Every metric is macro-AP (mean average-precision over benchmark sources) on the
represented (id_test) and transfer (transfer_test) splits, anchored to each
checkpoint's OWN unmodified base -- identical estimand to the paper.
"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

COMBINED = "artifacts/starting_type_adaptation_v1/scores/combined.parquet"
REP, TRANS = "id_test", "transfer_test"
STRESS = ["stress_harmbench", "stress_orbench"]

FAMILIES = {
    "qwen":    ["qwen25_15b", "qwen3_4b", "qwen3guard_gen_06b", "qwen3guard_gen_4b"],
    "smollm":  ["smollm2_17b", "smollm3_3b"],
    "gemma":   ["shieldgemma_2b"],
    "granite": ["granite_guardian_31_2b"],
    "llama":   ["llama_guard_3_1b"],
    "mistral": ["wildguard_7b"],
}
CK2FAM = {ck: fam for fam, cks in FAMILIES.items() for ck in cks}
GENERAL = ["qwen25_15b", "qwen3_4b", "smollm2_17b", "smollm3_3b"]
NULLCK = "llama_guard_3_1b"  # 20-token pruned tied head; LoRA cannot move it

df = pd.read_parquet(COMBINED, columns=[
    "sample_id", "source", "split", "gold", "starting_model_key",
    "starting_type", "adaptation", "seed", "score_raw"])
CKS = sorted(df.starting_model_key.unique())


def macro_ap(frame, score_col):
    """Mean AP over sources within one split frame (needs gold, source)."""
    aps = []
    for _, g in frame.groupby("source"):
        if g.gold.nunique() > 1:
            aps.append(average_precision_score(g.gold.values, g[score_col].values))
    return float(np.mean(aps)) if aps else np.nan


def aligned(d, split):
    """Per (checkpoint, split): base + seed-mean SFT/KL margins + per-seed matrices."""
    x = d[d.split == split]
    base = (x[x.adaptation == "unmodified"][["sample_id", "source", "gold", "score_raw"]]
            .rename(columns={"score_raw": "base"}).set_index("sample_id"))
    sft = x[x.adaptation == "sft"].pivot_table(index="sample_id", columns="seed", values="score_raw")
    kl = x[x.adaptation == "kl_sft"].pivot_table(index="sample_id", columns="seed", values="score_raw")
    m = base.copy()
    m["sft_sm"] = sft.mean(axis=1)     # seed-ensemble of SFT
    m["kl_sm"] = kl.mean(axis=1)       # seed-ensemble of KL-SFT
    # within-checkpoint ensembles (equal-weight margin averaging)
    m["base_sft"] = 0.5 * m.base + 0.5 * m.sft_sm            # = Act II composition
    m["base_kl"] = 0.5 * m.base + 0.5 * m.kl_sm
    m["sft_kl"] = 0.5 * m.sft_sm + 0.5 * m.kl_sm
    m["base_sft_kl"] = (m.base + m.sft_sm + m.kl_sm) / 3.0
    return m, sft, kl


def single_ap(m, seedmat):
    """Mean over seeds of the per-seed macro-AP (a typical single fine-tune)."""
    aps = []
    for s in seedmat.columns:
        f = m[["source", "gold"]].join(seedmat[s].rename("sc"), how="inner")
        aps.append(macro_ap(f.rename(columns={"sc": "x"}), "x"))
    return float(np.mean(aps))


METHODS = ["base", "sft_single", "kl_single", "sft_seedens", "kl_seedens",
           "base_sft", "base_kl", "sft_kl", "base_sft_kl"]

rows = []
for ck in CKS:
    d = df[df.starting_model_key == ck]
    rec = {"ck": ck, "family": CK2FAM[ck], "type": d.starting_type.iloc[0]}
    for split, tag in [(REP, "rep"), (TRANS, "trans")]:
        m, sft, kl = aligned(d, split)
        rec[f"base_{tag}"] = macro_ap(m, "base")
        rec[f"sft_single_{tag}"] = single_ap(m, sft)
        rec[f"kl_single_{tag}"] = single_ap(m, kl)
        for col in ["sft_sm", "kl_sm", "base_sft", "base_kl", "sft_kl", "base_sft_kl"]:
            nm = {"sft_sm": "sft_seedens", "kl_sm": "kl_seedens"}.get(col, col)
            rec[f"{nm}_{tag}"] = macro_ap(m, col)
    rows.append(rec)
R = pd.DataFrame(rows)


def eqfam_mean(frame, col):
    """Equal-family mean: average within family, then across families."""
    return frame.groupby("family")[col].mean().mean()


def summarize(frame, label):
    print(f"\n{'='*92}\n{label}   (n_checkpoints={len(frame)}, n_families={frame.family.nunique()})\n{'='*92}")
    print(f"{'method':<14} {'rep_AP':>8} {'Δrep':>8} | {'trans_AP':>9} {'Δtrans':>8}   note")
    base_rep = eqfam_mean(frame, "base_rep"); base_tr = eqfam_mean(frame, "base_trans")
    print(f"{'base':<14} {base_rep:>8.4f} {0.0:>8.4f} | {base_tr:>9.4f} {0.0:>8.4f}   reference")
    notes = {
        "sft_single": "typical single SFT",
        "sft_seedens": "avg 5 SFT seeds",
        "kl_single": "typical single KL-SFT",
        "kl_seedens": "avg 5 KL-SFT seeds",
        "base_sft": "= Act II composition",
        "base_kl": "base + KL-SFT",
        "sft_kl": "SFT + KL-SFT",
        "base_sft_kl": "base+SFT+KL-SFT (3)",
    }
    for meth in METHODS[1:]:
        r = eqfam_mean(frame, f"{meth}_rep"); t = eqfam_mean(frame, f"{meth}_trans")
        free = "  <-- FREE LUNCH" if (r - base_rep) > -0.002 and (t - base_tr) > 0.002 else ""
        print(f"{meth:<14} {r:>8.4f} {r-base_rep:>+8.4f} | {t:>9.4f} {t-base_tr:>+8.4f}   {notes[meth]}{free}")


summarize(R, "ALL 10 CHECKPOINTS (equal-family mean)")
summarize(R[R.ck != NULLCK], "EXCL. degenerate llama_guard_3_1b null cell")
summarize(R[R.ck.isin(GENERAL)], "GENERAL checkpoints only (Act I panel)")
summarize(R[R.type == "purpose_built"], "PURPOSE-BUILT released guards only")

# ---- headline answers to the user's question ----
def em(frame, c): return eqfam_mean(frame, c)
F = R[R.ck != NULLCK]  # drop null cell for the honest headline
print(f"\n{'#'*92}\nHEADLINE (equal-family, excl. null cell)\n{'#'*92}")
print(f"Seed-ensembling SFT   : Δtrans {em(F,'sft_seedens_trans')-em(F,'sft_single_trans'):+.4f} vs single SFT ; "
      f"Δrep {em(F,'sft_seedens_rep')-em(F,'sft_single_rep'):+.4f}")
print(f"Seed-ensembling KL-SFT: Δtrans {em(F,'kl_seedens_trans')-em(F,'kl_single_trans'):+.4f} vs single KL ; "
      f"Δrep {em(F,'kl_seedens_rep')-em(F,'kl_single_rep'):+.4f}")
print(f"base+SFT vs SFT       : Δtrans {em(F,'base_sft_trans')-em(F,'sft_seedens_trans'):+.4f} ; "
      f"Δrep {em(F,'base_sft_rep')-em(F,'sft_seedens_rep'):+.4f}  (recover transfer at rep cost?)")
print(f"base+SFT vs base      : Δtrans {em(F,'base_sft_trans')-em(F,'base_trans'):+.4f} ; "
      f"Δrep {em(F,'base_sft_rep')-em(F,'base_rep'):+.4f}")

R.to_csv("artifacts/starting_type_adaptation_v1/analysis/ensembling_percheckpoint.csv", index=False)
print("\nwrote per-checkpoint CSV.")

# ---- committed JSON (auditable; macros bind to this, never hand-typed) ----
import json


def panel_json(frame):
    d = {"n_ck": int(len(frame)), "n_fam": int(frame.family.nunique())}
    d["base"] = {"rep": eqfam_mean(frame, "base_rep"), "trans": eqfam_mean(frame, "base_trans")}
    d["methods"] = {}
    for meth in METHODS[1:]:
        r = eqfam_mean(frame, f"{meth}_rep"); t = eqfam_mean(frame, f"{meth}_trans")
        d["methods"][meth] = {"rep": r, "trans": t,
                              "drep": r - d["base"]["rep"], "dtrans": t - d["base"]["trans"]}
    return d


out = {
    "estimand": ("macro-AP on raw logit margin (unsafe-safe); equal-family mean; anchored to each "
                 "checkpoint's own unmodified base; represented=id_test, transfer=transfer_test; "
                 "within-checkpoint ensembles average raw margins (scale-comparable, shared head)"),
    "source_scores": COMBINED,
    "panels": {
        "all10": panel_json(R),
        "excl_null": panel_json(R[R.ck != NULLCK]),
        "general": panel_json(R[R.ck.isin(GENERAL)]),
        "purpose_built": panel_json(R[R.type == "purpose_built"]),
    },
}
json.dump(out, open("artifacts/starting_type_adaptation_v1/analysis/ensembling_point.json", "w"), indent=2)
print("wrote ensembling_point.json")
