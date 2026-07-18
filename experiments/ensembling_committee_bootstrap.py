#!/usr/bin/env python3
"""Part 2 of the ensembling analysis: (A) cross-model COMMITTEE ensembles, and
(B) family-clustered bootstrap CIs on the headline within-checkpoint deltas.

Committee = the strongest "ensemble to generalize" form: combine DIFFERENT guards
on the same row. Cross-model margins are not scale-comparable, so each guard's
score is rank-normalized to a percentile per (checkpoint, split, source), then
percentiles are averaged across guards -> committee score -> macro-AP.

Bootstrap mirrors the paper: Poisson(1) weight per near-duplicate `family_id`
cluster; models held fixed; seeds held at their mean (the seed-ensemble point
result shows seed variance contributes only ~0.02). 2.5/97.5 percentiles.

Writes artifacts/starting_type_adaptation_v1/analysis/ensembling_committee_bootstrap.json.
"""
import json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

COMBINED = "artifacts/starting_type_adaptation_v1/scores/combined.parquet"
OUT = "artifacts/starting_type_adaptation_v1/analysis/ensembling_committee_bootstrap.json"
REP, TRANS = "id_test", "transfer_test"
FAMILIES = {
    "qwen": ["qwen25_15b", "qwen3_4b", "qwen3guard_gen_06b", "qwen3guard_gen_4b"],
    "smollm": ["smollm2_17b", "smollm3_3b"], "gemma": ["shieldgemma_2b"],
    "granite": ["granite_guardian_31_2b"], "llama": ["llama_guard_3_1b"],
    "mistral": ["wildguard_7b"]}
CK2FAM = {ck: f for f, cks in FAMILIES.items() for ck in cks}
GENERAL = ["qwen25_15b", "qwen3_4b", "smollm2_17b", "smollm3_3b"]
NULLCK = "llama_guard_3_1b"
REPS, BOOT_SEED = 1500, 20260716

df = pd.read_parquet(COMBINED, columns=[
    "sample_id", "source", "split", "gold", "starting_model_key",
    "adaptation", "seed", "score_raw", "family_id"])
ALL10 = sorted(df.starting_model_key.unique())


def seedmean(ck, adap, split):
    x = df[(df.starting_model_key == ck) & (df.adaptation == adap) & (df.split == split)]
    if adap == "unmodified":
        return x[["sample_id", "source", "gold", "score_raw", "family_id"]].set_index("sample_id")
    piv = x.pivot_table(index="sample_id", columns="seed", values="score_raw")
    meta = x.groupby("sample_id")[["source", "gold", "family_id"]].first()
    meta["score_raw"] = piv.mean(axis=1)
    return meta


def macro_ap(frame, col="score_raw"):
    aps = [average_precision_score(g.gold.values, g[col].values)
           for _, g in frame.groupby("source") if g.gold.nunique() > 1]
    return float(np.mean(aps)) if aps else float("nan")


# ---------- (A) COMMITTEE ----------
def committee(cks, adaps, split):
    """Rank-percentile average across guards `cks` over regimes `adaps`."""
    parts = []
    for ck in cks:
        for adap in adaps:
            m = seedmean(ck, adap, split).copy()
            m["pct"] = m.groupby("source")["score_raw"].rank(pct=True)
            parts.append(m[["source", "gold", "pct"]].reset_index())
    com = pd.concat(parts).groupby(["sample_id", "source", "gold"])["pct"].mean().reset_index()
    return macro_ap(com.rename(columns={"pct": "score_raw"}))


def best_single(cks, adap, split):
    return max(macro_ap(seedmean(ck, adap, split)) for ck in cks)


committee_json = {}
print("=" * 80 + "\nCROSS-MODEL COMMITTEE (rank-percentile average)\n" + "=" * 80)
for name, cks in [("general", GENERAL), ("all10", ALL10)]:
    committee_json[name] = {}
    print(f"\n-- committee over {name} ({len(cks)} guards) --")
    for adap, lab in [("unmodified", "base"), ("sft", "SFT"), ("kl_sft", "KL-SFT")]:
        cr, ctr = committee(cks, [adap], REP), committee(cks, [adap], TRANS)
        bs = best_single(cks, adap, TRANS)
        committee_json[name][adap] = {"rep": cr, "trans": ctr,
                                      "best_single_trans": bs, "vs_best_single_trans": ctr - bs}
        print(f"committee-of-{lab:<7} rep={cr:.4f} trans={ctr:.4f}  best_single={bs:.4f} Δ={ctr-bs:+.4f}")
    dv = committee(cks, ["unmodified", "sft"], TRANS)
    committee_json[name]["diverse_bases_plus_sfts_trans"] = dv
    print(f"diverse committee (bases+SFTs) trans={dv:.4f}")


# ---------- (B) BOOTSTRAP (excl-null, equal-family) ----------
CKS = [c for c in ALL10 if c != NULLCK]
cache = {}
for ck in CKS:
    for split in (REP, TRANS):
        base = seedmean(ck, "unmodified", split)
        sft = df[(df.starting_model_key == ck) & (df.adaptation == "sft") & (df.split == split)] \
            .pivot_table(index="sample_id", columns="seed", values="score_raw")
        idx = base.index.intersection(sft.index)
        base, sft = base.loc[idx], sft.loc[idx]
        cache[(ck, split)] = {
            "source": base.source.values, "gold": base.gold.values.astype(int),
            "fam": base.family_id.values, "base": base.score_raw.values,
            "sft_seeds": [sft[c].values for c in sft.columns], "sft_sm": sft.mean(axis=1).values}

all_fams = sorted({f for ck in CKS for f in cache[(ck, REP)]["fam"]} |
                  {f for ck in CKS for f in cache[(ck, TRANS)]["fam"]})
fam_pos = {f: i for i, f in enumerate(all_fams)}


def wm(entry, score, wmap):
    w = np.array([wmap[f] for f in entry["fam"]])
    src, y = entry["source"], entry["gold"]
    aps = []
    for u in np.unique(src):
        mm = src == u
        if len(np.unique(y[mm])) > 1:
            aps.append(average_precision_score(y[mm], score[mm], sample_weight=w[mm]))
    return np.mean(aps) if aps else np.nan


def efm(perck, key):
    byfam = {}
    for ck in CKS:
        byfam.setdefault(CK2FAM[ck], []).append(perck[ck][key])
    return np.mean([np.mean(v) for v in byfam.values()])


rng = np.random.default_rng(BOOT_SEED)
draws = {k: [] for k in ["seedens_gain_trans", "basesft_vs_base_trans",
                         "basesft_vs_sft_trans", "basesft_vs_sft_rep"]}
for _ in range(REPS):
    wv = rng.poisson(1.0, size=len(all_fams)).astype(float)
    wmap = {f: wv[fam_pos[f]] for f in all_fams}
    perck = {}
    for ck in CKS:
        rec = {}
        for split, tag in ((REP, "rep"), (TRANS, "trans")):
            e = cache[(ck, split)]
            rec[("base", tag)] = wm(e, e["base"], wmap)
            rec[("sft_sm", tag)] = wm(e, e["sft_sm"], wmap)
            rec[("base_sft", tag)] = wm(e, 0.5 * e["base"] + 0.5 * e["sft_sm"], wmap)
            rec[("sft_single", tag)] = np.mean([wm(e, s, wmap) for s in e["sft_seeds"]])
        perck[ck] = rec
    draws["seedens_gain_trans"].append(efm(perck, ("sft_sm", "trans")) - efm(perck, ("sft_single", "trans")))
    draws["basesft_vs_base_trans"].append(efm(perck, ("base_sft", "trans")) - efm(perck, ("base", "trans")))
    draws["basesft_vs_sft_trans"].append(efm(perck, ("base_sft", "trans")) - efm(perck, ("sft_sm", "trans")))
    draws["basesft_vs_sft_rep"].append(efm(perck, ("base_sft", "rep")) - efm(perck, ("sft_sm", "rep")))

boot_json = {"reps": REPS, "rng_seed": BOOT_SEED, "panel": "excl_null", "deltas": {}}
print("\n" + "=" * 80 + f"\nBOOTSTRAP (Poisson per family_id; equal-family; excl null; reps={REPS})\n" + "=" * 80)
print(f"{'delta':<26}{'mean':>9}{'lcb2.5':>9}{'ucb97.5':>9}   verdict")
for k, v in draws.items():
    v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5])
    verdict = "sig>0" if lo > 0 else ("sig<0" if hi < 0 else "n.s.")
    boot_json["deltas"][k] = {"mean": float(v.mean()), "lcb": float(lo), "ucb": float(hi), "verdict": verdict}
    print(f"{k:<26}{v.mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}   {verdict}")

json.dump({"committee": committee_json, "bootstrap": boot_json,
           "source_scores": COMBINED}, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
