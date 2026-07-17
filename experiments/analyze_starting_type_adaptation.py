#!/usr/bin/env python
"""Non-HARKing analyzer for the starting-type adaptation study (proposal Sec 3/6/9/10).

This is the confirmatory analyzer that replaces the post-hoc-verdict defect of
``experiments/analyze_klsft.py`` (which auto-selects a "best" beta and derives verdict
wording after seeing the point estimates). Here EVERY interpretation string is selected
ONLY by locked, bootstrap-bound predicate outcomes via a claim-registry mapping; no branch
ever inspects the sign of a point estimate.

Reads the starting-type score parquet emitted by ``eval_starting_type_adaptation.py``
(columns include: ``starting_model_key``, ``starting_type``, ``adaptation`` in
{unmodified, sft, kl_sft}, ``seed``, ``kl_beta``, ``source``, ``split``, ``gold``,
``family_id``, ``score_raw``). It scores on the RAW logit margin (never the saturating
sigmoid) using the canonical tie-aware ``guard_research.metrics.average_precision``.

Estimands (proposal Sec 6), with a regime ``R`` = evaluation split (represented = id_test,
adaptation-held-out = transfer_test) and ``M(i,a,R)`` = macro-AP (mean over the benchmark
``source`` values in that split of tie-aware AP):

  * within-checkpoint movement  Delta_a(i,R) = mean_r[M(i,a,R,r) - M(i,U,R)]  for a in {sft, kl};
  * seed-paired KL preservation P(i,R) = mean_r[M(i,kl,R,r) - M(i,sft,R,r)];
  * two-dimensional movement vectors theta_a,i = (Delta_a(i,represented), Delta_a(i,heldout));
  * equal-FAMILY means: average checkpoints within a model family first (both Qwen sizes are one
    family), then average across families -- Delta_sft(f,R), P(f,R);
  * headroom-normalized gains Delta/(1 - AP_U), reported ALONGSIDE the raw deltas (Sec 3 ceiling
    confound).

Confirmatory predicates (proposal Sec 3):

  H_gain     = mean_f Delta_sft(f, represented)
  H_conc     = mean_f [Delta_sft(f, represented) - Delta_sft(f, heldout)]
  H_preserve = mean_f P(f, heldout)
  H_cost     = mean_f P(f, represented)

Each carries a ONE-SIDED 97.5% bootstrap LOWER bound (the 2.5th percentile) from a family-aware
paired bootstrap that resamples evaluation near-duplicate families (Poisson(1) weights over
``family_id``) and training seeds while holding MODEL IDENTITIES FIXED. These bounds are therefore
CONDITIONAL ON THE FIXED MODEL PANEL: they carry only eval-row and seed uncertainty, not
between-model-family uncertainty, and one dominant family can drive the equal-family mean.

Gates (Bonferroni across the two RQ families -> one-sided 97.5% per family controls FWER 0.05):

  RQ1 supported  iff  LCB(H_gain) > 0   AND  LCB(H_conc) > 0
  RQ2 supported  iff  LCB(H_preserve) > 0  AND  LCB(H_cost) > -m,   m = 0.02 AP

Usage:
  python experiments/analyze_starting_type_adaptation.py \
      --scores artifacts/starting_type_adaptation_v1/scores/scores.parquet \
      --registry configs/starting_type_adaptation_v1.yaml \
      --out artifacts/starting_type_adaptation_v1/analysis
  python experiments/analyze_starting_type_adaptation.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402  (weighted_metric)
from guard_research.metrics import average_precision as AP  # noqa: E402

ANALYSIS_CODE_VERSION = "starting_type_adaptation_analysis_v1"

# Regime = evaluation split. Represented sources are scored on id_test; adaptation-held-out
# (transfer) sources on transfer_test -- the same split->regime map Paper A uses.
DEFAULT_REPRESENTED_SPLIT = "id_test"
DEFAULT_HELDOUT_SPLIT = "transfer_test"
DEFAULT_PRIMARY_BETA = 0.5           # predeclared primary KL-SFT arm (proposal Sec 5.5)
NONINF_MARGIN = 0.02                 # m: max tolerated represented-source loss (proposal Sec 3)
DEFAULT_BOOT_REPS = 10000
DEFAULT_RNG_SEED = 20260716
REGIMES = ("represented", "heldout")

# Columns tolerated under either the canonical eval schema or the proposal shorthand.
_KEY_COLS = ("starting_model_key", "starting_key")
_FAMILY_COLS = ("family_id", "family")

PANEL_CONDITIONAL_LIMITATION = (
    "Bootstrap lower bounds are CONDITIONAL ON THE FIXED MODEL PANEL: the resample draws "
    "evaluation near-duplicate families (Poisson weights) and training seeds while holding model "
    "identities fixed, so intervals carry only eval-row and seed uncertainty -- not between-model-"
    "family uncertainty. With few families a single dominant family can drive the equal-family "
    "mean; read H_gain/H_conc/H_preserve/H_cost as fixed-panel summaries, not as generalization "
    "over a population of purpose-built guards."
)


class ScoreSchemaError(ValueError):
    """The score table lacks the columns/conditions this analyzer requires."""


# ======================================================================================
# schema + family-map resolution
# ======================================================================================
def _pick_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ScoreSchemaError(f"score table lacks a {what} column (any of {candidates})")


def resolve_family_map(registry_path):
    """starting_key -> model family, from the study registry (both Qwen sizes share one family)."""
    import starting_type_common as S
    reg = S.load_registry(registry_path)
    return {k: v.get("family") for k, v in reg["checkpoints"].items()}


def load_scores(path):
    import pandas as pd
    return pd.read_parquet(path)


# ======================================================================================
# panel construction (source-aware; macro-AP is mean over `source` within a split)
# ======================================================================================
def _cell(sub, fam_index, score_col):
    return {
        "scores": sub[score_col].to_numpy(float),
        "gold": sub["gold"].to_numpy(int),
        "fam_idx": np.array([fam_index[str(f)] for f in sub["_family"]], int),
    }


def build_panel(df, *, represented_split=DEFAULT_REPRESENTED_SPLIT,
                heldout_split=DEFAULT_HELDOUT_SPLIT, primary_beta=DEFAULT_PRIMARY_BETA,
                family_map):
    """Return (panel, families) where

        panel[key] = {"model_family": str,
                      regime: {"sources": [...], "seeds": [...],
                               "unmodified": {source: cell},
                               "sft": {seed: {source: cell}},
                               "kl":  {seed: {source: cell}}}}

    and ``families`` is the ordered GLOBAL list of eval ``family_id`` values over the analyzed
    regimes (one Poisson weight per entry drives the paired bootstrap). ``cell`` carries the RAW
    ``score_raw`` margin, gold, and per-row global-family index.
    """
    key_col = _pick_col(df, _KEY_COLS, "starting-checkpoint key")
    fam_col = _pick_col(df, _FAMILY_COLS, "eval-family")
    for req in ("adaptation", "seed", "kl_beta", "source", "split", "gold", "score_raw"):
        if req not in df.columns:
            raise ScoreSchemaError(f"score table lacks required column {req!r}")

    df = df.copy()
    df["_family"] = df[fam_col].astype(str)
    valid = {"unmodified", "sft", "kl_sft"}
    extra = set(df["adaptation"].unique()) - valid
    if extra:
        raise ScoreSchemaError(f"unexpected adaptation labels: {sorted(extra)}")

    regime_split = {"represented": represented_split, "heldout": heldout_split}
    sub_all = df[df["split"].isin(list(regime_split.values()))]
    if sub_all.empty:
        raise ScoreSchemaError(
            f"no rows in the analyzed splits {sorted(set(regime_split.values()))}")
    families = sorted(sub_all["_family"].unique())
    fam_index = {f: i for i, f in enumerate(families)}

    panel = {}
    for key in sorted(df[key_col].astype(str).unique()):
        mf = family_map.get(key)
        if mf is None:
            raise ScoreSchemaError(
                f"no model family for starting key {key!r}; supply --registry or a family_map")
        d_key = df[df[key_col].astype(str) == key]
        entry = {"model_family": str(mf)}
        for regime, split in regime_split.items():
            d_reg = d_key[d_key["split"] == split]
            u = d_reg[d_reg["adaptation"] == "unmodified"]
            s = d_reg[d_reg["adaptation"] == "sft"]
            k = d_reg[(d_reg["adaptation"] == "kl_sft")
                      & (np.isclose(d_reg["kl_beta"].astype(float), float(primary_beta)))]
            sources = sorted(set(u["source"].astype(str)) | set(s["source"].astype(str)))
            seeds = sorted(set(int(x) for x in s["seed"].unique())
                           & set(int(x) for x in k["seed"].unique()))

            def _bysrc(frame, seed=None):
                f = frame if seed is None else frame[frame["seed"].astype(int) == int(seed)]
                out = {}
                for src in sources:
                    ss = f[f["source"].astype(str) == src]
                    if not ss.empty:
                        out[src] = _cell(ss, fam_index, "score_raw")
                return out

            entry[regime] = {
                "sources": sources, "seeds": seeds,
                "unmodified": _bysrc(u),
                "sft": {sd: _bysrc(s, sd) for sd in seeds},
                "kl": {sd: _bysrc(k, sd) for sd in seeds},
            }
        panel[key] = entry
    return panel, families


# ======================================================================================
# metric core
# ======================================================================================
def _ap(cell, weights):
    w = None if weights is None else weights[cell["fam_idx"]]
    return C.weighted_metric(AP, cell["scores"], cell["gold"], w)


def _macro_ap(source_cells, weights):
    """Mean over benchmark sources of tie-aware AP; single-class sources drop out (nan)."""
    vals = []
    for cell in source_cells.values():
        v = _ap(cell, weights)
        if not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _norm(delta, ap_u):
    """Headroom-normalized gain Delta / (1 - AP_U)."""
    head = 1.0 - ap_u
    if not np.isfinite(ap_u) or head <= 1e-9:
        return float("nan")
    return float(delta / head)


def _checkpoint_stats(entry, *, weights=None, seed_pick=None):
    """Per-checkpoint stats for both regimes. ``seed_pick`` (list of indices into the checkpoint's
    seed list, with replacement) drives the seed bootstrap; None uses each seed once."""
    out = {}
    for regime in REGIMES:
        reg = entry[regime]
        seeds = reg["seeds"]
        idxs = list(range(len(seeds))) if seed_pick is None else list(seed_pick)
        ap_u = _macro_ap(reg["unmodified"], weights)
        sft = [_macro_ap(reg["sft"][seeds[j]], weights) for j in idxs]
        kl = [_macro_ap(reg["kl"][seeds[j]], weights) for j in idxs]
        sft_mean = float(np.mean(sft)) if sft else float("nan")
        kl_mean = float(np.mean(kl)) if kl else float("nan")
        delta_sft = sft_mean - ap_u
        delta_kl = kl_mean - ap_u
        preservation = float(np.mean([kl[t] - sft[t] for t in range(len(idxs))])) if idxs else float("nan")
        out[regime] = {
            "ap_u": ap_u, "sft_mean": sft_mean, "kl_mean": kl_mean,
            "delta_sft": delta_sft, "delta_kl": delta_kl, "P": preservation,
            "delta_sft_norm": _norm(delta_sft, ap_u), "delta_kl_norm": _norm(delta_kl, ap_u),
        }
    return out


def _family_means(per_checkpoint, panel, field):
    """Average ``field`` within each model family, keyed by regime -> family -> value."""
    fams = {}
    for key, ck in per_checkpoint.items():
        fams.setdefault(panel[key]["model_family"], []).append(key)
    out = {regime: {} for regime in REGIMES}
    for regime in REGIMES:
        for fam, keys in fams.items():
            vals = [per_checkpoint[k][regime][field] for k in keys]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            out[regime][fam] = float(np.mean(vals)) if vals else float("nan")
    return out


def _eqfam_mean(fam_dict_regime):
    vals = [v for v in fam_dict_regime.values()
            if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def compute_hypotheses(panel, *, weights=None, seed_pick_by_key=None):
    """Return per-checkpoint, per-family, and the four (+ held-out SFT) H statistics for ONE draw.

    seed_pick_by_key: {key: [indices]} or None. For the bootstrap the same seed pick is applied
    to both the sft and kl arms of a checkpoint (seed-paired preservation)."""
    per_ck = {}
    for key, entry in panel.items():
        sp = None if seed_pick_by_key is None else seed_pick_by_key[key]
        per_ck[key] = _checkpoint_stats(entry, weights=weights, seed_pick=sp)

    fam_delta_sft = _family_means(per_ck, panel, "delta_sft")
    fam_delta_kl = _family_means(per_ck, panel, "delta_kl")
    fam_P = _family_means(per_ck, panel, "P")
    fam_delta_sft_norm = _family_means(per_ck, panel, "delta_sft_norm")

    h_gain = _eqfam_mean(fam_delta_sft["represented"])
    h_held_sft = _eqfam_mean(fam_delta_sft["heldout"])
    h_gain_norm = _eqfam_mean(fam_delta_sft_norm["represented"])
    h_held_sft_norm = _eqfam_mean(fam_delta_sft_norm["heldout"])
    # H_conc as the equal-family mean of the per-family (rep - held) gap.
    conc_by_fam = {f: fam_delta_sft["represented"][f] - fam_delta_sft["heldout"][f]
                   for f in fam_delta_sft["represented"]}
    conc_norm_by_fam = {f: fam_delta_sft_norm["represented"][f] - fam_delta_sft_norm["heldout"][f]
                        for f in fam_delta_sft_norm["represented"]}
    h_conc = _eqfam_mean(conc_by_fam)
    h_conc_norm = _eqfam_mean(conc_norm_by_fam)
    h_preserve = _eqfam_mean(fam_P["heldout"])
    h_cost = _eqfam_mean(fam_P["represented"])

    return {
        "per_checkpoint": per_ck,
        "per_family": {
            "delta_sft": fam_delta_sft, "delta_kl": fam_delta_kl, "P": fam_P,
            "delta_sft_norm": fam_delta_sft_norm,
        },
        "H": {
            "H_gain": h_gain, "H_conc": h_conc, "H_preserve": h_preserve, "H_cost": h_cost,
            "H_held_sft": h_held_sft,
            "H_gain_norm": h_gain_norm, "H_conc_norm": h_conc_norm,
            "H_held_sft_norm": h_held_sft_norm,
        },
    }


# ======================================================================================
# family-aware paired bootstrap (proposal Sec 9); model identities FIXED
# ======================================================================================
def _all_cells(panel):
    for entry in panel.values():
        for regime in REGIMES:
            reg = entry[regime]
            for src_dict in [reg["unmodified"], *reg["sft"].values(), *reg["kl"].values()]:
                for cell in src_dict.values():
                    yield cell


def _weights_valid(panel, w):
    """A draw is valid iff every (checkpoint, regime) has >= 1 benchmark source that remains
    two-class under the family weights (so its macro-AP is defined)."""
    for entry in panel.values():
        for regime in REGIMES:
            reg = entry[regime]
            ok = False
            for cell in reg["unmodified"].values():
                g = cell["gold"]; fi = cell["fam_idx"]
                if w[fi[g == 1]].sum() > 0 and w[fi[g == 0]].sum() > 0:
                    ok = True
                    break
            if not ok:
                return False
    return True


def bootstrap(panel, families, *, reps=DEFAULT_BOOT_REPS, rng_seed=DEFAULT_RNG_SEED,
              max_redraw=5000):
    rng = np.random.default_rng(rng_seed)
    n_fam = len(families)
    keys = list(panel.keys())
    n_seeds = {k: len(panel[k]["represented"]["seeds"]) for k in keys}
    stat_names = ["H_gain", "H_conc", "H_preserve", "H_cost", "H_held_sft",
                  "H_gain_norm", "H_conc_norm", "H_held_sft_norm"]
    samples = {s: np.empty(reps) for s in stat_names}
    total_redraw = 0
    for rep in range(reps):
        redraws = 0
        while True:
            w = rng.poisson(1.0, size=n_fam).astype(float)
            if _weights_valid(panel, w):
                break
            redraws += 1
            total_redraw += 1
            if redraws > max_redraw:
                raise RuntimeError("bootstrap: exceeded redraw cap (score panel too sparse?)")
        seed_pick = {k: rng.integers(0, max(1, n_seeds[k]), size=n_seeds[k]) for k in keys}
        H = compute_hypotheses(panel, weights=w, seed_pick_by_key=seed_pick)["H"]
        for s in stat_names:
            samples[s][rep] = H[s]

    def summarize(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"lcb975_one_sided": float("nan"), "ucb975_one_sided": float("nan"),
                    "ci95_two_sided": [float("nan"), float("nan")], "std": float("nan"),
                    "n_finite": 0}
        return {
            "lcb975_one_sided": float(np.percentile(arr, 2.5)),
            "ucb975_one_sided": float(np.percentile(arr, 97.5)),
            "ci95_two_sided": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "n_finite": int(arr.size),
        }

    return {
        "reps": reps, "rng_seed": rng_seed, "n_families": n_fam,
        "redraws": int(total_redraw),
        "rejected_fraction": (float(total_redraw / (reps + total_redraw)) if reps else 0.0),
        "stats": {s: summarize(samples[s]) for s in stat_names},
    }


# ======================================================================================
# claim registry: interpretation selected ONLY by locked bound predicates (NON-HARKing)
# ======================================================================================
# Wording is keyed by tuples of LOCKED, bootstrap-bound predicate booleans -- NEVER by inspecting
# a point-estimate sign. The held-out-loss distinction is itself a locked one-sided-bound predicate
# (UCB(H_held_sft) < 0), not a raw point sign, so no branch reads a point estimate.
RQ1_CLAIMS = {
    # key: (p_gain, p_conc, p_held_sft_loss)  -- only consulted when RQ1 gate passes
    (True, True, True): (
        "Our ordinary SFT protocol further specialized these released guards toward represented "
        "sources, with a bound-confirmed adaptation-held-out loss (fixed-panel, this recipe)."),
    (True, True, False): (
        "The gains are concentrated toward represented sources; no adaptation-held-out loss is "
        "established at the locked bound (fixed-panel, this recipe)."),
}
RQ1_NOT_SUPPORTED = (
    "The preregistered further-specialization hypothesis was not supported; checkpoint-level "
    "movements remain descriptive.")
RQ2_CLAIMS = {
    # key: (p_preserve, p_cost)
    (True, True): (
        "Anchoring to the released guard preserved more adaptation-held-out transfer than ordinary "
        "SFT at an acceptable represented-source cost (fixed-panel, this recipe)."),
    (True, False): (
        "KL-SFT trades adaptation gain for retention; it preserved held-out transfer but the "
        "represented-source non-inferiority margin was not met -- it is not a free improvement."),
    (False, True): (
        "The preregistered anti-forgetting hypothesis was not supported under the locked "
        "coefficient and recipe."),
    (False, False): (
        "The preregistered anti-forgetting hypothesis was not supported under the locked "
        "coefficient and recipe."),
}


def claim_checks(boot, *, margin=NONINF_MARGIN, primary_beta=DEFAULT_PRIMARY_BETA):
    """Evaluate the RQ1/RQ2 gates and select interpretation strings PURELY from locked bound
    predicates. This function reads only bootstrap bounds -- never a point estimate sign."""
    st = boot["stats"]
    p_gain = st["H_gain"]["lcb975_one_sided"] > 0.0
    p_conc = st["H_conc"]["lcb975_one_sided"] > 0.0
    p_preserve = st["H_preserve"]["lcb975_one_sided"] > 0.0
    p_cost = st["H_cost"]["lcb975_one_sided"] > -float(margin)
    # locked bound predicate (NOT a point sign): is the equal-family SFT held-out movement
    # significantly negative?
    p_held_sft_loss = st["H_held_sft"]["ucb975_one_sided"] < 0.0

    rq1 = bool(p_gain and p_conc)
    rq2 = bool(p_preserve and p_cost)

    rq1_wording = (RQ1_CLAIMS[(True, True, bool(p_held_sft_loss))] if rq1
                   else RQ1_NOT_SUPPORTED)
    rq2_wording = RQ2_CLAIMS[(bool(p_preserve), bool(p_cost))]

    return {
        "primary_beta": float(primary_beta),
        "non_inferiority_margin_m": float(margin),
        "multiplicity": (
            "Bonferroni across the two RQ families: one-sided 97.5% per-family lower bounds control "
            "familywise alpha = 0.05."),
        "predicates": {
            "p_gain_lcb_gt_0": bool(p_gain),
            "p_conc_lcb_gt_0": bool(p_conc),
            "p_preserve_lcb_gt_0": bool(p_preserve),
            "p_cost_lcb_gt_neg_m": bool(p_cost),
            "p_held_sft_loss_ucb_lt_0": bool(p_held_sft_loss),
        },
        "RQ1": {
            "criterion": "LCB(H_gain) > 0 AND LCB(H_conc) > 0",
            "supported": rq1,
            "H_gain_lcb": st["H_gain"]["lcb975_one_sided"],
            "H_conc_lcb": st["H_conc"]["lcb975_one_sided"],
            "interpretation": rq1_wording,
        },
        "RQ2": {
            "criterion": f"LCB(H_preserve) > 0 AND LCB(H_cost) > -{margin}",
            "supported": rq2,
            "H_preserve_lcb": st["H_preserve"]["lcb975_one_sided"],
            "H_cost_lcb": st["H_cost"]["lcb975_one_sided"],
            "interpretation": rq2_wording,
        },
        "interpretation_source": (
            "locked claim-registry keyed by bootstrap-bound predicates; NO point-estimate sign "
            "was consulted to select wording (non-HARKing)."),
        "panel_conditional_limitation": PANEL_CONDITIONAL_LIMITATION,
    }


# ======================================================================================
# orchestration
# ======================================================================================
def _movement_vectors(per_checkpoint, panel):
    out = {}
    for key, ck in per_checkpoint.items():
        out[key] = {
            "model_family": panel[key]["model_family"],
            "theta_sft": [ck["represented"]["delta_sft"], ck["heldout"]["delta_sft"]],
            "theta_kl": [ck["represented"]["delta_kl"], ck["heldout"]["delta_kl"]],
        }
    return out


def analyze(df, *, family_map, represented_split=DEFAULT_REPRESENTED_SPLIT,
            heldout_split=DEFAULT_HELDOUT_SPLIT, primary_beta=DEFAULT_PRIMARY_BETA,
            reps=DEFAULT_BOOT_REPS, rng_seed=DEFAULT_RNG_SEED):
    panel, families = build_panel(
        df, represented_split=represented_split, heldout_split=heldout_split,
        primary_beta=primary_beta, family_map=family_map)
    point = compute_hypotheses(panel)
    boot = bootstrap(panel, families, reps=reps, rng_seed=rng_seed)
    checks = claim_checks(boot, primary_beta=primary_beta)
    return {
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "config": {
            "represented_split": represented_split, "heldout_split": heldout_split,
            "primary_beta": float(primary_beta), "non_inferiority_margin_m": NONINF_MARGIN,
            "bootstrap_reps": reps, "rng_seed": rng_seed,
            "metric": "macro-AP over benchmark sources; RAW logit margin; tie-aware AP",
            "model_families": sorted({v["model_family"] for v in panel.values()}),
            "n_checkpoints": len(panel), "n_eval_families": len(families),
        },
        "point_estimates": {
            "H": point["H"],
            "per_family": point["per_family"],
            "per_checkpoint": {
                k: {r: point["per_checkpoint"][k][r] for r in REGIMES} for k in panel},
            "movement_vectors": _movement_vectors(point["per_checkpoint"], panel),
        },
        "bootstrap": boot,
        "claim_checks": checks,
        "panel_conditional_limitation": PANEL_CONDITIONAL_LIMITATION,
    }


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def write_results(out_dir, results):
    os.makedirs(out_dir, exist_ok=True)
    payload = _jsonify(results)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "claim_checks.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify(results["claim_checks"]), f, indent=2, sort_keys=True)
    return out_dir


# ======================================================================================
# synthetic fixture (shared by --self-test and tests/)
# ======================================================================================
def make_synthetic_scores(*, kl_equals_sft_heldout=False, kl_beta=DEFAULT_PRIMARY_BETA,
                          seeds=(42, 43, 44, 45, 46), n_per_source=40, n_eval_families=4,
                          rng_seed=7):
    """Build a synthetic starting-type score table with KNOWN deltas over 3 model families
    (4 checkpoints), U/SFT/KL x 5 seeds x 2 splits, 2 benchmark sources per split.

    Represented (id_test): U weak, SFT strong (positive gain), KL == SFT (zero represented cost).
    Held-out (transfer_test): U strong, SFT weak (transfer loss), KL between (positive preservation)
    -- unless ``kl_equals_sft_heldout`` (the beta==0 identity case), where KL == SFT everywhere so
    every preservation P is exactly zero.
    """
    import pandas as pd
    rng = np.random.default_rng(rng_seed)
    # 4 checkpoints across 3 model families (famQ has two checkpoints -> exercises within-family mean)
    checkpoints = [("ckptQ1", "famQ"), ("ckptQ2", "famQ"), ("ckptR", "famR"), ("ckptS", "famS")]
    family_map = {k: f for k, f in checkpoints}
    splits = {"represented": DEFAULT_REPRESENTED_SPLIT, "heldout": DEFAULT_HELDOUT_SPLIT}
    sources = {"represented": ["src_rep_a", "src_rep_b"], "heldout": ["src_held_a", "src_held_b"]}
    # per-condition class separation (higher -> higher AP)
    sep = {
        "represented": {"unmodified": 0.8, "sft": 2.6, "kl": 2.6},
        "heldout": {"unmodified": 2.6, "sft": 0.8, "kl": 1.9},
    }

    rows = []

    def gen(gold, signal):
        return signal * (2.0 * gold - 1.0) + rng.normal(0.0, 1.0, size=len(gold))

    for key, mf in checkpoints:
        for regime, split in splits.items():
            for src in sources[regime]:
                gold = np.array([i % 2 for i in range(n_per_source)], int)
                fam = np.array([f"efam-{i % n_eval_families}" for i in range(n_per_source)])

                def emit(adaptation, seed, beta, scores):
                    for i in range(n_per_source):
                        rows.append({
                            "sample_id": f"{key}:{split}:{src}:{adaptation}:{seed}:{i}",
                            "content_sha256": f"sha-{src}-{i}",
                            "source": src, "split": split, "gold": int(gold[i]),
                            "family_id": fam[i],
                            "starting_model_key": key, "model_revision": "synthetic",
                            "starting_type": "purpose_built", "adaptation": adaptation,
                            "condition_id": f"{key}:{adaptation}:{seed}",
                            "seed": int(seed), "kl_beta": beta,
                            "adapter_sha256": None if adaptation == "unmodified" else f"ad-{key}-{seed}",
                            "score_raw": float(scores[i]),
                            "probability_raw": float(1.0 / (1.0 + np.exp(-scores[i]))),
                        })

                emit("unmodified", -1, None, gen(gold, sep[regime]["unmodified"]))
                for sd in seeds:
                    sft_scores = gen(gold, sep[regime]["sft"])
                    emit("sft", sd, 0.0, sft_scores)
                    if (regime == "represented") or kl_equals_sft_heldout:
                        # KL identical to SFT on this cell -> preservation is exactly zero
                        kl_scores = sft_scores.copy()
                    else:
                        kl_scores = gen(gold, sep[regime]["kl"])
                    emit("kl_sft", sd, float(kl_beta), kl_scores)

    return pd.DataFrame(rows), family_map


# ======================================================================================
# self-test
# ======================================================================================
def _self_test(reps=400):
    print("=== analyze_starting_type_adaptation self-test (synthetic) ===")
    # --- Scenario 1: designed so RQ1 and RQ2 are supported ---
    df, fmap = make_synthetic_scores(kl_equals_sft_heldout=False)
    res = analyze(df, family_map=fmap, reps=reps, rng_seed=DEFAULT_RNG_SEED)
    H = res["point_estimates"]["H"]
    st = res["bootstrap"]["stats"]
    checks = res["claim_checks"]
    print(f"  point  H_gain={H['H_gain']:+.4f} H_conc={H['H_conc']:+.4f} "
          f"H_preserve={H['H_preserve']:+.4f} H_cost={H['H_cost']:+.4f}")
    print(f"  LCB    H_gain={st['H_gain']['lcb975_one_sided']:+.4f} "
          f"H_conc={st['H_conc']['lcb975_one_sided']:+.4f} "
          f"H_preserve={st['H_preserve']['lcb975_one_sided']:+.4f} "
          f"H_cost={st['H_cost']['lcb975_one_sided']:+.4f}")
    print(f"  norm   H_gain_norm={H['H_gain_norm']:+.4f} H_conc_norm={H['H_conc_norm']:+.4f}")
    assert H["H_gain"] > 0 and H["H_conc"] > 0, H
    assert H["H_preserve"] > 0, H
    assert st["H_gain"]["lcb975_one_sided"] > 0, "expected RQ1 gain LCB > 0"
    assert st["H_conc"]["lcb975_one_sided"] > 0, "expected RQ1 conc LCB > 0"
    assert st["H_preserve"]["lcb975_one_sided"] > 0, "expected RQ2 preserve LCB > 0"
    assert st["H_cost"]["lcb975_one_sided"] > -NONINF_MARGIN, "expected RQ2 cost LCB > -m"
    assert checks["RQ1"]["supported"] is True, checks["RQ1"]
    assert checks["RQ2"]["supported"] is True, checks["RQ2"]
    assert math.isfinite(H["H_gain_norm"]) and H["H_gain_norm"] > 0
    # headroom-normalized gain must exceed the raw gain (0 < 1 - AP_U < 1)
    assert H["H_gain_norm"] >= H["H_gain"] - 1e-9, (H["H_gain_norm"], H["H_gain"])
    print(f"  [ok] RQ1 supported={checks['RQ1']['supported']} RQ2 supported={checks['RQ2']['supported']}")
    print(f"       RQ1: {checks['RQ1']['interpretation']}")
    print(f"       RQ2: {checks['RQ2']['interpretation']}")

    # --- Scenario 2: beta==0 identity (KL == SFT everywhere) -> H_preserve == 0 EXACTLY ---
    df0, fmap0 = make_synthetic_scores(kl_equals_sft_heldout=True, kl_beta=0.0)
    res0 = analyze(df0, family_map=fmap0, primary_beta=0.0, reps=reps, rng_seed=DEFAULT_RNG_SEED)
    H0 = res0["point_estimates"]["H"]
    assert H0["H_preserve"] == 0.0, f"beta=0 identity broken: H_preserve={H0['H_preserve']!r}"
    assert H0["H_cost"] == 0.0, f"beta=0 identity broken: H_cost={H0['H_cost']!r}"
    st0 = res0["bootstrap"]["stats"]
    assert st0["H_preserve"]["lcb975_one_sided"] == 0.0, st0["H_preserve"]
    assert res0["claim_checks"]["RQ2"]["supported"] is False, "beta=0 must not support preservation"
    print(f"  [ok] beta=0 identity: H_preserve==0 exactly, RQ2 supported="
          f"{res0['claim_checks']['RQ2']['supported']}")

    # --- interpretation is predicate-driven: forcing predicates flips wording deterministically ---
    fake_boot = {"stats": {
        "H_gain": {"lcb975_one_sided": -1.0, "ucb975_one_sided": 1.0},
        "H_conc": {"lcb975_one_sided": -1.0, "ucb975_one_sided": 1.0},
        "H_preserve": {"lcb975_one_sided": -1.0, "ucb975_one_sided": 1.0},
        "H_cost": {"lcb975_one_sided": -1.0, "ucb975_one_sided": 1.0},
        "H_held_sft": {"lcb975_one_sided": -1.0, "ucb975_one_sided": 1.0},
    }}
    c = claim_checks(fake_boot)
    assert c["RQ1"]["supported"] is False and c["RQ1"]["interpretation"] == RQ1_NOT_SUPPORTED
    assert c["RQ2"]["supported"] is False
    print("  [ok] claim registry maps locked predicates -> fixed wording (non-HARKing)")
    print("ALL PASSED")
    return True


# ======================================================================================
# CLI
# ======================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", help="scores parquet from eval_starting_type_adaptation")
    ap.add_argument("--registry", default=os.path.join(
        str(C.REPO_ROOT), "configs", "starting_type_adaptation_v1.yaml"),
        help="study registry YAML for the starting_key -> model-family map")
    ap.add_argument("--out", default=None, help="analysis output directory")
    ap.add_argument("--represented-split", default=DEFAULT_REPRESENTED_SPLIT)
    ap.add_argument("--heldout-split", default=DEFAULT_HELDOUT_SPLIT)
    ap.add_argument("--primary-beta", type=float, default=DEFAULT_PRIMARY_BETA)
    ap.add_argument("--reps", type=int, default=DEFAULT_BOOT_REPS)
    ap.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0
    if not args.scores:
        print("[analyze-sta] --scores is required (or use --self-test)", file=sys.stderr)
        return 2

    df = load_scores(args.scores)
    family_map = resolve_family_map(args.registry)
    results = analyze(
        df, family_map=family_map, represented_split=args.represented_split,
        heldout_split=args.heldout_split, primary_beta=args.primary_beta,
        reps=args.reps, rng_seed=args.rng_seed)
    checks = results["claim_checks"]
    print(f"[analyze-sta] families={results['config']['model_families']} "
          f"checkpoints={results['config']['n_checkpoints']} "
          f"eval_families={results['config']['n_eval_families']}")
    print(f"[analyze-sta] RQ1 supported={checks['RQ1']['supported']}: {checks['RQ1']['interpretation']}")
    print(f"[analyze-sta] RQ2 supported={checks['RQ2']['supported']}: {checks['RQ2']['interpretation']}")
    if args.out:
        write_results(args.out, results)
        print(f"[analyze-sta] wrote results.json + claim_checks.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
