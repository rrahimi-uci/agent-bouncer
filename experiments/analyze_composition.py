#!/usr/bin/env python
"""Composition ("Compose, Don't Tune") analysis: base + tuned guard ensembles.

Reproducibly regenerates the numbers behind the composition result from a lock-bound,
row-keyed score table. For each of the four checkpoints it evaluates the untuned base,
the SFT adapters (mean of per-seed metrics), and several *composed* guards (combine the
base and one SFT adapter per seed), on the represented and dataset-held-out regimes.

Outputs (into --out):
  - composition.json     : point estimates (all combiners), bootstrap CIs (primary
                           combiner), leave-one-benchmark-out, matched-FPR operating
                           point, and two single-permutation shuffle diagnostics;
  - composition.md       : human-readable summary tables.
  - composition_metadata.json: runtime/source/input verification and output hashes.

Metric: benchmark-macro tie-aware Average Precision (guard_research.metrics), macro over
a regime's benchmarks then mean over the 4-checkpoint panel (SFT/ensemble also over seeds).
Uncertainty: the same hierarchical PAIRED bootstrap as analyze_paper_a_sft.py -- 4
checkpoints fixed; resample the 5 SFT seed indices within each checkpoint; one Poisson(1)
weight per GLOBAL family_id; weighted tie-aware AP -> macro -> panel; percentile CIs.

The analyzer supports both the explicit legacy compatibility path and a strict v2 lock.
Strict v2 release-cache mode consumes the verified text-free public identity manifests.
Neither path is prospective-confirmatory: the v2 run repairs execution provenance, but
part of its transfer cohort was inspected during earlier development. WiSE-FT weight
interpolation is out of scope here because it is not an output-space score operation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score  # canonical tie-aware AP (weighted form)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from guard_research.metrics import average_precision  # noqa: E402  (scores, labels)
from guard_research.thresholds import select_threshold  # noqa: E402
import analyze_paper_a_sft as paper_a_analysis  # noqa: E402
import paper_a_common as C  # noqa: E402

MODELS = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]
REP = ["toxicchat", "prompt_injections", "jailbreak_classification"]
TR = ["jailbreakbench", "xstest", "wildguardtest", "wildjailbreak"]
REGIMES = {"represented": ("id_test", REP), "transfer": ("transfer_test", TR)}
CAL_SPLIT = "calibration"
PROTOTYPE_REPS = 4000
PROTOTYPE_RNG_SEED = 20260712
PROTOTYPE_SHUFFLE_RNG_SEED = 20260714
PROTOTYPE_TARGET_FPR = 0.05
PROTOTYPE_PRIMARY_COMBINER = "calibrated_avg"
COMPOSITION_ANALYSIS_SOURCE_FILES = (
    "experiments/analyze_composition.py",
    "experiments/analyze_paper_a_sft.py",
    "experiments/paper_a_common.py",
    "guard_research/__init__.py",
    "guard_research/metrics.py",
    "guard_research/thresholds.py",
)


# ----------------------------------------------------------------------------- weighted AP
def wap(scores, labels, weights=None):
    """Weighted tie-aware AP. Unweighted path uses the canonical guard_research wrapper;
    the weighted path calls the same sklearn function with sample_weight (Poisson family
    weights). NaN if single-class."""
    y = np.asarray(labels, float)
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    if weights is None:
        return average_precision(scores, labels)
    return float(average_precision_score(y, np.asarray(scores, float), sample_weight=np.asarray(weights, float)))


# ----------------------------------------------------------------------------- data
def load(scores_path, *, strict=False):
    df = pd.read_parquet(scores_path)
    if strict:
        # Strict validation must see the exact Parquet values. Filtering labels or coercing a
        # fractional seed before validate_score_artifacts would normalize tampered evidence.
        return df
    df = df[df["gold"].isin([0, 1])].copy()
    df["gold"] = df["gold"].astype(int)
    df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    return df


def composition_analysis_source_hashes(repo_root=None):
    """Hash every source file imported by the downstream composition analysis."""
    return C.execution_source_hashes(
        repo_root=repo_root, required_files=COMPOSITION_ANALYSIS_SOURCE_FILES)


def composition_parameter_record(reps, rng_seed, target_fpr, *, nonfinal=False):
    """Validate fixed prototype settings; they are not part of the Paper A lock."""
    record = {
        "reps": int(reps),
        "rng_seed": int(rng_seed),
        "shuffle_rng_seed": PROTOTYPE_SHUFFLE_RNG_SEED,
        "target_fpr": float(target_fpr),
        "primary_combiner": PROTOTYPE_PRIMARY_COMBINER,
    }
    expected = {
        "reps": PROTOTYPE_REPS,
        "rng_seed": PROTOTYPE_RNG_SEED,
        "shuffle_rng_seed": PROTOTYPE_SHUFFLE_RNG_SEED,
        "target_fpr": PROTOTYPE_TARGET_FPR,
        "primary_combiner": PROTOTYPE_PRIMARY_COMBINER,
    }
    mismatches = [key for key in expected if record[key] != expected[key]]
    if mismatches and not nonfinal:
        raise C.ArtifactContractError(
            "canonical composition parameters differ from the fixed prototype constants: "
            + ", ".join(mismatches) + "; use --nonfinal with an external output path")
    record["status"] = (
        "nonfinal_override_not_lock_bound" if mismatches
        else "fixed_prototype_constants_not_paper_a_lock")
    return record


def composition_analysis_attestation(parameter_record):
    """Describe this unlocked downstream analysis without minting a Paper B lock."""
    runtime = C.runtime_environment("cpu")
    return {
        "analysis_source_hashes": composition_analysis_source_hashes(),
        "analysis_runtime_environment": runtime,
        "analysis_runtime_sha256": C.canonical_obj_sha256(runtime),
        "analysis_parameters": dict(parameter_record),
    }


def validate_composition_paths(
        lock, scores_path, out_dir, *, release_cache=False, nonfinal=False):
    """Keep release, full-artifact, and legacy outputs in disjoint namespaces."""
    paths = C.artifact_paths(lock)
    strict = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    expected_scores = os.path.join(paths["scores"], "scores.parquet")
    output_name = "composition" if release_cache or not strict else "composition-full"
    expected_out = os.path.join(paths["analysis"], output_name)
    if C.resolved_path(scores_path) != C.resolved_path(expected_scores):
        raise C.ArtifactContractError(
            f"composition scores must use the lock-authoritative path: {expected_scores}")
    if nonfinal:
        protected = {
            paths["root"], C.DEFAULT_ARTIFACTS["root"], C.DEFAULT_ARTIFACTS_V2["root"],
        }
        if any(C.path_is_within(out_dir, root) for root in protected):
            raise C.ArtifactContractError(
                "nonfinal composition output must be outside canonical artifact roots")
    elif C.resolved_path(out_dir) != C.resolved_path(expected_out):
        raise C.ArtifactContractError(
            f"composition output must use the mode-authoritative path: {expected_out}")
    return {"scores_path": str(C.resolved_path(scores_path)),
            "out_dir": str(C.resolved_path(out_dir))}


def verify_locked_scoring_manifests(lock):
    """Rehash the five scoring manifests before joining their rows to scores.

    Composition intentionally does not verify the current checkout against Paper A's
    execution-source hashes: the downstream analyzer was added after that execution.
    It must nevertheless verify the exact manifest bytes on which its score matrix rests.
    """
    root = C.resolved_path(C.artifact_paths(lock)["manifests"])
    locked = (lock.get("manifests") or {}).get("splits") or {}
    report = {}
    for filename in paper_a_analysis.SCORING_SPLIT_FILES.values():
        record = locked.get(filename) or {}
        path = root / filename
        if not path.is_file():
            raise C.ArtifactContractError(f"locked scoring manifest is missing: {path}")
        observed_sha = C.sha256_file(path)
        if observed_sha != record.get("sha256"):
            raise C.ArtifactContractError(
                f"scoring manifest hash mismatch for {filename}: "
                f"locked={record.get('sha256')} observed={observed_sha}")
        observed_rows = len(C.read_jsonl(path))
        try:
            locked_rows = int(record["rows"])
        except (KeyError, TypeError, ValueError):
            locked_rows = -1
        if observed_rows != locked_rows:
            raise C.ArtifactContractError(
                f"scoring manifest row-count mismatch for {filename}: "
                f"locked={locked_rows} observed={observed_rows}")
        report[filename] = {"sha256": observed_sha, "rows": observed_rows}
    return report


def load_verified(
        scores_path, lock_path, *, allow_legacy=False, release_cache=False,
        nonfinal=False):
    """Validate the immutable Paper A evidence used by this downstream analysis.

    Default strict-v2 analysis rehashes/joins the five raw scoring manifests, run metadata,
    adapters, score metadata, and score bytes. ``release_cache=True`` is an explicit reduced
    artifact path: it verifies the final lock and text-free public release, then permits only
    the raw manifests/run metadata/adapters to be absent. Paper A's original source bundle is
    verified separately because this downstream analyzer postdates that execution.
    """
    if release_cache and allow_legacy:
        raise C.ArtifactContractError(
            "--release-cache cannot be combined with legacy lock compatibility")
    lock = C.load_lock(lock_path, allow_legacy=allow_legacy, verify_files=False)
    strict = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    if release_cache and (not strict or lock.get("finalization_status") != "final"):
        raise C.ArtifactContractError(
            "--release-cache requires a strict final v2 lock")
    release_verification = None
    if release_cache:
        release_verification = C.verify_release_cache_lock(lock)
        release_contract = release_verification.get("release_contract") or {}
        if not (release_contract.get("release_sha256")
                and release_contract.get("release_file_sha256")
                and release_contract.get("anchor_path")):
            raise C.ArtifactContractError(
                "release-cache analysis lacks the tracked RELEASE.json trust root")
    if strict:
        paper_a_analysis.validate_analysis_runtime(
            lock, nonfinal=nonfinal, release_cache=release_cache)

    metadata_path = os.path.join(os.path.dirname(os.path.abspath(scores_path)), "metadata.json")
    verified = C.verify_score_artifact(
        scores_path, metadata_path, lock, allow_legacy=allow_legacy)
    df = load(scores_path, strict=strict)
    if strict and release_cache:
        scoring_manifests = release_verification["public_release"]["splits"]
        manifest_rows = paper_a_analysis.load_public_scoring_manifest_rows(lock)
    elif strict:
        scoring_manifests = verify_locked_scoring_manifests(lock)
        manifest_rows = paper_a_analysis.load_locked_scoring_manifest_rows(lock)
    else:
        scoring_manifests = None
        manifest_rows = None
    matrix = paper_a_analysis.validate_score_artifacts(
        df, lock, verified["metadata"], manifest_rows=manifest_rows,
        release_cache=release_cache)
    report = {
        "scores_sha256": verified["scores_sha256"],
        "metadata_sha256": verified["metadata_sha256"],
        "metadata_filename": verified["metadata_filename"],
        "bound": bool(verified["bound"]),
        "legacy": bool(verified["legacy"]),
        "release_cache": bool(release_cache),
        "paper_a_source_tree_verification": (
            "unrecoverable_legacy_source" if not strict
            else "separate_immutable_source_bundle" if release_cache
            else "separate_source_bundle"),
        "scoring_manifests": scoring_manifests,
        "matrix": matrix,
    }
    if release_cache:
        sources = release_verification["execution_sources"]
        report["release_cache_verification"] = {
            "mode": "strict_v2_score_only_release_cache",
            "release_contract": release_verification["release_contract"],
            "public_manifest_sha256": release_verification["public_release"]["sha256"],
            "public_splits": release_verification["public_release"]["splits"],
            "score_and_metadata_hashes_reverified": bool(
                verified["bound"] and not verified["legacy"]
                and verified["scores_sha256"] and verified["metadata_sha256"]),
            "current_analysis_source_hashes": composition_analysis_source_hashes(),
            "original_paper_a_execution_source": {
                "aggregate_sha256": sources["aggregate_sha256"],
                "verification": sources[
                    "original_paper_a_execution_source_verification"],
                "current_checkout_files_reverified": False,
            },
            "raw_manifest_files_locally_reverified": False,
            "run_metadata_and_adapter_bytes_locally_reverified": False,
        }
    return df, lock, report


def build(df, seeds):
    """data[model][split][source] = dict(gold, fam(str[]), base{cal,raw,logit},
    sft{seed:{cal,raw,logit}}), all aligned to the base row order (by sample_id)."""
    data = {}
    for mk in MODELS:
        data[mk] = {}
        for split in ("id_test", "transfer_test", CAL_SPLIT):
            data[mk][split] = {}
            srcs = REP if split in ("id_test", CAL_SPLIT) else TR
            for src in srcs:
                b = df[(df.model_key == mk) & (df.condition == "base") & (df.split == split) & (df.source == src)]
                b = b.sort_values("sample_id")
                if b.empty:
                    raise C.ArtifactContractError(
                        f"composition input lacks base cell {mk}/{split}/{src}")
                order = b["sample_id"].tolist()
                entry = {
                    "gold": b["gold"].to_numpy(int),
                    "fam": [str(f) for f in b["family_id"]],
                    "base": {"cal": b["probability_calibrated"].to_numpy(float),
                             "raw": b["probability_raw"].to_numpy(float),
                             "logit": b["score_raw"].to_numpy(float)},
                    "sft": {},
                }
                for s in seeds:
                    sf = df[(df.model_key == mk) & (df.condition == "sft") & (df.seed == s)
                            & (df.split == split) & (df.source == src)].set_index("sample_id").reindex(order)
                    if sf.isna().any().any():
                        raise C.ArtifactContractError(
                            f"composition input has incomplete SFT cell {mk}/seed_{s}/{split}/{src}")
                    entry["sft"][s] = {"cal": sf["probability_calibrated"].to_numpy(float),
                                       "raw": sf["probability_raw"].to_numpy(float),
                                       "logit": sf["score_raw"].to_numpy(float)}
                data[mk][split][src] = entry
    return data


# ----------------------------------------------------------------------------- combiners
def combiner_score(entry, s, name, pit=None):
    """Composed per-row score for combiner `name` on this benchmark entry, for ONE SFT
    seed `s`. (Seed averaging happens at the AP level in `macro`, matching Paper A's
    mean-of-per-seed-AP estimand -- i.e. base composed with a *single* adapter.)"""
    b = entry["base"]
    if name == "base":
        return b["cal"]
    sft = entry["sft"][s]
    if name == "sft":
        return sft["cal"]
    if name == "calibrated_avg":
        return 0.5 * (b["cal"] + sft["cal"])
    if name == "raw_avg":
        return 0.5 * (b["raw"] + sft["raw"])
    if name == "logit_avg":
        return 0.5 * (b["logit"] + sft["logit"])
    if name == "max_cal":
        return np.maximum(b["cal"], sft["cal"])
    if name == "pit_avg":
        fb, fs = pit
        return 0.5 * (fb(b["logit"]) + fs(sft["logit"]))
    if name.startswith("convex:"):
        w = float(name.split(":")[1])
        return w * sft["cal"] + (1.0 - w) * b["cal"]
    raise ValueError(name)


def macro(data, mk, split, sources, seeds, name, pit=None):
    """Benchmark-macro AP for guard `name`: per source, mean-of-per-seed AP (base has no
    seeds); then mean over sources. Equivalent to Paper A's seed-mean-of-macro-AP."""
    vals = []
    for src in sources:
        e = data[mk][split].get(src)
        if e is None:
            continue
        if name == "base":
            ap = wap(e["base"]["cal"], e["gold"])
        else:
            aps = [
                wap(combiner_score(
                    e, s, name, pit=(pit.get((mk, src, s)) if pit else None)), e["gold"])
                for s in seeds
            ]
            aps = [a for a in aps if not math.isnan(a)]
            ap = float(np.mean(aps)) if aps else float("nan")
        if not math.isnan(ap):
            vals.append(ap)
    return float(np.mean(vals)) if vals else float("nan")


def panel(data, split, sources, seeds, name, pit=None):
    return float(np.mean([macro(data, mk, split, sources, seeds, name, pit=pit) for mk in MODELS]))


# ----------------------------------------------------------------------------- PIT (calibration-only)
def fit_pit(data, seeds):
    """Empirical-CDF maps fit on calibration only, separately for every adapter seed."""
    pit = {}
    for mk in MODELS:
        base_c = []
        sft_c = {seed: [] for seed in seeds}
        for src in REP:
            e = data[mk][CAL_SPLIT].get(src)
            if e is None:
                continue
            base_c.append(e["base"]["logit"])
            for seed in seeds:
                sft_c[seed].append(e["sft"][seed]["logit"])
        if not base_c:
            continue
        bs = np.sort(np.concatenate(base_c))
        fb = lambda x, bs=bs: np.searchsorted(bs, np.asarray(x, float), side="right") / max(len(bs), 1)
        for seed in seeds:
            ss = np.sort(np.concatenate(sft_c[seed]))
            fs = lambda x, ss=ss: np.searchsorted(
                ss, np.asarray(x, float), side="right") / max(len(ss), 1)
            for src in REP + TR:
                pit[(mk, src, seed)] = (fb, fs)
    return pit


# ----------------------------------------------------------------------------- point estimates
def point_estimates(data, seeds, combiners, pit):
    out = {}
    for name in combiners:
        out[name] = {}
        for regime, (split, srcs) in REGIMES.items():
            per_model = {mk: macro(data, mk, split, srcs, seeds, name, pit=pit) for mk in MODELS}
            out[name][regime] = {"per_model": per_model, "panel": float(np.mean(list(per_model.values())))}
    return out


def select_convex_w(data, seeds):
    """Pick w on calibration only; no reported test split may select the combiner."""
    best_w, best = 0.0, -1.0
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        m = panel(data, CAL_SPLIT, REP, seeds, f"convex:{w}")
        if m > best:
            best, best_w = m, float(w)
    return best_w


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(data, seeds, reps, rng_seed, name="calibrated_avg"):
    """Paired hierarchical bootstrap of the composed guard's advantage. Poisson(1) weight
    per global family + resample seed indices within each checkpoint. Reports, per regime,
    per-model, per-benchmark, and panel percentile CIs for (ensemble - SFT) and
    (ensemble - base)."""
    if name != "calibrated_avg":
        raise ValueError(
            "bootstrap currently supports only the primary calibrated_avg combiner")
    rng = np.random.default_rng(rng_seed)
    fams = sorted({f for mk in MODELS for split in ("id_test", "transfer_test")
                   for e in data[mk][split].values() for f in e["fam"]})
    fam_idx = {f: i for i, f in enumerate(fams)}
    n_fam = len(fams)
    for mk in MODELS:
        for split in ("id_test", "transfer_test"):
            for e in data[mk][split].values():
                e["_fi"] = np.array([fam_idx[f] for f in e["fam"]], int)
    ns = len(seeds)

    entries = [(mk, split, e) for mk in MODELS for split in ("id_test", "transfer_test")
               for e in data[mk][split].values()]

    def valid(w):
        # only benchmarks that actually contain both classes must keep both classes weighted;
        # a genuinely single-class benchmark (none here) is skipped rather than hanging forever.
        for _mk, _sp, e in entries:
            g, fi = e["gold"], e["_fi"]
            has_pos, has_neg = (g == 1).any(), (g == 0).any()
            if has_pos and has_neg and (w[fi[g == 1]].sum() <= 0 or w[fi[g == 0]].sum() <= 0):
                return False
        return True

    keys = ["ens_minus_sft", "ens_minus_base"]
    samp = {
        regime: {
            key: {
                **{mk: np.empty(reps) for mk in MODELS},
                "panel": np.empty(reps),
                "per_benchmark": {src: np.empty(reps) for src in REGIMES[regime][1]},
            }
            for key in keys
        }
        for regime in REGIMES
    }

    redraws = 0
    progress_every = max(1, reps // 10)
    for rep in range(reps):
        tries = 0
        while True:
            w = rng.poisson(1.0, size=n_fam).astype(float)
            if valid(w):
                break
            redraws += 1
            tries += 1
            if tries > 2000:
                raise RuntimeError("bootstrap: exceeded redraw cap (data too sparse?)")
        pick = {mk: rng.integers(0, ns, size=ns) for mk in MODELS}
        for regime, (split, srcs) in REGIMES.items():
            d_es, d_eb = [], []
            source_deltas = {src: {"es": [], "eb": []} for src in srcs}
            for mk in MODELS:
                model_es, model_eb = [], []
                for src in srcs:
                    e = data[mk][split][src]
                    row_weights = w[e["_fi"]]
                    base_b = wap(e["base"]["cal"], e["gold"], row_weights)
                    sft_b = float(np.mean([
                        wap(e["sft"][seeds[j]]["cal"], e["gold"], row_weights)
                        for j in pick[mk]
                    ]))
                    ens_b = float(np.mean([
                        wap(0.5 * (e["base"]["cal"] + e["sft"][seeds[j]]["cal"]),
                            e["gold"], row_weights)
                        for j in pick[mk]
                    ]))
                    es, eb = ens_b - sft_b, ens_b - base_b
                    model_es.append(es)
                    model_eb.append(eb)
                    source_deltas[src]["es"].append(es)
                    source_deltas[src]["eb"].append(eb)
                model_es_mean = float(np.mean(model_es))
                model_eb_mean = float(np.mean(model_eb))
                samp[regime]["ens_minus_sft"][mk][rep] = model_es_mean
                samp[regime]["ens_minus_base"][mk][rep] = model_eb_mean
                d_es.append(model_es_mean)
                d_eb.append(model_eb_mean)
            samp[regime]["ens_minus_sft"]["panel"][rep] = float(np.mean(d_es))
            samp[regime]["ens_minus_base"]["panel"][rep] = float(np.mean(d_eb))
            for src in srcs:
                samp[regime]["ens_minus_sft"]["per_benchmark"][src][rep] = float(
                    np.mean(source_deltas[src]["es"]))
                samp[regime]["ens_minus_base"]["per_benchmark"][src][rep] = float(
                    np.mean(source_deltas[src]["eb"]))
        if reps >= 1000 and (rep + 1) % progress_every == 0:
            print(f"[composition] bootstrap {rep + 1}/{reps}", flush=True)

    def summ(a):
        return {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)),
                "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                "lcb95": float(np.percentile(a, 5)), "ucb95": float(np.percentile(a, 95))}

    out = {"reps": reps, "rng_seed": rng_seed, "n_families": n_fam, "redraws": redraws, "combiner": name}
    for regime in REGIMES:
        out[regime] = {
            key: {
                "panel": summ(samp[regime][key]["panel"]),
                "per_model": {mk: summ(samp[regime][key][mk]) for mk in MODELS},
                "per_benchmark": {
                    src: summ(values)
                    for src, values in samp[regime][key]["per_benchmark"].items()
                },
            }
            for key in keys
        }
    return out


# ----------------------------------------------------------------------------- leave-one-benchmark-out
def loo_benchmark(data, seeds, name="calibrated_avg", pit=None):
    out = {}
    for regime, (split, srcs) in REGIMES.items():
        full_eb = panel(data, split, srcs, seeds, name, pit) - panel(data, split, srcs, seeds, "base", pit)
        full_es = panel(data, split, srcs, seeds, name, pit) - panel(data, split, srcs, seeds, "sft", pit)
        loo = {}
        for drop in srcs:
            keep = [s for s in srcs if s != drop]
            eb = panel(data, split, keep, seeds, name, pit) - panel(data, split, keep, seeds, "base", pit)
            es = panel(data, split, keep, seeds, name, pit) - panel(data, split, keep, seeds, "sft", pit)
            loo[drop] = {"ens_minus_base": eb, "ens_minus_sft": es}
        out[regime] = {"full": {"ens_minus_base": full_eb, "ens_minus_sft": full_es}, "loo": loo,
                       "ens_minus_base_sign_stable": all(np.sign(v["ens_minus_base"]) == np.sign(full_eb) for v in loo.values()),
                       "ens_minus_sft_sign_stable": all(np.sign(v["ens_minus_sft"]) == np.sign(full_es) for v in loo.values())}
    return out


# ----------------------------------------------------------------------------- matched-FPR operating point
def operating_point(data, seeds, target_fpr, name="calibrated_avg"):
    """Evaluate one base or one adapter-at-a-time, then average seed metrics.

    This deliberately matches the Paper A mean-of-per-seed estimand. Averaging five
    adapters' probabilities first would describe a different, six-model deployment.
    """
    def guard_score(entry, guard, seed):
        if guard == "base":
            return entry["base"]["cal"]
        if guard == "sft":
            return entry["sft"][seed]["cal"]
        if guard == name:
            return 0.5 * (entry["base"]["cal"] + entry["sft"][seed]["cal"])
        raise ValueError(guard)

    def cal_scores(mk, guard, seed):
        s, y = [], []
        for src in REP:
            e = data[mk][CAL_SPLIT].get(src)
            if e is None:
                continue
            s.append(guard_score(e, guard, seed)); y.append(e["gold"])
        return np.concatenate(s), np.concatenate(y)

    out = {
        "target_fpr": target_fpr,
        "estimand": "base_once; SFT/composition mean_of_per_seed_operating_metrics",
    }
    for guard in ("base", "sft", name):
        units = [None] if guard == "base" else list(seeds)
        thresholds = {
            (mk, seed): select_threshold(
                *cal_scores(mk, guard, seed), target_fpr=target_fpr)
            for mk in MODELS for seed in units
        }
        g = {}
        for regime, (split, srcs) in REGIMES.items():
            tpr_m, fpr_m = [], []
            fp = tp = nneg = npos = 0
            for mk in MODELS:
                for seed in units:
                    selected = thresholds[(mk, seed)]
                    threshold = selected.get("threshold")
                    cutoff = float("inf") if threshold is None else float(threshold)
                    for src in srcs:
                        e = data[mk][split][src]
                        sc = guard_score(e, guard, seed)
                        gd = e["gold"]
                        pos, neg = gd == 1, gd == 0
                        if pos.sum():
                            tpr_m.append(float((sc[pos] >= cutoff).mean()))
                        if neg.sum():
                            fpr_m.append(float((sc[neg] >= cutoff).mean()))
                        tp += int((sc[pos] >= cutoff).sum()); npos += int(pos.sum())
                        fp += int((sc[neg] >= cutoff).sum()); nneg += int(neg.sum())
            g[regime] = {"macro_tpr": float(np.mean(tpr_m)) if tpr_m else float("nan"),
                         "macro_fpr": float(np.mean(fpr_m)) if fpr_m else float("nan"),
                         "pooled_tpr": tp / npos if npos else float("nan"),
                         "pooled_fpr": fp / nneg if nneg else float("nan"),
                         "n_seed_units_per_model": len(units)}
        out[guard] = g
    return out


# ----------------------------------------------------------------------------- shuffle diagnostics
def shuffle_null(data, seeds, rng_seed=20260714, name="calibrated_avg"):
    """Two single-permutation diagnostics, preserving the seed estimand.

    - ``signal`` permutes SFT scores across all rows within each model/benchmark, breaking
      their alignment with labels in this randomization.
    - ``complementarity`` permutes SFT scores within each gold class. This preserves the
      labeled score multisets (and therefore marginal AP) while breaking row-level pairing
      with the base scores.

    A single draw is a descriptive sensitivity ablation, not a randomization distribution.
    It can narrow explanations but cannot establish a causal mechanism or prove that the gain
    is not an averaging artifact. The historical output keys retain ``_null_`` for compatibility.
    """
    def panel_eb(split, srcs, mode, rng):
        vals = []
        for mk in MODELS:
            mv = []
            for src in srcs:
                e = data[mk][split][src]
                base_ap = wap(e["base"]["cal"], e["gold"])
                seed_deltas = []
                for seed in seeds:
                    ps = e["sft"][seed]["cal"].copy()
                    if mode == "signal":
                        ps = ps[rng.permutation(len(ps))]
                    elif mode == "complementarity":
                        for cls in (0, 1):
                            idx = np.where(e["gold"] == cls)[0]
                            ps[idx] = ps[idx][rng.permutation(len(idx))]
                    ens = 0.5 * (e["base"]["cal"] + ps)
                    seed_deltas.append(wap(ens, e["gold"]) - base_ap)
                mv.append(float(np.mean(seed_deltas)))
            mv = [v for v in mv if not math.isnan(v)]
            if mv:
                vals.append(float(np.mean(mv)))
        return float(np.mean(vals))

    out = {}
    for regime, (split, srcs) in REGIMES.items():
        out[regime] = {
            "real_ens_minus_base": panel_eb(split, srcs, "real", np.random.default_rng(rng_seed)),
            "signal_null_ens_minus_base": panel_eb(split, srcs, "signal", np.random.default_rng(rng_seed)),
            "complementarity_null_ens_minus_base": panel_eb(split, srcs, "complementarity", np.random.default_rng(rng_seed + 1)),
        }
    return out


# ----------------------------------------------------------------------------- render
def render_md(res):
    evidence = ("legacy execution artifact" if res["legacy"]
                else "clean v2 execution artifact; retrospective cohort")
    L = [f"# Composition analysis — Compose, Don't Tune ({evidence})", ""]
    L.append(f"Scores: `{res['scores_sha256'][:16]}…`  ·  seeds {res['seeds']}  ·  "
             f"bootstrap reps {res['bootstrap']['reps']} (rng {res['bootstrap']['rng_seed']}).")
    L.append(f"Lock: `{res['lock_sha256'][:16]}…`  ·  analysis status: "
             f"**{res['analysis_status']}**.")
    L += ["", "## Panel macro-AP by combiner (represented / transfer)", "",
          "| Combiner | represented | transfer |", "|---|---:|---:|"]
    pe = res["point_estimates"]
    for name in res["combiner_order"]:
        r = pe[name]["represented"]["panel"]; t = pe[name]["transfer"]["panel"]
        L.append(f"| {name} | {r:.3f} | {t:.3f} |")
    L += ["", "## Per-model transfer macro-AP (base / SFT / composed calibrated_avg)", "",
          "| Model | base | SFT | composed |", "|---|---:|---:|---:|"]
    for mk in MODELS:
        b = pe["base"]["transfer"]["per_model"][mk]
        s = pe["sft"]["transfer"]["per_model"][mk]
        c = pe["calibrated_avg"]["transfer"]["per_model"][mk]
        L.append(f"| {mk} | {b:.3f} | {s:.3f} | {c:.3f} |")
    bt = res["bootstrap"]
    L += ["", "## Bootstrap CIs — composed(calibrated_avg) advantage (panel)", "",
          "| Regime | ens − SFT [95% CI] | ens − base [95% CI] |", "|---|---|---|"]
    for regime in REGIMES:
        es = bt[regime]["ens_minus_sft"]["panel"]; eb = bt[regime]["ens_minus_base"]["panel"]
        L.append(f"| {regime} | {es['mean']:+.3f} [{es['ci95'][0]:+.3f}, {es['ci95'][1]:+.3f}] | "
                 f"{eb['mean']:+.3f} [{eb['ci95'][0]:+.3f}, {eb['ci95'][1]:+.3f}] |")
    L += ["", "### Per-model transfer ens − base [95% CI]", ""]
    for mk in MODELS:
        eb = bt["transfer"]["ens_minus_base"]["per_model"][mk]
        L.append(f"- {mk}: {eb['mean']:+.3f} [{eb['ci95'][0]:+.3f}, {eb['ci95'][1]:+.3f}]")
    L += ["", "### Per-benchmark transfer advantages [95% CI]", "",
          "| Benchmark | ensemble − SFT | ensemble − base |", "|---|---:|---:|"]
    for src in TR:
        es = bt["transfer"]["ens_minus_sft"]["per_benchmark"][src]
        eb = bt["transfer"]["ens_minus_base"]["per_benchmark"][src]
        L.append(
            f"| {src} | {es['mean']:+.3f} [{es['ci95'][0]:+.3f}, {es['ci95'][1]:+.3f}] | "
            f"{eb['mean']:+.3f} [{eb['ci95'][0]:+.3f}, {eb['ci95'][1]:+.3f}] |")
    op = res["operating_point"]
    L += ["", f"## Matched-FPR operating point (target {op['target_fpr']:.0%}) — realized rates", "",
          "| Guard | regime | macro TPR | macro FPR | pooled FPR |", "|---|---|---:|---:|---:|"]
    for guard in ("base", "sft", "calibrated_avg"):
        for regime in REGIMES:
            g = op[guard][regime]
            L.append(f"| {guard} | {regime} | {g['macro_tpr']:.3f} | {g['macro_fpr']:.3f} | {g['pooled_fpr']:.3f} |")
    sn = res["shuffle_null"]
    L += ["", "## Single-permutation shuffle diagnostics (panel ens − base)", "",
          "| Regime | real | label-alignment shuffle | within-class row-pairing shuffle |",
          "|---|---:|---:|---:|"]
    for regime in REGIMES:
        L.append(f"| {regime} | {sn[regime]['real_ens_minus_base']:+.3f} | "
                 f"{sn[regime]['signal_null_ens_minus_base']:+.3f} | "
                 f"{sn[regime]['complementarity_null_ens_minus_base']:+.3f} |")
    L += ["", "*This is retrospective, precision-focused evidence, not a prospective "
          "confirmatory result. Clean v2 execution repairs provenance but does not erase prior "
          "exposure to part of the transfer cohort. WiSE-FT weight interpolation is out of "
          "scope for this output-space analyzer.*", ""]
    return "\n".join(L)


def write_composition_artifacts(
        res, out_dir, *, lock, input_verification, parameter_record,
        release_cache=False, nonfinal=False):
    """Write deterministic science separately from mode/runtime provenance."""
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "composition.json")
    markdown_path = os.path.join(out_dir, "composition.md")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    rendered = render_md(res)
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    legacy = bool(input_verification["legacy"])
    execution_mode = ("legacy" if legacy else
                      "release_cache" if release_cache else "full_artifact")
    metadata = {
        "composition_metadata_contract_version": 1,
        "analysis": "composition_v2",
        "execution_mode": execution_mode,
        "nonfinal": bool(nonfinal),
        "lock_sha256": lock["lock_sha256"],
        "scores_sha256": res["scores_sha256"],
        "analysis_source_sha256": C.sha256_file(__file__),
        **composition_analysis_attestation(parameter_record),
        "input_verification": input_verification,
        "outputs": {
            "composition.json": C.sha256_file(result_path),
            "composition.md": C.sha256_file(markdown_path),
        },
    }
    C.write_json(os.path.join(out_dir, "composition_metadata.json"), metadata)
    return rendered, metadata


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=os.path.join(
        _ROOT, "artifacts/paper_a_sft/LOCK.json"))
    ap.add_argument("--scores", default=os.path.join(_ROOT, "artifacts/paper_a_sft/scores/scores.parquet"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "artifacts/paper_a_sft/analysis/composition"))
    ap.add_argument("--reps", type=int, default=PROTOTYPE_REPS)
    ap.add_argument("--rng-seed", type=int, default=PROTOTYPE_RNG_SEED)
    ap.add_argument("--target-fpr", type=float, default=PROTOTYPE_TARGET_FPR)
    ap.add_argument("--allow-legacy-lock", action="store_true",
                    help="explicitly admit the immutable v1 score/lock compatibility path")
    ap.add_argument(
        "--release-cache", action="store_true",
        help=("strict-final-v2 score-only mode using bound text-free public manifests; "
              "raw manifests, run metadata, and adapters may be absent"))
    ap.add_argument(
        "--nonfinal", action="store_true",
        help=("permit prototype-parameter overrides only when --out is outside all "
              "canonical artifact roots"))
    args = ap.parse_args(argv)

    if args.reps <= 0:
        ap.error("--reps must be positive")
    if not (0.0 < args.target_fpr < 1.0):
        ap.error("--target-fpr must lie strictly between 0 and 1")
    if args.release_cache and args.allow_legacy_lock:
        ap.error("--release-cache cannot be combined with --allow-legacy-lock")
    try:
        parameter_record = composition_parameter_record(
            args.reps, args.rng_seed, args.target_fpr, nonfinal=args.nonfinal)
    except C.ArtifactContractError as exc:
        ap.error(str(exc))
    df, lock, input_verification = load_verified(
        args.scores, args.lock, allow_legacy=args.allow_legacy_lock,
        release_cache=args.release_cache, nonfinal=args.nonfinal)
    validate_composition_paths(
        lock, args.scores, args.out, release_cache=args.release_cache,
        nonfinal=args.nonfinal)
    sha = input_verification["scores_sha256"]
    seeds = list(input_verification["matrix"]["seeds"])
    data = build(df, seeds)
    pit = fit_pit(data, seeds)

    w = select_convex_w(data, seeds)
    combiners = ["base", "sft", "calibrated_avg", "raw_avg", "logit_avg", "max_cal", "pit_avg", f"convex:{w}"]
    pe = point_estimates(data, seeds, combiners, pit)
    # normalize the convex key name for reporting
    pe["convex_blind"] = pe.pop(f"convex:{w}")
    pe["convex_blind"]["selected_w_on_calibration"] = w
    order = ["base", "sft", "calibrated_avg", "raw_avg", "logit_avg", "max_cal", "pit_avg", "convex_blind"]

    print("[composition] point estimates done; running bootstrap…", flush=True)
    bt = bootstrap(
        data, seeds, args.reps, args.rng_seed, name=PROTOTYPE_PRIMARY_COMBINER)
    loo = loo_benchmark(data, seeds, pit=pit)
    op = operating_point(data, seeds, args.target_fpr)
    sn = shuffle_null(
        data, seeds, rng_seed=parameter_record["shuffle_rng_seed"])

    legacy = bool(input_verification["legacy"])
    res = {"analysis": "composition_v2", "scores_sha256": sha, "seeds": seeds,
           "legacy": legacy, "analysis_status": ("legacy_retrospective_estimation"
                                                    if legacy else
                                                    "clean_v2_retrospective_estimation"),
           "prospective_confirmatory": False,
           "lock_sha256": lock["lock_sha256"],
           "lock_contract_version": lock.get("lock_contract_version"),
           "analysis_mode": lock.get("analysis_mode"),
           "analysis_parameters": parameter_record,
           "combiner_order": order, "convex_selected_w": w,
           "convex_selection_split": CAL_SPLIT,
           "point_estimates": pe, "bootstrap": bt, "leave_one_benchmark_out": loo,
           "operating_point": op, "shuffle_null": sn,
           "note": "Retrospective estimation only. Clean v2 execution does not make the "
                   "previously exposed transfer cohort prospective. WiSE-FT is out of scope."}

    rendered, _metadata = write_composition_artifacts(
        res, args.out, lock=lock, input_verification=input_verification,
        parameter_record=parameter_record, release_cache=args.release_cache,
        nonfinal=args.nonfinal)
    print(rendered)
    print(f"\n[composition] wrote {args.out}/composition.json, composition.md, "
          "and composition_metadata.json")


if __name__ == "__main__":
    main()
