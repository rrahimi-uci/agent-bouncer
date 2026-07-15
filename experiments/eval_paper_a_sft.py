#!/usr/bin/env python
"""Paper A scorer (plan sec 10).

Scores the 4 untuned bases ONCE each plus all 20 trained SFT adapters (24 model
bundles) on calibration / id_test / transfer_test / OR-Bench benign stress /
HarmBench positive stress, using the single-forward two-token logprob head
(reuses experiments/guard_eval_pipeline.py idioms):

    s(x) = z_unsafe(x, t_last) - z_safe(x, t_last)          (score_raw)
    p(unsafe|x) = softmax([z_unsafe, z_safe])[0]            (probability_raw)

Emits per-row records (plan sec 10.2) to scores/scores.parquet + metadata.json.
Base scores are computed once per checkpoint and reused. Cache validity (plan
sec 10.3) matches manifest hash, ordered sample ids, content hashes, model /
tokenizer / adapter / prompt / score-code hashes and dtype+device policy --
NEVER row count alone. Temperature is fit on CALIBRATION rows only (plan sec
10.4); a conservative 5% FPR threshold is taken from
guard_research.thresholds.select_threshold on calibration data.

Usage:
  python experiments/eval_paper_a_sft.py --lock artifacts/paper_a_sft_v2/LOCK.json \
    --out artifacts/paper_a_sft_v2/scores
  # local plumbing test without models/GPU:
  python experiments/eval_paper_a_sft.py --lock LOCK.json --out /tmp/scores \
    --synthetic --allow-incomplete --nonfinal
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import hashlib
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402

SCORE_CODE_VERSION = "paper_a_sft_scorer_v2"
# Manifests scored (train.jsonl is deliberately excluded).
SCORING_MANIFESTS = [
    ("calibration.jsonl", "calibration"),
    ("id_test.jsonl", "id_test"),
    ("transfer_test.jsonl", "transfer_test"),
    ("orbench_safe_stress.jsonl", "stress_orbench"),
    ("harmbench_positive_stress.jsonl", "stress_harmbench"),
]

SCORE_COLUMNS = [
    "sample_id", "content_sha256", "source", "split", "gold", "family_id",
    "model_key", "model_revision", "condition", "seed", "adapter_sha256",
    "prompt_sha256", "safe_token_id", "unsafe_token_id", "safe_logit",
    "unsafe_logit", "score_raw", "probability_raw", "probability_calibrated",
    "threshold_id", "prediction", "original_token_count", "scored_token_count",
    "truncated", "truncation_strategy", "latency_ms",
]


# --------------------------------------------------------------------------------------
# manifest loading (calibration + eval; verify content hashes)
# --------------------------------------------------------------------------------------
def load_scoring_rows(manifests_dir: str, limit: int | None) -> list[dict]:
    rows = []
    for fn, split in SCORING_MANIFESTS:
        p = os.path.join(manifests_dir, fn)
        if not os.path.exists(p):
            raise SystemExit(f"[eval] scoring manifest missing: {p}")
        recs = C.read_jsonl(p)
        if limit:
            recs = recs[:limit]
        for r in recs:
            text = C.row_text(r)
            if not r.get("sample_id") or not r.get("family_id") or not r.get("content_sha256"):
                raise SystemExit(f"[eval] incomplete manifest identity in {p}")
            observed_content = C.content_sha256(text)
            if observed_content != r.get("content_sha256"):
                raise SystemExit(
                    f"[eval] content hash mismatch for sample {r.get('sample_id')}: "
                    f"stored={r.get('content_sha256')} observed={observed_content}")
            rows.append({
                "sample_id": r.get("sample_id"),
                "content_sha256": r.get("content_sha256"),
                "source": r.get("source"),
                "split": split,
                "gold": C.to_gold(r.get("label")),
                "family_id": r.get("family_id"),
                "text": text,
            })
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("[eval] duplicate sample_id across scoring manifests (fail closed).")
    return rows


def manifest_fingerprints(manifests_dir: str, rows: list[dict]) -> dict:
    per_split = {}
    for fn, split in SCORING_MANIFESTS:
        p = os.path.join(manifests_dir, fn)
        per_split[split] = C.sha256_file(p) if os.path.exists(p) else None
    return {
        "manifest_sha256": C.sha256_ordered([f"{k}={v}" for k, v in sorted(per_split.items())]),
        "sample_ids_fingerprint": C.sha256_ordered([str(r["sample_id"]) for r in rows]),
        "content_fingerprint": C.sha256_ordered([str(r["content_sha256"]) for r in rows]),
        "per_split_manifest_sha256": per_split,
        "n_rows": len(rows),
    }


# --------------------------------------------------------------------------------------
# calibration (plan sec 10.4): one positive temperature, NLL on calibration only
# --------------------------------------------------------------------------------------
def fit_temperature(score_raw: np.ndarray, gold: np.ndarray) -> dict:
    """Fit p = sigmoid(score_raw / T), T>0, minimizing binary NLL on calibration rows."""
    s = np.asarray(score_raw, float)
    y = np.asarray(gold, float)
    stats = {"n": int(s.size), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    try:
        from guard_research.metrics import brier as _brier, log_loss_ as _ll
    except Exception:
        def _brier(p, yy): p = np.asarray(p); yy = np.asarray(yy); return float(np.mean((p - yy) ** 2))
        def _ll(p, yy, eps=1e-12):
            p = np.clip(np.asarray(p), eps, 1 - eps); yy = np.asarray(yy)
            return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

    def sig(x): return 1.0 / (1.0 + np.exp(-x))
    p_before = sig(s)
    if s.size == 0 or y.min() == y.max():
        stats.update({"temperature": 1.0, "status": "single_class_or_empty",
                      "nll_before": _ll(p_before, y) if s.size else None,
                      "nll_after": _ll(p_before, y) if s.size else None,
                      "brier_before": _brier(p_before, y) if s.size else None,
                      "brier_after": _brier(p_before, y) if s.size else None})
        return stats
    from scipy.optimize import minimize

    def nll(u):  # u = log T ; T = exp(u) > 0
        T = np.exp(u[0])
        p = np.clip(sig(s / T), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B")
    success = bool(res.success)
    try:
        T = float(np.exp(res.x[0]))
    except (AttributeError, IndexError, TypeError, ValueError):
        T = float("nan")
    if not success or not np.isfinite(T) or T <= 0 or not np.isfinite(nll([np.log(T)])):
        stats.update({
            "temperature": None,
            "status": "optimization_failed",
            "optim_success": success,
            "optim_message": str(getattr(res, "message", "optimizer returned no message")),
            "nll_before": _ll(p_before, y),
            "nll_after": None,
            "brier_before": _brier(p_before, y),
            "brier_after": None,
        })
        return stats
    p_after = sig(s / T)
    stats.update({
        "temperature": T, "status": "ok", "optim_success": True,
        "optim_message": str(getattr(res, "message", "")),
        "nll_before": _ll(p_before, y), "nll_after": _ll(p_after, y),
        "brier_before": _brier(p_before, y), "brier_after": _brier(p_after, y),
    })
    return stats


def calibrated_prob(score_raw: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(score_raw, float) / max(T, 1e-6))))


# --------------------------------------------------------------------------------------
# scoring backends
# --------------------------------------------------------------------------------------
def _synthetic_logits(content_sha, gold, model_key, condition, seed):
    """Deterministic pseudo-logits for local plumbing tests (no model load).

    SFT bundles get slightly stronger class separation than base so that AP
    deltas and calibration behave sensibly end-to-end."""
    key = f"{content_sha}|{model_key}|{condition}|{seed}"
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(h)
    sep = 1.6 if condition == "sft" else 1.0
    sign = (2 * int(gold) - 1)
    unsafe_logit = sep * 0.5 * sign + rng.normal(0, 0.6)
    safe_logit = -sep * 0.5 * sign + rng.normal(0, 0.6)
    return float(safe_logit), float(unsafe_logit)


def score_bundle(lock, rows, model_key, condition, seed, adapter_dir_path,
                 adapter_sha256, device, dtype, batch_size, synthetic):
    """Return (per_row_logit_dicts, prompt_template_sha256, decision_tokens)."""
    m = C.lock_model_panel(lock)[model_key]
    max_len = int(lock.get("recipe", {}).get("max_length", 1024))

    if synthetic:
        prompt_sha = lock.get("prompt", {}).get("per_model_template_sha256", {}).get(
            model_key) or f"synthetic_template::{model_key}"
        dtoks = {"safe_id": 0, "unsafe_id": 1}
        out = []
        for r in rows:
            safe_l, unsafe_l = _synthetic_logits(r["content_sha256"], r["gold"],
                                                 model_key, condition, seed)
            out.append({"safe_logit": safe_l, "unsafe_logit": unsafe_l,
                        "original_token_count": len(str(r["text"]).split()),
                        "scored_token_count": min(len(str(r["text"]).split()), max_len),
                        "truncated": False, "truncation_strategy": "none",
                        "latency_ms": 0.0})
        return out, prompt_sha, dtoks

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    build_prompt, _ = C.require_prompts()

    tok = AutoTokenizer.from_pretrained(
        m["model_id"], revision=m["tokenizer_revision"],
        trust_remote_code=bool(m.get("trust_remote_code", True)))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"; tok.truncation_side = "left"
    dt = C.resolve_decision_tokens(tok)
    prompt_sha = C.template_sha256(tok)

    locked_dtype = str(dtype or m.get("dtype", "bfloat16"))
    torch_dtype = C.torch_dtype_from_name(torch, locked_dtype)
    model_kwargs = {
        "revision": m["model_revision"],
        "dtype": torch_dtype,
        "trust_remote_code": bool(m.get("trust_remote_code", True)),
    }
    if m.get("attn_implementation"):
        model_kwargs["attn_implementation"] = m["attn_implementation"]
    base = AutoModelForCausalLM.from_pretrained(m["model_id"], **model_kwargs)
    model = (PeftModel.from_pretrained(base, adapter_dir_path) if condition == "sft" else base)
    model = model.eval().to(device)

    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            rendered = [C.budgeted_prompt(tok, build_prompt, r["text"], max_len)
                        for r in chunk]
            prompts = [item[0] for item in rendered]
            trunc_stats = [item[1] for item in rendered]
            if not all(stat["wrapper_preserved"] for stat in trunc_stats):
                raise C.ArtifactContractError("scoring prompt lost classifier wrapper")
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=False,
                      add_special_tokens=False).to(device)
            if int(enc["attention_mask"].sum(1).max()) > max_len:
                raise C.ArtifactContractError("budgeted scoring prompt exceeds locked max_length")
            t0 = time.time()
            logits = model(**enc).logits
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.time() - t0) * 1000.0 / max(1, len(chunk))
            last = enc["attention_mask"].sum(1) - 1
            picked = logits[torch.arange(len(chunk)), last]
            for j, r in enumerate(chunk):
                sl = float(picked[j, dt["safe_id"]]); ul = float(picked[j, dt["unsafe_id"]])
                stat = trunc_stats[j]
                out.append({"safe_logit": sl, "unsafe_logit": ul,
                            "original_token_count": int(stat["original_token_count"]),
                            "scored_token_count": int(stat["scored_token_count"]),
                            "truncated": bool(stat["truncated"]),
                            "truncation_strategy": stat["truncation_strategy"],
                            "latency_ms": float(dt_ms)})
    del model, base
    return out, prompt_sha, dt


# --------------------------------------------------------------------------------------
# assemble one bundle's rows (logits -> calibrated probs + threshold + prediction)
# --------------------------------------------------------------------------------------
def assemble_bundle(lock, rows, logits, model_key, model_revision, condition, seed,
                    adapter_sha256, prompt_sha, dtoks, target_fpr):
    safe_id = dtoks.get("safe_id"); unsafe_id = dtoks.get("unsafe_id")
    score_raw = np.array([lg["unsafe_logit"] - lg["safe_logit"] for lg in logits], float)
    prob_raw = 1.0 / (1.0 + np.exp(-score_raw))
    gold = np.array([r["gold"] for r in rows], int)
    split = np.array([r["split"] for r in rows])

    cal_mask = split == "calibration"
    cal_stats = fit_temperature(score_raw[cal_mask], gold[cal_mask])
    if cal_stats.get("status") != "ok" or cal_stats.get("optim_success") is not True:
        raise C.ArtifactContractError(
            f"temperature fitting failed for {model_key}/{condition}/seed_{seed}: "
            f"{cal_stats.get('optim_message', cal_stats.get('status'))}")
    T = cal_stats.get("temperature", 1.0)
    cal_stats["source_composition"] = _source_composition(rows, cal_mask)
    prob_cal = calibrated_prob(score_raw, T)

    threshold_id = f"{model_key}:{condition}:{seed}:fpr{target_fpr}"
    thr_info = {"threshold_id": threshold_id}
    cal_scores = prob_cal[cal_mask]; cal_labels = gold[cal_mask]
    if cal_labels.size and cal_labels.min() != cal_labels.max():
        select_threshold = C.require_select_threshold()
        raw = select_threshold(cal_scores.tolist(), cal_labels.tolist(), target_fpr)
        norm = C.normalize_threshold_result(raw)
        raw_record = dict(norm["extra"])
        if norm["status"] == "ok":
            raw_record["threshold"] = float(norm["threshold"])
        elif norm["status"] == "PREDICT_NONE":
            raw_record.update({
                "threshold": None,
                "threshold_kind": "positive_infinity_predict_none",
            })
        thr_info.update({"status": norm["status"], "raw": raw_record})
        thr = norm["threshold"]
        if norm["status"] != "ok" or thr is None:
            thr = float("inf")  # conservative: predict no positives
            thr_info["status"] = norm["status"] or "NO_FEASIBLE_THRESHOLD"
    else:
        thr = float("inf")
        thr_info["status"] = "NO_CALIBRATION_TWO_CLASS"
    thr_info["threshold_value"] = (None if np.isinf(thr) else float(thr))
    pred = (prob_cal >= thr).astype(int)

    recs = []
    for i, r in enumerate(rows):
        recs.append({
            "sample_id": r["sample_id"], "content_sha256": r["content_sha256"],
            "source": r["source"], "split": r["split"], "gold": int(r["gold"]),
            "family_id": r["family_id"], "model_key": model_key,
            "model_revision": model_revision, "condition": condition,
            "seed": (int(seed) if seed is not None else -1), "adapter_sha256": adapter_sha256,
            "prompt_sha256": prompt_sha, "safe_token_id": safe_id, "unsafe_token_id": unsafe_id,
            "safe_logit": float(logits[i]["safe_logit"]),
            "unsafe_logit": float(logits[i]["unsafe_logit"]),
            "score_raw": float(score_raw[i]), "probability_raw": float(prob_raw[i]),
            "probability_calibrated": float(prob_cal[i]), "threshold_id": threshold_id,
            "prediction": int(pred[i]), "original_token_count": logits[i]["original_token_count"],
            "scored_token_count": logits[i]["scored_token_count"],
            "truncated": bool(logits[i]["truncated"]),
            "truncation_strategy": logits[i].get("truncation_strategy", "none"),
            "latency_ms": float(logits[i]["latency_ms"]),
        })
    return recs, {
        "calibration": cal_stats,
        "threshold": thr_info,
        "truncation": {
            "strategy": C.TRUNCATION_STRATEGY,
            "n_truncated": int(sum(bool(row["truncated"]) for row in recs)),
            "n_examples": len(recs),
            "classifier_wrapper_preserved": True,
            "assistant_generation_prefix_preserved": True,
        },
    }


def _source_composition(rows, mask):
    comp = {}
    for r, keep in zip(rows, mask):
        if keep:
            comp.setdefault(r["source"], {"n": 0, "pos": 0})
            comp[r["source"]]["n"] += 1
            comp[r["source"]]["pos"] += int(r["gold"])
    return comp


# --------------------------------------------------------------------------------------
# completeness + cache
# --------------------------------------------------------------------------------------
def collect_adapters(lock, runs_root, allow_legacy=False):
    """Return revalidated adapter records; stored hashes are never trusted alone."""
    out = {}
    for mk in C.MODEL_KEYS:
        for s in C.lock_seeds(lock):
            rd = C.run_dir(runs_root, mk, s)
            validation = C.validate_run_artifact(
                lock, mk, s, rd, allow_legacy=allow_legacy,
                recompute_adapter_hash=True)
            meta = validation.get("metadata") or {}
            run_meta_path = os.path.join(rd, "run_meta.json")
            info = {
                "adapter_dir": validation["adapter_dir"],
                "adapter_sha256": validation["adapter_sha256"],
                "run_meta_path": (os.path.relpath(run_meta_path, C.REPO_ROOT)
                                  if C.path_is_within(run_meta_path, C.REPO_ROOT)
                                  else str(pathlib.Path(run_meta_path).resolve())),
                "run_meta_sha256": (C.sha256_file(run_meta_path)
                                    if os.path.isfile(run_meta_path) else None),
                "status": meta.get("status", "missing"),
                "valid": bool(validation["valid"]),
                "issues": list(validation["issues"]),
            }
            out[(mk, s)] = info
    return out


def build_expected_meta(lock, fps, model_rev, tok_rev, adapter_sha, run_meta_sha,
                        prompt_sha, dtype, device, batch_size, producer_runtime_sha256):
    return {
        "manifest_sha256": fps["manifest_sha256"],
        "sample_ids_fingerprint": fps["sample_ids_fingerprint"],
        "content_fingerprint": fps["content_fingerprint"],
        "model_revision": model_rev, "tokenizer_revision": tok_rev,
        "adapter_sha256": adapter_sha, "run_meta_sha256": run_meta_sha,
        "prompt_sha256": prompt_sha,
        "score_code_version": SCORE_CODE_VERSION, "dtype": dtype, "device_policy": device,
        "batch_size": int(batch_size),
        "producer_runtime_sha256": producer_runtime_sha256,
        "n_rows": fps["n_rows"], "lock_sha256": lock.get("lock_sha256"),
    }


def _read_cache(parquet_path, meta_path):
    if not (os.path.exists(parquet_path) and os.path.exists(meta_path)):
        return None, None
    import pandas as pd
    try:
        meta = C.read_json(meta_path)
        expected_hash = meta.get("parquet_sha256")
        if not expected_hash or C.sha256_file(parquet_path) != expected_hash:
            return None, meta
        return pd.read_parquet(parquet_path), meta
    except Exception:
        return None, None


def _cache_rows_match(cached_df, rows, expected) -> tuple[bool, list[str]]:
    issues = []
    if cached_df is None:
        return False, ["parquet_missing_or_hash_mismatch"]
    columns_ok = list(cached_df.columns) == SCORE_COLUMNS
    if not columns_ok:
        issues.append("columns")
    if len(cached_df) != expected.get("n_rows"):
        issues.append("n_rows")
    if columns_ok and len(cached_df) == len(rows):
        identity_fields = (
            ("sample_id", str), ("content_sha256", str), ("source", str),
            ("split", str), ("gold", int), ("family_id", str),
        )
        for field, normalize in identity_fields:
            try:
                observed = [normalize(value) for value in cached_df[field].tolist()]
                wanted = [normalize(row[field]) for row in rows]
            except (KeyError, TypeError, ValueError):
                issues.append(f"{field}_identity")
                continue
            if observed != wanted:
                issues.append(f"{field}_identity")
    return not issues, issues


def validate_combined_scores(df, rows, model_keys, seeds, bundle_meta=None,
                             require_complete=True, target_fpr=None) -> None:
    """Fail closed before publishing the combined score artifact."""
    missing_columns = [column for column in SCORE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise C.ArtifactContractError(f"combined score table lacks columns: {missing_columns}")
    keys = ["model_key", "condition", "seed", "sample_id"]
    if bool(df.duplicated(keys).any()):
        raise C.ArtifactContractError("combined score table has duplicate bundle/sample keys")
    expected_bundles = {(mk, "base", -1) for mk in model_keys}
    expected_bundles.update((mk, "sft", int(seed)) for mk in model_keys for seed in seeds)
    observed_bundles = set(zip(df["model_key"], df["condition"], df["seed"].astype(int)))
    if require_complete and observed_bundles != expected_bundles:
        raise C.ArtifactContractError(
            f"combined score bundles mismatch: missing={sorted(expected_bundles-observed_bundles)} "
            f"extra={sorted(observed_bundles-expected_bundles)}")
    expected_ids = [str(row["sample_id"]) for row in rows]
    for bundle, group in df.groupby(["model_key", "condition", "seed"], sort=False):
        if group["sample_id"].astype(str).tolist() != expected_ids:
            raise C.ArtifactContractError(f"bundle {bundle} has incomplete/reordered sample identities")
    numeric = ["safe_logit", "unsafe_logit", "score_raw", "probability_raw",
               "probability_calibrated", "latency_ms"]
    if not np.isfinite(df[numeric].to_numpy(float)).all():
        raise C.ArtifactContractError("combined score table contains non-finite numeric evidence")
    if not np.allclose(df["score_raw"].to_numpy(float),
                       df["unsafe_logit"].to_numpy(float) - df["safe_logit"].to_numpy(float),
                       rtol=0.0, atol=1e-10):
        raise C.ArtifactContractError("score_raw does not equal unsafe_logit-safe_logit")
    expected_raw_probability = 1.0 / (1.0 + np.exp(-df["score_raw"].to_numpy(float)))
    if not np.allclose(df["probability_raw"].to_numpy(float), expected_raw_probability,
                       rtol=0.0, atol=1e-10):
        raise C.ArtifactContractError("probability_raw does not equal sigmoid(score_raw)")
    if bundle_meta is not None:
        for (model_key, condition, seed), group in df.groupby(
                ["model_key", "condition", "seed"], sort=False):
            key = (f"{model_key}:base" if condition == "base" else
                   f"{model_key}:sft:seed_{int(seed)}")
            meta = bundle_meta.get(key)
            if not isinstance(meta, dict):
                raise C.ArtifactContractError(f"missing bundle metadata for {key}")
            row_adapter_values = set(group["adapter_sha256"].dropna().astype(str))
            if condition == "base":
                if row_adapter_values or meta.get("adapter_sha256") is not None:
                    raise C.ArtifactContractError(f"base bundle has adapter identity for {key}")
            elif row_adapter_values != {str(meta.get("adapter_sha256"))}:
                raise C.ArtifactContractError(
                    f"bundle adapter identity differs from score rows for {key}")
            calibration = meta.get("calibration") or {}
            if calibration.get("status") != "ok" or calibration.get("optim_success") is not True:
                raise C.ArtifactContractError(f"calibration optimizer did not succeed for {key}")
            temperature = float(calibration.get("temperature", 1.0))
            if not np.isfinite(temperature) or temperature <= 0:
                raise C.ArtifactContractError(f"invalid calibration temperature for {key}")
            expected_calibrated = 1.0 / (
                1.0 + np.exp(-(group["score_raw"].to_numpy(float) / max(temperature, 1e-6))))
            if not np.allclose(group["probability_calibrated"].to_numpy(float),
                               expected_calibrated, rtol=0.0, atol=1e-10):
                raise C.ArtifactContractError(
                    f"probability_calibrated is inconsistent with bundle temperature for {key}")
            threshold = meta.get("threshold") or {}
            threshold_id = threshold.get("threshold_id")
            if not threshold_id or set(group["threshold_id"].astype(str)) != {str(threshold_id)}:
                raise C.ArtifactContractError(f"threshold identity mismatch for {key}")
            threshold_value = threshold.get("threshold_value")
            if target_fpr is not None:
                calibration_rows = group[group["split"] == "calibration"]
                selected = C.normalize_threshold_result(C.require_select_threshold()(
                    calibration_rows["probability_calibrated"].astype(float).tolist(),
                    calibration_rows["gold"].astype(int).tolist(),
                    float(target_fpr),
                ))
                selected_value = selected["threshold"]
                if threshold.get("status") != selected["status"]:
                    raise C.ArtifactContractError(
                        f"threshold status is not canonical for locked target FPR in {key}")
                if selected["status"] in ("NO_FEASIBLE_THRESHOLD", "PREDICT_NONE"):
                    if threshold_value is not None:
                        raise C.ArtifactContractError(
                            f"threshold value must be null for {key}")
                elif selected_value is None or threshold_value is None or not np.isclose(
                        float(threshold_value), float(selected_value), rtol=0, atol=1e-12):
                    raise C.ArtifactContractError(
                        f"threshold value is not canonical for locked target FPR in {key}")
            cutoff = float("inf") if threshold_value is None else float(threshold_value)
            expected_prediction = (expected_calibrated >= cutoff).astype(int)
            if not np.array_equal(group["prediction"].to_numpy(int), expected_prediction):
                raise C.ArtifactContractError(
                    f"prediction is inconsistent with calibrated probability/threshold for {key}")
    if require_complete:
        expected_rows = len(rows) * len(expected_bundles)
        if len(df) != expected_rows:
            raise C.ArtifactContractError(
                f"combined score row count mismatch: expected={expected_rows} observed={len(df)}")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paper A scorer (plan sec 10).")
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out", default=None, help="scores output dir (default: lock artifact path)")
    ap.add_argument("--manifests-dir", default=None)
    ap.add_argument("--base-scores-dir", default=None)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None,
                    help="development-only dtype override; final scoring uses each locked dtype")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="rows per manifest (debug)")
    ap.add_argument("--synthetic", action="store_true",
                    help="fabricate deterministic logits (no model load) for plumbing tests")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="score even if fewer than 20/20 adapters are present")
    ap.add_argument("--nonfinal", action="store_true",
                    help="explicit development output; required for limit/dtype/incomplete/synthetic")
    ap.add_argument("--allow-legacy-lock", action="store_true",
                    help="explicitly use the historical v1 lock (never upgrades it)")
    ap.add_argument("--strict-cache", action="store_true",
                    help="error on cache mismatch instead of recomputing")
    ap.add_argument("--force", action="store_true", help="ignore caches and recompute")
    args = ap.parse_args(argv)

    nonfinal_reasons = []
    if args.synthetic:
        nonfinal_reasons.append("synthetic")
    if args.allow_incomplete:
        nonfinal_reasons.append("allow_incomplete")
    if args.limit is not None:
        nonfinal_reasons.append("row_limit")
    if args.dtype is not None:
        nonfinal_reasons.append("dtype_override")
    if nonfinal_reasons and not args.nonfinal:
        print(f"[eval] {nonfinal_reasons} require --nonfinal and an explicit --out",
              file=sys.stderr)
        return 2
    if args.nonfinal and not args.out:
        print("[eval] --nonfinal requires an explicit --out directory", file=sys.stderr)
        return 2
    try:
        lock = C.load_lock(
            args.lock, allow_legacy=args.allow_legacy_lock, verify_files=True,
            manifests_dir=args.manifests_dir)
    except (C.ArtifactContractError, FileNotFoundError) as exc:
        print(f"[eval] lock verification failed: {exc}", file=sys.stderr)
        return 2
    apaths = C.artifact_paths(lock)
    strict_lock = int(lock.get("lock_contract_version", 1)) >= C.LOCK_CONTRACT_VERSION
    if not strict_lock and not args.nonfinal:
        print("[eval] historical v1 score/run artifacts are immutable; legacy evaluation "
              "requires --nonfinal and an output outside all canonical artifact roots",
              file=sys.stderr)
        return 2
    if strict_lock and not args.nonfinal:
        software_issues = C.protocol_software_issues(
            C.software_versions(), lock.get("software_versions"))
        if software_issues:
            print(f"[eval] runtime software differs from LOCK.json: {software_issues}",
                  file=sys.stderr)
            return 2
    if strict_lock and not args.nonfinal:
        requested = {
            "manifests": args.manifests_dir,
            "scores": args.out,
            "base_scores": args.base_scores_dir,
            "runs": args.runs_root,
        }
        for key, override in requested.items():
            if override is None:
                continue
            if C.resolved_path(override) != C.resolved_path(apaths[key]):
                print(f"[eval] final v2 {key} path is lock-authoritative: {apaths[key]}",
                      file=sys.stderr)
                return 2
    manifests_dir = C.abspath(args.manifests_dir or apaths["manifests"])
    out_dir = C.abspath(args.out or apaths["scores"])
    base_dir = C.abspath(args.base_scores_dir or (
        os.path.join(out_dir, "base_scores") if args.nonfinal else apaths["base_scores"]))
    runs_root = C.abspath(args.runs_root or apaths["runs"])
    if args.nonfinal:
        protected_roots = {
            apaths["root"], C.DEFAULT_ARTIFACTS["root"], C.DEFAULT_ARTIFACTS_V2["root"],
        }
        for label, write_path in (("--out", out_dir), ("--base-scores-dir", base_dir)):
            if any(C.path_is_within(write_path, root) for root in protected_roots):
                print(f"[eval] nonfinal {label} must be outside canonical v1/v2 artifact roots",
                      file=sys.stderr)
                return 2
    device = args.device or (_default_device())
    dtype_override = "synthetic" if args.synthetic else args.dtype
    device_policy = "synthetic" if args.synthetic else device
    producer_runtime_details = C.runtime_environment(device_policy)
    producer_runtime = {
        "sha256": C.canonical_obj_sha256(producer_runtime_details),
        "details": producer_runtime_details,
    }
    target_fpr = float(lock.get("operating_point", {}).get("target_fpr", C.DEFAULT_TARGET_FPR))
    models = C.lock_model_panel(lock)
    seeds = C.lock_seeds(lock)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(base_dir, exist_ok=True)

    rows = load_scoring_rows(manifests_dir, args.limit)
    fps = manifest_fingerprints(manifests_dir, rows)
    print(f"[eval] scoring rows={len(rows)} across "
          f"{len(set(r['split'] for r in rows))} splits | device={device_policy}")

    # completeness gate (plan sec 9.3 / 10.1): require 20/20 valid adapters
    adapters = collect_adapters(lock, runs_root, allow_legacy=args.allow_legacy_lock)
    complete = all(a["valid"] for a in adapters.values())
    if not complete and not (args.allow_incomplete or args.synthetic):
        invalid = [f"{mk}/seed_{s}:{','.join(a['issues'])}" for (mk, s), a in adapters.items()
                   if not a["valid"]]
        raise SystemExit(f"[eval] refusing to score: {len(invalid)} adapters invalid "
                         f"(need 20/20 revalidated). Invalid: {invalid[:6]}...")

    all_recs = []
    bundle_meta = {}

    # ---- bases: score once per checkpoint, reuse cache ----
    for mk in C.MODEL_KEYS:
        m = models[mk]
        model_dtype = dtype_override or str(m.get("dtype", "bfloat16"))
        prompt_sha_locked = lock.get("prompt", {}).get("per_model_template_sha256", {}).get(mk)
        expected = build_expected_meta(
            lock, fps, m["model_revision"], m["tokenizer_revision"], None, None,
            prompt_sha_locked, model_dtype, device_policy, args.batch_size,
            producer_runtime["sha256"])
        pq = os.path.join(base_dir, mk, "base_scores.parquet")
        mp = os.path.join(base_dir, mk, "base_scores.meta.json")
        recs, meta = _score_or_reuse(
            lock, rows, mk, "base", None, None, None, None, prompt_sha_locked,
            expected, pq, mp, args, device, model_dtype, target_fpr)
        all_recs.extend(recs); bundle_meta[f"{mk}:base"] = meta

    # ---- sft adapters: 4 x 5 ----
    for mk in C.MODEL_KEYS:
        m = models[mk]
        model_dtype = dtype_override or str(m.get("dtype", "bfloat16"))
        prompt_sha_locked = lock.get("prompt", {}).get("per_model_template_sha256", {}).get(mk)
        for s in seeds:
            info = adapters[(mk, s)]
            adapter_sha = info["adapter_sha256"] if not args.synthetic else f"synthetic::{mk}:{s}"
            if not args.synthetic and not info["valid"]:
                if not args.allow_incomplete:
                    raise SystemExit(f"[eval] adapter {mk}/seed_{s} is invalid: {info['issues']}.")
                print(f"  [skip] {mk}/seed_{s} issues={info['issues']}"); continue
            expected = build_expected_meta(
                lock, fps, m["model_revision"], m["tokenizer_revision"], adapter_sha,
                info["run_meta_sha256"], prompt_sha_locked, model_dtype, device_policy,
                args.batch_size, producer_runtime["sha256"])
            pq = os.path.join(out_dir, "sft", mk, f"seed_{s}.parquet")
            mp = os.path.join(out_dir, "sft", mk, f"seed_{s}.meta.json")
            recs, meta = _score_or_reuse(
                lock, rows, mk, "sft", s, info["adapter_dir"], adapter_sha, info,
                prompt_sha_locked, expected, pq, mp, args, device, model_dtype, target_fpr)
            all_recs.extend(recs); bundle_meta[f"{mk}:sft:seed_{s}"] = meta

    # ---- write combined scores.parquet + metadata.json ----
    import pandas as pd
    df = pd.DataFrame(all_recs, columns=SCORE_COLUMNS)
    validate_combined_scores(
        df, rows, list(C.MODEL_KEYS), seeds, bundle_meta=bundle_meta,
        require_complete=not args.allow_incomplete, target_fpr=target_fpr)
    scores_path = os.path.join(out_dir, "scores.parquet")
    scores_tmp = scores_path + ".tmp.parquet"
    df.to_parquet(scores_tmp, engine="pyarrow", index=False)
    os.replace(scores_tmp, scores_path)
    scores_sha256 = C.sha256_file(scores_path)
    legacy_lock = int(lock.get("lock_contract_version", 1)) < C.LOCK_CONTRACT_VERSION
    finalization_status = ("nonfinal" if args.nonfinal else
                           "legacy_unverified" if legacy_lock else "final")
    adapter_inventory = {
        key: {
            "adapter_sha256": record.get("adapter_sha256"),
            "run_meta_path": record.get("run_meta_path"),
            "run_meta_sha256": record.get("run_meta_sha256"),
        }
        for key, record in bundle_meta.items() if ":sft:" in key
    }
    metadata = {
        "score_artifact_contract_version": 2,
        "finalization_status": finalization_status,
        "nonfinal_reasons": nonfinal_reasons,
        "lock_sha256": lock.get("lock_sha256"), "score_code_version": SCORE_CODE_VERSION,
        "execution_sources_sha256": lock.get(
            "execution_sources", {}).get("aggregate_sha256"),
        "created_utc": C.utcnow(), "device_policy": device_policy,
        "dtype_by_model": {mk: (dtype_override or models[mk].get("dtype", "bfloat16"))
                           for mk in C.MODEL_KEYS},
        "target_fpr": target_fpr, "n_rows_total": len(df), "columns": SCORE_COLUMNS,
        "manifest_fingerprints": fps, "seeds": seeds, "models": models,
        "bundles": bundle_meta, "software_versions": C.software_versions(),
        "runtime_environment": producer_runtime_details,
        "producer_runtime": producer_runtime,
        "batch_size": int(args.batch_size),
        "adapter_inventory": adapter_inventory,
        "n_bundles": len(bundle_meta), "synthetic": bool(args.synthetic),
        "scores_sha256": scores_sha256,
        "scores_filename": os.path.basename(scores_path),
    }
    metadata_path = os.path.join(out_dir, "metadata.json")
    C.write_json(metadata_path, metadata)
    if finalization_status in ("final", "legacy_unverified"):
        C.verify_score_artifact(scores_path, metadata_path, lock,
                                allow_legacy=args.allow_legacy_lock)
    print(f"[eval] wrote {scores_path} ({len(df)} rows, {len(bundle_meta)} bundles)")
    print(f"[eval]   scores_sha256={scores_sha256}")
    print(f"[eval] wrote {metadata_path}")
    return 0


def _score_or_reuse(lock, rows, mk, condition, seed, adapter_dir_path, adapter_sha,
                    adapter_identity, prompt_sha_locked, expected, pq, mp, args, device,
                    dtype, target_fpr):
    tag = f"{mk}:{condition}" + (f":seed_{seed}" if seed is not None else "")
    if not args.force:
        cached_df, cached_meta = _read_cache(pq, mp)
        ok, mism = C.cache_is_valid(cached_meta.get("cache") if cached_meta else None, expected)
        rows_ok, row_issues = _cache_rows_match(cached_df, rows, expected)
        if ok and rows_ok:
            print(f"  [cache] reuse {tag}")
            return cached_df.to_dict("records"), cached_meta.get("bundle_meta", {})
        mism = list(mism) + row_issues
        if cached_meta is not None and args.strict_cache:
            raise SystemExit(f"[eval] cache mismatch for {tag} on {mism} (--strict-cache).")
        if cached_meta is not None:
            print(f"  [recompute] {tag} cache invalid on {mism}")
    logits, prompt_sha, dtoks = score_bundle(
        lock, rows, mk, condition, seed, adapter_dir_path, adapter_sha,
        device, dtype, args.batch_size, args.synthetic)
    if prompt_sha_locked and prompt_sha != prompt_sha_locked and not args.synthetic:
        raise SystemExit(f"[eval] prompt template drift for {mk}: lock={prompt_sha_locked} "
                         f"observed={prompt_sha}")
    m = C.lock_model_panel(lock)[mk]
    recs, meta = assemble_bundle(lock, rows, logits, mk, m["model_revision"], condition, seed,
                                 adapter_sha, prompt_sha, dtoks, target_fpr)
    meta.update({
        "adapter_sha256": adapter_sha,
        "run_meta_path": (adapter_identity or {}).get("run_meta_path"),
        "run_meta_sha256": (adapter_identity or {}).get("run_meta_sha256"),
        "batch_size": int(expected["batch_size"]),
        "producer_runtime_sha256": expected["producer_runtime_sha256"],
    })
    # write per-bundle cache
    import pandas as pd
    os.makedirs(os.path.dirname(pq), exist_ok=True)
    tmp = pq + ".tmp.parquet"
    pd.DataFrame(recs, columns=SCORE_COLUMNS).to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, pq)
    exp = dict(expected); exp["adapter_sha256"] = adapter_sha; exp["prompt_sha256"] = prompt_sha
    C.write_json(mp, {"cache": exp, "bundle_meta": meta, "tag": tag,
                      "parquet_sha256": C.sha256_file(pq), "created_utc": C.utcnow()})
    print(f"  [scored] {tag} (T={meta['calibration'].get('temperature'):.3f} "
          f"thr={meta['threshold'].get('status')})")
    return recs, meta


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
