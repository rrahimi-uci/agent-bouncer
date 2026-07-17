#!/usr/bin/env python
"""Orchestrate the starting-type 2x3 adaptation grid (proposal Sec 5,6,8,9,12,13).

Loops selected starting checkpoints x EXPLICIT conditions {unmodified, sft, kl_sft} x seeds
(x KL betas) from the study registry (configs/starting_type_adaptation_v1.yaml) and calls
experiments.starting_type_common.adapt(...) for each trained cell into a resumable run tree:

    <out-root>/<starting_key>/<condition>/seed_<r>[/beta<b>]/{adapter,run_meta.json}

Design invariants (proposal Sec 12/13):
  * conditions are EXPLICIT and never encoded as `sft` + a nullable beta; kl_sft records its beta;
  * `unmodified` is a single no-train reference cell with the locked seed=-1 sentinel (not a replicate);
  * expected per-checkpoint cardinality is 1 U + 5 SFT + 5 KL-SFT (primary beta) [+ 5 per sensitivity
    beta]; --dry-run enumerates + validates that grid and every condition_id's uniqueness;
  * resumable: a trained cell whose adapter is already present is skipped unless --force;
  * the KL reference for any checkpoint is that SAME unmodified checkpoint (adapt() via disable_adapter);
  * the harmonized train manifest is bound by the recipe's `train_manifest_from` hash reference
    (reuses the frozen Paper A paper_a_sft_v2 manifest); this CLI never writes into that artifact.

Nonfinal by default: pilot/smoke output is tagged nonfinal unless --final is given, and any recipe
override (--max-steps / --limit-train) forces nonfinal. This module does NOT mutate the Paper A files
or the running KL-SFT sweep.

Usage:
  # enumerate + validate the full 2x3 grid without loading any model:
  python experiments/run_starting_type_adaptation.py --dry-run
  # one real (nonfinal) SFT cell:
  python experiments/run_starting_type_adaptation.py \
      --checkpoints qwen25_15b --conditions sft --seeds 42 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402
import starting_type_common as S  # noqa: E402

DEFAULT_REGISTRY = os.path.join(C.REPO_ROOT, "configs", "starting_type_adaptation_v1.yaml")
DEFAULT_OUT_ROOT = os.path.join("artifacts", "starting_type_adaptation_v1", "runs")


# --------------------------------------------------------------------------------------
# train manifest: bind by the recipe's train_manifest_from hash reference (reuse paper_a)
# --------------------------------------------------------------------------------------
def resolve_train_manifest(recipe: dict) -> tuple[str, str]:
    """Resolve + hash the harmonized train manifest named by recipe.train_manifest_from.

    Returns (abs_train_jsonl_path, train_manifest_sha256). When the referenced artifact ships a
    LOCK.json with a train_manifest_sha256, the on-disk file is verified against it (fail closed)."""
    ref = str(recipe.get("train_manifest_from") or "").strip()
    if not ref:
        raise SystemExit("[start] recipe.train_manifest_from is required")
    artifact_root = C.abspath(os.path.join("artifacts", ref))
    manifests_dir = C.artifact_paths_for_root(artifact_root)["manifests"]
    train_path = os.path.join(manifests_dir, "train.jsonl")
    if not os.path.exists(train_path):
        raise SystemExit(f"[start] train manifest missing: {train_path}")
    observed = C.sha256_file(train_path)
    lock_path = os.path.join(artifact_root, "LOCK.json")
    if os.path.exists(lock_path):
        try:
            locked = C.read_json(lock_path).get("train_manifest_sha256")
        except Exception:
            locked = None
        if locked and locked != observed:
            raise SystemExit(
                f"[start] train manifest hash mismatch vs {ref} LOCK.json: "
                f"locked={locked} observed={observed}")
    return train_path, observed


def load_train_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    for r in C.read_jsonl(path):
        rows.append({"text": C.row_text(r), "gold": C.to_gold(r.get("label")),
                     "sample_id": r.get("sample_id")})
    if limit is not None:
        rows = rows[:limit]
    return rows


# --------------------------------------------------------------------------------------
# grid enumeration + cardinality validation (no model load; safe for --dry-run)
# --------------------------------------------------------------------------------------
def _beta_tag(beta: float) -> str:
    return ("%g" % float(beta)).replace(".", "p").replace("-", "m")


def cell_dir(out_root: str, key: str, condition: str, seed: int, beta) -> str:
    if condition == "unmodified":
        return os.path.join(out_root, key, "unmodified", f"seed_{seed}")
    if condition == "sft":
        return os.path.join(out_root, key, "sft", f"seed_{seed}")
    return os.path.join(out_root, key, "kl_sft", f"seed_{seed}", f"beta{_beta_tag(beta)}")


def enumerate_grid(reg: dict, out_root: str, keys, conditions, seeds, betas) -> list[dict]:
    """Enumerate every expected cell for the selected slice. `seeds`/`betas` are the trained-cell
    seeds and the kl_sft betas. `unmodified` is a single seed=-1 no-train cell per checkpoint."""
    ckpts = reg["checkpoints"]
    cells = []
    for key in keys:
        ck = ckpts[key]
        common = {"starting_key": key, "starting_type": ck["starting_type"],
                  "contract": ck.get("contract"), "model_id": ck.get("model_id")}
        if "unmodified" in conditions:
            cells.append({**common, "condition": "unmodified", "method": None,
                          "seed": -1, "beta": None,
                          "condition_id": S.condition_id(key, "unmodified", -1, None),
                          "out_dir": cell_dir(out_root, key, "unmodified", -1, None)})
        if "sft" in conditions:
            for r in seeds:
                cells.append({**common, "condition": "sft", "method": "sft",
                              "seed": int(r), "beta": None,
                              "condition_id": S.condition_id(key, "sft", r, None),
                              "out_dir": cell_dir(out_root, key, "sft", int(r), None)})
        if "kl_sft" in conditions:
            for b in betas:
                for r in seeds:
                    cells.append({**common, "condition": "kl_sft", "method": "kl_sft",
                                  "seed": int(r), "beta": float(b),
                                  "condition_id": S.condition_id(key, "kl_sft", r, b),
                                  "out_dir": cell_dir(out_root, key, "kl_sft", int(r), b)})
    return cells


def validate_grid(cells: list[dict], keys, conditions, seeds, betas) -> dict:
    """Assert the enumerated grid matches the expected cardinality and has unique condition_ids.

    Expected per checkpoint: (1 if unmodified) + (n_seeds if sft) + (n_seeds*n_betas if kl_sft)."""
    n_u = 1 if "unmodified" in conditions else 0
    n_sft = len(seeds) if "sft" in conditions else 0
    n_kl = (len(seeds) * len(betas)) if "kl_sft" in conditions else 0
    per_ckpt = n_u + n_sft + n_kl
    expected_total = per_ckpt * len(keys)
    counts = {"unmodified": 0, "sft": 0, "kl_sft": 0}
    for c in cells:
        counts[c["condition"]] += 1
    ids = [c["condition_id"] for c in cells]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    problems = []
    if len(cells) != expected_total:
        problems.append(f"total {len(cells)} != expected {expected_total}")
    if counts["unmodified"] != n_u * len(keys):
        problems.append(f"unmodified {counts['unmodified']} != {n_u * len(keys)}")
    if counts["sft"] != n_sft * len(keys):
        problems.append(f"sft {counts['sft']} != {n_sft * len(keys)}")
    if counts["kl_sft"] != n_kl * len(keys):
        problems.append(f"kl_sft {counts['kl_sft']} != {n_kl * len(keys)}")
    if dupes:
        problems.append(f"duplicate condition_ids: {dupes[:6]}")
    if problems:
        raise SystemExit("[start] grid cardinality invalid: " + "; ".join(problems))
    return {"n_checkpoints": len(keys), "per_checkpoint": per_ckpt, "total_cells": expected_total,
            "counts": counts, "n_unique_condition_ids": len(set(ids))}


# --------------------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------------------
def _write_unmodified_meta(cell: dict, ck: dict, train_sha: str) -> None:
    """Record the unmodified reference cell. seed=-1 is a locked sentinel, not a replicate; the
    method-specific fields (kl_beta, adapter hashes, achieved KL) are null (proposal Sec 13)."""
    os.makedirs(cell["out_dir"], exist_ok=True)
    C.write_json(os.path.join(cell["out_dir"], "run_meta.json"), {
        "starting_key": cell["starting_key"], "starting_type": cell["starting_type"],
        "model_id": ck.get("model_id"), "model_revision": ck.get("model_revision"),
        "tokenizer_revision": ck.get("tokenizer_revision"), "contract": ck.get("contract"),
        "condition": "unmodified", "condition_id": cell["condition_id"],
        "seed": -1, "kl_beta": None, "adapter_sha256": None, "initial_adapter_sha256": None,
        "train_manifest_sha256": train_sha, "out_dir": cell["out_dir"],
        "status": "unmodified_reference", "note": "no-train reference cell; scored by the evaluator",
        "recorded_utc": C.utcnow(),
    })


def run_cell(cell: dict, ck: dict, recipe: dict, train_rows, device, train_sha, force: bool) -> str:
    """Execute (or skip) one grid cell. Returns a status string."""
    if cell["condition"] == "unmodified":
        meta_path = os.path.join(cell["out_dir"], "run_meta.json")
        if os.path.exists(meta_path) and not force:
            return "skip_present"
        _write_unmodified_meta(cell, ck, train_sha)
        return "unmodified_reference"
    adir = C.adapter_dir(cell["out_dir"])
    if C.adapter_is_present(adir) and not force:
        return "skip_present"
    meta = S.adapt(
        ckpt=ck, contract_name=ck["contract"], train_rows=train_rows,
        method=cell["method"], beta=(cell["beta"] if cell["beta"] is not None else 0.0),
        seed=cell["seed"], data_order_seed=int(recipe.get("data_order_seed", 42)),
        recipe=recipe, out_dir=cell["out_dir"], device=device)
    return meta.get("status", "unknown")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help="run tree root (default artifacts/starting_type_adaptation_v1/runs)")
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help="starting_key subset (default: all registry checkpoints)")
    ap.add_argument("--conditions", nargs="+", default=None, choices=list(S.CONDITIONS),
                    help="condition subset (default: all -> unmodified sft kl_sft)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="trained-cell seeds (default: recipe.seeds)")
    ap.add_argument("--betas", nargs="+", type=float, default=None,
                    help="kl_sft betas (default: [recipe.kl.primary_beta])")
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (default: auto)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override recipe max_steps (smoke/dev only -> forces nonfinal)")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="cap train rows (smoke/dev only -> forces nonfinal)")
    ap.add_argument("--force", action="store_true", help="retrain/rewrite even if the cell is present")
    ap.add_argument("--final", action="store_true",
                    help="mark this run final (default: nonfinal pilot/smoke output)")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate + validate the grid; load no models, train nothing")
    args = ap.parse_args(argv)

    reg = S.load_registry(args.registry)
    recipe = dict(reg["recipe"])
    ckpts = reg["checkpoints"]

    keys = args.checkpoints or list(ckpts.keys())
    unknown = [k for k in keys if k not in ckpts]
    if unknown:
        raise SystemExit(f"[start] unknown checkpoint(s): {unknown}; have {list(ckpts.keys())}")
    conditions = tuple(args.conditions) if args.conditions else S.CONDITIONS
    seeds = args.seeds if args.seeds is not None else list(recipe.get("seeds", C.DEFAULT_SEEDS))
    betas = (args.betas if args.betas is not None
             else [float(recipe.get("kl", {}).get("primary_beta"))])

    cells = enumerate_grid(reg, C.abspath(args.out_root), keys, conditions, seeds, betas)
    summary = validate_grid(cells, keys, conditions, seeds, betas)

    nonfinal_reasons = []
    if not args.final:
        nonfinal_reasons.append("default_nonfinal")
    if args.max_steps is not None:
        nonfinal_reasons.append("max_steps_override")
    if args.limit_train is not None:
        nonfinal_reasons.append("limit_train")
    finalization = "final" if (args.final and args.max_steps is None
                               and args.limit_train is None) else "nonfinal"

    print(f"[start] registry={os.path.relpath(args.registry, C.REPO_ROOT)} "
          f"checkpoints={len(keys)} conditions={list(conditions)} seeds={seeds} betas={betas}")
    print(f"[start] grid: {summary['total_cells']} cells "
          f"({summary['counts']['unmodified']} unmodified + {summary['counts']['sft']} sft + "
          f"{summary['counts']['kl_sft']} kl_sft) | unique condition_ids="
          f"{summary['n_unique_condition_ids']} | finalization={finalization}")

    if args.dry_run:
        for c in cells:
            print(f"  [plan] {c['condition_id']:<48} -> {os.path.relpath(c['out_dir'], C.REPO_ROOT)}")
        print(f"[start] dry-run OK: {summary['total_cells']} cells, cardinality validated")
        return 0

    if args.max_steps is not None:
        recipe["max_steps"] = int(args.max_steps)
    train_path, train_sha = resolve_train_manifest(recipe)
    train_rows = load_train_rows(train_path, args.limit_train)
    print(f"[start] train manifest={os.path.relpath(train_path, C.REPO_ROOT)} "
          f"rows={len(train_rows)} sha={train_sha[:16]} | nonfinal_reasons={nonfinal_reasons}")

    tallies: dict[str, int] = {}
    for c in cells:
        status = run_cell(c, ckpts[c["starting_key"]], recipe, train_rows, args.device,
                          train_sha, args.force)
        tallies[status] = tallies.get(status, 0) + 1
        print(f"  [{status}] {c['condition_id']}")

    C.write_json(os.path.join(C.abspath(args.out_root), "run_manifest.json"), {
        "study_id": reg.get("study_id"), "registry": os.path.relpath(args.registry, C.REPO_ROOT),
        "finalization_status": finalization, "nonfinal_reasons": nonfinal_reasons,
        "checkpoints": keys, "conditions": list(conditions), "seeds": seeds, "betas": betas,
        "train_manifest_from": recipe.get("train_manifest_from"),
        "train_manifest_sha256": train_sha, "grid": summary, "status_tally": tallies,
        "recipe_max_steps": recipe.get("max_steps"), "device": args.device,
        "created_utc": C.utcnow(),
    })
    print(f"[start] done: {tallies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
