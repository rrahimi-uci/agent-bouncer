#!/usr/bin/env python
"""Production run-tree scorer for the starting-type adaptation study (proposal Sec 12).

The smoke CLI in experiments/eval_starting_type_adaptation.py scores ONE cell. This orchestrator
walks a trained run tree (produced by run_starting_type_adaptation.py) for one or more starting
checkpoints, scores EVERY cell (unmodified + sft + kl_sft) under the checkpoint's native contract
against the SAME held-out scoring rows, and writes one per-checkpoint parquet:

    <scores-dir>/sta_scores_<starting_key>.parquet   (+ sta_scores_<key>.metadata.json)

concatenating all cells' per-row records (schema == eval_starting_type_adaptation.SCORE_COLUMNS).
This is the GCS-uploadable unit (one parquet per VM), the analyzer's input. The scoring rows are
the frozen Paper A scoring manifests (calibration + id_test [+ ood]); the per-frame temperature
calibration in score_condition is fit on each cell's own calibration split.

Nonfinal by default; --final marks a claim-bearing run. Never mutates Paper A artifacts, the KL-SFT
sweep, or the training run tree (read-only over adapters).

Usage:
  # score a trained checkpoint's whole tree against the Paper A scoring manifests:
  python experiments/run_eval_starting_type_adaptation.py --checkpoints qwen3guard_gen_06b \
      --out-root artifacts/starting_type_adaptation_v1/runs \
      --manifests-dir artifacts/paper_a_sft_v2/manifests \
      --scores-dir artifacts/starting_type_adaptation_v1/scores --device cuda --final
  # plumbing self-test (tiny model, synthetic rows, unmodified only -> no adapter needed):
  python experiments/run_eval_starting_type_adaptation.py --checkpoints smollm2_17b \
      --conditions unmodified --synthetic-rows 12 --scores-dir /tmp/sta_scores --device cpu
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
import run_starting_type_adaptation as R  # noqa: E402  (enumerate_grid, cell_dir)
import eval_starting_type_adaptation as E  # noqa: E402  (score_condition, SCORE_COLUMNS)
import eval_paper_a_sft as EA  # noqa: E402  (load_scoring_rows)

DEFAULT_REGISTRY = os.path.join(C.REPO_ROOT, "configs", "starting_type_adaptation_v1.yaml")
DEFAULT_OUT_ROOT = os.path.join("artifacts", "starting_type_adaptation_v1", "runs")
DEFAULT_SCORES_DIR = os.path.join("artifacts", "starting_type_adaptation_v1", "scores")


def score_checkpoint(reg: dict, key: str, out_root: str, rows: list[dict], *, device: str,
                     batch_size: int, conditions, seeds, betas, force: bool = False) -> tuple:
    """Score every enumerated cell of one checkpoint's run tree. Returns (records, tally)."""
    ck = dict(reg["checkpoints"][key])
    ck["starting_key"] = key
    ck.setdefault("max_length", reg.get("recipe", {}).get("max_length", 1024))
    contract = ck.get("contract")
    cells = R.enumerate_grid(reg, C.abspath(out_root), [key], conditions, seeds, betas)

    all_recs: list[dict] = []
    tally: dict[str, int] = {}

    def bump(k):
        tally[k] = tally.get(k, 0) + 1

    for cell in cells:
        cond = cell["condition"]
        if cond == "unmodified":
            adapter_dir = None
        else:
            adapter_dir = C.adapter_dir(cell["out_dir"])
            if not C.adapter_is_present(adapter_dir):
                bump("missing_adapter")
                print(f"  [miss] {cell['condition_id']} (no adapter at "
                      f"{os.path.relpath(adapter_dir, C.REPO_ROOT)})")
                continue
        try:
            recs = E.score_condition(
                ckpt=ck, contract=contract, adapter_dir=adapter_dir, condition=cond,
                seed=cell["seed"], beta=(cell["beta"] if cell["beta"] is not None else 0.5),
                rows=rows, device=device, batch_size=batch_size)
        except Exception as exc:  # a single bad cell must not sink the whole checkpoint
            bump("scoring_error")
            print(f"  [err ] {cell['condition_id']}: {type(exc).__name__}: {exc}")
            continue
        all_recs.extend(recs)
        bump(cond)
        print(f"  [ok  ] {cell['condition_id']} -> {len(recs)} rows")
    return all_recs, tally


def write_checkpoint_scores(scores_dir: str, key: str, recs: list[dict], *, reg: dict,
                            finalization: str, row_source: str, device: str, tally: dict) -> str:
    import pandas as pd
    os.makedirs(scores_dir, exist_ok=True)
    df = pd.DataFrame(recs, columns=E.SCORE_COLUMNS)
    path = os.path.join(scores_dir, f"sta_scores_{key}.parquet")
    tmp = path + ".tmp.parquet"
    df.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)
    C.write_json(os.path.join(scores_dir, f"sta_scores_{key}.metadata.json"), {
        "score_code_version": E.SCORE_CODE_VERSION,
        "study_id": reg.get("study_id"), "starting_key": key,
        "starting_type": reg["checkpoints"][key].get("starting_type"),
        "contract": reg["checkpoints"][key].get("contract"),
        "finalization_status": finalization, "row_source": row_source, "device": device,
        "n_rows": len(df), "n_condition_ids": int(df["condition_id"].nunique()) if len(df) else 0,
        "conditions_scored": (sorted(df["adaptation"].unique().tolist()) if len(df) else []),
        "status_tally": tally, "columns": E.SCORE_COLUMNS, "created_utc": C.utcnow(),
    })
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="trained run tree root")
    ap.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR, help="per-checkpoint parquet output")
    ap.add_argument("--checkpoints", nargs="+", default=None, help="starting_key subset")
    ap.add_argument("--conditions", nargs="+", default=None, choices=list(S.CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--betas", nargs="+", type=float, default=None)
    ap.add_argument("--manifests-dir", default=None,
                    help="Paper A scoring manifests dir (calibration/id_test/ood rows)")
    ap.add_argument("--synthetic-rows", type=int, default=0,
                    help="score N fabricated rows instead (dev/plumbing -> forces nonfinal)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="mark this scoring run final (requires --manifests-dir)")
    args = ap.parse_args(argv)

    reg = S.load_registry(args.registry)
    recipe = reg["recipe"]
    ckpts = reg["checkpoints"]
    keys = args.checkpoints or list(ckpts.keys())
    unknown = [k for k in keys if k not in ckpts]
    if unknown:
        raise SystemExit(f"[eval-tree] unknown checkpoint(s): {unknown}")
    conditions = tuple(args.conditions) if args.conditions else S.CONDITIONS
    seeds = args.seeds if args.seeds is not None else list(recipe.get("seeds", C.DEFAULT_SEEDS))
    betas = (args.betas if args.betas is not None
             else [float(recipe.get("kl", {}).get("primary_beta"))])

    if args.synthetic_rows:
        rows = E.synthetic_rows(args.synthetic_rows)
        row_source = f"synthetic:{args.synthetic_rows}"
        finalization = "nonfinal"
    elif args.manifests_dir:
        rows = EA.load_scoring_rows(args.manifests_dir, args.limit)
        row_source = os.path.relpath(args.manifests_dir, C.REPO_ROOT)
        finalization = "final" if args.final else "nonfinal"
    else:
        raise SystemExit("[eval-tree] provide --manifests-dir DIR (real) or --synthetic-rows N (dev)")

    device = args.device or S._default_device()
    print(f"[eval-tree] checkpoints={keys} conditions={list(conditions)} seeds={seeds} "
          f"betas={betas} rows={len(rows)} source={row_source} device={device} "
          f"finalization={finalization}")

    written = {}
    for key in keys:
        print(f"[eval-tree] scoring {key} ...")
        recs, tally = score_checkpoint(
            reg, key, args.out_root, rows, device=device, batch_size=args.batch_size,
            conditions=conditions, seeds=seeds, betas=betas, force=args.force)
        path = write_checkpoint_scores(
            args.scores_dir, key, recs, reg=reg, finalization=finalization,
            row_source=row_source, device=device, tally=tally)
        written[key] = {"path": os.path.relpath(path, C.REPO_ROOT), "n_rows": len(recs),
                        "tally": tally}
        print(f"[eval-tree] wrote {os.path.relpath(path, C.REPO_ROOT)} "
              f"({len(recs)} rows) tally={tally}")

    print(f"[eval-tree] done: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
