#!/usr/bin/env python
"""KL-regularized SFT sweep (anti-forgetting control for Act I).

Trains KL-regularized LoRA-SFT guards (loss = CE + beta * KL(pi_theta || pi_base) on the
completion tokens; see run_paper_a_sft.train_one_cell) for a grid of (seed, beta) on ONE
checkpoint, then scores each adapter through the SAME margin + calibration primitives the
canonical evaluator uses (eval_paper_a_sft.score_bundle / assemble_bundle), so KL scores are
byte-for-byte comparable to the committed base/SFT scores.

This is a NON-CANONICAL research variant: it modifies the objective, so it runs against the
lock WITHOUT execution-source verification (verify_files=False) and writes outside the canonical
artifact roots. The train-manifest sha256 is still enforced. beta==0 reproduces vanilla SFT.

Usage (one checkpoint per process; parallelize across GPUs with CUDA_VISIBLE_DEVICES):
  python experiments/run_klsft_sweep.py --lock artifacts/paper_a_sft_v2/LOCK.json \
      --model-key qwen3_4b --seeds 42 43 44 45 46 --betas 0.1 0.5 1.0 \
      --out-root artifacts/paper_a_klsft --device cuda
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_a_common as C  # noqa: E402
from run_paper_a_sft import train_one_cell, train_manifest_path  # noqa: E402
from eval_paper_a_sft import load_scoring_rows, score_bundle, assemble_bundle, _default_device  # noqa: E402


def _beta_tag(beta: float) -> str:
    return ("%g" % float(beta)).replace(".", "p").replace("-", "m")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--model-key", required=True, choices=list(C.MODEL_KEYS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--out-root", default="artifacts/paper_a_klsft")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override recipe steps (debug/smoke only; real runs omit -> recipe 300)")
    ap.add_argument("--limit", type=int, default=None, help="rows per manifest (debug)")
    ap.add_argument("--no-score", action="store_true", help="train only, skip scoring")
    ap.add_argument("--force", action="store_true", help="retrain even if adapter present")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    # Non-canonical objective: load the lock for its recipe/panel/manifests WITHOUT execution-source
    # verification (the trainer source is deliberately modified). Self-hash + structure still enforced.
    lock = C.load_lock(args.lock, verify_files=False)
    mk = args.model_key
    m = C.lock_model_panel(lock)[mk]
    model_revision = m["model_revision"]
    device = args.device or _default_device()
    target_fpr = float(lock.get("operating_point", {}).get("target_fpr", C.DEFAULT_TARGET_FPR))
    manifests_dir = C.abspath(C.artifact_paths(lock)["manifests"])

    train_path = train_manifest_path(lock, None)
    if not os.path.exists(train_path):
        print(f"[klsft] train manifest missing: {train_path}", file=sys.stderr)
        return 2
    if C.sha256_file(train_path) != lock.get("train_manifest_sha256"):
        print("[klsft] refusing mismatched train manifest", file=sys.stderr)
        return 2

    rows = None if args.no_score else load_scoring_rows(manifests_dir, args.limit)
    out_root = C.abspath(args.out_root)
    runs_root = os.path.join(out_root, "runs")
    scores_dir = os.path.join(out_root, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    all_recs = []
    for beta in args.betas:
        for seed in args.seeds:
            tag = f"{mk} seed {seed} beta {beta}"
            out_dir = os.path.join(runs_root, mk, f"beta{_beta_tag(beta)}", f"seed_{seed}")
            adir = C.adapter_dir(out_dir)
            if args.force or not C.adapter_is_present(adir):
                t0 = time.time()
                meta = train_one_cell(lock, mk, seed, out_dir, train_path,
                                      steps=args.max_steps, device=device,
                                      run_kind="nonfinal", kl_beta=beta)
                if meta.get("status") != "completed":
                    print(f"[klsft] TRAIN FAILED {tag}: {meta.get('failure_reason')}", file=sys.stderr)
                    continue
                print(f"[klsft] trained {tag} kl={meta.get('final_kl')} "
                      f"({round(time.time()-t0,1)}s)", flush=True)
            else:
                print(f"[klsft] reuse adapter {tag}", flush=True)

            if args.no_score:
                continue
            import json
            meta = json.load(open(os.path.join(out_dir, "run_meta.json")))
            adapter_sha = meta.get("adapter_sha256") or C.sha256_dir(adir)
            t0 = time.time()
            logits, prompt_sha, dtoks = score_bundle(
                lock, rows, mk, "sft", seed, adir, adapter_sha,
                device, m.get("dtype", "bfloat16"), args.batch_size, False)
            recs, _ = assemble_bundle(
                lock, rows, logits, mk, model_revision, "sft", seed, adapter_sha,
                prompt_sha, dtoks, target_fpr)
            for r in recs:
                r["kl_beta"] = float(beta)
            all_recs.extend(recs)
            print(f"[klsft] scored {tag}: {len(recs)} rows ({round(time.time()-t0,1)}s)", flush=True)

    if args.no_score:
        print("[klsft] train-only done.")
        return 0
    df = pd.DataFrame(all_recs)
    out_pq = os.path.join(scores_dir, f"klsft_scores_{mk}.parquet")
    df.to_parquet(out_pq, index=False)
    print(f"[klsft] wrote {out_pq}: {len(df)} rows, "
          f"betas={sorted(df['kl_beta'].unique())}, seeds={sorted(df['seed'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
