#!/usr/bin/env python
"""External-validation eval: score compact instruction checkpoints on ExpGuard as
prompt-safety guards, using the canonical guard_research prompt + decision tokens
(byte-parity with Paper A / the mortgage baseline).

ExpGuard = 6rightjade/expguardmix, config `expguardtest` (expert-annotated, domains
finance / healthcare / law). We classify the INPUT PROMPT (`prompt_label` in {safe,unsafe}),
matching Paper A's prompt-only task, and report aggregate + per-domain AP + AUROC.

This is the pre-planned "Optional External Validation: ExpGuard" from Paper A: does the
base-vs-SFT specialization/transfer pattern recur on an external expert-labeled source?

Redistribution: ExpGuard is gated/licensed, so we commit ONLY per-row guard scores
(the z_unsafe-z_safe -> prob) keyed by a text-free row hash, plus a text-free
{hash -> label, domain} index. No prompt text is written.

Modes:
  (default)        score each checkpoint on ExpGuard, write per-row scores + summary.
  --from-scores    recompute metrics from committed per-row scores (NO GPU, NO dataset).
  --mock           assign deterministic pseudo-scores (offline smoke; no model, no dataset
                   download if a cached labels index exists) to validate the data/metrics path.

Reproducibility: `--from-scores` regenerates every ExpGuard number in the report from the
committed score files under out_dir; it needs neither a GPU nor ExpGuard access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "artifacts" / "expguard_external"
DATASET = "6rightjade/expguardmix"
TEST_FILE = "expguardtest.parquet"
DOMAINS = ("finance", "healthcare", "law")

# Fixed panel = Paper A's four checkpoints at their pinned revisions.
PANEL = [
    ("qwen25_15b_base", "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 8),
    ("smollm2_17b_base", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674", 8),
    ("smollm3_3b_base", "HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", 4),
    ("qwen3_4b_base", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", 4),
]


def _load_hf_token() -> str | None:
    for p in (REPO / ".env",):
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN")


def _row_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_expguard(limit: int | None = None, parquet_path: str | None = None) -> list[dict]:
    """Load the ExpGuard test split -> [{id, prompt, label(int), domain}].
    Reads a local parquet if parquet_path is given (tokenless, e.g. on a GPU VM); otherwise
    fetches from HF (gated dataset -> needs HF_TOKEN)."""
    import pandas as pd

    if parquet_path:
        df = pd.read_parquet(parquet_path)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET, TEST_FILE, repo_type="dataset", token=_load_hf_token())
        df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        prompt = str(r["prompt"])
        rows.append({
            "id": _row_id(prompt),
            "prompt": prompt,
            "label": 1 if str(r["prompt_label"]).strip().lower() == "unsafe" else 0,
            "domain": str(r["domain"]).strip().lower(),
        })
    # de-duplicate by id (identical prompts) keeping first
    seen, uniq = set(), []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"]); uniq.append(row)
    if limit:
        uniq = uniq[:limit]
    return uniq


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def score_checkpoint(model_id: str, revision: str, rows: list[dict], *, dtype: str,
                     batch_size: int, device: str | None) -> dict[str, float]:  # pragma: no cover - GPU/MPS
    """Guard score = the raw margin z_unsafe - z_safe at the decision position, via the canonical
    guard_research prompt + decision tokens (byte-parity with Paper A / the mortgage LogitDiffGuard).

    We store the RAW margin, not sigmoid(margin). AP/AUROC are rank metrics, and a confident guard
    saturates sigmoid to exactly 1.0/0.0 for |margin| > ~37 (e.g. Qwen3-4B ties ~1000 rows at 1.0),
    which collapses the model's ranking of confident predictions into one tie and makes the metric
    non-reproducible from the stored (rounded) probabilities. The margin never saturates, so
    ---from-scores reproduces the live number exactly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from guard_research.prompts import build_prompt, select_decision_tokens

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    # bf16 is unreliable on MPS; use fp32 off-CUDA for correctness.
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"
    td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=td, trust_remote_code=True).to(device).eval()
    dt = select_decision_tokens(tok)
    safe_id, unsafe_id = dt["safe_id"], dt["unsafe_id"]
    out: dict[str, float] = {}
    # device-aware allocator hygiene: torch.mps/torch.cuda .empty_cache (None on CPU).
    # Without this the MPS allocator cache balloons across ~hundreds of batches -> unified-memory
    # swap -> the process thrashes ("stuck"). One model per fresh process is the primary guard;
    # this is belt-and-suspenders within a model's batch loop.
    empty_cache = getattr(getattr(torch, device, None), "empty_cache", None)
    with torch.no_grad():
        for step, i in enumerate(range(0, len(rows), batch_size)):
            chunk = rows[i:i + batch_size]
            prompts = [build_prompt(tok, r["prompt"]) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            for k, r in enumerate(chunk):
                lz = logits[k, last[k]]
                out[r["id"]] = float(lz[unsafe_id]) - float(lz[safe_id])  # raw margin (higher = unsafe)
            del logits, enc
            if empty_cache and step % 16 == 0:
                empty_cache()
    del model
    if empty_cache:
        empty_cache()
    return out


def compute_metrics(scores: dict[str, float], labels: dict[str, dict]) -> dict:
    from guard_research.metrics import average_precision, auroc

    def _metrics(ids):
        s = [scores[i] for i in ids if i in scores]
        y = [labels[i]["label"] for i in ids if i in scores]
        n, pos = len(y), sum(y)
        if n == 0 or pos == 0 or pos == n:
            return {"ap": None, "auroc": None, "n": n, "n_pos": pos}
        return {"ap": round(average_precision(s, y), 4), "auroc": round(auroc(s, y), 4),
                "n": n, "n_pos": pos}

    all_ids = list(labels.keys())
    res = {"overall": _metrics(all_ids)}
    for dom in DOMAINS:
        res[dom] = _metrics([i for i in all_ids if labels[i]["domain"] == dom])
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_expguard_external", description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--from-scores", action="store_true", help="recompute metrics from committed scores (no GPU/dataset)")
    ap.add_argument("--mock", action="store_true", help="deterministic pseudo-scores (offline smoke)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke)")
    ap.add_argument("--parquet-path", default=None, help="local ExpGuard test parquet (tokenless; e.g. on a GPU VM)")
    ap.add_argument("--only", default=None,
                    help="comma-separated panel names to (re)score in THIS process; others reused from disk. "
                         "Run one model per process to avoid multi-load hangs / allocator balloon.")
    ap.add_argument("--force", action="store_true", help="re-score even if a scores file already exists")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels_index.json"  # text-free: id -> {label, domain}

    # ---- resolve labels index (from committed file, or by loading ExpGuard) ----
    if args.from_scores or (args.mock and labels_path.exists()):
        labels = json.loads(labels_path.read_text())
        rows = None
    else:
        rows = load_expguard(limit=args.limit, parquet_path=args.parquet_path)
        labels = {r["id"]: {"label": r["label"], "domain": r["domain"]} for r in rows}
        labels_path.write_text(json.dumps(labels, indent=0))
        print(f"[expguard] loaded {len(rows)} rows; domains={sorted({r['domain'] for r in rows})}; "
              f"prevalence={sum(l['label'] for l in labels.values())}/{len(labels)}")

    table = []
    for name, model_id, revision, bs in PANEL:
        scores_path = out / f"scores_{name}.json"
        if args.from_scores:
            if not scores_path.exists():
                print(f"[expguard] SKIP {name}: no committed scores at {scores_path.name}")
                continue
            scores = {k: float(v) for k, v in json.loads(scores_path.read_text()).items()}
        elif args.mock:
            # deterministic pseudo-score from the row hash (offline smoke only)
            scores = {rid: (int(rid, 16) % 1000) / 1000.0 for rid in labels}
        else:
            selected = only is None or name in only
            if selected and (args.force or not scores_path.exists()):
                print(f"[expguard] scoring {name} ({model_id}@{revision[:8]}) on {len(rows)} rows "
                      f"[dtype={args.dtype} device={args.device or 'auto'}] ...")
                scores = score_checkpoint(model_id, revision, rows, dtype=args.dtype,
                                          batch_size=bs, device=args.device)
                scores_path.write_text(json.dumps({k: round(v, 6) for k, v in scores.items()}, indent=0))
            elif scores_path.exists():
                print(f"[expguard] REUSE {name}: existing scores at {scores_path.name}")
                scores = {k: float(v) for k, v in json.loads(scores_path.read_text()).items()}
            else:
                print(f"[expguard] SKIP {name}: not selected (--only) and no existing scores")
                continue
        m = compute_metrics(scores, labels)
        row = {"guard": name, "overall_ap": m["overall"]["ap"], "overall_auroc": m["overall"]["auroc"],
               **{f"{d}_ap": m[d]["ap"] for d in DOMAINS}, "per_domain": m}
        table.append(row)
        print(f"  {name}: overall AP={m['overall']['ap']} | "
              + " ".join(f"{d}={m[d]['ap']}" for d in DOMAINS))

    summary = {"dataset": DATASET, "split": TEST_FILE, "task": "prompt_label",
               "n_rows": len(labels), "domains": list(DOMAINS), "panel": [p[0] for p in PANEL],
               "table": table}
    (out / "baseline_expguard.json").write_text(json.dumps(summary, indent=2))
    print(f"[expguard] wrote {out/'baseline_expguard.json'} ({len(table)} guards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
