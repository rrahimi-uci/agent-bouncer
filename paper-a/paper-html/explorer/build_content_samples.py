#!/usr/bin/env python
"""Extract a small, balanced, REAL sample of prompts into a local-only JS file,
which index.html loads via <script src> (works offline on file://, unlike fetch).

The generated output is gitignored and must remain local: several sources are
non-commercial or gated. Each entry records the true total and how many local rows
are shown, and the page links to the authoritative source for the complete set.

Sources (all already local):
  data/frozen_eval_rows.json                     6 in-house strata + 4 novel sets (the exact scored rows; gitignored)
  paper-a/paper-html/explorer/sources/guard_benchmark_hard.jsonl   the mortgage set (this work; tracked)
  data/benchmarks/openai_moderation.jsonl        landscape-only reference set (gitignored)

Run from repo root:  python3 paper-a/paper-html/explorer/build_content_samples.py
"""
import json, os, re
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))          # paper-a/paper-html/explorer
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # repo root
OUT = os.path.join(HERE, "samples.local.js")
PER = 30        # rows shown per benchmark (label-balanced + tag-diversified)
TRUNC = 320     # truncate prompt text to keep the page browsable

def clip(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t[:TRUNC] + " …" if len(t) > TRUNC else t

def even(xs, k):
    """k items evenly spaced across xs (deterministic, no RNG)."""
    if k <= 0 or not xs: return []
    if len(xs) <= k: return list(xs)
    step = len(xs) / k
    return [xs[int(i * step)] for i in range(k)]

def diverse(rows, k):
    """Up to k rows spread across distinct tags: round-robin one per tag per round,
    even-spaced within each tag so we don't cluster on near-identical prompts."""
    if len(rows) <= k: return list(rows)
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(r.get("tag", ""), []).append(r)
    queues = {g: even(v, min(len(v), k)) for g, v in groups.items()}
    out, qi, order = [], {g: 0 for g in groups}, list(groups)
    while len(out) < k:
        progressed = False
        for g in order:
            if qi[g] < len(queues[g]):
                out.append(queues[g][qi[g]]); qi[g] += 1; progressed = True
                if len(out) >= k: break
        if not progressed: break
    return out[:k]

def balanced(rows, per=PER):
    """Interleave an unsafe/safe-balanced, tag-diversified sample (deterministic)."""
    pos = [r for r in rows if r["g"] == 1]
    neg = [r for r in rows if r["g"] == 0]
    if not neg: return diverse(pos, per)
    if not pos: return diverse(neg, per)
    npos = min(len(pos), per // 2)
    nneg = min(len(neg), per - npos)
    npos = min(len(pos), per - nneg)          # backfill if one class is short
    P, N = diverse(pos, npos), diverse(neg, nneg)
    out = []
    for i in range(max(len(P), len(N))):       # interleave unsafe/safe
        if i < len(P): out.append(P[i])
        if i < len(N): out.append(N[i])
    return out

SAMPLES = {}

def add(bid, rows, total, extra=None):
    seen, uniq = set(), []                     # drop exact-duplicate (clipped) prompts
    for r in rows:
        if r["t"] in seen: continue
        seen.add(r["t"]); uniq.append(r)
    s = balanced(uniq)
    entry = {"n": total, "shown": len(s), "trunc": TRUNC, "rows": s}
    if extra: entry.update(extra)
    SAMPLES[bid] = entry

# ---- frozen eval rows: 6 in-house strata + 4 novel sets ----
fr = json.load(open(os.path.join(ROOT, "data/frozen_eval_rows.json")))
by = {}
for t, g, s in zip(fr["test_texts"], fr["gold"], fr["strata"]):
    by.setdefault(s, []).append({"t": clip(t), "g": int(g)})
for stratum, rows in by.items():
    add(stratum, rows, len(rows))
for nk, nv in fr["novel"].items():
    rows = [{"t": clip(t), "g": int(g)} for t, g in zip(nv["texts"], nv["gold"])]
    add(nk, rows, len(rows))

# ---- mortgage (this work): rich labels; carry trap_type as a tag ----
mp = os.path.join(HERE, "sources", "guard_benchmark_hard.jsonl")
mrows, mtot = [], 0
for line in open(mp):
    line = line.strip()
    if not line: continue
    d = json.loads(line); mtot += 1
    # mortgage set uses label_binary in {"allow","flag"} (safe/unsafe); category "safe" for benign
    g = 0 if (d.get("label_binary") == "allow" or d.get("label_category") == "safe") else 1
    mrows.append({"t": clip(d["text"]), "g": g, "tag": d.get("trap_type", "")})
add("mortgage", mrows, mtot)

# ---- openai moderation (landscape reference): carry hazard as a tag ----
op = os.path.join(ROOT, "data/benchmarks/openai_moderation.jsonl")
if os.path.exists(op):
    orows, otot = [], 0
    for line in open(op):
        line = line.strip()
        if not line: continue
        d = json.loads(line); otot += 1
        g = 1 if str(d.get("label")).lower() == "unsafe" else 0
        orows.append({"t": clip(d["text"]), "g": g, "tag": d.get("hazard", "")})
    add("openai_mod", orows, otot)

# ---- ExpGuardTest (CC BY-4.0): expert-annotated; use a local sample if pulled, else link ----
ep = os.path.join(HERE, "expguard_sample.local.json")
if os.path.exists(ep):
    ej = json.load(open(ep))
    erows = [{"t": clip(r["t"]), "g": int(r["g"]), "tag": r.get("tag", "")} for r in ej["rows"]]
    add("expguard", erows, ej.get("n", len(erows)),
        extra={"src": "https://huggingface.co/datasets/6rightjade/expguardmix"})
else:
    SAMPLES["expguard"] = {"n": 2275, "shown": 0,
        "note": "Expert-annotated (finance / healthcare / law). Browse the full set at the source.",
        "src": "https://huggingface.co/datasets/6rightjade/expguardmix"}

order = ["beavertails","toxicchat","prompt_injections","jailbreak_classification","jailbreakbench",
         "xstest","wildguardtest","wildjailbreak","orbench_hard","harmbench","expguard","mortgage","openai_mod"]
SAMPLES = {k: SAMPLES[k] for k in order if k in SAMPLES}

with open(OUT, "w") as f:
    f.write("// Auto-generated by build_content_samples.py — real sampled prompts per benchmark.\n")
    f.write("// Samples only (sources are variously licensed; full sets at each benchmark's source link).\n")
    f.write("window.SAMPLES = " + json.dumps(SAMPLES, ensure_ascii=False, indent=1) + ";\n")

print(f"wrote {OUT}")
for k, v in SAMPLES.items():
    print(f"  {k:26s} shown {v.get('shown',0):>2}/{v['n']}" + ("  [note]" if v.get('note') else ""))
