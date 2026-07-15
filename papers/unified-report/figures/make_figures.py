#!/usr/bin/env python
"""Generate the unified report's result visualizations from COMMITTED data, so every figure is
reproducible and matches the report's tables. Run: python figures/make_figures.py (from the report dir).

Figures:
  fig_act1_percheckpoint.pdf   per-checkpoint represented vs transfer SFT delta (the split)   <- tab_primary_gen.tex
  fig_act3_composition.pdf     per-checkpoint transfer macro-AP: base / SFT / composition      <- pilot_per_model_table.tex
  fig_mortgage_quadrant.pdf    the G x D 2x2 with row counts (G0/D1 payload; empty G1/D0)       <- benchmark jsonl `quadrant`
  fig_mortgage_baseline.pdf    zero-shot guards: AP.G, AP.D, and the Delta_context fairness gap <- out_eval/baseline_table.json
"""
from __future__ import annotations
import json, re, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
REPO = REPORT.parents[1]
GEN = REPORT / "generated"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": "--"})
BLUE, ORANGE, GREEN, RED, GREY = "#2563EB", "#EA580C", "#15803D", "#DC2626", "#94A3B8"


def _short(name):  # tidy checkpoint labels
    return (name.replace("-Instruct", "").replace("2.5-1.5B", "2.5\n1.5B")
            .replace("LM2-1.7B", "LM2\n1.7B").replace("LM3-3B", "LM3\n3B").replace("3-4B", "3\n4B"))


def act1_percheckpoint():
    """Grouped bars: represented-source delta vs transfer delta, per checkpoint (parse tab_primary_gen)."""
    rows = []
    for ln in (GEN / "tab_primary_gen.tex").read_text().splitlines():
        m = re.match(r"\s*([\w.\-]+)\s*&\s*[\d.\-]+\s*&\s*[\d.\-]+\s*&\s*([\-\d.]+)\s*\[.*?\]\s*&"
                     r"\s*[\d.\-]+\s*&\s*[\d.\-]+\s*&\s*([\-\d.]+)\s*\[", ln)
        if m and "aggregate" not in ln.lower():
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    if not rows:
        return
    labels = [_short(r[0]) for r in rows]
    rep = [r[1] for r in rows]; tr = [r[2] for r in rows]
    x = range(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([i - w/2 for i in x], rep, w, label="represented-source $\\Delta$", color=BLUE)
    ax.bar([i + w/2 for i in x], tr, w, label="dataset-held-out transfer $\\Delta$", color=ORANGE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("base$\\to$SFT change in macro-AP")
    ax.set_title("Act I: SFT lifts represented-source ranking uniformly,\nbut transfer change flips sign by checkpoint")
    for i, v in enumerate(rep): ax.text(i - w/2, v + 0.01, f"{v:+.2f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(tr): ax.text(i + w/2, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                                       va="bottom" if v >= 0 else "top", fontsize=7)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(HERE / "fig_act1_percheckpoint.pdf", metadata={"CreationDate": None}); plt.close(fig)


def act3_composition():
    """Grouped bars: transfer macro-AP for base / SFT / composition, per checkpoint (parse pilot_per_model_table)."""
    rows = []
    for ln in (GEN / "pilot_per_model_table.tex").read_text().splitlines():
        m = re.match(r"\s*([\w.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&", ln)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if not rows:
        return
    labels = [_short(r[0]) for r in rows]
    base = [r[1] for r in rows]; sft = [r[2] for r in rows]; comp = [r[3] for r in rows]
    x = range(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar([i - w for i in x], base, w, label="base", color=GREY)
    ax.bar(list(x), sft, w, label="SFT", color=ORANGE)
    ax.bar([i + w for i in x], comp, w, label="base+SFT composition", color=GREEN)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(min(base + sft + comp) - 0.05, 1.0)
    ax.set_ylabel("dataset-held-out transfer macro-AP")
    ax.set_title("Act III: composition recovers transfer above SFT\n(recovery, not dominance — it can dip below base, e.g. Qwen3-4B)")
    # legend BELOW the axis (clear of the 2-line x labels) so it never overlaps the bars
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(HERE / "fig_act3_composition.pdf", metadata={"CreationDate": None}); plt.close(fig)


def mortgage_quadrant():
    """G x D 2x2 with counts from the frozen benchmark `quadrant` field (all splits)."""
    from collections import Counter
    c = Counter()
    for f in glob.glob(str(REPO / "mortgage-benchmark/benchmark/v1_hmda2022/*.jsonl")):
        for ln in open(f):
            ln = ln.strip()
            if ln:
                c[json.loads(ln).get("quadrant", "?")] += 1
    # grid: rows D (0 top,1 bottom), cols G (0 left,1 right)
    cells = {("G0", "D0"): ("G0/D0", "benign", GREEN),
             ("G0", "D1"): ("G0/D1", "safe-looking, non-compliant — the payload", RED),
             ("G1", "D0"): ("G1/D0", "general harm only", GREY),
             ("G1", "D1"): ("G1/D1", "both", ORANGE)}
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for (g, d), (key, desc, col) in cells.items():
        gi = 0 if g == "G0" else 1
        di = 1 if d == "D0" else 0   # D0 on top row
        n = c.get(f"{g}{d}", 0)
        ax.add_patch(plt.Rectangle((gi, di), 1, 1, facecolor=col, alpha=0.16 if n else 0.05,
                                   edgecolor=col if n else GREY, lw=2 if key == "G0/D1" else 1))
        ax.text(gi + 0.5, di + 0.62, key, ha="center", va="center", fontweight="bold", fontsize=13)
        ax.text(gi + 0.5, di + 0.40, f"{n} rows", ha="center", va="center", fontsize=11)
        ax.text(gi + 0.5, di + 0.20, desc, ha="center", va="center", fontsize=7.5, style="italic", wrap=True)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["$G{=}0$ (looks safe)", "$G{=}1$ (generally unsafe)"])
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["$D{=}0$\n(allow)", "$D{=}1$\n(intervene)"])
    ax.set_title("Mortgage benchmark: the two labels $G\\times D$ and their 994-row quadrants")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(HERE / "fig_mortgage_quadrant.pdf", metadata={"CreationDate": None}); plt.close(fig)


def mortgage_baseline():
    """Per-guard AP.G, AP.D and the Delta_context fairness gap (from baseline_table.json)."""
    d = json.loads((REPO / "mortgage-benchmark/out_eval/baseline_table.json").read_text())
    tbl = d.get("table", [])
    if not tbl:
        return
    labels = [r["guard"].replace("_base", "").replace("_", "\n") for r in tbl]
    apg = [r.get("AP_G") or 0 for r in tbl]; apd = [r.get("AP_D") or 0 for r in tbl]
    dctx = [r.get("delta_context") or 0 for r in tbl]
    x = range(len(tbl)); w = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.bar([i - w/2 for i in x], apg, w, label="AP$\\cdot$G (general safety)", color=BLUE)
    ax1.bar([i + w/2 for i in x], apd, w, label="AP$\\cdot$D (mortgage policy)", color=GREEN)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("average precision"); ax1.set_ylim(0, 1.05)
    ax1.set_title("Zero-shot ranking (AP)")
    # legend below ax1 so it clears the (2-line) guard labels and never sits over a bar
    ax1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    thr = 0.1
    ax2.bar(list(x), dctx, 0.5, color=[RED if v > thr else GREY for v in dctx])
    ax2.axhline(thr, color=RED, lw=0.9, ls=":")
    ax2.text(len(tbl) - 0.5, thr, "  gap $>0.1$\n  (flagged)", ha="right", va="bottom",
             fontsize=6.5, color=RED)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("$\\Delta_{\\mathrm{context}}$  (0 = fair)")
    ax2.set_ylim(0, max(dctx + [thr]) * 1.35)
    ax2.set_title("Protected-pair gap")
    for i, v in enumerate(dctx): ax2.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Act IV: general guards rank mortgage violations only moderately, and fairness varies",
                 fontsize=10.5, y=1.00)
    fig.subplots_adjust(top=0.84, bottom=0.28, wspace=0.34, left=0.085, right=0.985)
    fig.savefig(HERE / "fig_mortgage_baseline.pdf", metadata={"CreationDate": None}); plt.close(fig)


def main():
    made = []
    for fn in (act1_percheckpoint, act3_composition, mortgage_quadrant, mortgage_baseline):
        try:
            fn(); made.append(fn.__name__)
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")
    print("generated:", ", ".join(made))


if __name__ == "__main__":
    main()
