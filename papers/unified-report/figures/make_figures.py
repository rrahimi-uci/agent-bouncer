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
                     "grid.alpha": 0.25, "grid.linestyle": "--",
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # embed TrueType, never Type 3
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


def _parse_primary():
    """Parse tab_primary_gen -> [(name, rep_base, rep_sft, tr_base, tr_sft)] for the 4 checkpoints."""
    rows = []
    for ln in (GEN / "tab_primary_gen.tex").read_text().splitlines():
        m = re.match(r"\s*([\w.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*[\-\d.]+\s*\[.*?\]\s*&"
                     r"\s*([\d.]+)\s*&\s*([\d.]+)\s*&", ln)
        if m and "aggregate" not in ln.lower():
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    return rows


def attractor():
    """The fine-tuning attractor (B.1): base scores spread wide; post-SFT scores collapse to a
    benchmark-fixed endpoint, so Delta = SFT - base is forced onto a slope-(-1) line."""
    import numpy as np
    rows = _parse_primary()
    if len(rows) < 3:
        return
    names = [_short(r[0]).replace("\n", "-") for r in rows]
    tb = np.array([r[3] for r in rows]); ts = np.array([r[4] for r in rows])
    AT = ts.mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.7))
    # -- left: base -> SFT convergence to the endpoint band --
    x = np.arange(len(rows))
    ax1.axhspan(AT - ts.std(ddof=1), AT + ts.std(ddof=1), color=GREEN, alpha=0.13, zorder=0)
    ax1.axhline(AT, color=GREEN, lw=1.2, ls="--", zorder=1, label=f"endpoint $A_T\\approx{AT:.2f}$")
    for i in x:
        ax1.annotate("", xy=(i, ts[i]), xytext=(i, tb[i]),
                     arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4, shrinkA=3, shrinkB=3))
    ax1.scatter(x, tb, s=55, color=BLUE, zorder=3, label="base")
    ax1.scatter(x, ts, s=55, color=ORANGE, zorder=3, label="after SFT")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=7.5)
    ax1.set_ylabel("dataset-held-out transfer macro-AP")
    ax1.set_ylim(0.72, 0.98)
    ax1.set_title("Fine-tuning is an attractor:\nwide base spread $\\to$ one endpoint band")
    ax1.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    # -- right: Delta vs base is forced onto slope -1 --
    dtr = ts - tb
    slope, intc = np.polyfit(tb, dtr, 1)
    yh = intc + slope * tb
    r2 = 1 - ((dtr - yh) ** 2).sum() / ((dtr - dtr.mean()) ** 2).sum()
    xs = np.linspace(tb.min() - 0.02, tb.max() + 0.02, 50)
    ax2.plot(xs, AT - xs, color=GREEN, lw=1.4, ls="--", label="prediction: slope $-1$")
    ax2.scatter(tb, dtr, s=55, color=BLUE, zorder=3)
    for i in range(len(rows)):
        ax2.annotate(names[i], (tb[i], dtr[i]), fontsize=6.5, xytext=(4, 3),
                     textcoords="offset points")
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xlabel("base transfer macro-AP")
    ax2.set_ylabel("$\\Delta$ transfer (SFT $-$ base)")
    ax2.set_title(f"So “stronger base specializes more”\nis arithmetic (fit slope ${slope:.2f}$, $R^2={r2:.2f}$)")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Act I, restated: after SFT the benchmark fixes the score; the checkpoint sets only the residual",
                 fontsize=10, y=1.02)
    fig.subplots_adjust(top=0.80, bottom=0.24, wspace=0.32, left=0.085, right=0.98)
    fig.savefig(HERE / "fig_attractor.pdf", metadata={"CreationDate": None}); plt.close(fig)


def prevalence():
    """MEASURED AP as a function of deployment prevalence, from the committed Paper A transfer scores:
    the four base guards' held-out ROC, reweighted to prevalence pi. Shows AP collapsing at low prevalence
    and the ranking re-spacing (Qwen2.5-1.5B leads SmolLM2-1.7B at balance but they cross at ~1%)."""
    import numpy as np, pandas as pd
    sp = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    if not sp.exists():
        return
    df = pd.read_parquet(sp, columns=["sample_id", "split", "source", "gold", "model_key",
                                       "condition", "probability_calibrated"])
    tr = df[(df.split == "transfer_test") & (df.condition == "base")].drop_duplicates(["model_key", "sample_id"])
    order = ["qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"]
    pretty = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B",
              "smollm3_3b": "SmolLM3-3B", "qwen3_4b": "Qwen3-4B"}
    cols = {"qwen25_15b": BLUE, "smollm2_17b": ORANGE, "smollm3_3b": GREEN, "qwen3_4b": RED}

    def ap_prev(frame, prev):  # macro-AP over the transfer sources at prevalence `prev`, from the empirical ROC
        aps = []
        for s in frame.source.unique():
            sub = frame[frame.source == s]; y = sub.gold.values.astype(int)
            o = np.argsort(-sub.probability_calibrated.values); y = y[o]
            P = y.sum(); N = len(y) - P
            if P == 0 or N == 0:
                continue
            tpr = np.cumsum(y) / P; fpr = np.cumsum(1 - y) / N
            prec = (prev * tpr) / (prev * tpr + (1 - prev) * fpr + 1e-12)
            aps.append(float(np.sum(prec * np.diff(np.concatenate([[0], tpr])))))
        return float(np.mean(aps)) if aps else float("nan")

    prevs = np.geomspace(0.005, 0.5, 40)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for mk in order:
        f = tr[tr.model_key == mk]
        ax.plot(prevs * 100, [ap_prev(f, p) for p in prevs], color=cols[mk], lw=2, label=pretty[mk])
    for p in (1, 5, 50):
        ax.axvline(p, color=GREY, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 5, 10, 50]); ax.set_xticklabels(["0.5%", "1%", "5%", "10%", "50%"])
    ax.set_xlabel("deployment prevalence of unsafe prompts (log scale)")
    ax.set_ylabel("transfer macro-AP")
    ax.set_ylim(0, 1.0)
    ax.set_title("The prevalence also chooses the winner (measured):\ntransfer AP collapses — and the ranking re-spaces — as positives get rare")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    fig.subplots_adjust(bottom=0.16, top=0.84, left=0.1, right=0.97)
    fig.savefig(HERE / "fig_prevalence.pdf", metadata={"CreationDate": None}); plt.close(fig)


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
    ax.set_title("Act II: composition recovers transfer above SFT\n(recovery, not dominance — it can dip below base, e.g. Qwen3-4B)")
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
    # no pass/fail threshold line: Delta_context is a signal, not a verdict (see text); bars are neutral
    ax2.bar(list(x), dctx, 0.5, color=BLUE)
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("$\\Delta_{\\mathrm{context}}$  (0 = score-invariant)")
    ax2.set_ylim(0, max(dctx) * 1.35)
    ax2.set_title("Protected-pair gap ($n{=}3$ pairs)")
    for i, v in enumerate(dctx): ax2.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Act III: instruction models (zero-shot) rank mortgage violations only moderately; protected-pair sensitivity varies (n=3)",
                 fontsize=10.5, y=1.00)
    fig.subplots_adjust(top=0.84, bottom=0.28, wspace=0.34, left=0.085, right=0.985)
    fig.savefig(HERE / "fig_mortgage_baseline.pdf", metadata={"CreationDate": None}); plt.close(fig)


def expguard_domains():
    """Per-domain AP (finance/health/law) for the 4 base guards on ExpGuard (baseline_expguard.json)."""
    p = REPO / "artifacts/expguard_external/baseline_expguard.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    tbl = {r["guard"]: r for r in d.get("table", [])}
    order = [g for g in ("qwen25_15b_base", "smollm2_17b_base", "smollm3_3b_base", "qwen3_4b_base") if g in tbl]
    if not order:
        return
    pretty = {"qwen25_15b_base": "Qwen2.5\n1.5B", "smollm2_17b_base": "SmolLM2\n1.7B",
              "smollm3_3b_base": "SmolLM3\n3B", "qwen3_4b_base": "Qwen3\n4B"}
    labels = [pretty[g] for g in order]
    fin = [tbl[g]["finance_ap"] for g in order]
    hea = [tbl[g]["healthcare_ap"] for g in order]
    law = [tbl[g]["law_ap"] for g in order]
    x = range(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.bar([i - w for i in x], fin, w, label="finance", color=BLUE)
    ax.bar(list(x), hea, w, label="health", color=GREEN)
    ax.bar([i + w for i in x], law, w, label="law", color=ORANGE)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("average precision (AP)")
    ax.set_title("Act III breadth: zero-shot base guards on ExpGuard\n"
                 "(finance / health / law --- ranking is not monotone in model size)")
    for xs, vals in (([i - w for i in x], fin), (list(x), hea), ([i + w for i in x], law)):
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.003, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(HERE / "fig_expguard_domains.pdf", metadata={"CreationDate": None}); plt.close(fig)


def diagrams():
    """Render the Graphviz flowcharts (.dot -> .png) if graphviz 'dot' is available; otherwise keep the
    committed PNGs (data-split construction + the paired experimental design)."""
    import shutil, subprocess
    dot = shutil.which("dot")
    if not dot:
        print("  [skip] diagrams: graphviz 'dot' not found; keeping committed PNGs")
        return
    for name in ("data_splits", "experiment_design"):
        src = HERE / f"{name}.dot"
        if src.exists():
            subprocess.run([dot, "-Tpng", "-Gdpi=150", str(src), "-o", str(HERE / f"{name}.png")], check=True)


def adaptation_plane():
    """Confirmatory adaptation study: the specialization plane. Represented vs held-out transfer
    macro-AP change (vs the same base) under SFT (o) and KL-SFT (triangle) for all 10 checkpoints,
    general vs released purpose-built guards. Reads the committed analysis results.json."""
    from matplotlib.lines import Line2D
    R = json.loads((REPO / "artifacts/starting_type_adaptation_v1/analysis/results.json").read_text())
    mv = R["point_estimates"]["movement_vectors"]
    GENERAL = {"qwen25_15b", "smollm2_17b", "smollm3_3b", "qwen3_4b"}
    LAB = {"qwen25_15b": "Qwen2.5-1.5B", "smollm2_17b": "SmolLM2-1.7B", "smollm3_3b": "SmolLM3-3B",
           "qwen3_4b": "Qwen3-4B", "qwen3guard_gen_06b": "Qwen3Guard-0.6B",
           "qwen3guard_gen_4b": "Qwen3Guard-4B", "granite_guardian_31_2b": "Granite-2B",
           "shieldgemma_2b": "ShieldGemma-2B", "llama_guard_3_1b": "Llama-Guard-3-1B",
           "wildguard_7b": "WildGuard-7B"}
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.axhline(0, color="#333", lw=0.8, zorder=1)
    ax.axvline(0, color="#333", lw=0.8, zorder=1)
    for k, v in mv.items():
        sft, kl = v["theta_sft"], v["theta_kl"]
        col = BLUE if k in GENERAL else GREEN
        if abs(sft[0]) < 1e-6 and abs(sft[1]) < 1e-6:  # Llama-Guard pruned-head null cell
            ax.scatter([0], [0], c=GREY, marker="x", s=45, zorder=4)
            ax.annotate(LAB.get(k, k) + " (null)", (0, 0), fontsize=6, color=GREY,
                        xytext=(4, 4), textcoords="offset points")
            continue
        ax.annotate("", xy=(kl[0], kl[1]), xytext=(sft[0], sft[1]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1, alpha=0.65), zorder=2)
        ax.scatter([sft[0]], [sft[1]], c=col, marker="o", s=34, zorder=3)
        ax.scatter([kl[0]], [kl[1]], c=col, marker="^", s=48, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.annotate(LAB.get(k, k), (sft[0], sft[1]), fontsize=6, color=col,
                    xytext=(4, -7), textcoords="offset points")
    ax.set_xlabel("Represented-source macro-AP change (vs. same base)")
    ax.set_ylabel("Held-out transfer macro-AP change")
    ax.set_title("Adaptation specialization plane: SFT ($\\circ$) $\\to$ KL-SFT ($\\triangle$)")
    ax.text(0.985, 0.03, "specialization\n(represented $\\uparrow$, transfer $\\downarrow$)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555")
    ax.legend(handles=[Line2D([0], [0], marker='o', color=BLUE, lw=0, label='general checkpoint'),
                       Line2D([0], [0], marker='o', color=GREEN, lw=0, label='purpose-built guard'),
                       Line2D([0], [0], marker='o', color='#555', lw=0, label='SFT'),
                       Line2D([0], [0], marker='^', color='#555', lw=0, label='KL-SFT')],
              fontsize=7, loc="upper left", framealpha=0.9)
    fig.savefig(HERE / "fig_adaptation_plane.pdf", metadata={"CreationDate": None}); plt.close(fig)


def ensembling_plane():
    """Ensembling plane: equal-family mean (Δrepresented, Δtransfer) vs each checkpoint's own base,
    for the within-checkpoint ensembles. Seed-ensembling a fine-tune (orange) stays BELOW the base
    transfer line; only crossing the specialization axis by adding the un-tuned base (green) lifts
    transfer above base. Reads the committed ensembling_point.json (excl-null panel)."""
    from matplotlib.lines import Line2D
    P = json.loads((REPO / "artifacts/starting_type_adaptation_v1/analysis/ensembling_point.json").read_text())
    m = P["panels"]["excl_null"]["methods"]

    def pt(k):
        return m[k]["drep"], m[k]["dtrans"]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.axhspan(0, 0.05, color=GREEN, alpha=0.06, zorder=0)
    ax.axhspan(-0.12, 0, color=RED, alpha=0.05, zorder=0)
    ax.axhline(0, color="#333", lw=0.9, zorder=1)
    ax.axvline(0, color="#333", lw=0.9, zorder=1)
    # A) ensemble a fine-tune with itself (redundant -> correlated) : orange, single -> 5-seed
    for single, ens, lab in [("sft_single", "sft_seedens", "SFT"), ("kl_single", "kl_seedens", "KL-SFT")]:
        x0, y0 = pt(single); x1, y1 = pt(ens)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2, alpha=0.8), zorder=2)
        ax.scatter([x0], [y0], c=ORANGE, marker="o", s=42, zorder=3)
        ax.scatter([x1], [y1], c=ORANGE, marker="s", s=46, edgecolors="white", linewidths=0.4, zorder=3)
        ax.annotate(f"{lab} single", (x0, y0), fontsize=6.5, color=ORANGE,
                    xytext=(2, -9), textcoords="offset points")
        ax.annotate(f"{lab} $\\times$5 seeds", (x1, y1), fontsize=6.5, color=ORANGE,
                    xytext=(2, 5), textcoords="offset points")
    # B) ensemble across the specialization axis (diverse -> decorrelated) : green
    x0, y0 = pt("sft_single"); x1, y1 = pt("base_sft")
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6), zorder=2)
    for k, lab, off in [("base_sft", "base$\\oplus$SFT", (6, 6)), ("base_kl", "base$\\oplus$KL", (4, -9)),
                        ("base_sft_kl", "", (0, 0)), ("sft_kl", "SFT$\\oplus$KL", (2, -9))]:
        x, y = pt(k)
        mk = "o" if k == "sft_kl" else "D"
        ax.scatter([x], [y], c=GREEN, marker=mk, s=48, edgecolors="white", linewidths=0.4, zorder=3)
        if lab:
            ax.annotate(lab, (x, y), fontsize=6.5, color=GREEN, xytext=off, textcoords="offset points")
    ax.scatter([0], [0], c=GREY, marker="*", s=150, zorder=4)
    ax.annotate("base", (0, 0), fontsize=7, color="#555", xytext=(4, 4), textcoords="offset points")
    ax.text(0.985, 0.965, "transfer recovered\n($\\geq$ base)", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=GREEN)
    ax.text(0.985, 0.03, "transfer below base", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=RED)
    ax.set_xlabel("Represented-source macro-AP change (vs. own base)")
    ax.set_ylabel("Held-out transfer macro-AP change")
    ax.set_title("Ensembling plane: only crossing the specialization axis recovers transfer")
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color=ORANGE, lw=0, label='fine-tune with itself (seeds)'),
        Line2D([0], [0], marker='D', color=GREEN, lw=0, label='base $\\oplus$ adapter (diverse)'),
        Line2D([0], [0], marker='*', color=GREY, lw=0, label='unmodified base')],
        fontsize=7, loc="lower left", framealpha=0.9)
    fig.savefig(HERE / "fig_ensembling_plane.pdf", metadata={"CreationDate": None}); plt.close(fig)


def main():
    made = []
    for fn in (act1_percheckpoint, attractor, act3_composition, mortgage_quadrant, mortgage_baseline,
               expguard_domains, prevalence, adaptation_plane, ensembling_plane, diagrams):
        try:
            fn(); made.append(fn.__name__)
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")
    print("generated:", ", ".join(made))


if __name__ == "__main__":
    main()
