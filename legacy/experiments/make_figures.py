#!/usr/bin/env python
"""Generate publication figures (vector PDF) for the ACM paper, from verified numbers.
dataviz rules: form-by-job, Okabe-Ito CVD-safe categorical palette (fixed order), thin marks,
direct value labels, recessive axes, CI error bars, legend for >=2 series. -> paper/figures/*.pdf
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
os.makedirs("paper/figures", exist_ok=True)
# Okabe-Ito (colorblind-safe). Fixed entity->hue assignment (never cycled).
C = {"guard":"#0072B2","base":"#56B4E9","llama":"#E69F00","shield":"#009E73","gpt":"#D55E00","kw":"#999999",
     "pos":"#0072B2","neg":"#D55E00","ink":"#222222","muted":"#666666","grid":"#DDDDDD"}
plt.rcParams.update({"font.family":"sans-serif","font.size":8,"axes.edgecolor":C["muted"],
    "axes.linewidth":0.8,"xtick.color":C["muted"],"ytick.color":C["muted"],"text.color":C["ink"],
    "axes.labelcolor":C["ink"],"figure.dpi":200})
def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y",color=C["grid"],linewidth=0.6,zorder=0); ax.set_axisbelow(True)
def lab(ax,bars,vals,fmt="{:.3f}",dy=0.006):
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+dy,fmt.format(v),ha="center",va="bottom",fontsize=7,color=C["ink"])

# ---- Fig 1: In-house pooled AUPRC (guard vs open guards); guard has CI ----
fig,ax=plt.subplots(figsize=(3.3,2.5))
sys=["Guard\n(ours, 3B)","ShieldGemma-2b","Llama-Guard-3-1B"]; ap=[0.833,0.710,0.624]; col=[C["guard"],C["shield"],C["llama"]]  # tie-aware
err=[[0.833-0.813],[0.854-0.833]]
b=ax.bar(sys,ap,color=col,width=0.62,zorder=3)
ax.errorbar([0],[0.833],yerr=err,fmt="none",ecolor=C["ink"],capsize=3,lw=1,zorder=4)
lab(ax,b,ap); ax.set_ylim(0,1.0); ax.set_ylabel("AUPRC (in-house pooled)"); clean(ax)
ax.set_title("Threshold-free discrimination (in-house)",fontsize=8.5,color=C["ink"])
plt.tight_layout(); plt.savefig("paper/figures/fig1_inhouse_auprc.pdf",bbox_inches="tight"); plt.close()

# ---- Fig 2: NOVEL held-out AUPRC (base vs tuned guard vs llama) with CIs -- the key cross result ----
fig,ax=plt.subplots(figsize=(3.3,2.5))
sys=["Base\n(zero-shot)","Guard\n(tuned)","Llama-Guard\n-3-1B"]; ap=[0.884,0.780,0.685]; col=[C["base"],C["guard"],C["llama"]]  # tie-aware
lo=[0.869,0.751,0.657]; hi=[0.899,0.809,0.715]
err=[[a-l for a,l in zip(ap,lo)],[h-a for a,h in zip(ap,hi)]]
b=ax.bar(sys,ap,color=col,width=0.62,zorder=3)
ax.errorbar(range(3),ap,yerr=err,fmt="none",ecolor=C["ink"],capsize=3,lw=1,zorder=4)
lab(ax,b,ap,dy=0.012); ax.set_ylim(0,1.0); ax.set_ylabel("AUPRC (3 balanced novel sets)"); clean(ax)
ax.set_title("Out-of-distribution: base > tuned > Llama-Guard",fontsize=8.5,color=C["ink"])
plt.tight_layout(); plt.savefig("paper/figures/fig2_novel_auprc.pdf",bbox_inches="tight"); plt.close()

# ---- Fig 3: Operating-point flip (grouped): native-F1 vs AUPRC vs matched-FPR F1 ----
fig,ax=plt.subplots(figsize=(3.5,2.6))
groups=["Native-thresh\nF1","Threshold-free\nAUPRC","Matched-FPR\nF1"]
guard=[0.794,0.833,0.581]; shield=[0.424,0.710,0.464]; llama=[0.673,0.624,0.360]  # AUPRC tie-aware
x=np.arange(3); w=0.26
b1=ax.bar(x-w,guard,w,label="Guard (ours)",color=C["guard"],zorder=3)
b2=ax.bar(x,shield,w,label="ShieldGemma-2b",color=C["shield"],zorder=3)
b3=ax.bar(x+w,llama,w,label="Llama-Guard-3-1B",color=C["llama"],zorder=3)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0,1.0); ax.set_ylabel("score"); clean(ax)
ax.legend(frameon=False,fontsize=6.5,loc="upper right",ncol=1)
ax.set_title("Ranking flip: ShieldGemma<Llama at native F1,\n>Llama under AUPRC / matched-FPR",fontsize=8,color=C["ink"])
# annotate the flip on the two open guards
plt.tight_layout(); plt.savefig("paper/figures/fig3_operating_point_flip.pdf",bbox_inches="tight"); plt.close()

# ---- Fig 4: Base->Tuned per-benchmark F1 delta (diverging) ----
fig,ax=plt.subplots(figsize=(3.4,2.7))
bench=["jailbreak_cls","toxicchat","prompt_inj","beavertails","jailbreakbench*","xstest*"]
delta=[0.641,0.150,0.138,0.024,0.012,-0.023]
colors=[C["pos"] if d>=0 else C["neg"] for d in delta]
y=np.arange(len(bench))[::-1]
b=ax.barh(y,delta,color=colors,height=0.6,zorder=3)
ax.set_yticks(y); ax.set_yticklabels(bench,fontsize=7)
ax.axvline(0,color=C["muted"],lw=0.8)
for yi,d in zip(y,delta): ax.text(d+(0.01 if d>=0 else -0.01),yi,f"{d:+.3f}",va="center",ha="left" if d>=0 else "right",fontsize=6.5,color=C["ink"])
ax.set_xlim(-0.15,0.75); ax.set_xlabel("F1 change from LoRA (tuned − base)"); clean(ax); ax.grid(axis="x",color=C["grid"],lw=0.6)
ax.spines["left"].set_visible(False)
ax.set_title("LoRA gains are in-distribution;\nheld-out (*) essentially flat",fontsize=8,color=C["ink"])
plt.tight_layout(); plt.savefig("paper/figures/fig4_base_vs_tuned.pdf",bbox_inches="tight"); plt.close()

# ---- Fig 5: Pareto — F1 vs single-request latency (guard vs GPT) ----
fig,ax=plt.subplots(figsize=(3.3,2.5))
ax.scatter([124],[0.794],s=70,color=C["guard"],zorder=3,label="Guard (ours, local)")
ax.scatter([512],[0.784],s=70,color=C["gpt"],marker="D",zorder=3,label="gpt-5.4-mini (API)")
ax.annotate("Guard\n0.794 F1 / 124 ms",(124,0.794),xytext=(150,0.74),fontsize=6.5,color=C["ink"])
ax.annotate("gpt-5.4-mini\n0.784 F1 / 512 ms",(512,0.784),xytext=(330,0.83),fontsize=6.5,color=C["ink"])
ax.set_xlim(0,600); ax.set_ylim(0.7,0.86); ax.set_xlabel("single-request latency (ms, batch=1)"); ax.set_ylabel("pooled F1"); clean(ax)
ax.set_title("Pareto: parity F1 at ~4x lower latency, local",fontsize=8.5,color=C["ink"])
plt.tight_layout(); plt.savefig("paper/figures/fig5_pareto.pdf",bbox_inches="tight"); plt.close()

print("figures written:")
for f in sorted(os.listdir("paper/figures")): print("  paper/figures/"+f)
