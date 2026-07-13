#!/usr/bin/env python
"""Ensemble probe: can combining models recover OOD (novel) ranking without losing in-house gains?

Uses ONLY cached per-row P(unsafe) (no model inference). Reconstructs the novel gold by replaying the
deterministic load_new() (balance seed=7, PC=400) used by guard_eval_pipeline.py, then sweeps convex
weights over model pairs and reports threshold-free AUPRC on the in-house pool (2018) and the balanced
novel pool (2020). AUPRC is rank-based, so we test BOTH probability-averaging and rank-averaging.

Endpoints are sanity-checked against the paper/summary AUPRCs before any ensemble claim is trusted.
Run from repo root:  .venv/bin/python legacy/experiments/ensemble_probe.py
"""
import os, json, random
import numpy as np

# ---- env / seeds (match guard_eval_pipeline) ----
def le(p):
    if os.path.exists(p):
        for l in open(p):
            l=l.strip()
            if l and not l.startswith("#") and "=" in l:
                k,v=l.split("=",1); os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
le("notebooks/.env"); le(".env"); HF=os.environ.get("HF_TOKEN")
SEED=42; random.seed(SEED); np.random.seed(SEED)
ND="notebooks/outputs/nb-smollm3-guard"; PC=400
def _norm(t): return " ".join((t or "").lower().split())
def balance(rows,k,seed=7):
    rng=random.Random(seed);s=[r for r in rows if r["label"]=="safe"];u=[r for r in rows if r["label"]=="unsafe"]
    rng.shuffle(s);rng.shuffle(u);n=min(k,len(s),len(u)) if k else min(len(s),len(u));o=s[:n]+u[:n];rng.shuffle(o);return o

# ---- metric (identical impl to pipeline) ----
def auprc(s,g):
    s=np.asarray(s,float);g=np.asarray(g,float);o=np.argsort(-s);g=g[o]
    tp=np.cumsum(g);fp=np.cumsum(1-g);P=g.sum()
    if P==0: return float('nan')
    pr=tp/(tp+fp);rc=tp/P;rc=np.r_[0,rc];pr=np.r_[1,pr];return float(np.sum((rc[1:]-rc[:-1])*pr[1:]))
def rank01(s):
    s=np.asarray(s,float);o=np.argsort(s);r=np.empty(len(s));r[o]=np.arange(len(s));return r/(len(s)-1)
def paired_boot(gold, sa, sb, B=3000, seed=0):
    """95% CI on auprc(sa)-auprc(sb), paired bootstrap over rows."""
    gold=np.asarray(gold);sa=np.asarray(sa);sb=np.asarray(sb);n=len(gold);rng=np.random.default_rng(seed);d=[]
    for _ in range(B):
        i=rng.integers(0,n,n); d.append(auprc(sa[i],gold[i])-auprc(sb[i],gold[i]))
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

# ---- reconstruct novel gold deterministically (dataset load only; no model) ----
from datasets import load_dataset
def load_new():
    N={}
    wg=load_dataset("allenai/wildguardmix","wildguardtest",split="test",token=HF)
    N["wildguardtest"]=balance([{"text":r["prompt"],"label":"unsafe" if r["prompt_harm_label"]=="harmful" else "safe"} for r in wg if r["prompt_harm_label"] in ("harmful","unharmful") and (r["prompt"] or "").strip()],PC)
    wj=load_dataset("allenai/wildjailbreak","eval",split="train",token=HF)
    N["wildjailbreak"]=balance([{"text":r["adversarial"],"label":"unsafe" if int(r["label"])==1 else "safe"} for r in wj if (r["adversarial"] or "").strip()],PC)
    hb=load_dataset("bench-llm/or-bench","or-bench-hard-1k",split="train",token=HF);tox=load_dataset("bench-llm/or-bench","or-bench-toxic",split="train",token=HF)
    N["orbench_hard"]=balance([{"text":r["prompt"],"label":"safe"} for r in hb if (r["prompt"] or "").strip()]+[{"text":r["prompt"],"label":"unsafe"} for r in tox if (r["prompt"] or "").strip()],PC)
    hbm=load_dataset("walledai/HarmBench","standard",split="train",token=HF)
    N["harmbench"]=[{"text":r["prompt"],"label":"unsafe"} for r in hbm if (r["prompt"] or "").strip()]
    return N

print("reconstructing novel gold (deterministic load_new)...", flush=True)
NEW=load_new()
balb=[b for b in NEW if b!="harmbench"]
gold_bal=np.concatenate([np.array([1 if r["label"]=="unsafe" else 0 for r in NEW[b]]) for b in balb])
n_bal=len(gold_bal); sizes={b:len(NEW[b]) for b in NEW}
# balanced rows are the first sum(balb) rows of the 2220 novel order (harmbench appended last)
mask_bal=slice(0,n_bal)
print(f"  novel sizes={sizes}  balanced n={n_bal}  (unsafe={int(gold_bal.sum())})")

L=lambda f: json.load(open(f"{ND}/{f}"))
def scores_bal(f, key=None):
    d=L(f); arr=np.array(d[key] if key else d); return arr[mask_bal]

# ---- load per-row scores ----
# in-house (2018), gold aligned across families (verified)
g_ih=np.array(L("base_smollm3_inhouse.json")["gold"])
IH={
 "smol_base": np.array(L("base_smollm3_inhouse.json")["test_scores"]),
 "smol_tuned": np.array(L("preds_corrected.json")["guard_test_cont"]),
 "qwen_base": np.array(L("qwen2.5-1.5b_base_inhouse_test.json")["scores"]),
 "qwen_tuned": np.array(L("qwen2.5-1.5b_guard_inhouse_test.json")["scores"]),
}
# novel balanced (2020)
NV={
 "smol_base": scores_bal("_cache_base_exp.json"),
 "smol_tuned": scores_bal("_cache_guard_exp.json"),
 "qwen_base": scores_bal("qwen2.5-1.5b_base_novel.json","scores"),
 "qwen_tuned": scores_bal("qwen2.5-1.5b_guard_novel.json","scores"),
 "llama": scores_bal("_cache_llama_exp.json"),
}

# ---- SANITY: endpoints must match the paper before trusting ensembles ----
print("\n=== SANITY (computed vs paper) ===")
exp={("smol","in"):{"base":0.696,"tuned":0.844},("smol","nov"):{"base":0.886,"tuned":0.781},
     ("qwen","in"):{"base":0.681,"tuned":0.821},("qwen","nov"):{"base":0.795,"tuned":0.758}}
ok=True
for fam in ("smol","qwen"):
    ib=auprc(IH[f"{fam}_base"],g_ih); it=auprc(IH[f"{fam}_tuned"],g_ih)
    nb=auprc(NV[f"{fam}_base"],gold_bal); nt=auprc(NV[f"{fam}_tuned"],gold_bal)
    print(f"  {fam}: in-house base {ib:.3f}/{exp[(fam,'in')]['base']}  tuned {it:.3f}/{exp[(fam,'in')]['tuned']}"
          f"  | novel base {nb:.3f}/{exp[(fam,'nov')]['base']}  tuned {nt:.3f}/{exp[(fam,'nov')]['tuned']}")
    for got,ref in [(nb,exp[(fam,'nov')]['base']),(nt,exp[(fam,'nov')]['tuned'])]:
        if abs(got-ref)>0.02: ok=False
print(f"  novel alignment {'OK' if ok else 'MISMATCH -> gold order wrong, do not trust'}")
assert ok, "novel endpoints do not match; gold reconstruction misaligned"

# ---- ensemble helpers ----
WS=np.round(np.linspace(0,1,11),2)
def sweep(gold, sA, sB, mode="prob"):
    """AUPRC of w*A + (1-w)*B across w; A/B raw prob (mode=prob) or rank-normalized (mode=rank)."""
    A = rank01(sA) if mode=="rank" else np.asarray(sA,float)
    B = rank01(sB) if mode=="rank" else np.asarray(sB,float)
    return {float(w): auprc(w*A+(1-w)*B, gold) for w in WS}

def report_pair(name, gold, sA, sB, aucA, aucB):
    print(f"\n--- {name}  (A endpoint={aucA:.3f}, B endpoint={aucB:.3f}) ---")
    out={}
    for mode in ("prob","rank"):
        sw=sweep(gold,sA,sB,mode); out[mode]=sw
        best_w=max(sw,key=sw.get); best=sw[best_w]
        mid=sw[0.5]
        print(f"  {mode:4s}: w=0.5 -> {mid:.3f} | best w={best_w:.2f} -> {best:.3f} | full curve: "
              +" ".join(f"{w:.1f}:{sw[w]:.3f}" for w in WS))
    return out

RESULTS={"novel":{}, "inhouse":{}, "meta":{"n_novel":int(n_bal),"n_inhouse":int(len(g_ih))}}

print("\n############ NOVEL / OOD (balanced 2020) ############")
# (1) base <-> tuned, same family  (does adding base recover OOD without wrecking in-house?)
for fam in ("smol","qwen"):
    RESULTS["novel"][f"{fam}: tuned<->base"]=report_pair(
        f"NOVEL {fam}: A=tuned  B=base", gold_bal, NV[f"{fam}_tuned"], NV[f"{fam}_base"],
        auprc(NV[f"{fam}_tuned"],gold_bal), auprc(NV[f"{fam}_base"],gold_bal))
# (2) cross-family tuned <-> tuned  (the literal "ensemble fine-tuned models")
RESULTS["novel"]["cross tuned⊕tuned"]=report_pair(
    "NOVEL cross-family: A=smol_tuned B=qwen_tuned", gold_bal, NV["smol_tuned"], NV["qwen_tuned"],
    auprc(NV["smol_tuned"],gold_bal), auprc(NV["qwen_tuned"],gold_bal))
# (3) cross-family base <-> base  (both OOD-strong)
RESULTS["novel"]["cross base⊕base"]=report_pair(
    "NOVEL cross-family: A=smol_base B=qwen_base", gold_bal, NV["smol_base"], NV["qwen_base"],
    auprc(NV["smol_base"],gold_bal), auprc(NV["qwen_base"],gold_bal))

# (4) model soups (equal-weight means), prob and rank
def soup(keys, store, gold, mode="rank"):
    arrs=[ (rank01(store[k]) if mode=="rank" else np.asarray(store[k],float)) for k in keys]
    return auprc(np.mean(arrs,0), gold)
print("\n--- NOVEL equal-weight soups (rank-avg) ---")
soups={
 "all4 {smol,qwen}x{base,tuned}":["smol_base","smol_tuned","qwen_base","qwen_tuned"],
 "both tuned":["smol_tuned","qwen_tuned"],
 "both base":["smol_base","qwen_base"],
 "smol base+tuned":["smol_base","smol_tuned"],
 "qwen base+tuned":["qwen_base","qwen_tuned"],
}
for nm,ks in soups.items():
    RESULTS["novel"].setdefault("soups",{})[nm]={"rank":soup(ks,NV,gold_bal,"rank"),"prob":soup(ks,NV,gold_bal,"prob")}
    print(f"  {nm:34s} rank {soup(ks,NV,gold_bal,'rank'):.3f} | prob {soup(ks,NV,gold_bal,'prob'):.3f}")

print("\n############ IN-HOUSE (2018) ############")
for fam in ("smol","qwen"):
    RESULTS["inhouse"][f"{fam}: tuned<->base"]=report_pair(
        f"INHOUSE {fam}: A=tuned  B=base", g_ih, IH[f"{fam}_tuned"], IH[f"{fam}_base"],
        auprc(IH[f"{fam}_tuned"],g_ih), auprc(IH[f"{fam}_base"],g_ih))
print("\n--- IN-HOUSE equal-weight soups (rank-avg) ---")
for nm,ks in soups.items():
    RESULTS["inhouse"].setdefault("soups",{})[nm]={"rank":soup(ks,IH,g_ih,"rank"),"prob":soup(ks,IH,g_ih,"prob")}
    print(f"  {nm:34s} rank {soup(ks,IH,g_ih,'rank'):.3f} | prob {soup(ks,IH,g_ih,'prob'):.3f}")

# ---- headline: for the deployable single model (tuned), does a base-blend Pareto-help? ----
print("\n############ HEADLINE: does base⊕tuned beat the pure TUNED model on BOTH axes? ############")
for fam in ("smol","qwen"):
    # Apples-to-apples: the blend is built from rank-transformed scores, so the tuned baseline must
    # be scored in the SAME rank space (rank01 breaks the many score ties by index, so auprc(rank01(x))
    # != auprc(x); comparing a rank-blend to the RAW tuned endpoint would make w=1.0 show a spurious gain).
    rt_in=rank01(IH[f"{fam}_tuned"]); rt_nov=rank01(NV[f"{fam}_tuned"])
    t_in=auprc(rt_in,g_ih); t_nov=auprc(rt_nov,gold_bal)
    b_in=auprc(IH[f"{fam}_base"],g_ih);  b_nov=auprc(NV[f"{fam}_base"],gold_bal)  # raw base endpoints (context)
    # rank-avg blend at each w; find w maximizing min-improvement over tuned on both axes
    best=None
    for w in WS:
        in_a=auprc(w*rt_in+(1-w)*rank01(IH[f"{fam}_base"]),g_ih)
        nv_a=auprc(w*rt_nov+(1-w)*rank01(NV[f"{fam}_base"]),gold_bal)
        # Pareto-dominates tuned iff both >= tuned (small tol) and at least one strictly >
        dom = (in_a>=t_in-1e-6) and (nv_a>=t_nov-1e-6) and ((in_a>t_in+1e-6) or (nv_a>t_nov+1e-6))
        score=min(in_a-t_in, nv_a-t_nov)  # worst-axis gain over tuned (0 at w=1.0 by construction)
        if best is None or score>best[0]: best=(score,w,in_a,nv_a,dom)
    _,w,in_a,nv_a,dom=best
    lo,hi=paired_boot(gold_bal, w*rt_nov+(1-w)*rank01(NV[f"{fam}_base"]), rt_nov)
    print(f"  {fam}: tuned(in={t_in:.3f},nov={t_nov:.3f})  base(in={b_in:.3f},nov={b_nov:.3f})")
    print(f"       best rank-blend w={w:.2f}: in={in_a:.3f} ({in_a-t_in:+.3f}), nov={nv_a:.3f} ({nv_a-t_nov:+.3f})"
          f"  Pareto-dominates tuned? {dom}")
    print(f"       novel gain over tuned 95% CI [{lo:+.3f},{hi:+.3f}] (excludes 0 => significant)")
    RESULTS.setdefault("headline",{})[fam]={"tuned_in":t_in,"tuned_nov":t_nov,"base_in":b_in,"base_nov":b_nov,
        "best_w":float(w),"blend_in":in_a,"blend_nov":nv_a,"pareto_dominates_tuned":bool(dom),
        "novel_gain_over_tuned_ci":[lo,hi]}

json.dump(RESULTS, open(f"{ND}/summary_ensemble_probe.json","w"), indent=2, default=float)
print(f"\nsaved -> {ND}/summary_ensemble_probe.json\nDONE_ENSEMBLE")
