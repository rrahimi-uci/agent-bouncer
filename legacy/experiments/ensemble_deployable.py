#!/usr/bin/env python
"""Deployable base+tuned ensemble: per-prompt, leak-free, no re-scoring.

The rank-average win in ensemble_probe.py used global test-set ranks (an oracle upper bound). Here we make
it DEPLOYABLE: each model's score is mapped through a fixed monotone transform (empirical CDF / probability
integral transform) fit ONLY on the in-distribution portion of the in-house pool, then averaged. That map is
applied per-prompt (no batch, no test info), so it is exactly what you'd ship. Evaluation is restricted to
rows DISJOINT from the fit set -- the in-house held-out benchmarks and the novel/OOD benchmarks -- so the
reported AUPRCs have no leakage. We report three points:
  raw prob-avg  (simplest per-prompt deployable, conservative)
  PIT-avg       (dev-style CDF normalization, the deployable recipe we recommend)
  rank-avg      (a non-deployable global-rank-average alternative, from ensemble_probe.py, for context;
                 NOT an upper bound -- the PIT ensemble can and does exceed it, e.g. Qwen 0.847 > 0.845)
A single model's AUPRC is invariant to any monotone map, so the base/tuned endpoints are unchanged.
Run from repo root:  .venv/bin/python legacy/experiments/ensemble_deployable.py
"""
import os, json, random
import numpy as np

def le(p):
    if os.path.exists(p):
        for l in open(p):
            l=l.strip()
            if l and not l.startswith("#") and "=" in l:
                k,v=l.split("=",1); os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
le("notebooks/.env"); le(".env"); HF=os.environ.get("HF_TOKEN")
random.seed(42); np.random.seed(42)
ND="notebooks/outputs/nb-smollm3-guard"; PC=400; HELD_OUT={"jailbreakbench","xstest"}
def _norm(t): return " ".join((t or "").lower().split())
def balance(rows,k,seed=7):
    rng=random.Random(seed);s=[r for r in rows if r["label"]=="safe"];u=[r for r in rows if r["label"]=="unsafe"]
    rng.shuffle(s);rng.shuffle(u);n=min(k,len(s),len(u)) if k else min(len(s),len(u));o=s[:n]+u[:n];rng.shuffle(o);return o
def auprc(s,g):  # tie-aware non-interpolated AP (sklearn-canonical; groups equal scores)
    from sklearn.metrics import average_precision_score
    g=np.asarray(g)
    if g.min()==g.max(): return float('nan')
    return float(average_precision_score(g, np.asarray(s,float)))
def paired_boot(gold, sa, sb, B=3000, seed=0):
    gold=np.asarray(gold);sa=np.asarray(sa);sb=np.asarray(sb);n=len(gold);rng=np.random.default_rng(seed);d=[]
    for _ in range(B):
        i=rng.integers(0,n,n); d.append(auprc(sa[i],gold[i])-auprc(sb[i],gold[i]))
    return float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

# ---- deployable monotone normalization: empirical CDF fit on a reference set ----
def fit_pit(ref):  return np.sort(np.asarray(ref,float))
def apply_pit(xs,s): return np.searchsorted(xs,np.asarray(s,float),side="right")/len(xs)

# ---- novel gold (deterministic replay; balanced 2020 = first rows, harmbench appended last) ----
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
print("reconstructing novel gold...", flush=True)
NEW=load_new(); balb=[b for b in NEW if b!="harmbench"]
gold_nov=np.concatenate([np.array([1 if r["label"]=="unsafe" else 0 for r in NEW[b]]) for b in balb]); n_bal=len(gold_nov)

L=lambda f: json.load(open(f"{ND}/{f}"))
sl=lambda f,k=None: (np.array(L(f)[k]) if k else np.array(L(f)))[:n_bal]
# in-house (2018) + strata
gold_ih=np.array(L("base_smollm3_inhouse.json")["gold"]); strata=np.array(L("preds_corrected.json")["strata"])
fit_mask=~np.isin(strata,list(HELD_OUT)); ho_mask=np.isin(strata,list(HELD_OUT))
IH={"smol_base":np.array(L("base_smollm3_inhouse.json")["test_scores"]),
    "smol_tuned":np.array(L("preds_corrected.json")["guard_test_cont"]),
    "qwen_base":np.array(L("qwen2.5-1.5b_base_inhouse_test.json")["scores"]),
    "qwen_tuned":np.array(L("qwen2.5-1.5b_guard_inhouse_test.json")["scores"])}
NV={"smol_base":sl("_cache_base_exp.json"),"smol_tuned":sl("_cache_guard_exp.json"),
    "qwen_base":sl("qwen2.5-1.5b_base_novel.json","scores"),"qwen_tuned":sl("qwen2.5-1.5b_guard_novel.json","scores")}
print(f"  novel balanced n={n_bal} | in-house held-out n={int(ho_mask.sum())} | fit(in-dist) n={int(fit_mask.sum())}")

# fit PIT per model on in-distribution in-house rows (leak-free for held-out + novel eval)
PIT={m:fit_pit(IH[m][fit_mask]) for m in IH}
def ens(members, store, mask=None, kind="pit"):
    arrs=[]
    for m in members:
        s=store[m] if mask is None else store[m][mask]
        arrs.append(apply_pit(PIT[m],s) if kind=="pit" else np.asarray(s,float))
    return np.mean(arrs,0)

PAIRS={"smol base+tuned":["smol_base","smol_tuned"],"qwen base+tuned":["qwen_base","qwen_tuned"],
       "cross tuned+tuned":["smol_tuned","qwen_tuned"],"all4":["smol_base","smol_tuned","qwen_base","qwen_tuned"]}
# (inhouse-pooled, novel) global-rank-average from ensemble_probe.py -- a reference ALTERNATIVE ensemble,
# NOT an upper bound (the deployable PIT ensemble exceeds it on Qwen and cross). cross novel = rank-avg best
# 0.796 (0.799 is the prob-avg best; do not confuse the two).
RANK_REF={"smol base+tuned":(0.847,0.907),"qwen base+tuned":(0.820,0.845),"cross tuned+tuned":(None,0.796),"all4":(0.794,0.898)}

def A(scores,gold): return round(auprc(scores,gold),3)
res={"endpoints":{}, "ensembles":{}, "meta":{"n_novel":int(n_bal),"n_inhouse_heldout":int(ho_mask.sum()),"n_fit_indist":int(fit_mask.sum())}}

print("\n=== ENDPOINTS (single model; AUPRC invariant to monotone PIT) ===")
print(f"{'model':14s} {'in-house HELD-OUT':>18s} {'novel/OOD':>11s}")
for m in IH:
    ho=A(IH[m][ho_mask],gold_ih[ho_mask]); nv=A(NV[m],gold_nov)
    res["endpoints"][m]={"inhouse_heldout":ho,"novel":nv}; print(f"{m:14s} {ho:18.3f} {nv:11.3f}")

print("\n=== DEPLOYABLE ENSEMBLES ===")
print(f"{'ensemble':18s} | {'in-house HELD-OUT':>26s} | {'novel / OOD':>26s} | rank-avg-ref(nov)")
print(f"{'':18s} | {'prob-avg':>12s} {'PIT-avg':>13s} | {'prob-avg':>12s} {'PIT-avg':>13s} |")
for nm,mem in PAIRS.items():
    ho_p=A(ens(mem,IH,ho_mask,"prob"),gold_ih[ho_mask]); ho_q=A(ens(mem,IH,ho_mask,"pit"),gold_ih[ho_mask])
    nv_p=A(ens(mem,NV,None,"prob"),gold_nov); nv_q=A(ens(mem,NV,None,"pit"),gold_nov)
    res["ensembles"][nm]={"inhouse_heldout":{"prob":ho_p,"pit":ho_q},"novel":{"prob":nv_p,"pit":nv_q}}
    print(f"{nm:18s} | {ho_p:12.3f} {ho_q:13.3f} | {nv_p:12.3f} {nv_q:13.3f} | {RANK_REF[nm][1]}")

# headline: deployable PIT base+tuned vs pure tuned on novel (CI), per family
print("\n=== HEADLINE: deployable PIT base+tuned vs pure tuned, on NOVEL (OOD) ===")
for fam in ("smol","qwen"):
    ens_nov=ens([f"{fam}_base",f"{fam}_tuned"],NV,None,"pit"); t_nov=NV[f"{fam}_tuned"]
    a_e=A(ens_nov,gold_nov); a_t=A(t_nov,gold_nov); a_b=A(NV[f"{fam}_base"],gold_nov)
    lo,hi=paired_boot(gold_nov,ens_nov,t_nov)
    lo2,hi2=paired_boot(gold_nov,ens_nov,NV[f"{fam}_base"])
    print(f"  {fam}: tuned {a_t:.3f} | base {a_b:.3f} | PIT-ens {a_e:.3f}  "
          f"(vs tuned {a_e-a_t:+.3f} CI[{lo:+.3f},{hi:+.3f}]; vs base {a_e-a_b:+.3f} CI[{lo2:+.3f},{hi2:+.3f}])")
    res.setdefault("headline",{})[fam]={"tuned":a_t,"base":a_b,"pit_ensemble":a_e,
        "gain_vs_tuned_ci":[lo,hi],"gain_vs_base_ci":[lo2,hi2]}

json.dump(res,open(f"{ND}/summary_ensemble_deployable.json","w"),indent=2,default=float)
print(f"\nsaved -> {ND}/summary_ensemble_deployable.json\nDONE_DEPLOYABLE")
