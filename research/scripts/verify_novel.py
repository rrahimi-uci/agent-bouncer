#!/usr/bin/env python
"""Re-ground the NOVEL held-out numbers (paper Table/prose: base 0.886, tuned 0.781, llama 0.701;
best-threshold F1 base 0.792 / tuned 0.796) directly from the cached continuous scores + reconstructed gold.
Reuses eval_novel_gaps.py's load_new() ordering; asserts the reconstructed row count matches the caches
(alignment guard). Does NOT load any model (network+CPU only). Also writes summary_novel_full.json,
which the original run never reached because ShieldGemma stalled before the aggregate step.
"""
import os, json, random, numpy as np, warnings; warnings.filterwarnings("ignore")
def le(p):
    if os.path.exists(p):
        for l in open(p):
            l=l.strip()
            if l and not l.startswith("#") and "=" in l:
                k,v=l.split("=",1); os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
le("notebooks/.env"); le(".env"); HF=os.environ.get("HF_TOKEN")
SEED=42; random.seed(SEED); np.random.seed(SEED)
PC=400; CACHE="notebooks/outputs/nb-smollm3-guard"
from datasets import load_dataset
def bal(rows,k,seed=7):
    rng=random.Random(seed); s=[r for r in rows if r["label"]=="safe"]; u=[r for r in rows if r["label"]=="unsafe"]
    rng.shuffle(s); rng.shuffle(u); n=min(k,len(s),len(u)); o=s[:n]+u[:n]; rng.shuffle(o); return o
def load_new():
    N={}
    wg=load_dataset("allenai/wildguardmix","wildguardtest",split="test",token=HF)
    N["wildguardtest"]=("guardrail",bal([{"text":r["prompt"],"label":"unsafe" if r["prompt_harm_label"]=="harmful" else "safe"} for r in wg if r["prompt_harm_label"] in ("harmful","unharmful") and (r["prompt"] or "").strip()],PC))
    wj=load_dataset("allenai/wildjailbreak","eval",split="train",token=HF)
    N["wildjailbreak"]=("red_team",bal([{"text":r["adversarial"],"label":"unsafe" if int(r["label"])==1 else "safe"} for r in wj if (r["adversarial"] or "").strip()],PC))
    hb=load_dataset("bench-llm/or-bench","or-bench-hard-1k",split="train",token=HF); tox=load_dataset("bench-llm/or-bench","or-bench-toxic",split="train",token=HF)
    N["orbench_hard"]=("over_refusal",bal([{"text":r["prompt"],"label":"safe"} for r in hb if (r["prompt"] or "").strip()]+[{"text":r["prompt"],"label":"unsafe"} for r in tox if (r["prompt"] or "").strip()],PC))
    hbm=load_dataset("walledai/HarmBench","standard",split="train",token=HF)
    N["harmbench"]=("red_team",[{"text":r["prompt"],"label":"unsafe"} for r in hbm if (r["prompt"] or "").strip()])
    return N

def auprc(s,g):
    s=np.asarray(s); o=np.argsort(-s); g=np.asarray(g)[o]; tp=np.cumsum(g); fp=np.cumsum(1-g); P=g.sum()
    if P==0: return float('nan')
    pr=tp/(tp+fp); rc=tp/P; rc=np.r_[0,rc]; pr=np.r_[1,pr]; return float(np.sum((rc[1:]-rc[:-1])*pr[1:]))
def ci(s,g,B=2000):
    r=np.random.default_rng(0); n=len(g); v=[auprc(np.asarray(s)[i],np.asarray(g)[i]) for i in (r.integers(0,n,n) for _ in range(B))]
    return float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))
def best_f1(s,g):
    s=np.asarray(s); g=np.asarray(g); best=0.0
    for t in np.unique(s):   # full precision (matches paper's Optimal-F1 grid)
        p=(s>=t).astype(int); tp=int(((p==1)&(g==1)).sum()); fp=int(((p==1)&(g==0)).sum()); fn=int(((p==0)&(g==1)).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f=2*pr*rc/(pr+rc) if pr+rc else 0
        best=max(best,f)
    return best

print("loading 4 novel benchmarks (reconstruct gold + order) ...", flush=True)
NEW=load_new(); gold={b:np.array([1 if r["label"]=="unsafe" else 0 for r in NEW[b][1]]) for b in NEW}
sizes={b:len(NEW[b][1]) for b in NEW}; total=sum(sizes.values())
print("rows:",sizes,"total",total)
gs=np.array(json.load(open(f"{CACHE}/_cache_guard_exp.json")))
bs_=np.array(json.load(open(f"{CACHE}/_cache_base_exp.json")))
ls=np.array(json.load(open(f"{CACHE}/_cache_llama_exp.json")))
assert len(gs)==len(bs_)==len(ls)==total, f"cache/gold row drift: caches={len(gs)} reconstructed={total}"
SC={"guard":gs,"base":bs_,"llama-guard":ls}; idx=0; per={n:{} for n in SC}
for b in NEW:
    n=sizes[b]
    for nm in SC: per[nm][b]=SC[nm][idx:idx+n]
    idx+=n
bal_b=[b for b in NEW if b!="harmbench"]
print("\n===== per-benchmark AUPRC =====")
for b in bal_b:
    print(f" {b} (n={sizes[b]}):", "  ".join(f"{nm}={auprc(per[nm][b],gold[b]):.3f}" for nm in SC))
print("\n===== AGGREGATE (3 balanced sets, excl HarmBench) =====")
res={}
for nm in SC:
    s=np.concatenate([per[nm][b] for b in bal_b]); g=np.concatenate([gold[b] for b in bal_b])
    lo,hi=ci(s,g); ap=auprc(s,g); bf=best_f1(s,g)
    res[nm]={"auprc":ap,"ci":[lo,hi],"best_f1":bf}
    print(f"  {nm:12s} AUPRC={ap:.3f} [{lo:.3f},{hi:.3f}]  best-F1={bf:.3f}  n={len(g)}")
print("\n HarmBench mean P(unsafe):", {nm:round(float(per[nm]['harmbench'].mean()),3) for nm in SC})
json.dump(res,open(f"{CACHE}/summary_novel_full.json","w"),indent=2,default=float)
print("saved -> summary_novel_full.json\nDONE_VERIFY_NOVEL")
