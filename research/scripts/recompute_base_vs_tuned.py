#!/usr/bin/env python
"""GROUP B re-ground fix: base-vs-tuned in-house comparison at CLEAN in-dist calibration for BOTH models.
Tuned clean preds come from preds_corrected.json (guard_pred_clean). Base continuous scores come from
base_smollm3_inhouse.json; we calibrate a BASE-SPECIFIC temperature+threshold on the base's OWN in-dist dev
(identical fit_T/choose_thr protocol as eval_corrected), then evaluate on the identical test rows.
This makes the base-vs-tuned delta a fair, same-decision-rule-family comparison and resolves the
'no base-specific calibration' caveat. Emits per-benchmark F1, deltas, aggregates, and McNemar.
"""
import json, numpy as np

def prf(g, p):
    g=np.asarray(g); p=np.asarray(p)
    tp=int(((p==1)&(g==1)).sum()); fp=int(((p==1)&(g==0)).sum())
    fn=int(((p==0)&(g==1)).sum()); tn=int(((p==0)&(g==0)).sum())
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    return {"precision":pr,"recall":rc,"f1":2*pr*rc/(pr+rc) if pr+rc else 0.0,"fpr":fp/(fp+tn) if fp+tn else 0.0,"n":len(g)}
def apply_temp(sc,T):
    sc=np.clip(np.asarray(sc,float),1e-6,1-1e-6); return 1/(1+np.exp(-(np.log(sc/(1-sc))/T)))
def fit_T(s,g):
    s=np.clip(np.asarray(s,float),1e-6,1-1e-6); z=np.log(s/(1-s)); y=np.asarray(g,float)
    return float(min(np.linspace(0.5,8.0,76),
        key=lambda T:-np.mean(y*np.log(np.clip(1/(1+np.exp(-z/T)),1e-9,1-1e-9))+(1-y)*np.log(np.clip(1-1/(1+np.exp(-z/T)),1e-9,1-1e-9)))))
def choose_thr(s,g):
    bt,bf=0.5,-1.0
    for t in np.unique(np.round(np.asarray(s,float),3)):
        m=prf(g,(np.asarray(s)>=t).astype(int))
        if m["f1"]>bf: bf,bt=m["f1"],float(t)
    return bt

ND="notebooks/outputs/nb-smollm3-guard"
tu=json.load(open(f"{ND}/preds_corrected.json"))
ba=json.load(open(f"{ND}/base_smollm3_inhouse.json"))
gold=np.array(tu["gold"]); strata=np.array(tu["strata"]); tuned_pred=np.array(tu["guard_pred_clean"])
assert tu["gold"]==ba["gold"], "gold row drift base vs tuned"
assert tu["strata"]==ba["strata"], "strata row drift base vs tuned"
HELD={"jailbreakbench","xstest"}

# ---- base: calibrate its OWN clean (in-dist dev) temperature+threshold, same protocol ----
bdev_s=np.array(ba["dev_scores"]); bdev_g=np.array(ba["dev_gold"]); btest_s=np.array(ba["test_scores"])
T_b=fit_T(bdev_s,bdev_g); THR_b=choose_thr(apply_temp(bdev_s,T_b),bdev_g)
base_pred=(apply_temp(btest_s,T_b)>=THR_b).astype(int)
print(f"BASE clean calibration (its own in-dist dev): T={T_b:.2f} THR={THR_b:.2f}  (n_dev={len(bdev_g)})")
print(f"TUNED clean calibration (from eval_corrected): T=2.10 THR=0.59\n")

print(f"{'benchmark':26s} {'axis':11s} {'base':>6s} {'tuned':>6s} {'delta':>7s}  n")
AXIS={"beavertails":"guardrail","toxicchat":"guardrail","prompt_injections":"red_team",
      "jailbreak_classification":"red_team","jailbreakbench":"red_team","xstest":"over_refusal"}
rows={}
for b in sorted(set(strata)):
    m=strata==b
    bf=prf(gold[m],base_pred[m])["f1"]; tf=prf(gold[m],tuned_pred[m])["f1"]
    ho="*" if b in HELD else " "
    print(f"{ho}{b:25s} {AXIS[b]:11s} {bf:6.3f} {tf:6.3f} {tf-bf:+7.3f}  {int(m.sum())}")
    rows[b]={"base":bf,"tuned":tf,"delta":tf-bf,"n":int(m.sum())}

def macro(bs,pred):
    return float(np.mean([prf(gold[strata==b],pred[strata==b])["f1"] for b in bs]))
ind=[b for b in set(strata) if b not in HELD]; hld=[b for b in set(strata) if b in HELD]
bp=prf(gold,base_pred); tp=prf(gold,tuned_pred)
print(f"\nPOOLED micro-F1:  base={bp['f1']:.4f}  tuned={tp['f1']:.4f}  delta={tp['f1']-bp['f1']:+.4f}")
print(f"macro-F1 IN-DIST: base={macro(ind,base_pred):.3f} tuned={macro(ind,tuned_pred):.3f} delta={macro(ind,tuned_pred)-macro(ind,base_pred):+.3f}")
print(f"macro-F1 HELD-OUT:base={macro(hld,base_pred):.3f} tuned={macro(hld,tuned_pred):.3f} delta={macro(hld,tuned_pred)-macro(hld,base_pred):+.3f}")
print(f"base per-held-out: "+", ".join(f"{b}={rows[b]['delta']:+.3f}" for b in ['jailbreakbench','xstest']))

# ---- paired-bootstrap 95% CI on the pooled F1 delta (tuned - base) ----
rng=np.random.default_rng(0); N=len(gold); deltas=[]
for _ in range(5000):
    idx=rng.integers(0,N,N)
    deltas.append(prf(gold[idx],tuned_pred[idx])["f1"]-prf(gold[idx],base_pred[idx])["f1"])
lo,hi=np.percentile(deltas,2.5),np.percentile(deltas,97.5)
print(f"paired-bootstrap 95% CI on pooled F1 delta: [{lo:+.3f}, {hi:+.3f}]  (point {tp['f1']-bp['f1']:+.4f})")

# ---- McNemar tuned vs base on matched rows ----
b01=int(((tuned_pred==gold)&(base_pred!=gold)).sum())  # tuned right, base wrong
b10=int(((tuned_pred!=gold)&(base_pred==gold)).sum())  # base right, tuned wrong
from math import comb
n=b01+b10; p_two=min(1.0,2*sum(comb(n,k) for k in range(min(b01,b10)+1))/(2**n)) if n>0 else 1.0
print(f"McNemar tuned-vs-base: tuned-only-correct={b01} base-only-correct={b10} exact two-sided p={p_two:.6f}")

# ---- machine-readable dump for grounding paper edits ----
json.dump({"base_calib":{"T":T_b,"THR":THR_b},"per_benchmark":rows,
           "pooled":{"base":bp["f1"],"tuned":tp["f1"],"delta":tp["f1"]-bp["f1"],"ci":[float(lo),float(hi)]},
           "macro":{"in_dist":{"base":macro(ind,base_pred),"tuned":macro(ind,tuned_pred)},
                    "held_out":{"base":macro(hld,base_pred),"tuned":macro(hld,tuned_pred)}},
           "mcnemar":{"tuned_only":b01,"base_only":b10,"p":p_two}},
          open(f"{ND}/base_vs_tuned_clean.json","w"),indent=2,default=float)
print("DONE_RECOMPUTE")
