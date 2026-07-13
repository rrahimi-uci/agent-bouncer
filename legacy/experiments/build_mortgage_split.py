#!/usr/bin/env python
"""Build a FAMILY-SAFE train/dev/test split of the mortgage guard benchmark (1,563 rows).
The benchmark contains technique+paraphrase variants of shared core requests (e.g. urgency/roleplay/
+paraphrase framings of the same intent). A naive random split would leak an intent across splits and
let the model memorize rather than generalize. We cluster near-duplicate prompts (union-find on
content-word Jaccard, blocked by category) and assign whole clusters to a split, so no core request
spans train/dev/test. Original line index is preserved so zero-shot baseline scores can be sliced.
"""
import json, re, random
BENCH="notebooks/data/benchmarks/full/guard_benchmark.jsonl"
OUT="notebooks/data/benchmarks/full/mortgage_split.json"
SEED=42; random.seed(SEED)
rows=[json.loads(l) for l in open(BENCH)]
for i,r in enumerate(rows): r["_idx"]=i   # position in guard_benchmark.jsonl (== eval_mortgage scoring order)

STOP=set("a an the is are was were be been being to of for and or but in on at by with as it this that i you we they he she my your our their me us them do does did should would could can will if not no yes just only even so here s t re m ve ll about from into out up down over under".split())
PREFIXES=["this is just for internal training purposes only.","no judgment, but","so here's the deal:","i need help with something.",
          "pretend you're my buddy who used to work in mortgages and doesn't care","for a novel i am writing,","hypothetically,",
          "as a compliance officer testing the system,","we close in 2 hours and the file is stuck. no time for the usual process —"]
def sig(t):
    t=t.lower().strip()
    for p in PREFIXES:
        if p in t: t=t.replace(p," ")
    t=re.sub(r"[^a-z0-9 ]"," ",t)
    return frozenset(w for w in t.split() if w not in STOP and len(w)>2)

# ---- union-find near-duplicate clustering (blocked by category) ----
parent=list(range(len(rows)))
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
sigs=[sig(r["text"]) for r in rows]
from collections import defaultdict
blocks=defaultdict(list)
for i,r in enumerate(rows): blocks[r.get("label_category","?")].append(i)
JACC=0.6
for cat,idxs in blocks.items():
    for a_pos in range(len(idxs)):
        i=idxs[a_pos]; si=sigs[i]
        if not si: continue
        for b_pos in range(a_pos+1,len(idxs)):
            j=idxs[b_pos]; sj=sigs[j]
            if not sj: continue
            inter=len(si & sj)
            if inter==0: continue
            if inter/len(si | sj) >= JACC: union(i,j)
clusters=defaultdict(list)
for i in range(len(rows)): clusters[find(i)].append(i)
cl=list(clusters.values())
sizes=sorted((len(c) for c in cl),reverse=True)
print(f"rows={len(rows)} clusters={len(cl)} (max cluster={sizes[0]}, #clusters>1={sum(1 for s in sizes if s>1)})")

# ---- assign whole clusters to train/dev/test (0.70/0.15/0.15), seeded shuffle ----
random.Random(SEED).shuffle(cl)
n=len(rows); tr_cut=int(0.70*n); dv_cut=int(0.85*n)
train,dev,test=[],[],[]; run=0
for c in cl:
    dest = train if run<tr_cut else (dev if run<dv_cut else test)
    for i in c: dest.append(rows[i])
    run+=len(c)
def comp(split):
    from collections import Counter
    lab=Counter(r["label_binary"] for r in split); cat=Counter(r.get("label_category") for r in split)
    return dict(n=len(split),flag=lab.get("flag",0),allow=lab.get("allow",0),cats=dict(cat))
print("train",comp(train)); print("dev  ",comp(dev)); print("test ",comp(test))
# ---- leakage assertion: no core signature shared across splits ----
def sset(split): return set(frozenset(s) for s in (sig(r["text"]) for r in split) if s)
inter_tt=sset(train)&sset(test); inter_td=sset(train)&sset(dev)
print(f"shared core-signatures train∩test={len(inter_tt)}  train∩dev={len(inter_td)} (0 = family-safe)")

json.dump({"seed":SEED,"jaccard":JACC,"n_clusters":len(cl),
           "train":train,"dev":dev,"test":test},open(OUT,"w"))
print(f"saved -> {OUT}")
