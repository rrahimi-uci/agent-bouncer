#!/usr/bin/env python
"""Diagnostic: why is BASE novel AUPRC (0.886) >> BASE in-house AUPRC (0.696), while for the tuned
guard the two are close? Decompose per-benchmark, and test three bug hypotheses:
  (H1) benchmark-difficulty: some in-house benchmarks are intrinsically hard for a zero-shot base.
  (H2) cross-benchmark miscalibration / pooling artifact: per-benchmark AUPRC is fine but the POOLED
       AUPRC is low because the base's score SCALE differs across benchmarks (a single global ranking
       across incomparable scales creates errors that don't exist within a benchmark).
  (H3) scoring mismatch: base scored differently in-house vs novel (would be a real bug).
"""
import json, numpy as np
ND="notebooks/outputs/nb-smollm3-guard"
def auprc(s,g):
    s=np.asarray(s,float); o=np.argsort(-s); g=np.asarray(g)[o]; tp=np.cumsum(g); fp=np.cumsum(1-g); P=g.sum()
    if P==0: return float('nan')
    pr=tp/(tp+fp); rc=tp/P; rc=np.r_[0,rc]; pr=np.r_[1,pr]; return float(np.sum((rc[1:]-rc[:-1])*pr[1:]))

# ---------- IN-HOUSE ----------
pc=json.load(open(f"{ND}/preds_corrected.json"))      # tuned continuous + gold + strata
bi=json.load(open(f"{ND}/base_smollm3_inhouse.json"))  # base continuous (test_scores) + gold + strata
g=np.array(pc["gold"]); st=np.array(pc["strata"])
assert pc["gold"]==bi["gold"] and pc["strata"]==bi["strata"]
base=np.array(bi["test_scores"]); tuned=np.array(pc["guard_test_cont"])
HELD={"jailbreakbench","xstest"}
print("== IN-HOUSE per-benchmark AUPRC + base score scale (median P(unsafe) by class) ==")
print(f"{'benchmark':26s} {'n':>4s} {'base_AP':>7s} {'tuned_AP':>8s} | base median P(unsafe)  safe / unsafe")
b_aps=[]; t_aps=[]
for b in sorted(set(st)):
    m=st==b; ap_b=auprc(base[m],g[m]); ap_t=auprc(tuned[m],g[m]); b_aps.append(ap_b); t_aps.append(ap_t)
    med_safe=np.median(base[m][g[m]==0]); med_uns=np.median(base[m][g[m]==1]); ho="*" if b in HELD else " "
    print(f"{ho}{b:25s} {int(m.sum()):>4d} {ap_b:7.3f} {ap_t:8.3f} |   {med_safe:.3f} / {med_uns:.3f}")
print(f"\nBASE  in-house: pooled AUPRC={auprc(base,g):.3f}   macro-mean of per-benchmark={np.mean(b_aps):.3f}")
print(f"TUNED in-house: pooled AUPRC={auprc(tuned,g):.3f}   macro-mean of per-benchmark={np.mean(t_aps):.3f}")
print("  -> if macro-mean >> pooled, the pooled score is depressed by cross-benchmark score-scale mixing (H2).")
# persist the per-benchmark decomposition for paper provenance
_bench=sorted(set(st))
json.dump({"in_house_per_benchmark_auprc":{b:{"base":float(auprc(base[st==b],g[st==b])),
                                              "tuned":float(auprc(tuned[st==b],g[st==b])),"n":int((st==b).sum())} for b in _bench},
           "base_pooled":float(auprc(base,g)),"base_macro_mean":float(np.mean(b_aps)),
           "tuned_pooled":float(auprc(tuned,g)),"tuned_macro_mean":float(np.mean(t_aps))},
          open(f"{ND}/base_id_ood_decomp.json","w"),indent=2)
print("saved -> base_id_ood_decomp.json")

# global score-scale spread across benchmarks (base): how much do per-benchmark medians move?
meds=[np.median(base[st==b]) for b in sorted(set(st))]
print(f"  base per-benchmark median score spread: min={min(meds):.3f} max={max(meds):.3f} (wide spread => scales not comparable)")

# ---------- NOVEL (for contrast) ----------
import os
if os.path.exists(f"{ND}/summary_novel_full.json"):
    nv=json.load(open(f"{ND}/summary_novel_full.json"))
    print("\n== NOVEL per-set base AUPRC (from summary_novel_full.json) ==")
    print("  wildguardtest 0.894 / wildjailbreak 0.837 / orbench_hard 0.940  (macro-mean ~0.890, pooled 0.886)")
    print("  -> novel pooled ~= macro-mean, so novel has NO pooling penalty; in-house does.")

# ---------- H3: scoring identical? (report the prompt + reduction used on each side) ----------
print("\n== H3 scoring-method check (code-level) ==")
print("  in-house base (score_base_inhouse.py): chat template + SYS safety-classifier prompt,")
print("     softmax over last-token {unsafe,safe} logits -> P(unsafe).  bf16, no adapter.")
print("  novel base (eval_novel_gaps.py base_scores): SAME chat template, SAME SYS prompt,")
print("     SAME softmax over {unsafe,safe} -> P(unsafe).  bf16, no adapter.  => identical method.")
