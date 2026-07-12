#!/usr/bin/env python
"""Emit notebooks/paper_reproduction.ipynb --- the canonical single-notebook paper entrypoint.

It does NOT reimplement the paper logic (that would drift); it ORCHESTRATES the committed paper scripts
and renders every table/figure from the JSON they emit. Two modes via env REPRO_MODE:
  - verify_cached (default): load existing summary_*.json and render tables/figures/validation (fast).
  - recompute: run each paper script to (re)produce the JSON first, then render (heavy: GPU/API/time).
Run from repo root:  python experiments/build_paper_reproduction_notebook.py
"""
import json, os
def md(*s): return {"cell_type":"markdown","metadata":{},"source":[l if l.endswith("\n") else l+"\n" for l in s]}
def code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                       "source":[l if l.endswith("\n") else l+"\n" for l in src.split("\n")]}
C=[]

C.append(md(
"# Paper reproduction — *The Benchmark Chooses the Winner*",
"",
"**Canonical single-notebook entrypoint.** Run All to regenerate every paper result family. This notebook",
"orchestrates the committed paper scripts (so its logic cannot drift from the paper) and renders all tables",
"and figures from the JSON those scripts emit.",
"",
"Two modes (set the `REPRO_MODE` env var before launching the kernel):",
"- `verify_cached` (default): load existing `outputs/nb-smollm3-guard/summary_*.json` and render fast.",
"- `recompute`: run each paper script to (re)produce the JSON first, then render. **Heavy** — needs a",
"  GPU for the local guards, an `OPENAI_API_KEY` for the GPT baseline, network for the open guards, and",
"  time (hours on a laptop; some open-guard/novel runs are GPU-pending per the paper).",
"",
"> Honesty note: this replaces the older `smollm3_guard_reproduction.ipynb` companion *demo* as the paper",
"> path. Full byte-exact `recompute` also depends on upstream Hugging Face datasets that are not revision",
"> pinned (see the repo README caveats).",
))

C.append(md("## 0 · Preflight & mode"))
C.append(code(
"import os, json, subprocess, sys\n"
"import pandas as pd\n"
"try:\n"
"    import torch; _cuda = torch.cuda.is_available(); _mps = getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()\n"
"except Exception:\n"
"    _cuda=_mps=False\n"
"NB_DIR = os.getcwd()\n"
"RESEARCH = NB_DIR if os.path.isdir(os.path.join(NB_DIR,'experiments')) else os.path.dirname(NB_DIR)\n"
"OUT = os.path.join(RESEARCH,'notebooks','outputs','nb-smollm3-guard')\n"
"MODE = os.environ.get('REPRO_MODE','verify_cached').lower()\n"
"assert MODE in ('verify_cached','recompute'), MODE\n"
"def _load_env(_p):\n"
"    if os.path.exists(_p):\n"
"        for _l in open(_p):\n"
"            _l=_l.strip()\n"
"            if _l and not _l.startswith('#') and '=' in _l:\n"
"                _k,_v=_l.split('=',1); os.environ.setdefault(_k.strip(), _v.strip().strip(chr(34)+chr(39)))\n"
"for _ef in (os.path.join(RESEARCH,'notebooks','.env'), os.path.join(RESEARCH,'.env')):\n"
"    _load_env(_ef)   # match the scripts: notebook preflight must see .env keys too, else it wrongly skips GPT\n"
"HAVE_OPENAI = bool(os.environ.get('OPENAI_API_KEY'))\n"
"print(f'MODE={MODE} | RESEARCH={RESEARCH}')\n"
"print(f'device: cuda={_cuda} mps={_mps} | OPENAI_API_KEY={\"set\" if HAVE_OPENAI else \"MISSING\"}')\n"
"if MODE=='recompute' and not _cuda:\n"
"    print('WARNING: recompute without CUDA is very slow on this box; local-guard scoring may take hours.')\n"
"if MODE=='recompute' and not HAVE_OPENAI:\n"
"    print('WARNING: no OPENAI_API_KEY — the GPT baseline / re-ground cells will be skipped.')\n"
"\n"
"def run_script(name, env=None):\n"
"    if MODE!='recompute':\n"
"        print(f'[verify] skipping experiments/{name} (using cached JSON)'); return\n"
"    print(f'[recompute] python experiments/{name}  (cwd={RESEARCH})', flush=True)\n"
"    p = subprocess.run([sys.executable,'-u',os.path.join('experiments',name)], cwd=RESEARCH,\n"
"                       env={**os.environ, **(env or {})})\n"
"    if p.returncode!=0: raise RuntimeError(f'experiments/{name} failed (exit {p.returncode})')\n"
"\n"
"def load_json(name, produced_by):\n"
"    fp=os.path.join(OUT,name)\n"
"    if not os.path.exists(fp):\n"
"        raise FileNotFoundError(f'{name} not found. Re-run with REPRO_MODE=recompute, or run experiments/{produced_by} first.')\n"
"    return json.load(open(fp))\n"
"HEADLINE={}  # collected for the final validation checklist"
))

C.append(md("## 1 · In-house corrected evaluation  (`eval_corrected.py`)",
            "Matched-FPR@0.10, threshold-free AUPRC/AUROC, clean in-dist-only calibration."))
C.append(code(
"run_script('eval_corrected.py')\n"
"s = load_json('summary_corrected.json','eval_corrected.py')\n"
"print('calibration:', s.get('calibration',{}).get('clean'))\n"
"print('guard batch=1 latency ms:', s.get('guard_batch1_latency_ms'))\n"
"mf = s.get('matched_fpr_0.10',{})\n"
"display(pd.DataFrame(mf).T.rename_axis('system')) if mf else print(s)"
))

C.append(md("## 2 · Novel / OOD evaluation  (`eval_novel_gaps.py` → `verify_novel.py`)",
            "Aggregate AUPRC over the 3 balanced novel sets: base > tuned guard > Llama-Guard."))
C.append(code(
"run_script('eval_novel_gaps.py'); run_script('verify_novel.py')\n"
"s = load_json('summary_novel_full.json','verify_novel.py')\n"
"rows=[{'system':k,'AUPRC':v.get('auprc'),'CI':v.get('ci'),'best_F1':v.get('best_f1')} for k,v in s.items()]\n"
"HEADLINE['novel_base_auprc']=s.get('base',{}).get('auprc'); HEADLINE['novel_guard_auprc']=s.get('guard',{}).get('auprc'); HEADLINE['novel_llama_auprc']=s.get('llama-guard',{}).get('auprc')\n"
"display(pd.DataFrame(rows).set_index('system'))"
))

C.append(md("## 3 · Base-vs-tuned decomposition  (`score_base_inhouse.py` → `recompute_base_vs_tuned.py`)"))
C.append(code(
"run_script('score_base_inhouse.py'); run_script('recompute_base_vs_tuned.py')\n"
"s = load_json('base_vs_tuned_clean.json','recompute_base_vs_tuned.py')\n"
"HEADLINE['base_vs_tuned_delta']=s.get('pooled',{}).get('delta')\n"
"print('pooled:', s.get('pooled')); print('base in-house AUPRC:', s.get('base_inhouse_auprc',{}).get('pooled'))\n"
"pb=s.get('per_benchmark',{})\n"
"display(pd.DataFrame(pb).T[['base','tuned','delta']]) if pb else None"
))

C.append(md("## 4 · Mortgage hardened case study  (`eval_mortgage_hard.py`)",
            "Loads the committed 334-row `guard_benchmark_hard.jsonl` directly (the superseded synthetic builders are NOT used)."))
C.append(code(
"run_script('eval_mortgage_hard.py')\n"
"s = load_json('summary_mortgage_hard.json','eval_mortgage_hard.py')\n"
"sy=s.get('systems',{})\n"
"cols=['auprc','opt_f1','acc','recall','precision','fpr']\n"
"df=pd.DataFrame({k:{c:v.get(c) for c in cols} for k,v in sy.items()}).T\n"
"HEADLINE['mortgage_sft_auprc']=sy.get('mortgage-sft',{}).get('auprc')\n"
"print(f\"n={s.get('n')} flag={s.get('flag')} allow={s.get('allow')}\"); display(df)"
))

C.append(md("## 5 · GPT baseline re-ground (abstain policy)  (`reground_gpt_inhouse.py`)",
            "Recomputes the in-house guard-vs-GPT F1 with the paper's *abstain* error policy (not fail-closed)."))
C.append(code(
"if MODE=='recompute' and not HAVE_OPENAI:\n"
"    print('skipped: no OPENAI_API_KEY')\n"
"else:\n"
"    run_script('reground_gpt_inhouse.py')\n"
"    s = load_json('summary_gpt_reground.json','reground_gpt_inhouse.py')\n"
"    HEADLINE['gpt_delta_f1']=s.get('delta_f1_guard_minus_gpt')\n"
"    print(f\"abstained={s.get('n_abstain')}/{s.get('n_rows')}  API errors={s.get('n_api_errors')}  preds changed vs old fail-closed={s.get('gpt_new_vs_old_changed_preds')}\")\n"
"    print(f\"gpt FPR old(fail-closed)={s['gpt_old_failclosed']['fpr']:.3f} -> new(abstain)={s['gpt_abstain']['fpr']:.3f}\")\n"
"    print(f\"Delta F1 (guard-gpt)={s['delta_f1_guard_minus_gpt']:+.3f} CI={s['delta_f1_ci']} McNemar p={s['mcnemar_p']:.3f}\")"
))

C.append(md("## 6 · Figures (drawn from the JSON above, not inline constants)"))
C.append(code(
"import matplotlib.pyplot as plt\n"
"# novel aggregate AUPRC\n"
"try:\n"
"    nv=load_json('summary_novel_full.json','verify_novel.py')\n"
"    order=['base','guard','llama-guard']; vals=[nv[k]['auprc'] for k in order]\n"
"    fig,ax=plt.subplots(figsize=(4,2.6)); ax.bar(order,vals,color=['#0072B2','#E69F00','#009E73'])\n"
"    [ax.text(i,v+.01,f'{v:.3f}',ha='center',fontsize=8) for i,v in enumerate(vals)]\n"
"    ax.set_ylim(0,1); ax.set_title('Novel aggregate AUPRC (from JSON)'); plt.tight_layout(); plt.show()\n"
"except Exception as e: print('novel fig skipped:',e)\n"
"# mortgage hardened AUPRC\n"
"try:\n"
"    mh=load_json('summary_mortgage_hard.json','eval_mortgage_hard.py')['systems']\n"
"    order=[k for k in ('base','mortgage-sft','general-guard') if k in mh]; vals=[mh[k]['auprc'] for k in order]\n"
"    fig,ax=plt.subplots(figsize=(4,2.6)); ax.bar(order,vals,color='#0072B2')\n"
"    [ax.text(i,v+.01,f'{v:.3f}',ha='center',fontsize=8) for i,v in enumerate(vals)]\n"
"    ax.set_ylim(0,1); ax.set_title('Hardened mortgage AUPRC (from JSON)'); plt.tight_layout(); plt.show()\n"
"except Exception as e: print('mortgage fig skipped:',e)"
))

C.append(md("## 7 · Validation checklist",
            "Compares the regenerated numbers against the paper's reported headline values."))
C.append(code(
"EXPECTED={'novel_base_auprc':0.886,'novel_guard_auprc':0.781,'novel_llama_auprc':0.701,\n"
"          'base_vs_tuned_delta':0.081,'mortgage_sft_auprc':0.924,'gpt_delta_f1':0.010}\n"
"rows=[]\n"
"for k,exp in EXPECTED.items():\n"
"    got=HEADLINE.get(k)\n"
"    ok = (got is not None) and abs(float(got)-exp)<=0.03\n"
"    rows.append({'metric':k,'expected':exp,'got':(round(float(got),3) if got is not None else None),'PASS(±0.03)':ok})\n"
"chk=pd.DataFrame(rows).set_index('metric'); display(chk)\n"
"n_ok=int(chk['PASS(±0.03)'].sum()); print(f'{n_ok}/{len(chk)} headline numbers match (tol 0.03).')\n"
"if MODE=='verify_cached': print('(verify_cached: numbers reflect the committed/produced JSON, not a fresh recompute.)')"
))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
os.makedirs("notebooks",exist_ok=True)
with open("notebooks/paper_reproduction.ipynb","w") as f: json.dump(nb,f,indent=1)
print("wrote notebooks/paper_reproduction.ipynb with",len(C),"cells")
