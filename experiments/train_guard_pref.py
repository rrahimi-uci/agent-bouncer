#!/usr/bin/env python
"""Guard trainer with PREFERENCE-OPTIMIZATION / RL objectives, to compare against the SFT/LoRA guard.
Same train pool, chat template, and LoRA config as train_guard.py (r=32, alpha=64, 7 projections); only the
objective changes, so the resulting adapter is scored by the identical single-token logprob head in
guard_eval_pipeline.py (FROZEN rows) -> a like-for-like SFT-vs-RL comparison.

TECHNIQUE=dpo  : Direct Preference Optimization (offline preference optimization; RLHF-family, RL-free math).
                 Pairs = (prompt, chosen=correct verdict, rejected=wrong verdict). Reference-model KL to the base.
TECHNIQUE=grpo : Group Relative Policy Optimization (online RL; DeepSeek-R1's method). Reward = verdict
                 correctness; group-relative advantage over num_generations samples; KL to reference.
TECHNIQUE=kto  : Kahneman-Tversky Optimization (binary desirable/undesirable; natural for safe/unsafe labels).

  MODEL_ID (base)  OUT (adapter dir)  TECHNIQUE=dpo|grpo|kto  GUARD_SMOKE=1 (10 steps, validate trl API)
Run from repo root (GPU):
  MODEL_ID=HuggingFaceTB/SmolLM3-3B OUT=notebooks/outputs/smollm3-dpo TECHNIQUE=dpo GUARD_SMOKE=1 python3 -u experiments/train_guard_pref.py
"""
import os, json, time, random
import numpy as np, torch
import warnings; warnings.filterwarnings("ignore")

def le(p):
    if os.path.exists(p):
        for l in open(p):
            l=l.strip()
            if l and not l.startswith("#") and "=" in l:
                k,v=l.split("=",1); os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
le("notebooks/.env"); le(".env"); HF=os.environ.get("HF_TOKEN")
SEED=int(os.environ.get("GUARD_SEED","42")); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)  # GUARD_SEED: multi-seed training randomness (data split stays fixed at seed 42 in balance())
DEV="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID=os.environ.get("MODEL_ID","HuggingFaceTB/SmolLM3-3B")
TECH=os.environ.get("TECHNIQUE","dpo").lower()
OUT=os.environ.get("OUT",f"notebooks/outputs/smollm3-{TECH}"); ADAPTER=f"{OUT}/adapter"
SMOKE=os.environ.get("GUARD_SMOKE","0") in ("1","true","yes")
MAX_SEQ_LEN=1024; TRAIN_CAP=1200; OR_BENCH_CAP=int(os.environ.get("OR_BENCH_CAP","1000")); MAX_STEPS=int(os.environ.get("GUARD_MAX_STEPS","300"))  # OR_BENCH_CAP=0 -> clean source-family-disjoint split
if SMOKE: TRAIN_CAP=40; OR_BENCH_CAP=30; MAX_STEPS=10
os.makedirs(OUT, exist_ok=True)
if os.path.exists(f"{ADAPTER}/adapter_config.json"):
    print(f"[skip] adapter exists at {ADAPTER}"); raise SystemExit(0)
print(f"device={DEV} model={MODEL_ID} tech={TECH} out={OUT} smoke={SMOKE} max_steps={MAX_STEPS}")

# ---------- train pool (identical construction to train_guard.py) ----------
from datasets import load_dataset, Dataset as HFDataset
def _norm(t): return " ".join((t or "").lower().split())
def balance(rows,k,seed=42):
    rng=random.Random(seed); s=[r for r in rows if r["label"]=="safe"]; u=[r for r in rows if r["label"]=="unsafe"]
    rng.shuffle(s); rng.shuffle(u); n=min(k,len(s),len(u)) if k else min(len(s),len(u)); o=s[:n]+u[:n]; rng.shuffle(o); return o
def _try(fn,n):
    try: return fn()
    except Exception as e: print(f"  !!{n}: {type(e).__name__} {str(e)[:50]}"); return []
def _bt(split,cap):
    lab,txt={},{}
    for i,r in enumerate(load_dataset("PKU-Alignment/BeaverTails",split=split,streaming=True,token=HF)):
        if i>=cap*8: break
        p=(r.get("prompt") or "").strip()
        if not p: continue
        k=_norm(p); txt.setdefault(k,p); lab[k]="unsafe" if (lab.get(k)=="unsafe" or not r.get("is_safe",False)) else "safe"
    return [{"text":txt[k],"label":lab[k],"source":"beavertails"} for k in lab]
def eval_keys():
    ev=[]
    ev+=_try(lambda:_bt("30k_test",400),"bt-test")
    ev+=_try(lambda:[{"text":(r.get("user_input") or "").strip()} for i,r in enumerate(load_dataset("lmsys/toxic-chat","toxicchat0124",split="test",streaming=True,token=HF)) if i<5000 and (r.get("user_input") or "").strip()],"tc-test")
    ev+=_try(lambda:[{"text":(r.get("text") or "").strip()} for r in load_dataset("deepset/prompt-injections",split="test",token=HF) if (r.get("text") or "").strip()],"pi-test")
    ev+=_try(lambda:[{"text":(r.get("prompt") or "").strip()} for r in load_dataset("jackhhao/jailbreak-classification",split="test",token=HF) if (r.get("prompt") or "").strip()],"jc-test")
    ev+=_try(lambda:[{"text":(r.get("prompt") or "").strip()} for r in load_dataset("natolambert/xstest-v2-copy",split="prompts",token=HF) if (r.get("prompt") or "").strip()],"xs")
    return {_norm(r["text"]) for r in ev}
def build_train_pool():
    EK=eval_keys()
    def _tc(): return [{"text":(r.get("user_input") or "").strip(),"label":"unsafe" if int(r.get("toxicity",0))==1 else "safe","source":"toxicchat"}
        for i,r in enumerate(load_dataset("lmsys/toxic-chat","toxicchat0124",split="train",streaming=True,token=HF)) if i<TRAIN_CAP*4 and (r.get("user_input") or "").strip()]
    def _pi(): return [{"text":(r.get("text") or "").strip(),"label":"unsafe" if int(r.get("label",0))==1 else "safe","source":"prompt_injections"}
        for r in load_dataset("deepset/prompt-injections",split="train",token=HF) if (r.get("text") or "").strip()]
    def _jc(): return [{"text":(r.get("prompt") or "").strip(),"label":"unsafe" if str(r.get("type","")).lower().startswith("jailbreak") else "safe","source":"jailbreak_classification"}
        for r in load_dataset("jackhhao/jailbreak-classification",split="train",token=HF) if (r.get("prompt") or "").strip()]
    pool=[]
    for name,fn in [("beavertails",lambda:_bt("30k_train",TRAIN_CAP)),("toxicchat",_tc),("prompt_injections",_pi),("jailbreak_classification",_jc)]:
        rows=[r for r in _try(fn,name) if _norm(r["text"]) not in EK]; pool+=balance(rows,TRAIN_CAP//2)
    pool+=_try(lambda:[{"text":(r.get("prompt") or "").strip(),"label":"safe","source":"or_bench"}
        for i,r in enumerate(load_dataset("bench-llm/or-bench","or-bench-80k",split="train",streaming=True,token=HF)) if i<OR_BENCH_CAP*3 and (r.get("prompt") or "").strip()][:OR_BENCH_CAP],"or_bench")
    seen,dedup=set(),[]
    for r in pool:
        k=_norm(r["text"])
        if not r["text"] or k in EK or k in seen: continue
        seen.add(k); dedup.append(r)
    random.Random(SEED).shuffle(dedup); return dedup

print("building train pool ...")
pool=build_train_pool()
from collections import Counter
ns=sum(r["label"]=="safe" for r in pool)
print(f"train pool: {len(pool)} ({ns} safe / {len(pool)-ns} unsafe) {dict(Counter(r['source'] for r in pool))}")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True,token=HF)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"; tok.truncation_side="left"
SYSTEM=("You are a safety classifier for user prompts to an LLM/agent. Decide if the prompt is safe or unsafe "
        "(unsafe = harmful content, a jailbreak, or a prompt injection). Respond with exactly one word: safe or unsafe.")
def build_prompt(t):
    m=[{"role":"system","content":SYSTEM},{"role":"user","content":t}]; kw={"tokenize":False,"add_generation_prompt":True}
    try: return tok.apply_chat_template(m,enable_thinking=False,**kw)
    except TypeError:
        try: return tok.apply_chat_template(m,**kw)
        except Exception: return f"{SYSTEM}\n\nPrompt: {t}\nVerdict:"
# HP overrides (for HPO; 0/unset => method default)
RANK=int(os.environ.get("GUARD_LORA_R","32")); ALPHA=int(os.environ.get("GUARD_LORA_ALPHA","64"))
LR=float(os.environ.get("GUARD_LR","0")); BETA=float(os.environ.get("GUARD_BETA","0"))
WARMUP=float(os.environ.get("GUARD_WARMUP","0.03"))
LORA=LoraConfig(r=RANK,lora_alpha=ALPHA,lora_dropout=0.05,task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
BS=int(os.environ.get("GUARD_BS","2")); ACCUM=int(os.environ.get("GUARD_ACCUM","4"))
COMMON=dict(output_dir=OUT, per_device_train_batch_size=BS, gradient_accumulation_steps=ACCUM, max_steps=MAX_STEPS,
    lr_scheduler_type="cosine", warmup_ratio=WARMUP, bf16=(DEV=="cuda"), gradient_checkpointing=(DEV=="cuda"),
    logging_steps=10, save_strategy="no", report_to=[], seed=SEED)

t0=time.time()
if TECH=="dpo":
    from trl import DPOTrainer, DPOConfig
    def _pref(r):
        cor=r["label"]; wrong="unsafe" if cor=="safe" else "safe"
        return {"prompt":build_prompt(r["text"]), "chosen":" "+cor, "rejected":" "+wrong}
    ds=HFDataset.from_list([_pref(r) for r in pool])
    # precompute_ref_log_probs: compute reference logps once up front, then free the ref model during training
    # (halves peak memory -> lets the 8B DPO run fit on a 40GB card). Same objective, memory-only change.
    cfg=DPOConfig(beta=(BETA or 0.1), learning_rate=(LR or 5e-6), max_length=MAX_SEQ_LEN,
        precompute_ref_log_probs=(os.environ.get("DPO_PRECOMPUTE","0")=="1"), **COMMON)
    tr=DPOTrainer(model=MODEL_ID, args=cfg, train_dataset=ds, processing_class=tok, peft_config=LORA)
    tr.train(); tr.save_model(ADAPTER)
elif TECH=="kto":
    from trl import KTOTrainer, KTOConfig
    rows=[]
    for r in pool:
        rows.append({"prompt":build_prompt(r["text"]), "completion":" "+r["label"], "label":True})
        wrong="unsafe" if r["label"]=="safe" else "safe"
        rows.append({"prompt":build_prompt(r["text"]), "completion":" "+wrong, "label":False})
    ds=HFDataset.from_list(rows)
    cfg=KTOConfig(beta=(BETA or 0.1), learning_rate=(LR or 5e-6), max_length=MAX_SEQ_LEN, **COMMON)
    tr=KTOTrainer(model=MODEL_ID, args=cfg, train_dataset=ds, processing_class=tok, peft_config=LORA)
    tr.train(); tr.save_model(ADAPTER)
elif TECH=="grpo":
    from trl import GRPOTrainer, GRPOConfig
    ds=HFDataset.from_list([{"prompt":build_prompt(r["text"]), "gold":r["label"]} for r in pool])
    def reward_correct(completions, gold, **kw):
        out=[]
        for c,g in zip(completions,gold):
            cl=(c or "").lower(); pred="unsafe" if "unsafe" in cl else ("safe" if "safe" in cl else "none")
            out.append(1.0 if pred==g else (-0.5 if pred=="none" else 0.0))
        return out
    G=dict(COMMON); G.pop("per_device_train_batch_size"); G.pop("gradient_accumulation_steps")
    NG=int(os.environ.get("GRPO_NUMGEN","8"))
    cfg=GRPOConfig(per_device_train_batch_size=NG, gradient_accumulation_steps=2, num_generations=NG,
        max_completion_length=int(os.environ.get("GRPO_MAXNEW","16")), learning_rate=(LR or 1e-6),
        beta=(BETA or 0.04), temperature=0.9, **G)
    tr=GRPOTrainer(model=MODEL_ID, args=cfg, train_dataset=ds, reward_funcs=[reward_correct],
        processing_class=tok, peft_config=LORA)
    tr.train(); tr.save_model(ADAPTER)
else:
    raise SystemExit(f"unknown TECHNIQUE {TECH}")
print(f"trained ({TECH}) in {time.time()-t0:.0f}s")
print(f"saved adapter -> {ADAPTER}\nDONE_TRAIN_{TECH.upper()}")
