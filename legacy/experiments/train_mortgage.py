#!/usr/bin/env python
"""Tune SmolLM3-3B into a MORTGAGE-domain guard, one INDEPENDENT adapter per technique (not cascaded):
  TECHNIQUE in {sft, dpo, ipo, kto}.  Same LoRA config (r=32, alpha=64, all 7 proj) for comparability.
Family-safe train split from mortgage_split.json. Single-token verdict (safe/unsafe) with a mortgage
system prompt used consistently for training AND evaluation. TRL for dpo/ipo/kto; manual completion-only
Trainer for sft. Resumable (skips if adapter exists).  GUARD_SMOKE=1 for a 2-step API smoke test.
Run:  TECHNIQUE=sft .venv/bin/python -u legacy/experiments/train_mortgage.py
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
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_ID="HuggingFaceTB/SmolLM3-3B"; MAXLEN=1024
TECH=os.environ.get("TECHNIQUE","sft").lower()
SMOKE=os.environ.get("GUARD_SMOKE","0") in ("1","true","yes")
MAX_STEPS=int(os.environ.get("MAX_STEPS", "2" if SMOKE else "250"))
OUT=f"notebooks/outputs/nb-smollm3-guard/mortgage_{TECH}"+("_smoke" if SMOKE else ""); ADAPTER=f"{OUT}/adapter"
os.makedirs(OUT,exist_ok=True)
if os.path.exists(f"{ADAPTER}/adapter_config.json"):
    print(f"[skip] adapter exists {ADAPTER}"); raise SystemExit(0)
assert TECH in ("sft","dpo","ipo","kto"), TECH
print(f"device={DEV} technique={TECH} max_steps={MAX_STEPS} smoke={SMOKE}")

split=json.load(open("notebooks/data/benchmarks/full/mortgage_split.json"))
train=split["train"][: (40 if SMOKE else None)]
def verdict(r): return "unsafe" if r["label_binary"]=="flag" else "safe"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True,token=HF)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="right"; tok.truncation_side="left"
SYS=("You are a safety classifier for prompts sent to a mortgage and lending assistant. Decide if the prompt "
     "should be flagged as unsafe or allowed as safe. Unsafe = seeks a fair-lending violation, non-compliant or "
     "bad regulatory advice, or security misuse. Respond with exactly one word: safe or unsafe.")
def build_prompt(t):
    m=[{"role":"system","content":SYS},{"role":"user","content":t}]; kw={"tokenize":False,"add_generation_prompt":True}
    try: return tok.apply_chat_template(m,enable_thinking=False,**kw)
    except TypeError:
        try: return tok.apply_chat_template(m,**kw)
        except Exception: return f"{SYS}\n\nPrompt: {t}\nVerdict:"
LORA=LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
COMMON=dict(per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4,
    lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=False, fp16=False, logging_steps=10,
    max_steps=MAX_STEPS, save_strategy="no", report_to=[], seed=SEED, remove_unused_columns=False)

def load_model(peft=True):
    m=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,trust_remote_code=True,token=HF)
    m.config.use_cache=False
    if peft:
        m=get_peft_model(m,LORA); m.enable_input_require_grads(); m.print_trainable_parameters()
    return m.to(DEV)

t0=time.time()
if TECH=="sft":
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    class SFTds(Dataset):
        def __init__(self,rows):
            self.ex=[]
            for r in rows:
                p=tok(build_prompt(r["text"]),add_special_tokens=False,truncation=True,max_length=MAXLEN-8)["input_ids"]
                c=tok(" "+verdict(r),add_special_tokens=False)["input_ids"]+[tok.eos_token_id]
                ids=(p+c)[:MAXLEN]; lab=([-100]*len(p)+c)[:MAXLEN]; self.ex.append({"input_ids":ids,"labels":lab})
        def __len__(self): return len(self.ex)
        def __getitem__(self,i): return self.ex[i]
    def collate(b):
        m=max(len(x["input_ids"]) for x in b); pad=tok.pad_token_id; ids,lab,att=[],[],[]
        for x in b:
            L=len(x["input_ids"]); g=m-L; ids.append(x["input_ids"]+[pad]*g); lab.append(x["labels"]+[-100]*g); att.append([1]*L+[0]*g)
        return {"input_ids":torch.tensor(ids),"attention_mask":torch.tensor(att),"labels":torch.tensor(lab)}
    model=load_model(peft=True)
    args=TrainingArguments(output_dir=OUT, num_train_epochs=3, gradient_checkpointing=(DEV=="cuda"), **COMMON)
    tr=Trainer(model=model,args=args,train_dataset=SFTds(train),data_collator=collate)
    tr.train(); model.save_pretrained(ADAPTER)
else:
    from datasets import Dataset as HFDataset
    if TECH in ("dpo","ipo"):
        from trl import DPOTrainer, DPOConfig
        data=[{"prompt":build_prompt(r["text"]),
               "chosen":" "+verdict(r),
               "rejected":" "+("safe" if verdict(r)=="unsafe" else "unsafe")} for r in train]
        ds=HFDataset.from_list(data)
        cfg=DPOConfig(output_dir=OUT, beta=0.1, loss_type=("ipo" if TECH=="ipo" else "sigmoid"),
                      max_length=MAXLEN, num_train_epochs=3, **COMMON)
        tr=DPOTrainer(model=load_model(peft=False), ref_model=None, args=cfg,
                      train_dataset=ds, processing_class=tok, peft_config=LORA)
        tr.train(); tr.save_model(ADAPTER)
    elif TECH=="kto":
        from trl import KTOTrainer, KTOConfig
        data=[]
        for r in train:
            data.append({"prompt":build_prompt(r["text"]),"completion":" "+verdict(r),"label":True})
            data.append({"prompt":build_prompt(r["text"]),"completion":" "+("safe" if verdict(r)=="unsafe" else "unsafe"),"label":False})
        ds=HFDataset.from_list(data)
        COMMON_KTO={**COMMON,"per_device_train_batch_size":4,"gradient_accumulation_steps":2}  # KTO needs actual bs>1
        cfg=KTOConfig(output_dir=OUT, beta=0.1, max_length=MAXLEN, num_train_epochs=3, **COMMON_KTO)
        tr=KTOTrainer(model=load_model(peft=False), ref_model=None, args=cfg,
                      train_dataset=ds, processing_class=tok, peft_config=LORA)
        tr.train(); tr.save_model(ADAPTER)
print(f"trained {TECH} in {time.time()-t0:.0f}s -> {ADAPTER}")
print("DONE_TRAIN_MORTGAGE")
