#!/usr/bin/env python
"""Agentic build on REAL HMDA data with a real LLM (OpenAI) driving generator + judge.

Bounded scope (few families, judge_samples=1) so the demo is cheap/fast; scale via config for
a full build. Writes out_real/ and prints a summary + a sample row per quadrant.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magen.store import load_config
from magen.pipeline import run_all

cfg = load_config()
cfg["llm"]["provider"] = "openai"
cfg["llm"]["generator_model"] = "gpt-5.4-mini"
cfg["llm"]["judge_model"] = "gpt-5.4-mini"
cfg["llm"]["temperature"] = 1.0          # gpt-5.x accepts only default temperature
cfg["llm"]["max_tokens"] = 900
cfg["llm"]["max_llm_calls"] = 400
cfg["hmda"]["source"] = "csv"
cfg["hmda"]["snapshot_path"] = "data/hmda_real/hmda_2022.csv"
cfg["hmda"]["sample_rows"] = 800
# match the states we actually downloaded (default config's CA/TX/... would exclude them all)
cfg["hmda"]["filters"]["states"] = ["DC", "DE", "RI", "VT", "WY", "AK",
                                    "ND", "SD", "MT", "NH", "HI", "ME"]
# a more even mix so all four quadrants appear at small family counts
cfg["generate"]["quadrant_mix"] = {"G0D0": 0.40, "G1D0": 0.20, "G0D1": 0.25, "G1D1": 0.15}
cfg["generate"]["variants_per_family"] = 2
cfg["generate"]["judge_samples"] = 1
cfg["generate"]["max_generation_retries"] = 1
cfg["generate"]["protected_pair_fraction"] = 0.34
cfg["strata"]["low_prevalence_stream"]["enabled"] = False   # skip the 1500-benign stream for the demo
cfg["output"]["root"] = "out_real"

summary = run_all(cfg, "out_real", n_families=8, strict_quadrants=False)

print("=== AGENTIC REAL-DATA RUN ===")
print(json.dumps(summary["generation"], indent=2))
print("validation_passed:", summary["validation_passed"],
      "| self_test_passed:", summary["self_test_passed"])

sc_path = "out_real/judge_self_consistency.json"
if os.path.exists(sc_path):
    print("judge_self_consistency:", json.load(open(sc_path)))

rows = [json.loads(l) for l in open("out_real/rows_split.jsonl")]
print("=== sample rows (real-data grounded, LLM-authored + LLM-judged) ===")
seen = set()
for r in rows:
    q = r["quadrant"]
    if q in seen:
        continue
    seen.add(q)
    print(f"[{q}] {r['id']} G={r['general_safety_gold']} D={r['mortgage_policy_gold']} "
          f"action={r['action_gold']} cards={r['policy_context']}")
    print("   prompt:", r["user_prompt"][:220])
    print("   grounding:", {k: r["hmda_grounding"].get(k) for k in
                            ("loan_purpose", "state", "ltv_band", "dti_band", "denial_reasons")})
print("DONE")
