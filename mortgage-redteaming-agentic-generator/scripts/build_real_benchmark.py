#!/usr/bin/env python
"""Build a HARD, real-HMDA-grounded mortgage guardrail benchmark, fully LLM-authored.

Every row is authored by the LLM from real HMDA loan facts (no templates) and labeled by the
LLM judge. Hardness follows Bowen, Price, Stein & Yang (2024): borderline/higher-risk files,
SUBTLE proxy violations, counterfactual protected-attribute minimal pairs, hard benign
negatives. Parallelized to reach ~1000 rows in a reasonable time.

Usage:  python scripts/build_real_benchmark.py [N_FAMILIES]   (default 180 -> ~1000 rows)
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from magen.store import load_config
from magen.pipeline import run_all

n_families = int(sys.argv[1]) if len(sys.argv) > 1 else 180

cfg = load_config()
cfg["llm"]["provider"] = "openai"
cfg["llm"]["generator_model"] = "gpt-5.4-mini"
cfg["llm"]["judge_model"] = "gpt-5.4-mini"
cfg["llm"]["temperature"] = 1.0
cfg["llm"]["max_tokens"] = 900
cfg["llm"]["concurrency"] = 12
cfg["llm"]["max_llm_calls"] = 12000
cfg["hmda"]["source"] = "csv"
cfg["hmda"]["snapshot_path"] = "data/hmda_real/hmda_2022.csv"
cfg["hmda"]["sample_rows"] = 3000
cfg["hmda"]["filters"]["states"] = ["DC", "DE", "RI", "VT", "WY", "AK",
                                    "ND", "SD", "MT", "NH", "HI", "ME"]
# emphasize the hard fair-lending G0/D1 content (the paper's focus); keep G1 modest
cfg["generate"]["quadrant_mix"] = {"G0D0": 0.40, "G1D0": 0.10, "G0D1": 0.38, "G1D1": 0.12}
cfg["generate"]["variants_per_family"] = 6
cfg["generate"]["judge_samples"] = 1
cfg["generate"]["max_generation_retries"] = 2
cfg["generate"]["protected_pair_fraction"] = 0.30
cfg["strata"]["low_prevalence_stream"]["enabled"] = False
cfg["output"]["root"] = "out_real"

summary = run_all(cfg, "out_real", n_families=n_families, strict_quadrants=False)

print("=== HARD REAL-DATA AGENTIC BENCHMARK ===")
print(json.dumps(summary["generation"], indent=2))
print("validation_passed:", summary["validation_passed"],
      "| self_test_passed:", summary["self_test_passed"])
sc = "out_real/judge_self_consistency.json"
if os.path.exists(sc):
    print("judge_self_consistency:", json.load(open(sc)))

rows = [json.loads(l) for l in open("out_real/rows_split.jsonl")]
print("final rows:", len(rows), "| by quadrant:", dict(Counter(r["quadrant"] for r in rows)))
print("by domain:", dict(Counter(r["domain"] for r in rows)))
print("protected pairs:", len({r["pair_id"] for r in rows if r["pair_id"]}))
print("DONE")
