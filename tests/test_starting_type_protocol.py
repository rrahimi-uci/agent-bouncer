#!/usr/bin/env python
"""Protocol / preregistration scaffolding tests for the starting-type adaptation study.

Loads every protocol JSON, asserts required keys are present, and checks internal consistency
with the authoring registry configs/starting_type_adaptation_v1.yaml (conditions, primary beta,
min families, seeds, panel membership) and with the claim predicates (m=0.02, Bonferroni,
predicate-selected interpretation rows). No models, no GPU.

Run directly: python tests/test_starting_type_protocol.py   (or via pytest)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import starting_type_common as A  # noqa: E402

PROTO = os.path.join(ROOT, "artifacts", "starting_type_adaptation_v1", "protocol")
REG = A.load_registry(os.path.join(ROOT, "configs", "starting_type_adaptation_v1.yaml"))
CK = REG["checkpoints"]


def _load(name):
    with open(os.path.join(PROTO, name), "r", encoding="utf-8") as f:
        return json.load(f)


def test_all_json_well_formed():
    for name in ("primary_contract.json", "claim_registry.json",
                 "model_benchmark_exposure_matrix.json", "scoring_contracts.json",
                 "policy_crosswalk.json"):
        d = _load(name)
        assert d["study_id"] == "starting_type_adaptation_v1", name
    # requirements + prereg exist
    assert os.path.exists(os.path.join(ROOT, "requirements-starting-type-adaptation.txt"))
    assert os.path.exists(os.path.join(ROOT, "docs", "starting-type-adaptation-prereg.md"))
    print("  [ok] all 5 protocol JSON well-formed; requirements + prereg present")


def test_primary_contract_consistent_with_yaml():
    pc = _load("primary_contract.json")
    for k in ("conditions", "recipe", "panels", "family_gate", "contracts", "estimands",
              "primary_contrasts", "decision_margins", "statistics", "confirmatory_hypotheses"):
        assert k in pc, k
    assert list(pc["conditions"]) == list(REG["conditions"])
    assert pc["recipe"]["kl"]["primary_beta"] == REG["recipe"]["kl"]["primary_beta"]
    assert pc["recipe"]["kl"]["sensitivity_beta_grid"] == REG["recipe"]["kl"]["sensitivity_beta"]
    assert pc["recipe"]["seeds"] == REG["recipe"]["seeds"]
    assert pc["family_gate"]["min_purpose_built_families"] == REG["min_purpose_built_families"]
    # panel membership exactly mirrors the registry
    gen = set(pc["panels"]["general"]["checkpoints"])
    pb = set(pc["panels"]["purpose_built"]["checkpoints"])
    assert gen == {k for k, v in CK.items() if v["starting_type"] == "general"}
    assert pb == {k for k, v in CK.items() if v["starting_type"] == "purpose_built"}
    for key, entry in {**pc["panels"]["general"]["checkpoints"],
                       **pc["panels"]["purpose_built"]["checkpoints"]}.items():
        assert entry["model_id"] == CK[key]["model_id"], key
        assert entry["contract"] == CK[key]["contract"], key
    assert pc["decision_margins"]["non_inferiority_margin_m"] == 0.02
    print(f"  [ok] primary_contract mirrors YAML: {len(gen)} general + {len(pb)} purpose-built, "
          f"beta {pc['recipe']['kl']['primary_beta']}, m={pc['decision_margins']['non_inferiority_margin_m']}")


def test_claim_registry_predicates():
    cr = _load("claim_registry.json")
    for k in ("predicates", "family_verdicts", "confirmatory_families", "interpretation_matrix",
              "non_inferiority", "beta_policy"):
        assert k in cr, k
    for name in ("H_gain", "H_conc", "H_preserve", "H_cost"):
        p = cr["predicates"][name]
        assert p["estimand"] and p["rule"] and p["family"] in ("RQ1", "RQ2"), name
    assert cr["predicates"]["H_cost"]["m"] == 0.02
    assert cr["non_inferiority"]["margin_m"] == 0.02
    assert cr["confirmatory_families"]["multiplicity"]["procedure"] == "bonferroni"
    assert cr["confirmatory_families"]["multiplicity"]["familywise_alpha"] == 0.05
    assert cr["beta_policy"]["auto_best_beta_selection"] == "forbidden"
    # every interpretation row is bound to a predicate + priority + wording
    rows = cr["interpretation_matrix"]["rows"]
    assert len(rows) >= 8
    for r in rows:
        assert r["predicate"] and r["wording"] and isinstance(r["priority"], int) and r["block"]
    # priorities disjoint within each block (predicate-selected, exactly one fires)
    from collections import defaultdict
    seen = defaultdict(set)
    for r in rows:
        assert r["priority"] not in seen[r["block"]], (r["block"], r["priority"])
        seen[r["block"]].add(r["priority"])
    print(f"  [ok] claim_registry: 4 predicates, Bonferroni alpha 0.05, m=0.02, "
          f"{len(rows)} predicate-bound interpretation rows with disjoint per-block priorities")


def test_exposure_matrix_template():
    em = _load("model_benchmark_exposure_matrix.json")
    allowed = em["allowed_exposure_status"]
    assert set(allowed) == {
        "documented_training_or_derivative", "constructionally_related_or_home_benchmark",
        "documented_author_evaluation", "no_documented_overlap", "unknown",
        "chronologically_post_pinned_weights"}
    assert em["n_rows"] == len(em["rows"]) == em["n_checkpoints"] * em["n_benchmarks"]
    assert em["n_checkpoints"] == len(CK)
    ckpts_seen = set()
    for r in em["rows"]:
        for f in em["required_fields_per_row"]:
            assert f in r, f
        assert r["exposure_status"] in allowed
        assert r["starting_model_key"] in CK
        ckpts_seen.add(r["starting_model_key"])
    assert ckpts_seen == set(CK)
    print(f"  [ok] exposure matrix: {em['n_rows']} rows = {em['n_checkpoints']} ckpts x "
          f"{em['n_benchmarks']} benchmarks; 6-status enum; all required fields")


def test_scoring_and_crosswalk():
    sc = _load("scoring_contracts.json")
    for v in ("study_native", "official_native", "common_sensitivity"):
        assert v in sc["views"], v
    assert sc["views"]["study_native"]["role"] == "primary"
    assert sc["views"]["study_native"]["store_raw_margins_not_only_sigmoid"] is True
    pcw = _load("policy_crosswalk.json")
    # ShieldGemma study-policy note present
    note = pcw["checkpoints"]["shieldgemma_2b"]["shieldgemma_study_policy_note"]
    assert "one locked broad study policy" in note.lower()
    # crosswalk covers every registry checkpoint
    assert set(pcw["checkpoints"]) == set(CK)
    print("  [ok] scoring_contracts 3 non-pooled views; policy_crosswalk covers all checkpoints "
          "incl ShieldGemma study-policy note")


if __name__ == "__main__":
    print("=== starting_type_adaptation protocol/prereg scaffolding tests ===")
    test_all_json_well_formed()
    test_primary_contract_consistent_with_yaml()
    test_claim_registry_predicates()
    test_exposure_matrix_template()
    test_scoring_and_crosswalk()
    print("ALL PASSED")
