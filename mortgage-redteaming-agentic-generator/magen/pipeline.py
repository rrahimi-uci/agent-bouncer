"""Orchestrator + CLI: ingest -> plan -> ground -> generate -> adversary -> judge ->
provenance/decontam -> split/seal -> validate -> evaluate -> package.

Run offline end-to-end with no keys/network/download:
    python -m magen --smoke
Production build (real HMDA + real judge):
    python -m magen --config config/default.yaml --phase all \
        --provider anthropic            # after setting hmda.source: csv + snapshot_path
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import __version__
from .cards import Design, load_design
from .llm import LLMClient, env_ready
from .planner import Cell, plan
from .ground import Grounder
from .generate import generate_prompt, llm_pair_template
from .adversary import mutate
from .judge import judge, Judgement, self_consistency
from . import provenance as prov
from .schema import Row, finalize, LEGAL_REVIEW_DEFAULT
from .split import assign_splits, check_isolation
from .validate import run_gates
from .evaluate import self_test
from .package import package
from .store import (load_config, load_dotenv, rng, abspath, ROOT, write_rows, read_rows,
                    write_json)

_DOMAIN_CODE = {"fair_lending": "FL", "atr_qm": "ATR", "disclosure": "DIS", "udaap": "UD",
                "fraud": "FRD", "privacy": "PRV", "benign": "BEN", "general_harm": "GEN"}


def _domain_for(cell: Cell, design: Design) -> tuple[str, str]:
    if cell.card_id:
        c = design.cards[cell.card_id]
        return c.domain, c.subcard
    if cell.quadrant == "G1D0":
        return "general_harm", cell.trap_type
    return "benign", cell.trap_type


def _mk_row(rid: str, cell: Cell, prompt: str, scenario: str, grounding: dict[str, Any],
            j: Judgement, version: str, pair_id: str = "") -> Row:
    # domain/subdomain are filled by the caller (needs the Design); left "" here.
    row = Row(
        id=rid, benchmark_version=version, track="guard", split="train",
        stratum=cell.stratum, domain="", subdomain="", trap_type=cell.trap_type,
        difficulty=cell.difficulty, scenario=scenario, policy_context=list(j.policy_context),
        user_prompt=prompt, general_safety_gold=j.general_safety_gold,
        mortgage_policy_gold=j.mortgage_policy_gold, action_gold=j.action_gold,
        severity=j.severity, rationale=j.rationale,
        source_ids=["HMDA-2022-Snapshot-National-Loan-Level (aggregate grounding)"],
        family_id=cell.family_id, hmda_grounding=grounding,
        pair_id=pair_id, protected_attribute=(cell.protected_attribute or ""),
        variant=(cell.variant or ""), legal_review_status=LEGAL_REVIEW_DEFAULT)
    return finalize(row)


def _accepts(j: Judgement, cell: Cell) -> bool:
    return (j.general_safety_gold == cell.target_G
            and j.mortgage_policy_gold == cell.target_D
            and j.realism_ok and j.consensus)


def _gen_single(cell: Cell, grounding: dict, design: Design, client: LLMClient,
                seed: int, vi: int, max_retries: int, judge_samples: int,
                errors: Counter) -> list[dict]:
    """Author + judge one variant of a non-pair cell; return [row-dict] or [] (a miss)."""
    for attempt in range(max_retries + 1):
        try:
            base, scenario = generate_prompt(cell, grounding, design, client)
            mutated = (mutate(base, cell, design, seed + attempt, vi)[0]
                       if client.provider == "mock" else base)
            j = judge(mutated, design, client, n_samples=judge_samples)
        except Exception:  # noqa: BLE001 - a bad LLM response is a retry, not a crash
            errors["_llm_errors"] += 1
            continue
        if _accepts(j, cell):
            return [dict(cell=cell, prompt=mutated, scenario=scenario, j=j, vi=vi, pair_id="",
                         grounding=grounding)]
    return []


def _gen_pair(members: list[Cell], grounding: dict, design: Design, client: LLMClient,
              seed: int, vi: int, max_retries: int, judge_samples: int,
              errors: Counter) -> list[dict]:
    """Author a benign COUNTERFACTUAL pair differing only in the protected token; return the
    two row-dicts, or [] unless BOTH variants judge benign G0/D0 (the fairness invariant)."""
    prot = next(m for m in members if m.variant == "protected")
    ref = next(m for m in members if m.variant == "reference")
    for attempt in range(max_retries + 1):
        try:
            if client.provider == "mock":
                pp, scenario = generate_prompt(prot, grounding, design, client)
                rp, _ = generate_prompt(ref, grounding, design, client)
            else:
                tmpl, scenario = llm_pair_template(prot, grounding, design, client)
                pp = tmpl.replace("{APPLICANT}", prot.protected_value or "a Black applicant")
                rp = tmpl.replace("{APPLICANT}", ref.reference_value or "a white applicant")
            jp = judge(pp, design, client, n_samples=judge_samples)
            jr = judge(rp, design, client, n_samples=judge_samples)
        except Exception:  # noqa: BLE001
            errors["_llm_errors"] += 1
            continue
        if _accepts(jp, prot) and _accepts(jr, ref):
            pid = f"{prot.pair_group}#{vi}"
            return [dict(cell=prot, prompt=pp, scenario=scenario, j=jp, vi=vi, pair_id=pid,
                         grounding=grounding),
                    dict(cell=ref, prompt=rp, scenario=scenario, j=jr, vi=vi, pair_id=pid,
                         grounding=grounding)]
    return []


def generate_rows(cfg: dict[str, Any], design: Design, factsheets, client: LLMClient,
                  seed: int, n_families: int) -> tuple[list[Row], dict[str, Any]]:
    cells = plan(design, cfg, n_families, seed)
    grounder = Grounder(factsheets, seed)
    gen = cfg.get("generate", {})
    max_retries = int(gen.get("max_generation_retries", 4))
    judge_samples = int(gen.get("judge_samples", 3))
    concurrency = int(cfg.get("llm", {}).get("concurrency", 8)) if client.provider != "mock" else 1

    pair_groups: dict[str, list[Cell]] = defaultdict(list)
    singles: list[Cell] = []
    for c in cells:
        (pair_groups[c.pair_group].append(c) if c.pair_group else singles.append(c))
    pair_grounding = {pg: grounder.ground(members[0]) for pg, members in pair_groups.items()}

    errors: Counter = Counter()
    tasks: list = []
    for c in singles:
        g = grounder.ground(c)
        for vi in range(c.n_variants):
            tasks.append(("single", c, g, vi))
    for pg, members in pair_groups.items():
        for vi in range(min(m.n_variants for m in members)):
            tasks.append(("pair", members, pair_grounding[pg], vi))

    def run(t) -> list[dict]:
        kind, obj, g, vi = t
        if kind == "single":
            return _gen_single(obj, g, design, client, seed, vi, max_retries, judge_samples, errors)
        return _gen_pair(obj, g, design, client, seed, vi, max_retries, judge_samples, errors)

    accepted: list[dict] = []
    n_expected = 0
    for t in tasks:
        n_expected += 2 if t[0] == "pair" else 1
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for res in ex.map(run, tasks):
            accepted.extend(res)

    # deterministic id assignment (content varies run-to-run; ordering + ids are stable)
    accepted.sort(key=lambda d: (d["cell"].family_id, d["cell"].variant or "", d["vi"], d["pair_id"]))
    rows: list[Row] = []
    rid_seq: Counter = Counter()
    counters: Counter = Counter()
    for d in accepted:
        c = d["cell"]
        domain, subdomain = _domain_for(c, design)
        code = _DOMAIN_CODE.get(domain, "GEN")
        rid_seq[code] += 1
        row = _mk_row(f"MGB-{code}-{rid_seq[code]:05d}", c, d["prompt"], d["scenario"],
                      d["grounding"], d["j"], __version__, d["pair_id"])
        row.domain, row.subdomain = domain, subdomain
        rows.append(row)
        counters[c.quadrant] += 1

    report = {"planned_cells": len(cells), "rows": len(rows),
              "generation_misses": n_expected - len(accepted),
              "llm_errors": errors["_llm_errors"], "by_quadrant": dict(counters),
              "llm_calls": client.calls, "provider": client.provider,
              "concurrency": concurrency}
    return rows, report


# ------------------------------------------------------------------ mock LLM handler
def _mock_handler_factory():
    """The mock provider never actually drives generation/judging (those short-circuit on
    provider=='mock'); this exists so LLMClient(mock) is constructible and any stray call is
    deterministic rather than a crash."""
    def handler(system: str, user: str) -> str:
        return "{}"
    return handler


# ------------------------------------------------------------------ phases
def phase_ingest(cfg, out, seed) -> list:
    from .hmda import load_records
    fs = load_records(cfg.get("hmda", {}), ROOT, rng(seed, "hmda"))
    write_json({"n_factsheets": len(fs),
                "source": cfg.get("hmda", {}).get("source"),
                "example": fs[0].scenario_line() if fs else None},
               os.path.join(out, "ingest_meta.json"))
    return fs


def _resolve_openai_models(client: LLMClient) -> None:
    """Pick real, available model ids for generator/judge (robust to exact id drift)."""
    try:
        ids = {m.id for m in client._client.models.list().data}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not list OpenAI models ({e}); using configured ids.",
              file=sys.stderr)
        return

    def pick(pref: str) -> str:
        if pref in ids:
            return pref
        for filt in (lambda i: i.startswith("gpt-5") and "mini" in i,
                     lambda i: i.startswith("gpt-5"),
                     lambda i: i.startswith("gpt-4o")):
            cands = sorted(i for i in ids if filt(i))
            if cands:
                return cands[0]
        return pref
    client.generator_model = pick(client.generator_model)
    client.judge_model = pick(client.judge_model)


def build_client(cfg) -> LLMClient:
    load_dotenv()                       # make .env OPENAI_API_KEY/HF_TOKEN available
    lcfg = dict(cfg.get("llm", {}))
    provider = lcfg.get("provider", "mock")
    if provider != "mock" and not env_ready(provider):
        print(f"[warn] provider={provider} but no API key in env; falling back to mock.",
              file=sys.stderr)
        lcfg["provider"] = "mock"
        provider = "mock"
    client = LLMClient(lcfg, mock_handler=_mock_handler_factory())
    if provider == "openai":
        _resolve_openai_models(client)
    return client


def run_all(cfg: dict[str, Any], out: str, n_families: int,
            strict_quadrants: bool = True) -> dict[str, Any]:
    seed = int(cfg.get("seed", 20260714))
    design = load_design()
    pending = design.signoff_pending()
    client = build_client(cfg)

    factsheets = phase_ingest(cfg, out, seed)
    rows, gen_report = generate_rows(cfg, design, factsheets, client, seed, n_families)
    write_rows(rows, os.path.join(out, "raw_rows.jsonl"))

    # provenance + decontamination
    prov.hash_rows(rows)
    rows, n_exact = prov.dedup_exact(rows)
    n_fams = prov.assign_content_families(rows,
                                          cfg.get("decontam", {}).get("jaccard_threshold"))
    decon = prov.decontaminate(rows, abspath(ROOT, cfg.get("decontam", {}).get(
        "general_sources_index", "")))
    rows = decon["kept"]
    write_json({"exact_dups_dropped": n_exact, "content_families": n_fams,
                "decontam": decon["report"]}, os.path.join(out, "provenance_report.json"))

    # split + seal
    split_counts = assign_splits(rows, cfg.get("split", {}).get("ratios", {}), seed, rng)
    iso = check_isolation(rows)
    write_rows(rows, os.path.join(out, "rows_split.jsonl"))

    # validate — the low-prev precision floor only applies when that stream is enabled
    lp_cfg = cfg.get("strata", {}).get("low_prevalence_stream", {})
    floor = int(lp_cfg.get("min_benign_units", 1500)) if lp_cfg.get("enabled") else 0
    gates = run_gates(rows, min_benign_low_prev=floor,
                      require_quadrant_coverage=strict_quadrants)
    gates["family_isolation_live"] = {"passed": not iso, "violations": iso[:10]}
    write_json(gates, os.path.join(out, "validation_report.json"))

    # evaluate (self-test: gold-as-preds must be perfect)
    st = self_test([r for r in rows if r.track == "guard"])
    write_json(st, os.path.join(out, "eval_selftest.json"))

    # judge self-consistency (online only; skipped cheaply in mock since deterministic)
    if client.provider != "mock":
        sc = self_consistency(rows[: min(60, len(rows))], design, client)
        write_json(sc, os.path.join(out, "judge_self_consistency.json"))

    # package
    license_note = ("LICENSE NOT YET SELECTED. HMDA public data are a U.S. Government work "
                    "(17 U.S.C. §105, no U.S. copyright); confirm no separate FFIEC/CFPB "
                    "terms-of-use restriction before redistributing generated prompts.")
    sources = [{"name": "HMDA 2022 Snapshot National Loan-Level Dataset",
                "publisher": "FFIEC / CFPB",
                "url": "https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/",
                "use": "aggregate/de-identified grounding only; no verbatim rows; no PII"}]
    pkg = package(rows, out, version=__version__, seed=seed,
                  license_note=license_note, sources=sources)

    summary = {"version": __version__, "seed": seed, "out": out,
               "signoff_pending_cards": pending, "generation": gen_report,
               "split_counts": split_counts, "validation_passed": gates["passed"],
               "self_test_passed": st.get("self_test_passed"),
               "package": {"dist_dir": pkg["dist_dir"], "written": pkg["written"],
                           "sealed": pkg["sealed"]}}
    write_json(summary, os.path.join(out, "run_summary.json"))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="magen", description=__doc__)
    ap.add_argument("--config", default=None, help="path to config yaml")
    ap.add_argument("--out", default=None, help="output root (default: config output.root)")
    ap.add_argument("--n-families", type=int, default=200,
                    help="number of authored families to plan (balanced stratum)")
    ap.add_argument("--provider", default=None, help="override llm.provider (mock|openai|anthropic)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast offline run (few families, small low-prev floor)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("llm", {})["provider"] = args.provider
    if args.smoke:
        cfg.setdefault("llm", {})["provider"] = cfg.get("llm", {}).get("provider", "mock")
        cfg.setdefault("hmda", {})["source"] = "bundled_sample"
        cfg.setdefault("generate", {})["variants_per_family"] = 3
        cfg.setdefault("strata", {}).setdefault("low_prevalence_stream", {})
        cfg["strata"]["low_prevalence_stream"]["min_benign_units"] = 30
        args.n_families = min(args.n_families, 24)

    out = args.out or abspath(ROOT, cfg.get("output", {}).get("root", "out"))
    os.makedirs(out, exist_ok=True)
    summary = run_all(cfg, out, args.n_families, strict_quadrants=not args.smoke)

    print(f"magen {summary['version']} → {out}")
    print(f"  rows: {summary['generation']['rows']}  quadrants: {summary['generation']['by_quadrant']}")
    print(f"  splits: {summary['split_counts']}")
    print(f"  validation_passed={summary['validation_passed']}  self_test_passed={summary['self_test_passed']}")
    if summary["signoff_pending_cards"]:
        print(f"  [!] {len(summary['signoff_pending_cards'])} policy cards await SME sign-off "
              f"(offline build; labels are policy-card-consistent, not legal authority).")
    return 0 if (summary["validation_passed"] and summary["self_test_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
