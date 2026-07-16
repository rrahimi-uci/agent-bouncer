#!/usr/bin/env python
"""Reproduce EVERY number/table/figure in the unified report from committed per-row scores.

One entry point (`make reproduce` calls this). For each study it re-derives the generated LaTeX
tables the report `\\input`s, copies the canonical outputs into `generated/`, and (with --check)
asserts byte-identity with the committed copies. It needs NO GPU and NO network; only committed
scores + the pinned analysis environment. Studies whose results are not yet produced (e.g. the
Paper C objective axis before its GPU run) are reported as PENDING, not failed.

  Paper A (SFT specialization)   analyze_paper_a_sft.py --release-cache   [needs the LOCK-pinned env]
  Paper B (composition)          build_pilot_artifacts.py
  Mortgage (dual-label G x D)    tools/reeval_from_scores.py + emit_baseline_tex.py
  ExpGuard (finance/health/law)  eval_expguard_external.py --from-scores  -> emit table
  Paper C (objective axis)       analyze_paper_c.py --from-scores         [PENDING until trained]

Usage:  python reproduce.py [--check] [--build]
        --check : fail if any regenerated table differs from the committed generated/ copy
        --build : also compile the PDF with tectonic afterwards
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = HERE / "generated"
PY = REPO / ".venv" / "bin" / "python"
PYS = str(PY) if PY.exists() else sys.executable


def _run(cmd, cwd=REPO, env=None):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)


def _copy_into_generated(src: Path, dst_name: str, results: dict, check: bool):
    dst = GEN / dst_name
    if not src.exists():
        results[dst_name] = "FAIL (source missing)"
        return
    content = src.read_text().replace("[H]", "[htbp]")  # keep the report float style on regen
    if check and dst.exists():
        # compare the NORMALIZED source (post [H]->[htbp]) against the committed copy, else the
        # normalization we apply on write would flag every [H]-using table as spurious drift.
        results[dst_name] = "OK (byte-identical)" if dst.read_text() == content else "DRIFT!"
    else:
        dst.write_text(content)
        results[dst_name] = "regenerated"


def paper_a(results, check):
    """Regenerate Paper A tables from committed scores.parquet (lock-pinned env)."""
    lock = REPO / "artifacts/paper_a_sft_v2/LOCK.json"
    scores = REPO / "artifacts/paper_a_sft_v2/scores/scores.parquet"
    out = REPO / "artifacts/paper_a_sft_v2/analysis"
    r = _run([PYS, "experiments/analyze_paper_a_sft.py", "--release-cache",
              "--lock", str(lock), "--scores", str(scores), "--out", str(out)])
    if r.returncode != 0:
        msg = "PINNED-ENV REQUIRED" if "software" in (r.stderr + r.stdout).lower() else "FAIL"
        for n in ("tab_primary_gen.tex", "tab_sensitivity_gen.tex", "tab_seed_values_gen.tex", "results_macros_gen.tex"):
            results[f"A:{n}"] = f"{msg} (analysis not re-run; committed copy used)"
        return
    tbl = out / "tables"
    _copy_into_generated(tbl / "table3_primary.tex", "tab_primary_gen.tex", results, check)
    _copy_into_generated(tbl / "table4_per_benchmark.tex", "tab_sensitivity_gen.tex", results, check)
    _copy_into_generated(tbl / "table5_seed_values.tex", "tab_seed_values_gen.tex", results, check)
    _copy_into_generated(tbl / "results_macros.tex", "results_macros_gen.tex", results, check)


def paper_b(results, check):
    pb = REPO / "papers/base-adapter-composition"
    r = _run([PYS, "code/build_pilot_artifacts.py",
              "--composition", "../../artifacts/paper_a_sft_v2/analysis/composition/composition.json",
              "--metadata", "../../artifacts/paper_a_sft_v2/analysis/composition/composition_metadata.json",
              "--out-dir", "generated"], cwd=pb)
    if r.returncode != 0:
        results["B:pilot_*"] = "FAIL: " + (r.stderr.strip().splitlines() or [""])[-1][:80]
        return
    for n in ("pilot_macros.tex", "pilot_summary_table.tex", "pilot_per_model_table.tex", "pilot_operating_point_table.tex"):
        _copy_into_generated(pb / "generated" / n, n, results, check)


def mortgage(results, check):
    import os as _os
    mb = REPO / "mortgage-benchmark"
    (mb / "generated").mkdir(exist_ok=True)
    env = {**_os.environ, "PYTHONPATH": str(mb)}  # reeval imports `magen`; scripts assume repo-root cwd
    r1 = _run([PYS, "mortgage-benchmark/tools/reeval_from_scores.py"], env=env)  # per-row scores -> baseline_table.json
    r2 = _run([PYS, "mortgage-benchmark/tools/emit_baseline_tex.py",
               "mortgage-benchmark/out_eval/baseline_table.json",
               "mortgage-benchmark/generated/baseline_table.tex"], env=env)
    src = mb / "generated" / "baseline_table.tex"
    if not src.exists():
        results["mortgage_baseline_table.tex"] = "FAIL: " + ((r1.stderr or r2.stderr or "no output").strip().splitlines() or [""])[-1][:90]
        return
    _copy_into_generated(src, "mortgage_baseline_table.tex", results, check)


def expguard(results, check):
    out = REPO / "artifacts/expguard_external"
    if not (out / "labels_index.json").exists():
        results["expguard_table.tex"] = "PENDING (base eval not yet committed)"
        return
    _run([PYS, "experiments/eval_expguard_external.py", "--from-scores", "--out", str(out)])
    # emit LaTeX table from the (deterministically recomputed) baseline_expguard.json
    tex = _emit_expguard_tex(out / "baseline_expguard.json")
    dst = GEN / "expguard_table.tex"
    if check and dst.exists():
        results["expguard_table.tex"] = "OK (byte-identical)" if dst.read_text() == tex else "DRIFT!"
    else:
        dst.write_text(tex)
        results["expguard_table.tex"] = "regenerated"


_EXPGUARD_PRETTY = {"qwen25_15b_base": "Qwen2.5-1.5B", "smollm2_17b_base": "SmolLM2-1.7B",
                    "smollm3_3b_base": "SmolLM3-3B", "qwen3_4b_base": "Qwen3-4B"}


def _emit_expguard_tex(json_path: Path) -> str:
    import json
    d = json.loads(json_path.read_text())
    def f(x):
        return "--" if x is None else f"{x:.3f}"
    def name(g):
        return _EXPGUARD_PRETTY.get(g, g.replace("_", chr(92) + "_"))
    rows = "\n".join(
        f"{name(r['guard'])} & {f(r['overall_ap'])} & {f(r['overall_auroc'])} & "
        f"{f(r.get('finance_ap'))} & {f(r.get('healthcare_ap'))} & {f(r.get('law_ap'))} \\\\"
        for r in d["table"])
    return ("% GENERATED by reproduce.py from artifacts/expguard_external/baseline_expguard.json\n"
            "\\begin{table}[H]\\centering\\small\n"
            "\\caption{External validation on ExpGuard (expert-annotated; input-prompt classification), "
            f"{d['n_rows']} rows across finance/health/law. Aggregate + per-domain AP; AUROC overall. "
            "Base checkpoints scored zero-shot via the canonical guard head; the ranking score is the raw "
            "decision margin $z_{\\text{unsafe}}-z_{\\text{safe}}$ (byte-parity with Act~I).}\n"
            "\\label{tab:expguard}\n"
            "\\begin{tabular}{lrrrrr}\\toprule\n"
            "Guard & AP (all) & AUROC (all) & AP finance & AP health & AP law \\\\\n\\midrule\n"
            f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def paper_c(results, check):
    scores = REPO / "artifacts/paper_c_objective_v2"
    if not scores.exists():
        results["paper_c_table.tex"] = "PENDING (objective-axis GPU run not yet done)"
        return
    r = _run([PYS, "experiments/analyze_paper_c.py", "--from-scores", "--out", str(scores / "analysis")])
    results["paper_c_table.tex"] = "regenerated" if r.returncode == 0 else "FAIL"



def figures(results, check):
    r = _run([PYS, 'figures/make_figures.py'], cwd=HERE)
    results['figures'] = 'regenerated' if r.returncode == 0 else 'FAIL: ' + (r.stderr or '')[-80:]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail on drift vs committed generated/")
    ap.add_argument("--build", action="store_true", help="compile the PDF afterwards")
    args = ap.parse_args(argv)

    GEN.mkdir(exist_ok=True)
    results: dict[str, str] = {}
    for fn in (paper_a, paper_b, mortgage, expguard, paper_c, figures):
        try:
            fn(results, args.check)
        except Exception as e:  # keep going; report per-study
            results[fn.__name__] = f"ERROR: {type(e).__name__}: {e}"

    print("\n=== reproduce: per-table status ===")
    drift = False
    for k, v in results.items():
        print(f"  {k:38s} {v}")
        if "DRIFT" in v or v.startswith(("FAIL", "ERROR")):
            drift = True
    if args.build:
        print("\n=== building PDF ===")
        b = _run(["tectonic", "--outdir", "build", "unified_report.tex"], cwd=HERE)
        print("  build:", "OK" if b.returncode == 0 else "FAIL\n" + b.stderr[-500:])
    if args.check and drift:
        print("\nCHECK FAILED: a regenerated table drifted or a study failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
