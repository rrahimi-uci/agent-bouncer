"""Phase: quality gates (mirrors docs spec §9 / runbook Phase 6).

Fail-closed. Returns a report dict with a top-level `passed` bool and per-gate detail.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import Row, validate_row, QUADRANTS
from .split import check_isolation


def run_gates(rows: list[Row], *, min_benign_low_prev: int = 1500,
              require_quadrant_coverage: bool = True) -> dict[str, Any]:
    gates: dict[str, Any] = {}

    # 1) schema validity
    schema_errs: list[str] = []
    for r in rows:
        for e in validate_row(r):
            schema_errs.append(f"{r.id}: {e}")
    gates["schema_valid"] = {"passed": not schema_errs,
                             "n_violations": len(schema_errs),
                             "examples": schema_errs[:10]}

    # 2) family isolation across splits
    iso = check_isolation(rows)
    gates["family_isolation"] = {"passed": not iso, "violations": iso[:10]}

    # 3) exact-dup freedom
    seen: dict[str, str] = {}
    dups = []
    for r in rows:
        if r.content_sha256 in seen:
            dups.append(f"{r.id} dup of {seen[r.content_sha256]}")
        else:
            seen[r.content_sha256] = r.id
    gates["no_exact_dups"] = {"passed": not dups, "n": len(dups), "examples": dups[:10]}

    # 4) all four quadrants populated in each eval split (guard track)
    per_split_quad: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.track == "guard":
            per_split_quad[r.split].add(r.quadrant)
    missing = {sp: sorted(set(QUADRANTS) - q)
               for sp, q in per_split_quad.items()
               if sp in ("dev", "public_test", "private_test") and set(QUADRANTS) - q}
    # coverage-per-eval-split is a production gate; small/smoke builds are too sparse to hit it
    gates["quadrants_populated"] = {"passed": (not missing) or (not require_quadrant_coverage),
                                    "missing": missing,
                                    "enforced": require_quadrant_coverage}

    # 5) protected minimal pairs balanced + both G0D0/PASS
    pairs: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.pair_id:
            pairs[r.pair_id].append(r)
    pair_errs = []
    for pid, members in pairs.items():
        variants = {m.variant for m in members}
        if variants != {"protected", "reference"}:
            pair_errs.append(f"{pid}: variants {sorted(variants)} != protected+reference")
        if any(m.quadrant != "G0D0" or m.action_gold != "PASS" for m in members):
            pair_errs.append(f"{pid}: a member is not G0D0/PASS")
    gates["protected_pairs"] = {"passed": not pair_errs, "n_pairs": len(pairs),
                                "examples": pair_errs[:10]}

    # 6) low-prevalence precision floor
    low_benign = sum(1 for r in rows
                     if r.track == "guard" and r.stratum == "low_prevalence_stream"
                     and r.final_intervention_gold == 0)
    gates["low_prevalence_floor"] = {"passed": low_benign >= min_benign_low_prev,
                                     "benign_units": low_benign,
                                     "floor": min_benign_low_prev}

    # 7) honesty constants
    bad_honesty = [r.id for r in rows if r.contains_real_pii or not r.synthetic]
    gates["honesty_constants"] = {"passed": not bad_honesty, "violations": bad_honesty[:10]}

    passed = all(g["passed"] for g in gates.values())
    return {"passed": passed, "n_rows": len(rows), "gates": gates}
