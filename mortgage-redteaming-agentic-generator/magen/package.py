"""Phase: package the release (mirrors runbook Phase 8).

Emits a distributable bundle plus a TEXT-FREE public index. Raw prompt text and the sealed
private test never enter the public index; the distributed splits carry text under whatever
license the builder selects (LICENSE must be chosen before release — see the data card).
"""
from __future__ import annotations

import hashlib
import os
from collections import Counter
from typing import Any

from .schema import Row
from .store import write_json, write_rows


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def public_index(rows: list[Row]) -> dict[str, Any]:
    """A recursively text-free snapshot: ids, hashes, family links, counts, labels — no prompt text."""
    entries = [{
        "id": r.id, "split": r.split, "track": r.track, "stratum": r.stratum,
        "domain": r.domain, "subdomain": r.subdomain, "trap_type": r.trap_type,
        "difficulty": r.difficulty, "quadrant": r.quadrant,
        "general_safety_gold": r.general_safety_gold,
        "mortgage_policy_gold": r.mortgage_policy_gold,
        "final_intervention_gold": r.final_intervention_gold,
        "action_gold": r.action_gold, "severity": r.severity,
        "policy_context": r.policy_context,
        "family_id": r.family_id, "content_family": r.content_family,
        "content_sha256": r.content_sha256,
        "pair_id": r.pair_id, "protected_attribute": r.protected_attribute,
        "variant": r.variant, "source_ids": r.source_ids,
    } for r in rows]
    return {
        "text_free": True,
        "n_rows": len(rows),
        "counts": {
            "by_split": dict(Counter(r.split for r in rows)),
            "by_quadrant": dict(Counter(r.quadrant for r in rows)),
            "by_domain": dict(Counter(r.domain for r in rows)),
            "by_trap_type": dict(Counter(r.trap_type for r in rows)),
            "by_stratum": dict(Counter(r.stratum for r in rows)),
        },
        "entries": entries,
    }


DATA_CARD = """# Mortgage Guardrail Benchmark — Data Card

**Version:** {version}
**Built:** deterministic (seed {seed}) by the agentic HMDA-grounded generator in this folder.

## What this is
A request-screening benchmark for a mortgage-specific safety guardrail. Each row is one
incoming request to a mortgage workflow, carrying TWO independent policy labels
(`general_safety_gold` G, `mortgage_policy_gold` D), a derived `final_intervention_gold`, an
`action_gold` lattice, quadrant, and severity. Scenarios are grounded in the public HMDA 2022
National Loan-Level Snapshot for realism; **no real individual record or PII is reproduced** —
grounding uses aggregate/de-identified fields only.

## Honesty / construct validity
- Prompts are **synthetic**; harmful requests are *represented for detection*, never operational
  recipes. `contains_real_pii=false` is a hard schema constant.
- Labels are **policy-card-consistent, not legally authoritative**. `legal_review_status`
  records this. Confirmatory fair-lending claims require the SME-adjudicated subset (not yet done).
- The `private_test` split is **sealed**: it is not in this bundle; only its text-free index is.

## Splits
{splits}

## License
{license}

## Reproduce
See the folder README. `make all` rebuilds the whole benchmark from the frozen design + seed.
"""


def package(rows: list[Row], out_dir: str, *, version: str, seed: int,
            license_note: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    dist_dir = os.path.join(out_dir, "dist")
    os.makedirs(dist_dir, exist_ok=True)

    # distributed splits (text-bearing) — sealed private_test excluded
    written = {}
    for split in ("train", "dev", "public_test"):
        srows = [r for r in rows if r.split == split]
        if srows:
            p = os.path.join(dist_dir, f"{split}.jsonl")
            written[split] = write_rows(srows, p)

    # sealed private test → custodian file (kept out of dist/)
    sealed = [r for r in rows if r.split == "private_test"]
    if sealed:
        write_rows(sealed, os.path.join(out_dir, "SEALED_private_test.jsonl"))

    # text-free public index (covers ALL rows incl. sealed, but no text)
    idx = public_index(rows)
    write_json(idx, os.path.join(dist_dir, "public_index.json"))

    # data card, schema pointer, sources, manifest, checksums
    splits_tbl = "\n".join(f"- `{k}`: {v} rows" for k, v in written.items()) + \
        (f"\n- `private_test` (sealed, not distributed): {len(sealed)} rows" if sealed else "")
    with open(os.path.join(dist_dir, "DATA_CARD.md"), "w") as fh:
        fh.write(DATA_CARD.format(version=version, seed=seed, splits=splits_tbl,
                                  license=license_note))
    write_json(sources, os.path.join(dist_dir, "SOURCES.json"))

    manifest = {"version": version, "seed": seed, "n_rows": len(rows),
                "distributed_splits": written, "sealed_rows": len(sealed),
                "public_index": "public_index.json"}
    write_json(manifest, os.path.join(dist_dir, "MANIFEST.json"))

    checks = {}
    for fn in sorted(os.listdir(dist_dir)):
        fp = os.path.join(dist_dir, fn)
        if os.path.isfile(fp):
            checks[fn] = _sha256_file(fp)
    with open(os.path.join(dist_dir, "CHECKSUMS.txt"), "w") as fh:
        for fn, h in checks.items():
            fh.write(f"{h}  {fn}\n")

    return {"dist_dir": dist_dir, "written": written, "sealed": len(sealed),
            "checksums": checks}
