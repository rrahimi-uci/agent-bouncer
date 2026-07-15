"""Phase: family-isolated split + sealed private test.

Splits are assigned at the near-dup `content_family` level so no lexical near-duplicate ever
crosses a split boundary. The `private_test` split is authored to be *sealed*: its rows are
written to a custodian-only file, and only a text-free index of it enters the public bundle.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import Row


def assign_splits(rows: list[Row], ratios: dict[str, float], seed: int,
                  rng_factory) -> dict[str, int]:
    """Assign each row's `.split` by whole content-families, honoring `ratios`.

    Splits are assigned QUADRANT-STRATIFIED: families are grouped by the quadrant of their
    rows (a family is single-quadrant by construction) and each quadrant's families are
    distributed across splits by `ratios`. This keeps the rare quadrants (G1D0, G1D1) present
    in the eval splits at scale, while never letting a content_family cross a split.
    Deterministic given `seed`. Returns per-split family counts.
    """
    fams: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        fams[r.content_family or r.id].append(r)

    # group families by their quadrant (interleave low-prev + balanced under the same quadrant)
    by_quad: dict[str, list[str]] = defaultdict(list)
    for cf, members in fams.items():
        by_quad[members[0].quadrant].append(cf)

    order = ["train", "dev", "public_test", "private_test"]
    counts: dict[str, int] = defaultdict(int)
    for quad in sorted(by_quad):
        fam_ids = sorted(by_quad[quad])
        rng_factory(seed, "split", quad).shuffle(fam_ids)
        total = len(fam_ids)
        targets = {k: int(round(ratios.get(k, 0.0) * total)) for k in order}
        drift = total - sum(targets.values())
        targets["train"] += drift
        idx = 0
        for split in order:
            for _ in range(targets[split]):
                if idx >= total:
                    break
                for r in fams[fam_ids[idx]]:
                    r.split = split
                counts[split] += 1
                idx += 1
        while idx < total:                     # leftovers → train
            for r in fams[fam_ids[idx]]:
                r.split = "train"
            counts["train"] += 1
            idx += 1
    return dict(counts)


def partition(rows: list[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        out[r.split].append(r)
    return dict(out)


def check_isolation(rows: list[Row]) -> list[str]:
    """Return violations: any content_family that appears in more than one split."""
    fam_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        fam_splits[r.content_family or r.id].add(r.split)
    return [f"content_family {cf} spans splits {sorted(sp)}"
            for cf, sp in fam_splits.items() if len(sp) > 1]
