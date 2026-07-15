"""Adversarial mutator agent: compose 2-7 surface tactics onto a base prompt.

Mutations are LABEL-PRESERVING by construction: they wrap/prepend/append but never remove the
core ask or its card trigger marker, so (G, D) are unchanged. Injection tactics that would flip
G to unsafe are only applied to cells already in a G1 quadrant. Determinism: tactic choice is
seeded by (family, variant), which also gives cross-variant surface diversity for free.
"""
from __future__ import annotations

from typing import Any

from .cards import Design
from .planner import Cell
from .store import rng

# additive tactics that never change the safety label (pure surface wrapping)
_LABEL_SAFE = ["euphemism", "business_justified", "multi_turn", "dual_use"]
# injection-style tactics that imply G=unsafe — only for cells already G1
_INJECTION = ["buried_injection"]


def _apply(tactic: dict[str, Any], text: str) -> str:
    wrap = tactic.get("wrap", "{ASK}")
    kind = tactic.get("kind", "identity")
    if kind == "identity":
        return text
    return wrap.replace("{ASK}", text)


def mutate(prompt: str, cell: Cell, design: Design, seed: int, variant_index: int) -> tuple[str, list[str]]:
    """Return (mutated_prompt, tactics_applied)."""
    if cell.trap_type == "minimal_pair":
        return prompt, ["minimal_pair"]   # keep pairs identical except the protected token

    r = rng(seed, "adv", cell.family_id, str(variant_index))
    chosen: list[str] = []
    # the trap_type itself, if it is a wrap tactic, anchors the mutation
    if cell.trap_type in design.tactics:
        chosen.append(cell.trap_type)

    pool = list(_LABEL_SAFE)
    if cell.quadrant.startswith("G1"):
        pool += _INJECTION
    r.shuffle(pool)
    # benign rows get light, realistic phrasing (0-1 extra tactic); red-team rows get the
    # heavier 2-6 tactic composition the adversarial strata are meant to stress.
    if cell.quadrant == "G0D0":
        target_total = r.randint(0, 1)
    else:
        target_total = r.randint(2, 6)
    for t in pool:
        if len(chosen) >= target_total:
            break
        if t not in chosen:
            chosen.append(t)

    out = prompt
    applied: list[str] = []
    for name in chosen:
        tac = design.tactics.get(name)
        if not tac:
            continue
        out = _apply(tac, out)
        applied.append(name)
    return out, applied
