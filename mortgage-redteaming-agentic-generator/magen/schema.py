"""Row schema, label derivation, and validation.

Mirrors the row schema in docs/mortgage-benchmark-build-spec.md §7. The two policy labels
are authoritative; `final_intervention_gold` and `quadrant` are DERIVED and must never be
authored independently (derive() enforces that). `action_gold` is a separate lattice and is
never substituted for the binary.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# ------------------------------------------------------------------ controlled vocabularies
TRACKS = ("guard", "capability")
GENERAL = ("safe", "unsafe")                 # general_safety_gold  (G)
POLICY = ("allow", "intervene")              # mortgage_policy_gold (D)
ACTIONS = ("PASS", "CONSTRAIN", "REVIEW", "BLOCK")
SEVERITY = ("none", "low", "med", "severe")
QUADRANTS = ("G0D0", "G1D0", "G0D1", "G1D1")
STRATA = ("balanced_challenge", "low_prevalence_stream")
DIFFICULTY = ("easy", "medium", "hard")
SPLITS = ("train", "dev", "public_test", "private_test")
VARIANTS = ("protected", "reference")

TRAP_TYPES = (
    "hard_negative", "minimal_pair", "euphemism", "buried_injection",
    "over_refusal_bait", "dual_use", "multi_turn", "coded_proxy", "business_justified",
    "direct", "benign_info", "occupancy_temptation",
)

LEGAL_REVIEW_DEFAULT = "synthetic_machine_validated_not_counsel_reviewed"


@dataclass
class Row:
    """One request-screening benchmark row."""
    # identity / bookkeeping
    id: str
    benchmark_version: str
    track: str                                # guard | capability
    split: str                                # train | dev | public_test | private_test
    stratum: str                              # balanced_challenge | low_prevalence_stream
    # taxonomy
    domain: str
    subdomain: str
    trap_type: str
    difficulty: str
    # content
    scenario: str                             # short human-readable situation
    policy_context: list[str]                 # policy card ids that apply (D**)
    user_prompt: str                          # the request x to be screened
    # labels (authoritative)
    general_safety_gold: str                  # safe | unsafe   (G)
    mortgage_policy_gold: str                 # allow | intervene (D)
    # labels (derived — set by derive())
    final_intervention_gold: int = 0
    quadrant: str = "G0D0"
    # handling / severity
    action_gold: str = "PASS"
    severity: str = "none"
    # explanation + provenance
    rationale: str = ""
    source_ids: list[str] = field(default_factory=list)
    family_id: str = ""                       # authored template family (label unit)
    content_family: str = ""                  # near-dup (MinHash) cluster id
    content_sha256: str = ""
    # HMDA grounding (PII-free fact sheet actually used; never a raw record)
    hmda_grounding: dict[str, Any] = field(default_factory=dict)
    # fairness minimal pairs (optional)
    pair_id: str = ""
    protected_attribute: str = ""
    variant: str = ""                         # protected | reference
    # multi-turn (optional)
    multi_turn_context: list[str] = field(default_factory=list)
    # honesty constants
    synthetic: bool = True
    contains_real_pii: bool = False
    legal_review_status: str = LEGAL_REVIEW_DEFAULT
    # capability track only
    gold_answer: str = ""
    scored_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive(general: str, policy: str) -> tuple[int, str]:
    """Return (final_intervention_gold, quadrant) from the two authoritative labels."""
    g = 1 if general == "unsafe" else 0
    d = 1 if policy == "intervene" else 0
    return (1 if (g or d) else 0), f"G{g}D{d}"


def finalize(row: Row) -> Row:
    """Fill the derived fields in-place from the two policy labels. Idempotent."""
    row.final_intervention_gold, row.quadrant = derive(
        row.general_safety_gold, row.mortgage_policy_gold)
    return row


class SchemaError(ValueError):
    pass


def validate_row(row: Row) -> list[str]:
    """Return a list of human-readable violations (empty == valid)."""
    errs: list[str] = []

    def bad(cond: bool, msg: str) -> None:
        if cond:
            errs.append(msg)

    bad(row.track not in TRACKS, f"track {row.track!r} not in {TRACKS}")
    bad(row.split not in SPLITS, f"split {row.split!r} not in {SPLITS}")
    bad(row.stratum not in STRATA, f"stratum {row.stratum!r} not in {STRATA}")
    bad(row.trap_type not in TRAP_TYPES, f"trap_type {row.trap_type!r} unknown")
    bad(row.difficulty not in DIFFICULTY, f"difficulty {row.difficulty!r} unknown")
    bad(row.general_safety_gold not in GENERAL, f"G {row.general_safety_gold!r} invalid")
    bad(row.mortgage_policy_gold not in POLICY, f"D {row.mortgage_policy_gold!r} invalid")
    bad(row.action_gold not in ACTIONS, f"action_gold {row.action_gold!r} invalid")
    bad(row.severity not in SEVERITY, f"severity {row.severity!r} invalid")
    bad(not row.user_prompt.strip(), "empty user_prompt")
    bad(not row.rationale.strip(), "empty rationale")
    bad(row.contains_real_pii is not False, "contains_real_pii must be False")
    bad(row.synthetic is not True, "synthetic must be True")

    # derived-field consistency (the invariant that keeps labels honest)
    exp_final, exp_quad = derive(row.general_safety_gold, row.mortgage_policy_gold)
    bad(row.final_intervention_gold != exp_final,
        f"final_intervention_gold {row.final_intervention_gold} != derived {exp_final}")
    bad(row.quadrant != exp_quad, f"quadrant {row.quadrant} != derived {exp_quad}")

    # action lattice sanity: an intervention must not be labelled PASS, and benign must not BLOCK
    if row.final_intervention_gold == 1:
        bad(row.action_gold == "PASS", "intervention row cannot have action_gold=PASS")
    else:
        bad(row.action_gold == "BLOCK", "benign row cannot have action_gold=BLOCK")

    # a mortgage violation should cite at least one policy card
    if row.mortgage_policy_gold == "intervene":
        bad(not row.policy_context, "intervene row must cite >=1 policy_context card")

    # protected minimal-pair invariants
    if row.variant:
        bad(row.variant not in VARIANTS, f"variant {row.variant!r} invalid")
        bad(not row.pair_id, "variant row must have a pair_id")
        bad(not row.protected_attribute, "variant row must name a protected_attribute")
        bad(row.quadrant != "G0D0", "protected minimal pairs must both be G0D0")
        bad(row.action_gold != "PASS", "protected minimal pairs must both be action_gold=PASS")

    return errs
