"""Load + expose the frozen construct: policy cards (D01..D24) and the taxonomy.

The online judge reads `card.predicate` (the rubric text). The offline mock rubric keys on
`card.trigger_markers`. Nothing here authors labels — it only serves the human-frozen design.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from .store import ROOT

_CARDS_DIR = os.path.join(ROOT, "policy_cards")


@dataclass
class Card:
    id: str
    domain: str
    subcard: str
    predicate: str
    authority: str
    typical_quadrant: str
    default_action: str
    default_severity: str
    general_harm: bool
    trigger_markers: list[str]
    ask_templates: list[str]
    sme_signoff: bool = False
    contested: bool = False

    def rubric_text(self) -> str:
        """What the online judge sees: the predicate + authority, never the target label."""
        return (f"Policy card {self.id} ({self.domain}/{self.subcard}). "
                f"INTERVENE iff: {self.predicate} Authority (context only): {self.authority}.")


@dataclass
class Design:
    version: str
    cards: dict[str, Card]
    protected_attributes: list[str]
    roles: dict[str, Any]
    benign_asks: dict[str, list[str]]
    general_harm_asks: list[str]
    general_harm_markers: list[str]
    tactics: dict[str, Any]
    slot_banks: dict[str, Any]

    def cards_by_quadrant(self, quadrant: str) -> list[Card]:
        return [c for c in self.cards.values() if c.typical_quadrant == quadrant]

    def d1_cards(self, *, exclude_contested: bool = True) -> list[Card]:
        return [c for c in self.cards.values()
                if c.typical_quadrant.endswith("D1")
                and not (exclude_contested and c.contested)]

    def signoff_pending(self) -> list[str]:
        return [cid for cid, c in self.cards.items() if not c.sme_signoff]


def load_design() -> Design:
    with open(os.path.join(_CARDS_DIR, "cards.yaml")) as fh:
        cy = yaml.safe_load(fh)
    with open(os.path.join(_CARDS_DIR, "taxonomy.yaml")) as fh:
        tx = yaml.safe_load(fh)
    cards: dict[str, Card] = {}
    for c in cy["cards"]:
        cards[c["id"]] = Card(
            id=c["id"], domain=c["domain"], subcard=c["subcard"], predicate=c["predicate"],
            authority=c["authority"], typical_quadrant=c["typical_quadrant"],
            default_action=c["default_action"], default_severity=c["default_severity"],
            general_harm=bool(c.get("general_harm", False)),
            trigger_markers=list(c.get("trigger_markers", [])),
            ask_templates=list(c.get("ask_templates", [])),
            sme_signoff=bool(c.get("sme_signoff", False)),
            contested=bool(c.get("contested", False)))
    return Design(
        version=cy.get("version", ""), cards=cards,
        protected_attributes=list(cy.get("protected_attributes", [])),
        roles=tx.get("roles", {}), benign_asks=tx.get("benign_asks", {}),
        general_harm_asks=list(tx.get("general_harm_asks", [])),
        general_harm_markers=[m.lower() for m in tx.get("general_harm_markers", [])],
        tactics=tx.get("tactics", {}), slot_banks=tx.get("slot_banks", {}))
