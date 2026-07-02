"""Core domain contracts: the typed ``Verdict``, the hazard taxonomy, and the ``Guard``
protocol. Dependency-light and imported everywhere — nothing here pulls in torch/datasets."""

from .guard import Guard, KeywordGuard
from .schema import Decision, Surface, Verdict
from .taxonomy import HAZARD_CATEGORIES, Hazard

__all__ = [
    "Decision",
    "Surface",
    "Verdict",
    "Hazard",
    "HAZARD_CATEGORIES",
    "Guard",
    "KeywordGuard",
]
