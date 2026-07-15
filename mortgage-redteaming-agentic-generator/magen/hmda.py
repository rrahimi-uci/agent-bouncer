"""HMDA 2022 loan-level ingestion + PII-safe grounding.

Turns records from the public FFIEC/CFPB **HMDA 2022 Snapshot National Loan-Level Dataset**
into small, de-identified `FactSheet`s that the agents use to ground realistic mortgage
scenarios. Design choices (see README "Ethics"):

  * We ground on **coarse/derived fields only** (loan purpose, occupancy, action, denial
    reason, *banded* loan size / income / LTV / DTI, derived demographics, state) — never the
    exact dollar amounts or census tract. HMDA's public file is already de-identified, and we
    bucket further so no scenario can be traced to an individual.
  * `derived_race`/`derived_ethnicity`/`derived_sex` are used **only** to build the abstract
    "protected context" of counterfactual minimal pairs — never to justify differential
    treatment inside a benign scenario.

Offline mode reads the bundled synthetic HMDA-shaped sample; a real build points
`hmda.snapshot_path` at the downloaded CSV. See README "Getting the real HMDA data".
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

# ------------------------------------------------------------------ HMDA public code maps
# Standard HMDA data documentation codes (FFIEC/CFPB Filing Instructions Guide). Verify
# against the dataset's own documentation/data-dictionary for the release year in use.
ACTION_TAKEN = {
    1: "loan originated", 2: "approved but not accepted", 3: "application denied",
    4: "withdrawn by applicant", 5: "closed for incompleteness", 6: "purchased loan",
    7: "preapproval request denied", 8: "preapproval approved but not accepted",
}
LOAN_TYPE = {1: "conventional", 2: "FHA", 3: "VA", 4: "USDA/RHS"}
LOAN_PURPOSE = {1: "home purchase", 2: "home improvement", 31: "refinancing",
                32: "cash-out refinancing", 4: "other purpose", 5: "n/a"}
OCCUPANCY = {1: "principal residence", 2: "second residence", 3: "investment property"}
LIEN_STATUS = {1: "first lien", 2: "subordinate lien"}
PREAPPROVAL = {1: "preapproval requested", 2: "preapproval not requested"}
HOEPA = {1: "high-cost mortgage", 2: "not high-cost", 3: "n/a"}
DENIAL_REASON = {
    1: "debt-to-income ratio", 2: "employment history", 3: "credit history",
    4: "collateral", 5: "insufficient cash (downpayment/closing)", 6: "unverifiable information",
    7: "credit application incomplete", 8: "mortgage insurance denied", 9: "other", 10: "n/a",
}

# Columns we read from the snapshot (a small, fair-lending-relevant subset of the ~99 columns).
USED_COLUMNS = [
    "activity_year", "state_code", "action_taken", "loan_type", "loan_purpose",
    "occupancy_type", "lien_status", "preapproval", "hoepa_status",
    "loan_amount", "property_value", "income", "loan_to_value_ratio",
    "debt_to_income_ratio", "interest_rate", "rate_spread",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
    "business_or_commercial_purpose",
]


# ------------------------------------------------------------------ banding (extra de-ID)
def _num(v: Any) -> float | None:
    try:
        if v in (None, "", "NA", "Exempt", "1111"):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def loan_amount_band(v: Any) -> str:
    x = _num(v)
    if x is None:
        return "unknown"
    for hi, lab in [(1e5, "<$100k"), (2.5e5, "$100k-$250k"), (5e5, "$250k-$500k"),
                    (1e6, "$500k-$1M")]:
        if x < hi:
            return lab
    return ">$1M"


def income_band(v: Any) -> str:  # HMDA income is in $thousands
    x = _num(v)
    if x is None or x < 0:
        return "unknown"
    for hi, lab in [(40, "<$40k"), (75, "$40k-$75k"), (120, "$75k-$120k"),
                    (200, "$120k-$200k")]:
        if x < hi:
            return lab
    return ">$200k"


def ltv_band(v: Any) -> str:
    x = _num(v)
    if x is None:
        return "unknown"
    for hi, lab in [(80.0001, "<=80%"), (90, "80-90%"), (95, "90-95%"), (97.0001, "95-97%")]:
        if x < hi:
            return lab
    return ">97%"


def dti_band(v: Any) -> str:
    # HMDA DTI may be numeric or a string range ("36", "43", "50%-60%", ">60%", "<20%").
    if isinstance(v, str) and "%" in v:
        return v
    x = _num(v)
    if x is None:
        return "unknown"
    for hi, lab in [(36.0001, "<=36%"), (43.0001, "36-43%"), (50.0001, "43-50%")]:
        if x < hi:
            return lab
    return ">50%"


@dataclass
class FactSheet:
    """A PII-free, banded summary of one HMDA record used to ground a scenario."""
    record_id: str                       # synthetic index, not a real loan id
    state: str
    action: str                          # decoded action_taken
    loan_type: str
    loan_purpose: str
    occupancy: str
    lien: str
    hoepa: str
    loan_amount_band: str
    income_band: str
    ltv_band: str
    dti_band: str
    denial_reasons: list[str]            # decoded, only for denied actions
    # protected context (used ONLY for minimal pairs / representing bias to be detected)
    derived_race: str
    derived_ethnicity: str
    derived_sex: str
    applicant_age: str
    was_denied: bool = False
    raw_used: dict[str, Any] = field(default_factory=dict)  # decoded, banded — no exact PII

    def scenario_line(self) -> str:
        """A neutral one-line situation description safe to embed in prompts."""
        bits = [f"{self.loan_purpose} {self.loan_type} loan",
                f"on a {self.occupancy}",
                f"amount {self.loan_amount_band}",
                f"applicant income {self.income_band}",
                f"LTV {self.ltv_band}", f"DTI {self.dti_band}",
                f"in {self.state}"]
        s = "; ".join(bits) + f"; outcome: {self.action}"
        if self.was_denied and self.denial_reasons:
            s += f" (reasons: {', '.join(self.denial_reasons)})"
        return s


def to_factsheet(rec: dict[str, Any], record_id: str) -> FactSheet:
    def code(colmap, key):
        try:
            return colmap.get(int(float(rec.get(key))), str(rec.get(key)))
        except (TypeError, ValueError):
            return "unknown"

    denials = []
    action_i = None
    try:
        action_i = int(float(rec.get("action_taken")))
    except (TypeError, ValueError):
        pass
    denied = action_i in (3, 7)
    if denied:
        for k in ("denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4"):
            try:
                di = int(float(rec.get(k)))
            except (TypeError, ValueError):
                continue
            if di in DENIAL_REASON and di != 10:
                denials.append(DENIAL_REASON[di])

    fs = FactSheet(
        record_id=record_id,
        state=str(rec.get("state_code", "??")),
        action=code(ACTION_TAKEN, "action_taken"),
        loan_type=code(LOAN_TYPE, "loan_type"),
        loan_purpose=code(LOAN_PURPOSE, "loan_purpose"),
        occupancy=code(OCCUPANCY, "occupancy_type"),
        lien=code(LIEN_STATUS, "lien_status"),
        hoepa=code(HOEPA, "hoepa_status"),
        loan_amount_band=loan_amount_band(rec.get("loan_amount")),
        income_band=income_band(rec.get("income")),
        ltv_band=ltv_band(rec.get("loan_to_value_ratio")),
        dti_band=dti_band(rec.get("debt_to_income_ratio")),
        denial_reasons=denials,
        derived_race=str(rec.get("derived_race", "Race Not Available")),
        derived_ethnicity=str(rec.get("derived_ethnicity", "Ethnicity Not Available")),
        derived_sex=str(rec.get("derived_sex", "Sex Not Available")),
        applicant_age=str(rec.get("applicant_age", "8888")),
        was_denied=denied,
    )
    fs.raw_used = {"loan_purpose": fs.loan_purpose, "occupancy": fs.occupancy,
                   "action": fs.action, "denial_reasons": fs.denial_reasons,
                   "loan_amount_band": fs.loan_amount_band, "income_band": fs.income_band,
                   "ltv_band": fs.ltv_band, "dti_band": fs.dti_band, "state": fs.state}
    return fs


# ------------------------------------------------------------------ ingestion
def _passes_filters(rec: dict[str, Any], filt: dict[str, Any]) -> bool:
    st = filt.get("states")
    if st and str(rec.get("state_code")) not in set(st):
        return False
    acts = filt.get("actions")
    if acts:
        try:
            if int(float(rec.get("action_taken"))) not in set(acts):
                return False
        except (TypeError, ValueError):
            return False
    bcp = filt.get("business_or_commercial_purpose")
    if bcp:
        try:
            if int(float(rec.get("business_or_commercial_purpose"))) not in set(bcp):
                return False
        except (TypeError, ValueError):
            pass
    return True


def load_records(cfg: dict[str, Any], root: str, rng) -> list[FactSheet]:
    """Load + filter + seeded-sample HMDA records into FactSheets.

    `cfg` is the `hmda:` block. Reads the bundled sample or a real CSV (streamed in chunks so
    the multi-GB snapshot never loads whole). Returns FactSheets, not raw records.
    """
    path = cfg.get("snapshot_path")
    path = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HMDA source not found at {path}. In offline mode use the bundled sample "
            f"(config hmda.source: bundled_sample); for a real build download the FFIEC/CFPB "
            f"HMDA 2022 Snapshot and set hmda.snapshot_path (see README).")

    filt = cfg.get("filters", {}) or {}
    cap = int(cfg.get("sample_rows", 5000))
    chunksize = int(cfg.get("chunksize", 200000))

    kept: list[dict[str, Any]] = []
    # reservoir sample over a streamed read, so the real file need not fit in memory
    reservoir_rng = rng
    seen = 0

    def consider(rec: dict[str, Any]) -> None:
        nonlocal seen
        if not _passes_filters(rec, filt):
            return
        seen += 1
        if len(kept) < cap:
            kept.append(rec)
        else:
            j = reservoir_rng.randint(0, seen - 1)
            if j < cap:
                kept[j] = rec

    # pandas streaming if available; else the csv module
    try:
        import pandas as pd
        usecols = None  # tolerate schema vari/absent cols; select after read
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str,
                                 low_memory=False, usecols=usecols):
            for rec in chunk.to_dict("records"):
                consider(rec)
    except ImportError:  # pragma: no cover
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                consider(rec)

    return [to_factsheet(rec, f"HMDA-{i:06d}") for i, rec in enumerate(kept)]
