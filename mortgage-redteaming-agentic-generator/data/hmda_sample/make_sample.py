#!/usr/bin/env python
"""Emit a tiny, fully SYNTHETIC HMDA-2022-shaped CSV for offline testing.

This is NOT real HMDA data. It reproduces the public snapshot's COLUMN NAMES and CODED
VALUES so the loader (magen/hmda.py) exercises the same decode path it would on the real
file, but every value here is randomly generated and represents no real person or loan.

For a real build, download the FFIEC/CFPB HMDA 2022 Snapshot National Loan-Level Dataset and
point config `hmda.snapshot_path` at it (see the folder README). Deterministic: seed fixed.
"""
import csv
import os
import random

COLUMNS = [
    "activity_year", "state_code", "action_taken", "loan_type", "loan_purpose",
    "occupancy_type", "lien_status", "preapproval", "hoepa_status",
    "loan_amount", "property_value", "income", "loan_to_value_ratio",
    "debt_to_income_ratio", "interest_rate", "rate_spread",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
    "business_or_commercial_purpose",
]

STATES = ["CA", "TX", "NY", "GA", "IL"]
RACES = ["White", "Black or African American", "Asian",
         "American Indian or Alaska Native",
         "Native Hawaiian or Other Pacific Islander", "2 or more minority races",
         "Joint", "Race Not Available"]
ETHN = ["Hispanic or Latino", "Not Hispanic or Latino", "Joint", "Ethnicity Not Available"]
SEX = ["Male", "Female", "Joint", "Sex Not Available"]
AGE = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "8888"]


def make(n: int = 80, seed: int = 20260714) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        action = rng.choices([1, 3, 7], weights=[6, 3, 1])[0]
        denied = action in (3, 7)
        purpose = rng.choice([1, 1, 1, 31, 32, 2])
        loan_amt = rng.choice([85000, 155000, 225000, 315000, 405000, 615000, 1005000])
        prop_val = int(loan_amt * rng.uniform(1.05, 1.6))
        income = rng.choice([35, 55, 85, 110, 150, 240])          # $thousands
        ltv = rng.choice([65, 78, 85, 92, 96, 98])
        dti = rng.choice(["<20%", "28", "36", "43", "50%-60%", ">60%"])
        rows.append({
            "activity_year": 2022,
            "state_code": rng.choice(STATES),
            "action_taken": action,
            "loan_type": rng.choice([1, 1, 2, 3, 4]),
            "loan_purpose": purpose,
            "occupancy_type": rng.choice([1, 1, 1, 2, 3]),
            "lien_status": rng.choice([1, 1, 2]),
            "preapproval": rng.choice([1, 2]),
            "hoepa_status": rng.choice([2, 2, 2, 1, 3]),
            "loan_amount": loan_amt,
            "property_value": prop_val,
            "income": income,
            "loan_to_value_ratio": ltv,
            "debt_to_income_ratio": dti,
            "interest_rate": round(rng.uniform(3.5, 7.9), 3),
            "rate_spread": round(rng.uniform(-0.5, 3.0), 3),
            "denial_reason-1": rng.choice([1, 3, 4, 5]) if denied else 10,
            "denial_reason-2": rng.choice([2, 6, 10]) if denied else 10,
            "denial_reason-3": 10,
            "denial_reason-4": 10,
            "derived_race": rng.choice(RACES),
            "derived_ethnicity": rng.choice(ETHN),
            "derived_sex": rng.choice(SEX),
            "applicant_age": rng.choice(AGE),
            "business_or_commercial_purpose": 2,
        })
    return rows


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "hmda_2022_sample.csv")
    rows = make()
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic HMDA-shaped rows -> {out}")


if __name__ == "__main__":
    main()
