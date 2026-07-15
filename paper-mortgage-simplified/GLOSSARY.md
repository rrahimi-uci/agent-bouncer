# Glossary — Mortgage guardrail benchmark

**Guard / guardrail.** A classifier that screens an incoming request as `safe`/`unsafe` (or
allow/intervene) *before* the system acts on it.

**HMDA.** Home Mortgage Disclosure Act data — the large public U.S. dataset of mortgage
applications (purpose, amount, income, LTV, DTI, applicant race/ethnicity/sex, approve/deny, denial
reasons). We ground scenarios in the **2022** snapshot. We use only coarse, banded, de-identified
fields — never a real person's full record.

**General-safety label (G).** Is the request harmful independent of any domain — a jailbreak,
prompt injection, abuse? This is what an ordinary safety guard is trained for. `G0` = safe,
`G1` = unsafe.

**Mortgage-policy label (D).** Would honoring the request violate a mortgage rule — fair lending
(disparate treatment, redlining, steering, discriminatory pricing, proxy discrimination),
ability-to-repay, disclosure integrity, UDAAP, fraud (income/occupancy/appraisal), or borrower
privacy? `D0` = allow, `D1` = intervene. Defined by 24 written **policy cards**.

**Quadrants (G×D).** The four combinations. **G0/D1** — *safe-looking but a mortgage violation* —
is the benchmark's payload: a general guard rates it safe, but it should be stopped. **G1/D0**
(general harm only) is currently **empty** in v1.

**Policy card.** A short, benchmark-local "intervene iff …" rule with a pointer to the governing
law/reg. Marked `sme_signoff:false` until a subject-matter expert reviews it.

**Protected (minimal) pair.** Two benign requests identical except the applicant's protected-class
token (e.g. race). A fair guard must score them the same; the gap is **Δ_context** (target ≈ 0).
This tests the *guard's invariance*, not a lending decision.

**macro-AP (macro Average Precision).** Threshold-free ranking quality (does the guard put
should-intervene requests above the fine ones?), computed per benchmark then averaged. Reported
separately for G, D, and the combined decision.

**Operating point / miss rate.** Pick an alarm threshold for a target false-alarm rate on
development data, then count how many G0/D1 violations the guard still misses.

**LLM-judge vs SME-adjudicated.** The labels here are assigned by a language-model **judge** using
the policy cards, with agreement measured as *self-consistency*. **SME adjudication** — human
experts reviewing a sample and reporting Fleiss-κ agreement — is required before any confirmatory
fair-lending claim, and has not been done.

**Frozen benchmark.** Because the LLM generation is random, we don't try to regenerate it; we
**publish the exact 994-row dataset** (checksummed) as the fixed artifact. Results reproduce by
*scoring guards on it*, not by rebuilding it.
