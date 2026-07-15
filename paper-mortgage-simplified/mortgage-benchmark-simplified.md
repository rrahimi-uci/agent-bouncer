# A Mortgage Safety-Guardrail Benchmark — the plain-language version

*Why a general "is this prompt unsafe?" guard isn't enough for mortgage lending, and a benchmark
built from real loan data to prove it.* Reza Rahimi (JazzX AI). Plain-language companion to the
formal draft.

> **Read this first.** The benchmark's prompts are **made-up (synthetic)** and its right-answers
> were assigned by **an AI judge, not a human expert**. So it's good for *measuring how a guard
> behaves* on mortgage-style requests — it is **not** proof of discrimination by any real lender or
> model. Think "test set for guards," not "fair-lending audit."

---

## The idea in one paragraph

A normal safety guard asks *"is this prompt harmful?"* — jailbreaks, abuse, injections. But in
mortgage lending, a request can be totally polite and still ask the assistant to **break the law**:
to price by neighborhood demographics, to hide the real reason for a denial, to fudge income so a
loan qualifies. A general guard says "looks safe" and waves it through. We built a benchmark of
such requests, **grounded in real HMDA loan data**, where each request has **two labels**: *is it
generally harmful?* (G) and *would honoring it break mortgage policy?* (D). The interesting cases
are **"safe-looking but a mortgage violation"** — and that's exactly what general guards miss.

---

> **Background: the words you need.**
> - **Guard.** A small model that scores a request as safe/unsafe.
> - **HMDA.** Public U.S. mortgage data (loan purpose, amounts, income, race/ethnicity of
>   applicants, approve/deny, denial reasons). We use it to make the scenarios *realistic*.
> - **Two labels, G and D.** **G** = generally harmful (jailbreak/abuse) — a normal guard's job.
>   **D** = would this break a mortgage rule (fair lending, ability-to-repay, disclosures, fraud,
>   privacy). A request can be G-safe but D-violating.
> - **Quadrants.** Combining G and D gives four boxes. The key one is **G0/D1**: *not* generally
>   harmful, *but* a mortgage-policy violation.
> - **macro-AP.** A score (0–1) for how well a guard ranks the "should-intervene" requests above
>   the fine ones. Higher = better.

---

## How the benchmark is built

- **Grounded in real loans.** We take real (de-identified, banded) HMDA records — e.g. "FHA
  cash-out refi, investment property, income under $40k, LTV over 97%, prior denial for
  collateral" — and write realistic requests around them. We never copy a real person's record;
  we sample *distributions* and check nothing traces to one loan.
- **Hard on purpose.** Following a 2024 study of LLM mortgage bias, we focus on **borderline,
  higher-risk files** (where bias and discretion actually happen) and make violations **subtle** —
  "just use your judgment," coded references to a neighborhood — not cartoonish.
- **Made by a team of AI agents.** A planner picks what to cover; a "grounder" pulls a loan fact
  sheet; a "generator" writes the request; an "adversary" adds realistic disguises; a
  **rubric-bound judge** assigns the labels from 24 written policy cards.
- **Fairness pairs.** For some benign requests we write **two versions identical except one word**
  — the applicant's protected group (e.g. "a Black applicant" vs "a white applicant"). A fair guard
  must treat both the same. We measure the gap (should be ≈ 0).

## What's in it (994 requests)

| | |
|---|---|
| **G0/D1** (safe-looking mortgage violation — the payload) | **502** |
| **G0/D0** (benign, incl. hard "don't over-block me" cases) | **450** |
| **G1/D1** (harmful *and* a violation) | **42** |
| **G1/D0** (general harm only) | **0** *(gap — see below)* |
| Domains: fair-lending 204, fraud 112, UDAAP 90, disclosure 66, ability-to-repay 54, privacy 18 | |
| Protected fairness pairs | 39 |
| Splits: train 604 / dev 149 / public-test 146 / sealed test 95 | |

## What we test, and what we expect to see

We run guards over the benchmark and score them with the same tie-aware metrics as the companion
Paper A. A guard gives one "unsafe" number per request; we check how well it separates the
should-intervene requests for **G**, for **D**, and for the combined decision, plus **how many
G0/D1 violations it misses** at a sensible alarm threshold, plus the **fairness-pair gap**.

The expected result (already confirmed by the offline plumbing test): a general guard **does okay
on G but falls apart on D** — it misses almost all the safe-looking mortgage violations. That's the
whole point: *general safety ≠ mortgage compliance.*

<!-- BASELINE_TABLE_START -->
*(Baseline numbers for the base checkpoints + off-the-shelf guards are filled in from the committed
scoring run; the harness is verified end-to-end.)*
<!-- BASELINE_TABLE_END -->

## Is this real? (the honesty box)

- **The labels are an AI judge's, not an expert's.** Every policy card is marked
  "not-counsel-reviewed." Before anyone makes a fair-lending *claim*, a human expert has to review
  a sample and we report the human-agreement number. This benchmark is a measuring stick, not a
  verdict.
- **One box is empty** (G1/D0) — the AI wouldn't write plain jailbreaks — so the 2×2 is only
  three-quarters filled.
- **You can't regenerate it** (the AI writing step is random), so we **freeze and publish the exact
  dataset**; results are reproduced by *scoring guards on the frozen data*, not by rebuilding it.
- **No domain-trained guard yet** — the natural next step is to fine-tune a mortgage guard and show
  it catches G0/D1 (at some cost to general skill), tying back to the "compose, don't tune" idea.

## What it means

General guards and mortgage compliance are **different problems**. This benchmark makes that
measurable, with realistic, hard, real-data-grounded requests and a fairness-invariance test. It's
an honest first artifact with a clear checklist to become a validated instrument: expert
adjudication, fill the empty quadrant, decontaminate, and add a fine-tuned mortgage guard.

---

*Frozen benchmark: `mortgage-redteaming-agentic-generator/benchmark/v1_hmda2022/` (994 rows,
checksummed, verified PII-free). Reproduce baselines with `magen/score_guards.py`.*
