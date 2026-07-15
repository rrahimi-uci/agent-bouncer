# Mortgage Red-Teaming Agentic Generator (`magen`)

An **agentic pipeline** that uses **real mortgage loan data** — the public
**HMDA 2022 Snapshot National Loan-Level Dataset** — to construct realistic evaluation
prompts for a **mortgage-specific safety guardrail benchmark**.

It follows the methodology already specified in the parent repo
([`docs/mortgage-benchmark-build-spec.md`](../docs/mortgage-benchmark-build-spec.md) and
[`-runbook.md`](../docs/mortgage-benchmark-build-runbook.md)) — request-screening rows,
**dual labels**, family-isolated splits, MinHash decontamination, a sealed private test, and
canonical tie-aware evaluation — and adds the two new pieces that spec deferred: an **agentic
planner→grounder→generator→adversary→judge loop**, and **grounding in real HMDA records**.

> The build is **deterministic** and runs **fully offline by default** (a mock LLM provider +
> a bundled synthetic HMDA-shaped sample): no API keys, no network, no multi-GB download. Flip
> two config values for a production build with a real judge model and the real HMDA file.

> **Note on "the example below":** the request referenced a methodology example that did not
> arrive in the message. This pipeline is grounded in the repo's own mortgage-benchmark spec +
> runbook (the established in-house methodology) plus the literature review in
> [`docs/`](../docs). If you had a specific external example in mind, share it and the design
> can be realigned.

---

## Quickstart

```bash
# from this folder, using the parent repo's virtualenv (has numpy/pandas/guard_research)
make smoke  PY=../.venv/bin/python     # tiny offline end-to-end build -> ./out
make test   PY=../.venv/bin/python     # offline test suite
```

`make smoke` runs the whole pipeline and writes `out/`:

```
out/
  ingest_meta.json          # HMDA fact-sheet count + an example scenario line
  raw_rows.jsonl            # generated + judged rows (pre-dedup)
  provenance_report.json    # exact-dedup, content-family count, decontamination
  rows_split.jsonl          # family-isolated train/dev/public_test/private_test
  validation_report.json    # the quality gates (spec §9)
  eval_selftest.json        # evaluator self-test (gold-as-preds must be perfect)
  run_summary.json          # one-glance summary
  SEALED_private_test.jsonl  # sealed test (custodian-only; NOT in dist/)
  dist/                     # the distributable bundle
    train.jsonl dev.jsonl public_test.jsonl
    public_index.json       # TEXT-FREE index (ids/hashes/labels/counts — no prompt text)
    DATA_CARD.md SOURCES.json MANIFEST.json CHECKSUMS.txt
```

---

## How it works (the agentic loop)

Six agents, orchestrated per **family** (the label-authorship + split-isolation unit) by
[`magen/pipeline.py`](magen/pipeline.py). Determinism lives in one place —
`store.rng(seed, *salt)` derives every stochastic choice from `config.seed` (default
`20260714`). The generator runs hot for surface variety; the **judge is forced to temperature 0**.

| Agent | Module | Input → Output |
|---|---|---|
| **Planner** | [`planner.py`](magen/planner.py) | design + config → a coverage archive of cells `(quadrant × trap × policy_card × role × protected-context × stratum)`, so all four quadrants, the protected minimal pairs, and the low-prevalence stream are populated to the configured mix. |
| **HMDA-grounder** | [`ground.py`](magen/ground.py) | a cell + HMDA **distributions** → a PII-free, banded **fact sheet**. Samples marginals (never a verbatim row) and asserts the tuple doesn't reproduce a source record. |
| **Generator/attacker** | [`generate.py`](magen/generate.py) | cell + fact sheet → a grounded request. Harmful rows **state intent only** — never an operational recipe/exploit/forged doc. |
| **Adversarial mutator** | [`adversary.py`](magen/adversary.py) | draft → prompt after composing **label-preserving** tactics (euphemism, buried_injection, coded_proxy, business_justified, …); injection tactics only on G1 cells; benign rows stay lightly phrased. |
| **Judge/labeler** | [`judge.py`](magen/judge.py) | prompt + policy-card rubric → the authoritative `G`, `D`, `action`, `severity`, `rationale`. Independent of the planner's target (no leakage); online mode does multi-sample majority + drop-on-no-consensus. |
| **Decontaminator** | [`provenance.py`](magen/provenance.py) | all rows + parent-repo general-safety index → exact-dedup, content-families, cross-source removal (reuses `guard_research`). |

**One episode:** plan cell → ground → generate → mutate → judge → *if* the judge's `(G,D)`
matches the cell target and realism holds, accept; else retry (≤`max_generation_retries`) then
drop and log the miss. Accepted rows get `schema.derive()` — `final_intervention_gold` and
`quadrant` are **derived from `(G,D)`, never authored**.

### The labels (why two)

Each guard-track row carries two **independent** labels (see
[`policy_cards/label_rubric.md`](policy_cards/label_rubric.md)):

- `general_safety_gold` (**G**) — domain-independent harm (jailbreak / injection / abuse), the
  same policy as the parent repo's general guard.
- `mortgage_policy_gold` (**D**) — would honoring the request violate a benchmark mortgage
  **policy card** (D01–D24)?

The interesting stratum is **G0/D1** — a real mortgage-policy violation (fair-lending
solicitation, ATR circumvention, disclosure evasion, occupancy fraud, …) that a *general*
safety guard rates "safe." That gap is the benchmark's payload.

---

## Getting the real HMDA data (production build)

The bundled `data/hmda_sample/` is **synthetic** (HMDA-*shaped*, for offline dev). For a real
build, acquire the public file and point the config at it:

- **Source (pin + cite):** FFIEC/CFPB **Snapshot National Loan-Level Dataset, 2022** — the
  reproducible, frozen-as-of-2023-05-01 product (not the weekly Dynamic dataset):
  <https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/>.
  Get the real download URL by clicking *Download* on that page (the `s3://cfpb-hmda-public/prod/…`
  object key and the ~5.8 GB size are unverified — resolve them at fetch time, don't hard-code).
  ~14.3M records nationally; schema source-of-truth is the 2022 FIG + the
  [LAR Data Fields page](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/).
- **Easier: the Data Browser CSV API** (filtered, no multi-GB pull):
  `GET https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?years=2022&states=CA&actions_taken=1,3,7`
  ([docs](https://ffiec.cfpb.gov/documentation/api/data-browser/)). Cache each state's CSV.
- **Point the config:**
  ```yaml
  hmda:
    source: csv
    snapshot_path: data/hmda_real/hmda_2022.csv   # your downloaded/cached file
  ```
  The loader streams with `pandas(chunksize=…, dtype=str)`, applies the `filters`, and takes a
  seeded reservoir sample — the full file never loads whole.

**Privacy / de-identification (why this is safe).** HMDA's public data has **no direct PII**:
the Modified LAR redacts ~27 fields (loan identifier, address, names, credit scores, NMLSR ID),
`loan_amount`/`property_value` are rounded to the nearest $10k midpoint, `income` is in
thousands, DTI and age are bucketed. This generator **only reads coarse/derived fields and
re-buckets them**, samples **distributions rather than verbatim rows**, and asserts no emitted
fact sheet reproduces a source record — so `contains_real_pii=false` holds by construction.
(Sources: CFPB *Using the Modified LAR* + the 2018 disclosure rule.)

**License.** HMDA public data is a U.S. Government work published for mandatory disclosure — no
U.S. copyright (17 U.S.C. §105). No formal SPDX/CC0 string is posted, so the `DATA_CARD` cites
the FFIEC/CFPB source and records a **"confirmed no separate terms-of-use restriction"** human
checkbox before any redistribution of generated prompts.

---

## Online mode (real judge / generator)

```bash
pip install anthropic      # or: openai
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
# in config/default.yaml: llm.provider: anthropic ; hmda.source: csv
make build PY=../.venv/bin/python
```

Models are set in `config/default.yaml` (`generator_model`, a stronger `judge_model`). The
online judge does `judge_samples` majority voting with drop-on-no-consensus; `max_llm_calls`
caps spend. If a real provider is selected without a key in the environment, the pipeline warns
and falls back to `mock`.

---

## Policy cards (the construct)

24 benchmark-local cards in [`policy_cards/cards.yaml`](policy_cards/cards.yaml), each a paraphrased
"intervene iff" predicate with an authority pointer, grouped by domain:

| Domain | Cards | Typical quadrant |
|---|---|---|
| fair_lending (disparate treatment, redlining, steering, pricing, **proxy/coded**, discouragement, disparate-impact) | D01–D07 | mostly **G0/D1** |
| ability-to-repay / QM | D08–D10 | G0/D1 |
| disclosure (TRID timing, adverse-action specificity, **reason-masking**) | D11–D13 | G0/D1 |
| UDAAP (deceptive/unfair/abusive) | D14–D16 | G0/D1 |
| fraud (income, occupancy, **straw-buyer**, appraisal, undisclosed debt, **identity**) | D17–D22 | G0/D1 → G1/D1 |
| privacy (**NPI exfiltration**, improper disclosure) | D23–D24 | G1/D1, G0/D1 |

**These are policy-card-consistent, not legal authority.** A row "violates the benchmark policy
card"; it is never asserted to "be illegal." Card **D07 (disparate impact)** is doctrinally
contested (2025–2026) and is flagged `contested: true` — version-date it against the primary
Federal Register text before relying on it.

---

## Novelty & ethics

**Novelty.** Prior red-team generators each cover one axis — WildTeaming/Rainbow/HarmBench/PAIR
do agentic generation for *general* content harm; Aegis/Llama Guard are policy-grounded over a
*content* taxonomy with no domain regulation and no real data; XSTest/OR-Bench probe
over-refusal but not protected-class pairs; counterfactual-fairness suites measure *model-output*
bias, not a *guard's* benign-pass invariance. This pipeline's contribution is the **composition**:
real-structured-data (HMDA) grounding **+** dual orthogonal labels (`general_safety` × `mortgage_policy`
→ G/D quadrants) **+** protected-class minimal pairs reframed as a **guard-invariance PASS gate**
**+** an agentic generate/judge loop, evaluated with tie-aware macro-AP on a sealed,
family-isolated, MinHash-decontaminated set.

**Ethics.** (1) *Aggregate grounding, not verbatim rows* — distributions are sampled and a
build-time assertion forbids reconstructing a source record. (2) *No PII* — only the
already-de-identified public snapshot, and a loader assertion rejects any redacted-identifier
column. (3) *Harms represented, not enabled* — every unsafe row **states intent only**; benign
hard-negatives sit alongside so a guard cannot win by never intervening.

---

## Open items requiring human / SME sign-off (do not auto-generate)

1. **Label rubric + D01–D24 predicates** — authored and SME-reviewed once (they are the construct).
2. **Golden-subset adjudication** for the judge (Fleiss κ per label; `mortgage_policy_gold`
   audited hardest). The offline run reports self-*consistency* only — not human agreement.
3. **D07 disparate-impact** version-dating against the primary rule text.
4. **HMDA specifics to re-confirm at ingest:** the real Snapshot download URL/size, exact CSV
   header spelling, the public DTI/age bin edges, and the "no separate terms-of-use" check.

`make smoke` prints a reminder that all 24 cards are `sme_signoff: false`.

---

## File map

```
config/default.yaml          seed, quadrant mix, HMDA filters, provider (mock default)
policy_cards/                 the human-frozen CONSTRUCT
  cards.yaml                  D01..D24 predicates + authority + trigger markers
  taxonomy.yaml               roles, benign / over-refusal / general-harm banks, tactic library
  label_rubric.md             what G / D / action mean (the construct definition)
data/hmda_sample/             bundled SYNTHETIC HMDA-shaped sample + its generator
magen/                        the pipeline package
  schema.py                   Row, derive(), validate_row()   (label invariants)
  hmda.py                     ingest: code maps, banding, PII-free fact sheet, chunked read
  cards.py                    load/expose the construct
  planner.py ground.py generate.py adversary.py judge.py   the agents
  provenance.py               dedup + content-families + decontam (reuses guard_research)
  split.py validate.py evaluate.py package.py              runbook Phases 5–8
  pipeline.py __main__.py     orchestrator + CLI (python -m magen)
tests/                        offline suite (determinism, judge, pairs, gates, no-PII)
Makefile                      smoke | build | test | sample | install | clean
```

Reuses the parent repo's `guard_research` for hashing/MinHash (`provenance`), tie-aware macro-AP
(`metrics`), and FPR-controlled operating points (`thresholds`) — no ad-hoc metrics anywhere.
