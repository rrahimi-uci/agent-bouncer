./

# Architecture

Agent Bouncer sits **on the request path**, in front of your LLM/agent. It runs on *every*
call (and every agent step), so it must be small and fast — which is exactly why a
**small language model (SLM)** is the right tool, not a compromise. This document explains
how the whole system fits together, layer by layer, with diagrams.

---

## 1 · Where it runs

The bouncer screens input **before** it reaches the model, and (optionally) screens what the
model is about to do or return.

```mermaid
flowchart LR
    U["User / upstream agent"] -->|"prompt · tool_call · agent_output"| G

    subgraph G["🕶️ Agent Bouncer (tiny guard, target <30 ms CPU)"]
        direction TB
        H["Verdict = decision + hazard + score"]
    end

    G -->|"decision = safe → ALLOW"| M["LLM / Agent"]
    G -->|"decision = unsafe → BLOCK"| R["Safe refusal"]
    M -.->|"agent_output re-screened"| G
    G -.->|"metrics · latency (optional MLflow)"| O[("mlruns/")]
```

**Surfaces** (`agent_bouncer.core.schema.Surface`) — guarding *agents* means screening more than
user text:

| Surface          | What it protects against                                           |
| ---------------- | ------------------------------------------------------------------ |
| `user_prompt`  | content harm, prompt injection, jailbreaks in the incoming request |
| `tool_call`    | a dangerous action the agent is about to execute                   |
| `agent_output` | unsafe content the agent is about to return                        |

---

## 2 · The core contract: one `Verdict`

Every guard — heuristic, encoder, RL-tuned decoder, OpenAI, or an incumbent — returns the
**same typed object**, so the eval harness, serving layer, and reward functions all speak one
language. That single contract is what makes the whole scoreboard apples-to-apples.

```mermaid
classDiagram
    class Guard {
        <<protocol>>
        +str name
        +predict(text, surface) Verdict
    }
    class Verdict {
        +Decision decision
        +Hazard hazard
        +float score
        +str rationale
        +Surface surface
        +float latency_ms
        +blocked() bool
    }
    class Decision {
        <<enum>>
        safe
        unsafe
    }
    class Hazard {
        <<enum>>
        none
        violent_crimes
        hate
        prompt_injection
        jailbreak
    }
    Guard ..> Verdict : returns
    Verdict --> Decision
    Verdict --> Hazard
    KeywordGuard ..|> Guard
    EncoderGuard ..|> Guard
    DecoderGuard ..|> Guard
    OpenAIChatGuard ..|> Guard
```

- **`taxonomy.Hazard`** — one canonical label space (MLCommons-aligned content hazards +
  two agent-specific surfaces: `prompt_injection`, `jailbreak`).
- **`schema.Verdict`** — the I/O contract (`decision`, `hazard`, `score`, `rationale`, …).
- **`guard.Guard`** — a `Protocol`; anything with `.name` + `.predict()` is a guard.

---

## 3 · Repository map

```mermaid
flowchart TD
    subgraph C["core/ — contracts (dependency-light)"]
        TAX["core/taxonomy.py — Hazard"]
        SCH["core/schema.py — Decision·Surface·Verdict"]
        GRD["core/guard.py — Guard protocol + KeywordGuard"]
    end
    subgraph CFG["config/"]
        ENV["config/envfile.py — .env / settings"]
    end
    subgraph D["data/"]
        DAT["data/loaders.py — normalizers + loaders → unified JSONL"]
        SPL["data/split.py — disjoint splits + anti-leakage"]
        TS["data/training_sets.py — strategy-built train/test sets"]
    end
    subgraph MDL["models/ — guards"]
        ENC["models/encoder.py — EncoderGuard"]
        DEC["models/decoder.py — DecoderGuard + prompt/parse"]
        ENS["models/ensemble.py — EnsembleGuard + combine()"]
        REG["models/registry.py — base-model catalog"]
    end
    subgraph TR["training/"]
        SFT["training/sft.py — encoder + decoder LoRA"]
        GRPO["training/grpo.py — RLVR"]
        DPO["training/dpo.py — preference tuning"]
        REW["training/rewards.py — verifiable GRPO reward"]
        RUN["training/runner.py — train→version→record"]
    end
    subgraph EV["evaluation/"]
        MET["evaluation/metrics.py — P/R/F1 · FPR@benign · latency"]
        HAR["evaluation/harness.py — evaluate()"]
        BEN["evaluation/benchmarks.py — registry + runner"]
        OAI["evaluation/openai_guards.py — Moderation · GPT-4o-mini · GPT-5.2"]
        BAS["evaluation/baselines.py — Llama Guard · ShieldGemma · PromptGuard"]
        CUR["evaluation/curves.py — ROC · PR · AUC"]
        REP["evaluation/report.py — tables · model card"]
    end
    subgraph TRK["tracking/"]
        EXP["tracking/experiments.py — experiment store + versioning"]
        HW["tracking/hardware.py — cross-OS hardware snapshot"]
    end
    subgraph SV["serving/"]
        API["serving/api.py — FastAPI /screen + Studio"]
        DASH["serving/dashboard.html — Benchmark Studio UI"]
    end

    SCH --> GRD --> MET
    TAX --> SCH
    ENV --> DAT
    DAT --> HAR
    SPL --> TS --> SFT
    GRD --> ENC & DEC
    REW --> GRPO
    SFT --> ENC & DEC
    GRPO --> DEC
    DPO --> DEC
    REG --> RUN --> EXP
    HW --> EXP
    ENC & DEC --> ENS
    ENC & DEC & ENS & OAI & BAS --> HAR --> BEN --> REP
    BEN --> CUR
    HAR --> API --> DASH
```

| Module                   | Role                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `core/taxonomy`        | single hazard label space (content + injection/jailbreak)                                       |
| `core/schema`          | the`Verdict` contract every guard returns                                                     |
| `core/guard`           | `Guard` protocol + dependency-free reference `KeywordGuard`                                 |
| `config/envfile`       | auto-loads`.env` (OPENAI_API_KEY, HF_TOKEN) into the environment                              |
| `data/loaders`         | dataset loaders/normalizers → unified taxonomy (train sets**and** benchmarks)            |
| `data/split`           | deterministic train/test split +**anti-leakage** guards                                   |
| `data/training_sets`   | training-set**strategies** (balanced / mixed / over-refusal-aware / red-team)             |
| `models/*`             | trained guards:`EncoderGuard` (BERT), `DecoderGuard` (Qwen3/…), `ensemble`, `registry` |
| `models/registry`      | base-model catalog (Qwen3, DeepSeek-R1, SmolLM2, Gemma, …) + techniques                        |
| `models/ensemble`      | `EnsembleGuard` + pure `combine()` (union/intersection/majority/mean/weighted)              |
| `training/*`           | `sft.py`, `grpo.py`, `dpo.py`, `rewards.py` (label = reward), `runner.py`             |
| `training/runner`      | train→version→record and leakage-checked test→record orchestration                           |
| `evaluation/*`         | `metrics`, `harness`, benchmark registry, OpenAI + incumbent guards, ROC/AUC, reports       |
| `tracking/experiments` | experiment store + model versioning (JSON, no server)                                           |
| `tracking/hardware`    | CPU/GPU/memory/runtime snapshot per run (cross-OS)                                              |
| `serving`              | FastAPI`/screen` API **and** the Benchmark Studio dashboard (train/test/experiments)    |

---

## 4 · Data layer — one taxonomy, many sources

Each dataset labels harm in its own scheme. Pure **normalizer** functions map every source
onto one record shape — `{"text", "label", "hazard", "source"}` — so training and evaluation
are comparable across datasets. Normalizers are unit-tested with no network; **loaders** add
the Hugging Face download on top (lazy import).

```mermaid
flowchart LR
    subgraph SRC["Hugging Face datasets"]
        BT["BeaverTails"]; OM["OpenAI-Moderation eval"]; TC["ToxicChat"]
        PI["deepset prompt-injections"]; JC["jailbreak-classification"]
        JB["JailbreakBench"]; XS["XSTest"]
    end
    SRC --> N["normalize_*() — pure, tested"]
    N --> U["Unified record<br/>{text, label, hazard, source}"]
    U --> J["JSONL splits<br/>(data/ · data/demo/)"]
    U --> S["balanced subset<br/>(data/benchmarks/)"]
    J --> TRAIN["training"]
    S --> EVAL["benchmark suite"]
```

- **Positive class = `unsafe`.** Labels normalize to `safe` / `unsafe`; the hazard is a
  best-effort category (it doesn't affect binary P/R/F1).
- **Splits are deterministic** (`train_val_split`, `seed=42`). The `beavertails` benchmark uses
  the held-out `30k_test` split — disjoint from the demo training data (no leakage).
- **Gated sets** (WildGuardMix, HarmBench, AdvBench, Lakera PINT) need `HF_TOKEN`; the suite
  reports them as *not run* rather than fabricating numbers.

---

## 5 · Guards — three regimes + baselines

All guards implement the same `Guard` interface, so they drop into one harness.

```mermaid
flowchart TD
    subgraph OURS["Agent Bouncer guards"]
        A["A · Encoder<br/>ModernBERT/DistilBERT + head<br/>SFT · latency hero (~8 ms)"]
        B["B · Decoder-SFT<br/>Qwen3-0.6B + LoRA<br/>emits JSON verdict"]
        Cc["C · Decoder-GRPO<br/>RL from the SFT checkpoint<br/>reason-then-verdict"]
    end
    subgraph REF["Reference / incumbents / frontier"]
        K["KeywordGuard (regex baseline)"]
        L["Llama Guard · ShieldGemma · PromptGuard2  (gated)"]
        O["OpenAI Moderation · GPT-4o-mini · GPT-5.2 (low/medium/high reasoning)"]
    end
    A & B & Cc & K & L & O --> H["eval.harness.evaluate()"]
    H --> V["GuardMetrics<br/>P · R · F1 · FPR@benign · latency"]
```

| Regime                      | Model             | Idea                                            | Trade-off                                         |
| --------------------------- | ----------------- | ----------------------------------------------- | ------------------------------------------------- |
| **A — Encoder**      | DistilBERT 66M    | safety as classification                        | fastest (~8 ms), continuous score → real ROC/AUC |
| **B — SFT decoder**  | Qwen3-0.6B (LoRA) | instruction-style`safe/unsafe + hazard` JSON  | generalizes better, ~500 ms on CPU                |
| **C — GRPO decoder** | Qwen3-0.6B (RL)   | reason-then-verdict,**verifiable reward** | RLVR on top of SFT                                |

The decoder's **prompt / target / parse** format lives in one place (`models/decoder.py`) so
SFT, GRPO, DPO, and inference never drift. `build_prompt` → model → `parse_verdict` → `Verdict`;
unparseable output **fails closed** (treated as unsafe).

---

## 6 · Training — SFT, DPO, and RL (GRPO / RLVR)

```mermaid
flowchart LR
    D["unified JSONL"] --> SFTe["SFT · encoder (Trainer)"] --> Ae["outputs/demo-encoder"]
    D --> SFTd["SFT · decoder (TRL SFTTrainer + LoRA)"] --> Bd["outputs/*-decoder-sft"]
    Bd --> G["GRPO from SFT checkpoint"] --> Cg["outputs/grpo-qwen3-0.6b"]
    D --> DPOt["DPO · over-refusal preference pairs"] --> Bd
```

The headline experiment is **RLVR**: a guardrail has ground-truth labels, so the **label is the
reward** — no reward model needed.

```mermaid
flowchart TD
    P["prompt (from a labeled example)"] --> RO["policy generates N rollouts"]
    RO --> PA["parse_verdict() each rollout"]
    PA --> RWD["composite_reward(pred, gold)"]
    subgraph RWD["verifiable reward = Σ"]
        direction LR
        c["+ correctness"]; cat["+ hazard category"]; f["+ parseable format"]
        fp["− false-positive penalty"]; br["+ brevity"]
    end
    RWD --> ADV["group-relative advantage (GRPO)"] --> UPD["policy update (KL-regularized)"]
    UPD --> P
```

The **false-positive penalty** bakes the headline metric (don't over-block benign traffic)
directly into the objective. Reward functions are pure and unit-tested; the LoRA adapter is
merged after training so the RL model loads as a standalone guard, exactly like SFT.

---

## 6b · Training & experiment lifecycle

A dedicated training subsystem turns "run a script" into a tracked, reproducible,
**versioned** experiment — for any model in the registry (the Qwen3 SLMs plus
**DeepSeek-R1-1.5B, SmolLM2-1.7B, Gemma-1B**), with the same SFT + RL techniques.

```mermaid
flowchart LR
    REG["models/registry<br/>base model + arch + techniques"] --> CFG["configure params<br/>(epochs · lr · LoRA · max_steps)"]
    CFG --> TR["train_and_record<br/>SFT / GRPO / DPO"]
    TR --> V["versioned checkpoint<br/>outputs/models/<key>/<version>"]
    TR --> EXPT["experiment (kind=train)<br/>params · hardware · git · timing"]
    V --> EV["evaluate_and_record"]
    SEP["split · leakage guard<br/>drop train∩test items"] --> EV
    BM["benchmark suite"] --> EV
    EV --> EXPE["experiment (kind=eval)<br/>P/R/F1 · AUC · p90 · throughput · hardware"]
    EXPT & EXPE --> HIST["experiment history<br/>comparison · P90 graphs"]
```

- **Model registry** (`models/registry.py`) — base models + arch + supported techniques.
- **Versioning** (`tracking/experiments.py`) — every train writes `outputs/models/<key>/<version>/`.
- **Experiment store** — one JSON per run under `outputs/experiments/` + an index; captures
  params, dataset, **hardware** (`tracking/hardware.py`: CPU/GPU/memory/runtime), git commit, and metrics.
- **Dataset separation** (`data/split.py`) — deterministic disjoint splits + `assert_no_leakage`;
  at test time, any benchmark prompt found in the model's *training* data is **dropped and
  reported**, so a model is never scored on what it trained on.
- **Orchestration** (`training/runner.py`) — `train_and_record` / `evaluate_and_record`, driven
  by `scripts/train/run_training.py`, `scripts/eval/run_testing.py`, and the Studio's `/api/train` `/api/test`.

## 7 · Evaluation harness & the benchmark suite

One harness scores any guard on any labeled set; the benchmark suite is a **registry** of
standard datasets wired to that harness.

```mermaid
flowchart LR
    REG["BENCHMARKS registry<br/>(name → loader · axis · description)"] --> LB["load_benchmark()<br/>download + balanced subsample"]
    LB --> DS["records per benchmark"]
    GUARDS["selected guards"] --> RUN
    DS --> RUN["run_suite / evaluate()<br/>guard × benchmark → GuardMetrics"]
    RUN --> J["outputs/benchmark_results.json<br/>(merge-on-write)"]
    RUN --> CU["compute_curves → outputs/curves.json"]
    J --> RPT["report.render_benchmark_report → BENCHMARKS.md"]
    J --> UI["Benchmark Studio"]
    CU --> UI
```

Three axes are covered: **guardrail** (BeaverTails, OpenAI-Moderation, ToxicChat),
**red-teaming** (deepset prompt-injections, jailbreak-classification, JailbreakBench), and
**over-refusal** (XSTest). Every guard is scored on the *same* balanced subset per benchmark.

---

## 8 · Metrics & curves

`UNSAFE` is the positive class. Definitions in `evaluation/metrics.py` / `evaluation/curves.py`:

| Metric                            | Meaning                                                                        | Direction          |
| --------------------------------- | ------------------------------------------------------------------------------ | ------------------ |
| **Precision**               | of flagged prompts, how many were truly unsafe                                 | higher ↑          |
| **Recall**                  | of unsafe prompts, how many were caught                                        | higher ↑          |
| **F1**                      | harmonic mean of precision & recall                                            | higher ↑          |
| **`fpr_on_benign`**       | share of*benign* prompts wrongly blocked (**over-blocking**)           | **lower ↓** |
| **p50 / p90 / p95 latency** | per-request cost;**p90** is the tail-latency SLO number                  | lower ↓           |
| **throughput**              | single-stream queries/sec (1000 / mean latency)                                | higher ↑          |
| **ROC-AUC**                 | ranking quality; swept for continuous scores, single-operating-point otherwise | higher ↑          |

`fpr_on_benign` is first-class because over-blocking is what makes a guardrail unusable in
production — and it's the number incumbents underreport. It is also baked into the GRPO reward.

---

## 9 · Serving & the Benchmark Studio

`serving/api.py` exposes the guard as `POST /screen` **and** serves the Benchmark Studio — a
web UI to **browse benchmark contents**, **build training sets** (by strategy), **train /
test** SLM guards (streamed live), and **compare experiments** with hardware + P90 graphs.
Tabs: Overview · Benchmarks · Datasets · Train & Test · Experiments · ROC & AUC.

```mermaid
sequenceDiagram
    participant B as Browser (dashboard.html)
    participant API as FastAPI (serving/api.py)
    participant P as Subprocesses
    B->>API: GET /api/config, /api/results, /api/curves
    API-->>B: benchmarks · guard catalog · latest results
    B->>API: POST /api/run {benchmarks, guards, per_class}
    API->>P: run_benchmarks.py (--skip-decoder)
    loop each guard × benchmark
        P-->>API: stdout "[bench] guard: P=.. R=.. F1=.."
        API-->>B: SSE event (step · result · log)
        B->>B: update stepper · live table · console
    end
    API->>P: eval_added_guard.py (isolated decoders) → compute_curves.py
    P-->>API: done
    API-->>B: SSE done
    B->>API: GET /api/results, /api/curves (refresh)
    B->>B: render P/R/F1 · FPR · latency · ROC/AUC
```

Design choices that keep it robust: **process isolation** (a BERT encoder and a Qwen decoder
co-loaded in one process can deadlock the tokenizer thread-pool, so decoders run in their own
subprocess); **merge-on-write** (a partial run augments the scoreboard instead of clobbering
it); and **Chart.js vendored locally** (works offline).

---

## 10 · Reproducing end-to-end

```mermaid
flowchart LR
    s0["make setup"] --> s1["make data-demo"] --> s2["make demo<br/>(train encoder, beat baseline)"]
    s2 --> s3["make bench<br/>(download + score all guards)"]
    s3 --> s4["compute_curves → curves.json"]
    s4 --> s5["render → BENCHMARKS.md"]
    s5 --> s6["make serve / ./start.sh<br/>→ Benchmark Studio"]
```

Everything is seeded and deterministic; benchmark subsets are cached; API guards are skipped
(never faked) when a key is absent. The notebook
[`notebooks/agent_bouncer_studio.ipynb`](../notebooks/agent_bouncer_studio.ipynb) drives the
same path from a single file.

---

## 11 · Extending it

- **Add a benchmark** — write a pure `normalize_*` in `data/loaders.py` (+ a loader), register a
  `Benchmark(name, loader, axis, description)` in `evaluation/benchmarks.py`. It flows into the
  suite, the report, and the dashboard automatically.
- **Add a guard** — implement `.name` + `.predict(text, *, surface) -> Verdict`. It's
  immediately scorable and comparable.
- **Add a dataset for training** — a `normalize_*` + loader, then point a training config's
  `data:` at the JSONL.
- **Tune the reward** — edit `RewardWeights` (e.g. raise `false_positive_penalty` to push
  over-blocking down further).
