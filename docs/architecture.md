# Architecture

Agent Bouncer sits **on the request path**, in front of your LLM/agent. Because it
runs on every call (and every agent step), it must be small and fast — which is
exactly why an SLM is the right tool, not a compromise.

```
                    ┌─────────────────────────────────────────┐
  user prompt ──▶   │  Agent Bouncer (tiny guard, <30ms)       │
  tool call   ──▶   │  1. deterministic checks                 │  ──▶ ALLOW ──▶ LLM / Agent
  agent output ─▶   │  2. small classifier clears ~97%         │
                    │  3. escalate hardest ~1-3% (optional)    │  ──▶ BLOCK ──▶ safe refusal
                    └─────────────────────────────────────────┘
                                    │
                                    ▼
                        MLflow: metrics, latency, traces
```

## Surfaces

Guarding *agents* means screening more than user text (`agent_bouncer.schema.Surface`):

- `user_prompt` — the incoming request (content safety, injection, jailbreak).
- `tool_call` — the action the agent wants to take before it executes.
- `agent_output` — what the agent is about to return.

## Components

| Module | Role |
|--------|------|
| `taxonomy` | The single hazard label space (MLCommons-aligned + injection/jailbreak). |
| `schema` | The `Verdict` contract every guard returns. |
| `guard` | `Guard` protocol + reference `KeywordGuard`. |
| `models` | Trained guards: `EncoderGuard` (ModernBERT), `DecoderGuard` (Qwen3/Llama). |
| `rewards` | Verifiable GRPO rewards (label = reward). |
| `metrics` | F1, AUPRC, **false-positive-on-benign**, latency. |
| `eval` | MLflow harness + GuardBench/PINT/XSTest adapters + incumbent baselines. |
| `serve` | FastAPI `/screen` endpoint. |
