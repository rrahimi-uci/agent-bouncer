# Datasets

All training data is chosen to be **license-compatible** with an Apache-2.0 release.

## Training

| Dataset | Size | License | Use |
|---------|------|---------|-----|
| WildGuardMix (`allenai/wildguardmix`) | ~92K | Apache-2.0 | Broad harm + refusals + jailbreaks |
| BeaverTails (`PKU-Alignment/BeaverTails`) | ~330K QA | Apache-2.0 (CC-BY-NC for some splits — check) | 14-category harm labels |
| Aegis 2.0 (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`) | ~30K | CC-BY-4.0 | Diverse safety taxonomy |

> ⚠️ **Avoid ToxicChat for the released model** — it is Vicuna-generated
> (non-commercial). It's fine for internal analysis only.

## Benign / over-refusal (for the false-positive metric)

| Dataset | Use |
|---------|-----|
| XSTest | Safe prompts that *look* unsafe — measures over-blocking |
| OR-Bench | Large-scale over-refusal benchmark |

## Evaluation (standard benchmark suite)

Run via `make bench` / `scripts/eval/run_benchmarks.py`; registered in
`agent_bouncer/evaluation/benchmarks.py` and normalized in `agent_bouncer/data/loaders.py`. All
are **ungated** (download without `HF_TOKEN`) and scored on class-balanced subsets by default —
or on the **full** sets with `run_benchmarks.py --full` (fetched by
`scripts/data/download_full_benchmarks.py`).

| Axis | Benchmark | HF dataset | Measures |
|------|-----------|------------|----------|
| Guardrail | BeaverTails (30k_test) | `PKU-Alignment/BeaverTails` | 14-category harmful-QA prompt safety |
| Guardrail | OpenAI-Moderation | `mmathys/openai-moderation-api-evaluation` | 8-category content moderation |
| Guardrail | ToxicChat | `lmsys/toxic-chat` | Real user-input toxicity (eval-only license) |
| Red-team | prompt-injections | `deepset/prompt-injections` | Prompt-injection attack detection |
| Red-team | jailbreak-classification | `jackhhao/jailbreak-classification` | Jailbreak vs. benign |
| Red-team | JailbreakBench | `JailbreakBench/JBB-Behaviors` | 100 harmful + 100 benign behaviors |
| Over-refusal | XSTest | `natolambert/xstest-v2-copy` | Safe-but-scary prompts → `fpr_on_benign` |

### Gated (need `HF_TOKEN` + license acceptance — reported as *not run*, never faked)

| Benchmark | Measures |
|-----------|----------|
| WildGuardMix (`allenai/wildguardmix`) | Broad harm + refusals + jailbreaks |
| HarmBench (`walledai/HarmBench`) | Red-teaming behaviors |
| AdvBench (`walledai/AdvBench`) | Adversarial harmful instructions |
| Lakera PINT | Prompt-injection / jailbreak (private benchmark) |

Mapping details for each dataset's native labels live next to the code in
`agent_bouncer/data.py`.
