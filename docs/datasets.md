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

## Evaluation (standard benchmarks)

| Benchmark | Measures |
|-----------|----------|
| GuardBench | 40-dataset guardrail benchmark (the standard; pip-installable pipeline) |
| Lakera PINT | Prompt-injection / jailbreak detection |
| XSTest | Over-refusal (feeds `fpr_on_benign`) |

Mapping details for each dataset's native labels live next to the code in
`agent_bouncer/data.py`.
