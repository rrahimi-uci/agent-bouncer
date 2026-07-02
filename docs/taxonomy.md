# Hazard taxonomy

One canonical label space (`agent_bouncer.core.taxonomy.Hazard`), aligned to the
MLCommons AILuminate hazard set and extended with two agent-specific surfaces.

| Hazard | Notes |
|--------|-------|
| `violent_crimes` | |
| `non_violent_crimes` | |
| `sex_crimes` | |
| `child_exploitation` | |
| `weapons_cbrne` | Chem/bio/radiological/nuclear/explosive |
| `suicide_self_harm` | |
| `hate` | |
| `privacy` | PII / data leakage |
| `sexual_content` | |
| `malicious_code` | Malware, exploits |
| `prompt_injection` | **agent-specific** |
| `jailbreak` | **agent-specific** |

Every source dataset is mapped onto this schema in `agent_bouncer.data` so that
training and evaluation are comparable across sources. When adding a dataset, add
its native-label → `Hazard` map there and document any judgement calls here.
