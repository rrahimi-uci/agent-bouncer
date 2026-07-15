# Reproducibility environments

The repository has two distinct environment records. They must not be described as
the same environment.

## Executed clean-v2 GPU environment

The final v2 lock records the exact scientific runtime used on the isolated GCP
runner:

| Component | Locked/recorded value |
|---|---|
| OS / kernel | Ubuntu 24.04 image; Linux 6.17.0-1020-gcp, glibc 2.39 |
| Python | CPython 3.12.3 |
| NumPy / SciPy | 2.5.0 / 1.18.0 |
| pandas / PyArrow | 3.0.3 / 24.0.0 |
| scikit-learn / Matplotlib | 1.9.0 / 3.11.0 |
| PyTorch / CUDA / cuDNN | 2.12.1+cu130 / 13.0 / 92000 |
| Transformers / PEFT / TRL | 5.12.1 / 0.19.1 / 1.7.0 |
| Accelerate / Datasets / Safetensors | 1.14.0 / 5.0.0 / 0.8.0 |
| GPU / driver | NVIDIA A100-SXM4-40GB / 580.159.03 |

Each of the 20 completed run records repeats the locked protocol package versions,
device, peak memory, timing, data and training seeds, token count, truncation
strategy, adapter digest, and clean source/lock commitments. The exact execution
source snapshot is distributed separately because the post-run analysis checkout
contains additional hardening.

## Current post-run pipeline and CPU analysis

Use Python 3.12 and [`requirements.txt`](../requirements.txt). That file is fully
pinned for release-cache analysis and for running the corrected pipeline code.
Run metadata records the OS, Python, PyTorch, CUDA runtime,
cuDNN, Transformers, PEFT, TRL, Accelerate, device, peak memory, and timing values
available to the process.

Final score metadata also records the producer-runtime fingerprint and batch size
used for every cache; changing either invalidates reuse. Analysis keeps the
scientific payload deterministic in `results.json` and writes the actual Python,
NumPy, SciPy, scikit-learn, pandas, pyarrow, and Matplotlib environment plus
execution-source hashes to sibling `analysis_metadata.json`.

## Historical GPU run behind the committed scores

The immutable historical run and score metadata report:

| Component | Recorded value |
|---|---|
| Python | 3.10.12 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| pyarrow | 25.0.0 |
| scikit-learn | 1.7.2 |
| PyTorch | 2.9.1+cu129 |
| Transformers | 4.56.2 |
| GPU | NVIDIA A100-SXM4-40GB |

The legacy metadata does not record a complete OS, NVIDIA driver, runtime CUDA,
container digest, Accelerate, PEFT, or TRL environment. Those values cannot be
recovered honestly after the fact. For that reason, `requirements.txt` is not
presented as the exact v1 result-producing GPU environment. The clean-v2 rerun
above supersedes v1 for publication evidence and carries the complete recorded
environment; the v1 tree remains archival only.
