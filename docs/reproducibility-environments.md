# Reproducibility environments

The repository has two distinct environment records. They must not be described as
the same environment.

## Current corrected pipeline and CPU analysis

Use Python 3.12 and [`requirements.txt`](../requirements.txt). That file is fully
pinned for regenerating the legacy analysis outputs and for running the corrected
pipeline code. New run metadata records the OS, Python, PyTorch, CUDA runtime,
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
presented as the exact result-producing GPU environment and a clean v2 rerun is
required for full environment reproducibility.
