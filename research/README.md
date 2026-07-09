# research/ — self-contained reproduction bundle

Everything needed to reproduce the paper *"Safety-Guard Rankings Are Operating-Point Artifacts"* lives in this folder. Nothing here depends on code or data outside `research/`.

```
research/
  notebooks/    standalone end-to-end pipeline + bundled data + outputs
    smollm3_guard_reproduction.ipynb  trains, calibrates, evaluates the guard top-to-bottom
    data/         bundled benchmarks (incl. guard_benchmark_hard.jsonl, mortgage_split.json)
    outputs/      cached scores + summary_*.json (every number in the paper traces here)
    .env          optional keys (OPENAI_API_KEY / HF_TOKEN)
    README.md
  paper/          ACM paper: safety_guard_operating_point_artifacts.{tex,pdf}, refs.bib, figures/, tables/
    Makefile, README.md, DRAFT.md, metrics_survey.md
  scripts/        producing scripts for the paper's experiments (see scripts/README.md)
  docs/           supporting design/results docs referenced by the notebook & paper
  requirements.txt  pinned Python deps
  .env.example
```

## Reproduce the notebook (the primary artifact)
The notebook is fully self-contained: all code is in the `.ipynb`, all data is bundled under `notebooks/data/`, and it writes to `notebooks/outputs/`. It finds its bundled data locally and never reaches outside `research/`.

```bash
cd research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example notebooks/.env      # optional: fill in keys for the gpt baseline / gated models
cd notebooks
# GUARD_SMOKE=1 → tiny proxy model, runs in minutes on CPU/MPS (default off CUDA);
# GUARD_SMOKE=0 → the real SmolLM3-3B guard (needs a GPU or a patient MPS box).
jupyter nbconvert --to notebook --execute smollm3_guard_reproduction.ipynb --output run.ipynb
```

## Reproduce the paper's extra experiments (scripts)
`scripts/` holds the drivers for the results the notebook does not itself produce (novel-benchmark OOD, base-vs-tuned decomposition, the mortgage case study, figure generation). **Run them from `research/`** so their `notebooks/…` and `paper/…` paths resolve:

```bash
cd research                          # cwd matters: paths are relative to here
python scripts/eval_mortgage_hard.py            # mortgage hardened-benchmark metrics
python scripts/build_hard_jsonl.mjs             # (node) rebuild the hard benchmark jsonl
python scripts/make_figures.py                  # regenerate paper/figures/*.pdf from outputs/*.json
```
Mortgage case-study chain: `build_mortgage_split.py` → `train_mortgage.py` (TECHNIQUE=sft) → `eval_mortgage_tuned.py`; hardened set: `wf_build_hard_benchmark_v2.mjs` → `build_hard_jsonl.mjs` → `eval_mortgage_hard.py`.

## Build the paper
`paper/` compiles standalone (figures are committed as vector PDFs — no scripts or Python needed):
```bash
cd research/paper && make            # tectonic -> safety_guard_operating_point_artifacts.pdf
make figures                         # optional: regenerate figures (needs the venv above)
```

## Provenance
Every number in the paper traces to `notebooks/outputs/nb-smollm3-guard/*.json`; the producing script for each is named in `paper/README.md` and `docs/`.
