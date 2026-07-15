# Paper A reproducibility contract

Paper A has one publication workflow: the strict final v2 contract under
`artifacts/paper_a_sft_v2/`. The v1 tree under `artifacts/paper_a_sft/` is an
archival compatibility artifact, not a fallback when v2 evidence is unavailable.
The completed v2 release binds lock SHA-256
`cabc8dee9b158773ce0be86f799ec3833c33c18787a2aa74d05ed1a261682c25`
to the 79,392-row score artifact SHA-256
`b941ddbaea7057ab1f224c510687ec5748916f5eca6a78e1d1f429e0ede5a1c3`.
The execution is clean and reproducible; its analysis remains retrospective and
precision-focused because prior development exposed part of the transfer cohort.

## Two v2 verification levels

### Full-artifact verification

Use this path on the execution archive that retains the locked raw manifests,
audit, run metadata, adapter directories, and the exact immutable execution-source
snapshot whose bytes match the lock:

```bash
python3.12 -m venv .venv && source .venv/bin/activate  # use the lock's exact patch
python -m pip install -r requirements.txt
make verify-lock PY=.venv/bin/python
make validate-runs PY=.venv/bin/python
make analyze PY=.venv/bin/python
make paper-sync PAPER_ANALYSIS=artifacts/paper_a_sft_v2/analysis
make paper-verify PAPER_ANALYSIS=artifacts/paper_a_sft_v2/analysis
```

This workflow rehashes the locked config, manifests, audit, public release,
execution sources, run metadata, adapters, score file, and score metadata before
regenerating and byte-comparing the manuscript inputs. It is valid only inside
that exact source snapshot/bundle with the locked Python patch and scientific
package versions. The post-run repository intentionally has different hardened
analysis sources and therefore uses only the public release-cache workflow.
The checksum-bound tracked-tree snapshot and exact GCP orchestration are retained
under [`artifacts/paper_a_sft_v2/provenance/`](../artifacts/paper_a_sft_v2/provenance/).

### Final release-cache verification

The smaller release cache is designed for CPU-only reproduction from a fresh
clone where licensed prompt text and large training artifacts cannot be
redistributed. A complete cache contains:

- `artifacts/paper_a_sft_v2/LOCK.json`, with a strict v2 contract and
  `finalization_status: final`;
- `artifacts/paper_a_sft_v2/RELEASE.json`, whose self-hash binds the exact lock,
  public-manifest tree, scores, score metadata, and verifier/analyzer sources;
- `configs/paper_a_sft_v2_release_anchor.json`, a separately tracked digest of
  the exact `RELEASE.json` bytes and self-hash;
- all files bound beneath `artifacts/paper_a_sft_v2/public_manifests/`;
- `artifacts/paper_a_sft_v2/scores/scores.parquet`; and
- `artifacts/paper_a_sft_v2/scores/metadata.json`, which binds the lock,
  producer/runtime identity, full bundle inventory, manifest fingerprints, and
  the SHA-256 of the combined score file.

From a fresh clone containing those release files:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install
make repro
make paper
```

`make repro` is an alias for `repro-release`. It invokes analysis with the
explicit `--release-cache` flag, then byte-compares all checked-in manuscript
inputs with the canonical v2 outputs **without synchronizing first**. This makes
stale paper files a failure instead of silently overwriting the comparison
baseline. `make paper-sync` is an explicit maintainer action; after using it,
rerun `make repro`. The release path never enables `--allow-legacy-lock` or reads
the v1 namespace. `make paper` repeats the comparisons before compilation.

The release-cache path verifies:

- the external config anchor, `RELEASE.json` self-hash, and every file/source
  digest under that contract, so mutually rewriting scores and metadata cannot
  silently mint a new result;
- the lock self-hash, strict structure, final status, config binding, and
  internally consistent original execution-source commitment;
- the hashes, row counts, raw-artifact commitments, redaction assertions, and
  supplemental records in the text-free public manifest release;
- the score-file digest, score metadata, complete expected bundle-by-sample
  matrix, row identities, manifest fingerprints, adapter/run-metadata
  commitments carried in metadata, and all analysis invariants;
- the same Python major/minor line as the final lock and exact locked versions
  of NumPy, pandas, PyArrow, scikit-learn, SciPy, and Matplotlib; training-only
  packages may be absent from this CPU path; and
- the hashes of every generated scientific output, while recording the actual
  analysis runtime and current analysis-source file hashes.

It intentionally does **not** claim to rehash locally absent raw manifests, run
metadata, adapter bytes, or the original GPU execution-source bytes. Those
omissions are machine-readable in
`artifacts/paper_a_sft_v2/analysis/analysis_metadata.json`. The lock's original
execution-source file hashes remain an immutable commitment; independent byte
verification of that commitment uses the separately archived source bundle.
This distinction prevents a later analysis-only checkout from being mislabeled
as the original GPU execution tree.

If any required release file is absent, or any binding, row, or runtime check
fails, `make repro` stops without falling back to historical results.

`make release-package` stages the public overlay under `dist/` from an explicit
allowlist. It includes only the trust anchor, strict release inputs, and selected
generated analysis outputs; it fails if raw/full-artifact directories or
symlinks appear. Raw manifests, adapters, run metadata, base-score caches, audit
inputs, and smoke outputs stay in the separately verified internal archive.

## Archived v1 compatibility

The historical score table can still be analyzed explicitly:

```bash
make repro-legacy
```

That target passes `--allow-legacy-lock`, writes only to
`artifacts/paper_a_sft/analysis/`, labels the output as legacy, and does not
overwrite v2 publication inputs. `make repro`, `make paper-sync`,
`make paper-verify`, and `make paper` use v2 paths only.

## Environment records

See [reproducibility-environments.md](reproducibility-environments.md) for the
separate current and historical environment records. The release cache records
its actual analysis environment in `analysis_metadata.json`; this does not
rewrite or weaken the original execution environment recorded by the lock.
