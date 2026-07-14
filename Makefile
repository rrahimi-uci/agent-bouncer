# Paper A pipeline — "The Benchmark Chooses the Winner"
# Steps 1–3 and 6 are CPU-only; steps 4–5 need a GPU.
# The historical score bundle is reproduced only through the explicit legacy target.

PY              ?= python3
CONFIG           = configs/paper_a_sft.yaml
LEGACY_ROOT      = artifacts/paper_a_sft
V2_ROOT          ?= artifacts/paper_a_sft_v2
MANIFESTS        = $(V2_ROOT)/manifests
PUBLIC_MANIFESTS = $(V2_ROOT)/public_manifests
MANIFEST_JSON    = $(MANIFESTS)/manifest.json
AUDIT            = $(V2_ROOT)/audit
LOCK             ?= $(V2_ROOT)/LOCK.json
SCORES           = $(V2_ROOT)/scores/scores.parquet
ANALYSIS         = $(V2_ROOT)/analysis
LEGACY_LOCK      = $(LEGACY_ROOT)/LOCK.json
LEGACY_SCORES    = $(LEGACY_ROOT)/scores/scores.parquet
LEGACY_ANALYSIS  = $(LEGACY_ROOT)/analysis
PAPER_ANALYSIS   ?= $(LEGACY_ANALYSIS)
PAPER_DIR        = paper-a

.DEFAULT_GOAL := help
.PHONY: help install install-all manifests manifests-legacy audit lock relock \
        verify-lock verify-legacy-lock train validate-runs eval analyze analyze-legacy repro \
        repro-legacy paper-sync paper-verify selftest test paper paper-html \
        legacy-explorer clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

install:      ## install CPU analysis plus tests
	$(PY) -m pip install -e ".[dev]"
install-all:  ## install training/scoring plus CPU analysis and tests
	$(PY) -m pip install -e ".[all]"

## --- clean v2 pipeline ------------------------------------------------------
manifests:  ## 1. pinned-HF, hash-ranked manifests + text-free public index
	$(PY) experiments/prepare_paper_a_manifests.py --config $(CONFIG) --out $(MANIFESTS) --public-out $(PUBLIC_MANIFESTS)
manifests-legacy:  ## reconstruct previously inspected seed-7 cohorts (explicit)
	$(PY) experiments/prepare_paper_a_manifests.py --config $(CONFIG) --out $(MANIFESTS) --public-out $(PUBLIC_MANIFESTS) --allow-legacy-frozen-cohorts
audit:      ## 2. independently recompute and fail-closed on split integrity
	$(PY) experiments/audit_paper_a_splits.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --out $(AUDIT)
lock:       ## 3. write a clean v2 lock without overwriting the historical lock
	$(PY) experiments/lock_paper_a_sft.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --manifests-dir $(MANIFESTS) --audit $(AUDIT)/audit.json --out $(LOCK) --require-tokenizer-probe
relock:     ## explicitly replace the clean v2-root LOCK.json (destructive)
	$(PY) experiments/lock_paper_a_sft.py --config $(CONFIG) --manifest $(MANIFEST_JSON) --manifests-dir $(MANIFESTS) --audit $(AUDIT)/audit.json --out $(LOCK) --require-tokenizer-probe --force
verify-lock:  ## verify lock self-hash and every bound file
	$(PY) -c "from experiments import paper_a_common as C; C.load_lock('$(LOCK)', verify_files=True); print('verified $(LOCK)')"
verify-legacy-lock:  ## verify historical self-hash without pretending it is v2
	$(PY) -c "from experiments import paper_a_common as C; C.load_lock('$(LEGACY_LOCK)', allow_legacy=True, verify_files=False); print('verified legacy $(LEGACY_LOCK)')"
train:      ## 4. train the 4x5 LoRA-SFT panel under the clean lock (GPU)
	$(PY) experiments/run_paper_a_sft.py train --lock $(LOCK)
validate-runs:  ## rehash and validate all 20 adapters against the clean lock
	$(PY) experiments/run_paper_a_sft.py validate-runs --lock $(LOCK) --strict
eval:       ## 5. score bases + validated adapters (GPU)
	$(PY) experiments/eval_paper_a_sft.py --lock $(LOCK)
analyze:    ## 6. strict complete-matrix analysis under the clean v2 contract
	$(PY) experiments/analyze_paper_a_sft.py --lock $(LOCK) --scores $(SCORES) --out $(ANALYSIS)

## --- historical score compatibility (no GPU) -------------------------------
analyze-legacy:  ## explicitly analyze the immutable v1 lock + committed scores
	$(PY) experiments/analyze_paper_a_sft.py --allow-legacy-lock --lock $(LEGACY_LOCK) --scores $(LEGACY_SCORES) --out $(LEGACY_ANALYSIS)
	$(MAKE) paper-sync PAPER_ANALYSIS=$(LEGACY_ANALYSIS)
repro-legacy: analyze-legacy  ## regenerate and sync every legacy paper output
	$(MAKE) paper-verify PAPER_ANALYSIS=$(LEGACY_ANALYSIS)
	@echo "reproduced legacy tables/figure in $(LEGACY_ANALYSIS) and verified paper copies"
repro: repro-legacy  ## compatibility alias; output remains explicitly labeled legacy

## --- generated paper inputs ------------------------------------------------
paper-sync:  ## copy canonical generated TeX/figure outputs into paper-a
	cp $(PAPER_ANALYSIS)/tables/table3_primary.tex $(PAPER_DIR)/tab_primary_gen.tex
	cp $(PAPER_ANALYSIS)/tables/table4_per_benchmark.tex $(PAPER_DIR)/tab_sensitivity_gen.tex
	cp $(PAPER_ANALYSIS)/tables/table5_seed_values.tex $(PAPER_DIR)/tab_seed_values_gen.tex
	cp $(PAPER_ANALYSIS)/tables/results_macros.tex $(PAPER_DIR)/results_macros_gen.tex
	cp $(PAPER_ANALYSIS)/figures/specialization_plane.pdf $(PAPER_DIR)/figures/specialization_plane.pdf
paper-verify:  ## fail if a paper-consumed generated file is stale
	cmp $(PAPER_ANALYSIS)/tables/table3_primary.tex $(PAPER_DIR)/tab_primary_gen.tex
	cmp $(PAPER_ANALYSIS)/tables/table4_per_benchmark.tex $(PAPER_DIR)/tab_sensitivity_gen.tex
	cmp $(PAPER_ANALYSIS)/tables/table5_seed_values.tex $(PAPER_DIR)/tab_seed_values_gen.tex
	cmp $(PAPER_ANALYSIS)/tables/results_macros.tex $(PAPER_DIR)/results_macros_gen.tex
	cmp $(PAPER_ANALYSIS)/figures/specialization_plane.pdf $(PAPER_DIR)/figures/specialization_plane.pdf

## --- verification / documents ---------------------------------------------
selftest:   ## synthetic end-to-end analysis check
	$(PY) experiments/analyze_paper_a_sft.py --self-test
test:       ## run unit and release-integrity tests
	$(PY) -m pytest
paper: paper-verify  ## verify generated inputs and build the PDF
	$(MAKE) -C $(PAPER_DIR)
paper-html: paper  ## build the nested HTML edition
	$(PY) $(PAPER_DIR)/paper-html/build.py
legacy-explorer:   ## regenerate archived broad-study explorer samples
	$(PY) $(PAPER_DIR)/paper-html/explorer/build_content_samples.py

clean:      ## remove Python and paper build caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	$(MAKE) -C $(PAPER_DIR) clean
