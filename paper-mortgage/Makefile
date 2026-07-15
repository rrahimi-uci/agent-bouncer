PAPER      := mortgage_guardrail_benchmark
TECTONIC   ?= tectonic
DOT        ?= dot
BUILD_DIR  ?= build
GEN        := ../mortgage-redteaming-agentic-generator
PYTHON     ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python3)

.DEFAULT_GOAL := pdf
.PHONY: pdf figure table clean

figure: figures/pipeline.png  ## render the construction-pipeline diagram

figures/pipeline.png: figures/pipeline.dot
	$(DOT) -Tpng -Gdpi=150 $< -o $@

table: generated/baseline_table.tex  ## regenerate the LaTeX baseline table from committed scores

generated/baseline_table.tex: $(GEN)/out_eval/baseline_table.json
	$(PYTHON) $(GEN)/tools/emit_baseline_tex.py $< $@

pdf: figure table  ## compile the 11pt article draft (same toolchain as paper-b)
	mkdir -p $(BUILD_DIR)
	$(TECTONIC) --keep-logs --outdir $(BUILD_DIR) $(PAPER).tex
	cp $(BUILD_DIR)/$(PAPER).pdf $(PAPER).pdf

clean:  ## remove build-only files (keeps committed PDF, figure, and generated table)
	rm -rf $(BUILD_DIR)
