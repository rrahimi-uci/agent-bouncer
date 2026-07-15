# The Benchmark Chooses the Winner — paper build
PAPER = benchmark_chooses_the_winner
TECTONIC ?= tectonic
.PHONY: all clean
all: $(PAPER).pdf
$(PAPER).pdf: $(PAPER).tex refs.bib tab_primary_gen.tex tab_sensitivity_gen.tex tab_seed_values_gen.tex results_macros_gen.tex $(wildcard figures/*.pdf)
	$(TECTONIC) $(PAPER).tex
# The figure + generated tables come from the analysis step, not from here:
#   make repro   (repo root) -> v2 artifacts + verified paper copies
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).xdv $(PAPER).pdf
	rm -rf build
