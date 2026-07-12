# agent-bouncer paper build
PAPER = benchmark_chooses_the_winner
.PHONY: all figures clean
all: $(PAPER).pdf
$(PAPER).pdf: $(PAPER).tex refs.bib $(wildcard figures/*.pdf)
	tectonic $(PAPER).tex
figures:
	cd .. && python experiments/make_figures.py
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
