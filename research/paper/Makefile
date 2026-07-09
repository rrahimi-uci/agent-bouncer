# agent-bouncer paper build
PAPER = safety_guard_operating_point_artifacts
.PHONY: all figures clean
all: $(PAPER).pdf
$(PAPER).pdf: $(PAPER).tex refs.bib $(wildcard figures/*.pdf)
	tectonic $(PAPER).tex
figures:
	cd .. && python scripts/make_figures.py
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
