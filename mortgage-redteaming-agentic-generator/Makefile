# Mortgage red-teaming agentic generator — one-command pipeline.
# Prefer the parent repo's venv so guard_research + numpy/pandas are present:
#   make smoke PY=../.venv/bin/python
PY ?= python3

.DEFAULT_GOAL := help
.PHONY: help install sample smoke build test clean

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## install deps (and the parent repo editable for guard_research)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ..   # exposes guard_research (metrics/provenance/thresholds)

sample:  ## regenerate the bundled SYNTHETIC HMDA-shaped sample (offline substrate)
	$(PY) data/hmda_sample/make_sample.py

smoke:  ## tiny fast OFFLINE end-to-end run (mock provider + bundled sample; no keys/network)
	$(PY) -m magen --smoke

build:  ## full build from config/default.yaml (set llm.provider + hmda.source first)
	$(PY) -m magen --config config/default.yaml --phase all

test:  ## run the offline test suite
	$(PY) -m pytest tests/ -q

clean:  ## remove build outputs and caches
	rm -rf out
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
