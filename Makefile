PYTHON ?= python3
EXPERIMENT ?= skew-t-50d-nu-0.5
.PHONY: list test quick quick-all paper
list:
	$(PYTHON) run_experiment.py --list
test:
	$(PYTHON) -m unittest discover -s tests -v
quick:
	$(PYTHON) run_experiment.py $(EXPERIMENT) --preset quick
quick-all:
	$(PYTHON) scripts/reproduce_all.py --preset quick --output-root results/quick
paper:
	$(PYTHON) scripts/reproduce_all.py --preset paper --output-root results/paper
