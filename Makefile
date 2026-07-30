.PHONY: list test quick paper

PYTHON ?= python3
EXPERIMENT ?= skew-t-50d-nu-0.5

list:
	$(PYTHON) run_experiment.py --list

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m compileall -q .

quick:
	$(PYTHON) run_experiment.py $(EXPERIMENT) --preset quick

paper:
	$(PYTHON) run_experiment.py $(EXPERIMENT) --preset paper
