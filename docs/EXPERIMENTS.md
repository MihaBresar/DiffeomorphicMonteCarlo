# Experiment specification

The current comparison suite is specified in `scripts/reproduce_comparison.py`, with target definitions in `dcs_bench/comparison_targets.py` and budgets in `scripts/reproduce_scs_100d.py`. See [the protocol](COMPARISONS.md). The other six experiments use named JSON files in `configs/`; the older 100D JSON is retained for compatibility. Each generic configuration contains the full paper target, methods, budgets, warmup, thinning, optimization, seeds, quantile levels, and output basename. `presets.json` lists explicit quick and affordable-reproduction overrides. Use `python run_experiment.py IDENTIFIER --preset paper --dry-run` to inspect the exact resolved input.

[README.md](../README.md) contains the catalogue. [RERUN.md](RERUN.md) maps each configuration to manuscript figures and the posterior table. [TARGETS.md](TARGETS.md) defines every density and coordinate transform, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines timing and metrics.

`experiments/` and `run_*.py` are compatibility entry points. They call the same implementation in `dcs_bench/`; there are no separate target copies to repair.
