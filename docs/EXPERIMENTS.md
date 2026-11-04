# Experiment specification

The current comparison suite is specified in `scripts/reproduce_comparison.py`, with target definitions in `dcs_bench/comparison_targets.py` and budgets in `scripts/reproduce_scs_100d.py`. See [the protocol](COMPARISONS.md). The other six experiments use named JSON files in `configs/`; the older 100D JSON is retained for compatibility. Each generic configuration contains the full paper target, methods, budgets, warmup, thinning, optimization, seeds, quantile levels, and output basename. `presets.json` lists explicit quick and affordable-reproduction overrides. Use `python run_experiment.py IDENTIFIER --preset paper --dry-run` to inspect the exact resolved input.

The version-4 comparison suite retains the original 100D and 500D skewed t(3)
targets. Both implementation modes
support VI on/off and record eight functionals: radial tail probability, mean
norm, the means of X₁/X₂/X₃, and the tails X₁ > 3, X₂ < −3 and X₃ > 3. The new
coordinate measurements run inside the sampling clock for every method. The added DCS-BW-rotate sampler mixes reflected ball steps and fixed-radius rotations with equal probability, in both JAX and Numba. Its two proposal scales share the existing pretuning budget and freeze before sampling.
Independent reference preparation occurs before fitting and sampling and is
recorded separately. Old radial-only results are retained as version-1 records,
version-2 records retain their radial and X₁ measurements, and version-3 records retain all eight measurements without the rotation mixture.

[README.md](../README.md) contains the catalogue. [RERUN.md](RERUN.md) maps each configuration to manuscript figures and the posterior table. [TARGETS.md](TARGETS.md) defines every density and coordinate transform, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines timing and metrics.

`experiments/` and `run_*.py` are compatibility entry points. They call the same implementation in `dcs_bench/`; there are no separate target copies to repair.
