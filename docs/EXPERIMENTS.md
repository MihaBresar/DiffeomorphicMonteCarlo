# Experiment specification

The current comparison suite is specified in `scripts/reproduce_comparison.py`, with target definitions in `dcs_bench/comparison_targets.py` and budgets in `scripts/reproduce_scs_100d.py`. See [the protocol](COMPARISONS.md). The other six experiments use named JSON files in `configs/`; the older 100D JSON is retained for compatibility. Each generic configuration contains the full paper target, methods, budgets, warmup, thinning, optimization, seeds, quantile levels, and output basename. `presets.json` lists explicit quick and affordable-reproduction overrides. Use `python run_experiment.py IDENTIFIER --preset paper --dry-run` to inspect the exact resolved input.

The version-10 JAX and Numba suites both contain DCS-RWM, DCS-BW, SCS,
DCS-BW-reflect and DCS-Billiard on the original 100D/500D skewed t(3) targets, with VI fitted for both map families.
They report only radial-tail probability and mean norm, using analytic
references, plus MSE, exact counts, production time, functional ESS and ESS/s.
Coordinate reference simulation and coordinate statistics are omitted.
Billiards use the fitted centre/R/β after Möbius with zero active deformation;
their own-kernel radial pilot runs in those coordinates and freezes before
production. SCS uses the original author VI and fixed h=0.1, with an optional acceptance-tuned control. All methods receive the same discarded warmup-transition count; fitting recipes differ and their costs are recorded. The Numba implementation shares its JAX pilot, then uses fused
native production. See [billiards](BILLIARD.md) and [ESS](ESS.md).

Every new VI-enabled SCS fit uses the pinned author optimizer, including
logistic regression, the symmetric 50D ν=0.5 example and diagnostic scripts.
SCS has a separate `scs_vi` budget: 1,000 updates on 2,000 fixed reference draws
for paper, 400 × 500 for reproduce, and 20 × 128 for quick. Its fit seed is 0,
with the author's unit initialization, Adam 0.01 and fitting-only coordinate
clip of ±1,000. In paired DCS/SCS experiments, DCS uses the same update
count and cloud size, and now reuses its own fixed cloud. Its initialization,
parameterization and gradient clipping remain DCS-specific. Other DCS
experiments keep their recorded fitting policy. Generic
SCS production remains Numba with acceptance-target 0.234 step tuning; the
100D/500D comparison retains its existing JAX transition and fixed-h=0.1
default. New five-method comparisons require VI; saved untuned-map reports
and lower-level diagnostic helpers retain their recorded behavior.

Historical protocol records retain their original methods and measurements.
New measurements require fresh runs; old thinned traces cannot recover
unthinned functional ESS. The other experiment targets and original figures
remain available.

[README.md](../README.md) contains the catalogue. [RERUN.md](RERUN.md) maps each configuration to manuscript figures and the posterior table. [TARGETS.md](TARGETS.md) defines every density and coordinate transform, and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) defines timing and metrics.

`experiments/` and `run_*.py` are compatibility entry points. They call the same implementation in `dcs_bench/`; there are no separate target copies to repair.
