# Running experiments

Open `Start.bat` on Windows, `Start.command` on macOS, or `START_HERE.py` in IDLE. Set up the environment and choose experiment, preset, seed and workers. Every start creates a fresh run. Keep the console open and prevent sleep.

Use Quick for installation checks, Smaller to estimate resource needs and Full for configured full budgets. **Seed 1–Seed 8** select reproducible sampling streams. See [the catalogue](EXPERIMENTS.md).

## Main comparison

DCS vs original SCS and Numba extension offer 100D/500D targets and six methods. All presets use the same maps and steps: Quick 2 × 0.05 seconds, Smaller 20 × 3 seconds, Full 50 × 20 seconds, with 1,500 separate warmup transitions. Zero-SCS keeps its pre-zeroing R/h; no new calibration occurs. See [the protocol](REPORTED_STUDY.md).

```sh
python run_experiment.py reported-zero-centre --target skew-t-100d --preset paper --workers 2 --output-dir results/dcs-scs-100d
python run_experiment.py reported-zero-centre --target skew-t-500d --preset paper --workers 2 --output-dir results/dcs-scs-500d
python run_experiment.py reported-zero-centre-numba --target skew-t-100d --preset paper --workers 2 --output-dir results/numba-100d
```

## Additional experiments

Identifiers are `skew-t-200d-nuts`, `skew-t-200d-baselines`, `skew-t-50d-nu-0.5`, `eight-schools`, `garch11` and `logistic-regression`. Their DCS and random-walk production uses Numba; NUTS retains its compiled implementation. All discard the first third after separate tuning.

```sh
python run_experiment.py skew-t-50d-nu-0.5 --preset paper --workers 2 --output-dir results/superheavy
python run_experiment.py skew-t-200d-baselines --preset paper --workers 2 --output-dir results/baselines
python scripts/reproduce_all.py --preset paper --workers 2 --output-root results/all-experiments
```

All seven runs the JAX main comparison and six additional experiments, with reference preparation, rather than every target/backend combination.

For an individual logistic run, prepare a reference with `python prepare_reference.py --preset paper --output-dir results/reference` and pass `--reference-dir results/reference` to the experiment. Reference checks require R-hat ≤1.01, bulk ESS ≥2,000, tail ESS ≥1,000, no divergences and no depth saturation in every parameter. A failed reference needs investigation.

## Reports

Open `paper_values.html`; CSV/JSON retain full precision and PNG/PDF figures are in `figures/`. To recreate a completed report:

```sh
python run_experiment.py --plot-only results/dcs-scs-100d
```

Keep worker allocation and computer load comparable across methods. Timed counts differ across hardware. See [parallel execution](PARALLEL.md), [figures](FIGURES.md) and [reproducibility](REPRODUCIBILITY.md).
