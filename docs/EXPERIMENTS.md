# Experiment guide

Run every experiment through the public CLI:

```bash
python run_experiment.py --list
python run_experiment.py <identifier> --preset quick
```

The `paper` preset deliberately supplies no overrides.  It preserves the
module-level settings that were present in the original simulation scripts;
the `quick` preset in [`configs/presets.json`](../configs/presets.json) reduces
chain counts, time budgets, optimizer work, warmup, and synthetic reference
size for smoke testing.

| Identifier | Module | Main diagnostics | Important interpretation note |
| --- | --- | --- | --- |
| `eight-schools` | `experiments.eight_schools` | First/second-moment MSE, per-chain RMSE, Q–Q plots | NUTS construction/warmup/draw time defines the projected sampler budget; fixed PosteriorDB summaries need provenance verification. |
| `garch11` | `experiments.garch11` | First/second-moment RMSE and Q–Q plots | NUTS construction/warmup/draw time defines the projected sampler budget; uses bundled data plus fixed PosteriorDB summaries. |
| `logistic-regression` | `experiments.logistic_regression_student_t` | Pooled-sampler Q–Q panels | The horizontal quantiles pool the evaluated samplers; this is a diagnostic, not independent truth. |
| `skew-t-50d-nu-0.5` | `experiments.skew_t_50d_nu_0_5` | Coordinate Q–Q, tail probability, log-norm metric | Synthetic Monte Carlo reference; validate the target/reference parameterization. |
| `skew-t-100d-fixed` | `experiments.skew_t_100d_fixed_iterations` | Coordinate Q–Q, tail probability, mean norm | Fixed number of transitions rather than a wall-clock budget. |
| `skew-t-200d-nuts` | `experiments.skew_t_200d_pbw_vs_nuts` | Coordinate Q–Q, tail probability, mean norm, ESS throughput | Uses BlackJAX NUTS; the recorded time budget excludes setup. |
| `skew-t-100d-baselines` | `experiments.skew_t_100d_pbw_rwm_nuts` | Coordinate Q–Q, tail probability, mean norm, ESS throughput | RWM diagnostic storage is intentionally partial to control memory. |

## Outputs and reproducibility controls

The CLI passes controls as `HEAVY_TAILS_*` environment variables so spawned
workers inherit the exact same configuration.  Common values are:

| CLI option | Environment variable |
| --- | --- |
| `--workers N` | `HEAVY_TAILS_WORKERS` |
| `--runs N` | `HEAVY_TAILS_NUM_RUNS` |
| `--time-budget SECONDS` | `HEAVY_TAILS_TIME_BUDGET` |
| `--iterations N` | `HEAVY_TAILS_NUM_ITERS` |
| `--reference-samples N` | `HEAVY_TAILS_REFERENCE_SAMPLES` |
| `--reference-batch-size N` | `HEAVY_TAILS_REFERENCE_BATCH_SIZE` |
| `--optimizer-steps N` | `HEAVY_TAILS_OPTIMIZATION_STEPS` |

`--reference-samples` is supported only by the synthetic skew-t experiments.
When no batch size is given, the runner selects a divisor of the requested
total; an explicit `--reference-batch-size` must evenly divide it.  The runner
rejects generic overrides that the selected experiment does not consume.
Module-specific settings, including warmup, can also be set as environment
variables; their final values appear in `configuration.json`.  The runner uses
a headless Matplotlib backend by default.  Add `--show` to retain the current
interactive backend and display figure windows.

## Naming note

This repository uses `DCS-BW/PBW` and `projected slice/HnR` where the source
code used multiple labels for closely related implementations.  It does not
assert that these are the paper's final canonical names.  Update the labels,
definitions, and citations together once the manuscript terminology is fixed.
