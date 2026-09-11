# What must be rerun

**Windows queue failure (`*.worker-*.json` missing):** download the updated
repository and start a fresh run; see [the no-command update steps](WINDOWS_QUEUE_FIX.md).
Completed runs need no new sampling for this queue-only fix.

**For the restored plot layouts and exact-number tables, no new sampling is
needed.** The command-line `--plot-only` option below rebuilds completed results.
Fresh launcher runs generate all original plot families and exact-number tables
automatically; see [the figure guide](FIGURES.md).

**To measure the faster kernels, new CPU scheduling, a different worker count,
or fixed-R tuning, start a new run.** Select **Full paper settings**, the desired
worker count, and **Start run**. Keep old results folders for comparison.
Do not resume an old run with these changed execution conditions. Full paper
figures have not been precomputed on your Ryzen; they must come from that run.

## Earlier scientific corrections

**Regenerate all five manuscript figures and both rows of the PosteriorDB table.** The previous aggregate-only outputs cannot repair changes to targets, adaptation, references, or timing. Original PNGs remain archival assets and are not updated results.

From the repository root, activate the Python 3.12 environment and install `requirements-cpu.txt`. The command below runs everything in the correct order with the default single worker. Add `--workers 4` for four parallel chains, or choose CPU workers in the launcher. Keep the same worker setting when resuming. See [parallel execution](PARALLEL.md); updating from the sequential-only code requires a new run directory.

```bash
python scripts/reproduce_all.py --preset paper --output-root results/paper
```

It prepares independent logistic references, runs the seven experiments, generates all figures and creates `results/paper/posteriordb_rmse.tex` and `.csv`. To resume after interruption, repeat with `--resume`. A changed scientific configuration/source requires a fresh output root.

## Individual experiments

| Manuscript output | Corrected run | Why new sampling is required |
|---|---|---|
| `NUTS_PBW_vs.png` | `skew-t-200d-nuts` | Shared exact target/transform, frozen tuning, synchronized timing, exact radial reference |
| `VsRWM_NUTS.png` | `skew-t-200d-baselines` | Dimension corrected from 100 to 200, plus shared corrections |
| `VsSCS.png` | `skew-t-100d-fixed` | ν corrected from 2 to 3; warmup separated from 500,000 frozen production steps |
| `SuperInfiniteFirstTest.png` | `skew-t-50d-nu-0.5` | Target corrected to α=0; symmetric map constraints enforced |
| `QQ-regresssion.png` | `logistic-regression` | 30-second budget and a separate independent reference, replacing the pooled reference |
| Table: Eight Schools | `eight-schools` | Common log-τ coordinates, pinned reference draws, correct chain-level RMSE and timing |
| Table: GARCH | `garch11` | Unintended conditional-uniform prior removed; correct four-parameter target, references and RMSE |

```bash
python run_experiment.py skew-t-200d-nuts --preset paper --output-dir results/paper/skew-t-200d-nuts
python run_experiment.py skew-t-200d-baselines --preset paper --output-dir results/paper/skew-t-200d-baselines
python run_experiment.py skew-t-100d-fixed --preset paper --output-dir results/paper/skew-t-100d-fixed
python run_experiment.py skew-t-50d-nu-0.5 --preset paper --output-dir results/paper/skew-t-50d-nu-0.5
python run_experiment.py eight-schools --preset paper --output-dir results/paper/eight-schools
python run_experiment.py garch11 --preset paper --output-dir results/paper/garch11
python prepare_reference.py --preset paper --output-dir results/paper/logistic-reference
python run_experiment.py logistic-regression --preset paper --reference-dir results/paper/logistic-reference --output-dir results/paper/logistic-regression
python scripts/collect_posterior_table.py results/paper
```

Each PNG and matching PDF is written to `results/paper/<experiment>/figures/`, retaining the exact manuscript image basename. Copy the five new PNGs into the manuscript folder only after inspecting the paper-run diagnostics. The table must replace the old numbers; do not retain conclusions about which sampler wins before seeing the rerun.

The logistic reference is generated with four independent NumPyro NUTS chains, each using 4,000 warmup and 100,000 production draws, target acceptance 0.95 and depth 12. Paper use requires recorded R-hat ≤1.01, bulk ESS ≥2,000, tail ESS ≥1,000, no divergences and no full-tree saturation in every parameter. Failure stops paper reference preparation; inspect the diagnostics and create a better reference in another directory. Thresholds are quality screens, not a proof of exactness.

## Replot or resume a single result

```bash
python run_experiment.py --plot-only results/paper/skew-t-100d-fixed
python run_experiment.py skew-t-100d-fixed --preset paper --output-dir results/paper/skew-t-100d-fixed --resume
```

Plot-only reads saved references and draws and never calls a sampler. Resume is between completed chains; an interrupted chain restarts with its original seed. Do not edit code/configurations during a run. For a changed seed, budget, target or scientific implementation, choose a new directory.

The `quick` profile is for execution checks; the `reproduce` profile is a smaller scientific run. Neither reproduces the paper design. See `configs/presets.json` for their explicit overrides.
