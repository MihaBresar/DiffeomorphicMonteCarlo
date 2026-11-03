# Running the updated experiments

On Windows, download the updated repository ZIP, extract it into a new folder,
then double-click **Start.bat**. On macOS, use **Start.command**. Click
**Set up environment**, select the experiment and run size, select CPU workers,
and click **Start run**. The launcher starts fresh runs automatically.

All seven experiments now discard the first third of their actual generated
sampling iterations **after separate step-size tuning**. Tuning never counts
inside that third. Sampling scales are frozen. Timed runs apply this rule to
the final measured iteration count; thinning keeps its original iteration grid.
For 500,000 iterations, discard 166,667 and use the remaining 333,333 before
thinning. At thinning 10 this stores 33,334 post-burn-in states.

## November 2026 matched SCS comparison

If you already ran the compiled, first-third-burn-in release described below,
only the **100D skewed t (ν=3) equal-time comparison** needs rerunning for this
update. It now keeps the authors' SCS transition and implements DCS in the same
JAX framework, with matched VI resources, shared step-size tuning and equivalent
output processing. Previous 100D timings are not measurements of this protocol.

The subsequent DCS efficiency update records implementation revision
`jax-dcs-exact-ball-v1` within the same matched protocol. It uses an equivalent
uniform-ball proposal generator, an exact β=2 map simplification and earlier
rejection of outside-ball proposals. The authors' SCS transition and all shared
fitting, tuning and recording policies remain unchanged. Rerun **all three
methods in the 100D comparison**, including SCS, to measure them under the same
machine conditions. The new BW generator preserves its proposal distribution
but changes seeded trajectories; existing graphs cannot be updated by changing
their time labels. No other experiment needs rerunning for this efficiency update.

## Earlier compiled-kernel and burn-in update

The compiled implementation and burn-in policy require fresh runs for **all
seven experiments**, including the two posterior table rows. Old runs remain
historical results. Changing labels to two decimals alone needs no sampling:
figures can be rebuilt from saved results without changing their statistics.
MSE labels use two decimal places in scientific notation, sample counts remain
exact integers, and CSV/JSON retain full precision.

| Launcher experiment | Current comparison | Outputs |
|---|---|---|
| 100D skewed t (ν=3): DCS-RWM, DCS-BW and SCS — equal time | 50 chains/method, 20 seconds/chain at Full paper settings | `VsSCS`, radial estimates, counts, runtime, bias/variance |
| 50D t: infinite first moment, equal samples | 50 chains/method, 500,000 iterations/chain | `SuperInfiniteFirstTest`, tail/log-norm and quantiles |
| 200D skew-t: DCS versus NUTS | Equal sampling time | `NUTS_PBW_vs` |
| 200D skew-t: random-walk baselines | Equal sampling time | `VsRWM_NUTS` |
| Eight Schools | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| GARCH(1,1) | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| Logistic regression: both priors | Equal sampling time, independent reference | `QQ-regresssion` |

The 100D comparison uses 2 chains of 0.05 seconds for Quick check, 20 chains of
2 seconds for Smaller reproduction, and 50 chains of 20 seconds for Full paper
settings. Both map families use equal fresh-batch VI budgets. All three methods
use separate dual-averaging warmup toward 0.234 using observed acceptances,
then freeze their steps. SCS invokes the unmodified author JAX transition;
DCS uses the same JAX framework and output processing.
See [the exact comparison protocol](SCS_100D.md). The identifier
`skew-t-100d-fixed` is retained only for compatibility; it now launches this
equal-time comparison.

The super-heavy-tail experiment retains equal iteration counts. Its mean norm
does not exist at ν=0.5, so use its tail probability, finite mean log norm and
marginal quantiles. These plots cannot be reconstructed by rescaling timed MSE.

**All seven experiments + posterior table** runs the complete sequence and
prepares new independent logistic references. Quick check only verifies that
execution works. Use Smaller reproduction first to estimate resource needs;
use Full paper settings for manuscript results. Keep worker counts and other
computer activity comparable across methods. The Ryzen 16-core/32-thread CPU
can use 32 workers where enough independent chains are available. See
[parallel execution](PARALLEL.md).

To compare fitted and untuned maps in other experiments, choose **Möbius map**
or **Vanilla map** and start a fresh run for each. Vanilla fixes R=1, centre=0
and Möbius=0 while preserving β and separate proposal tuning. The default 100D
comparison uses its specified fitting recipe automatically.

## Optional command-line equivalents

```bash
python scripts/reproduce_all.py --preset paper --workers 32 --output-root results/paper-new
python run_experiment.py skew-t-100d-fixed --preset paper --workers 32 --output-dir results/scs-time-new
python run_experiment.py skew-t-50d-nu-0.5 --preset paper --workers 32 --output-dir results/superheavy-new
python run_experiment.py --plot-only results/scs-time-new
```

The other experiment identifiers are `skew-t-200d-nuts`,
`skew-t-200d-baselines`, `eight-schools`, `garch11` and `logistic-regression`.
For an individual logistic run, prepare a reference first with
`python prepare_reference.py --preset paper --output-dir results/reference-new`
and pass `--reference-dir results/reference-new` to `run_experiment.py`.
Reference chains also discard their first third after separate tuning. Static
PosteriorDB reference draws retain their documented source processing.

Independent logistic references must meet the existing paper-quality checks:
R-hat ≤1.01, bulk ESS ≥2,000, tail ESS ≥1,000, no divergences or full-tree
saturation in every parameter. A failed quality screen requires a better
reference; the thresholds are not weakened by this update.

Original manuscript plot filenames are preserved in each run's `figures`
folder. Inspect diagnostics before replacing manuscript figures and both rows
of the posterior table. Earlier audit scripts and their saved reports preserve
their historical protocols; they are not additional launcher experiments.
