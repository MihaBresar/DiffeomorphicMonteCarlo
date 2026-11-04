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

## Current sampler comparisons

The previous three-method 100D and separate boundary launcher entries have
been replaced by two choices:

- **Four samplers: DCS-RWM, DCS-BW, SCS and DCS-BW-reflect**, using the same
  JAX framework and full-state processing (`timed-sampler-comparison-v2`).
- **Optimized Numba DCS-BW and reflected BW versus SCS**, using optimized
  DCS production and the unchanged author SCS JAX transition
  (`timed-numba-comparison-v2`). This comparison includes differences between
  the implementations and their output processing.

Both offer 100D and 500D skewed Student-t targets with ν=3 and DCS β=2. Each can
run **With VI** or **Without VI** for both map families. Every choice retains
separate proposal-step tuning and frozen sampling steps. See
[the current protocol](COMPARISONS.md).

Start a fresh run of **every method in the chosen comparison** for each target
and VI setting you want to assess. Old results remain valid records of their
saved protocols; replotting cannot turn them into the new comparison or
measure the optimized backend or recover all-iteration coordinate estimates.
The new coordinate measurements are part of the timing budget, so rerun every
method together even for an existing target. Keep older radial results as
historical records. Keep the same worker count and machine
conditions across methods. For a second independent sampling check, choose
a fresh seed batch; VI-on fits retain seed 42.

This comparison update does not by itself require rerunning the other six
experiments if they already used the compiled kernels and one-third burn-in
policy below. Formatting changes alone only require regenerating reports.

## Earlier compiled-kernel and burn-in update

The compiled implementation and burn-in policy require fresh runs for **all
seven experiments**, including the two posterior table rows. Old runs remain
historical results. Changing labels to two decimals alone needs no sampling:
figures can be rebuilt from saved results without changing their statistics.
MSE labels use two decimal places in scientific notation, sample counts remain
exact integers, and CSV/JSON retain full precision.

| Launcher experiment | Current comparison | Outputs |
|---|---|---|
| Four samplers: DCS-RWM, DCS-BW, SCS and DCS-BW-reflect | Selected target and VI setting; equal time, JAX | `VsSCS`, radial estimates, added coordinate mean/tail, counts, runtime, bias/variance |
| Optimized Numba DCS-BW and reflected BW versus SCS | Selected target and VI setting; equal time, mixed implementations | Same comparison figure families and exact-value tables |
| 50D t: infinite first moment, equal samples | 50 chains/method, 500,000 iterations/chain | `SuperInfiniteFirstTest`, tail/log-norm and quantiles |
| 200D skew-t: DCS versus NUTS | Equal sampling time | `NUTS_PBW_vs` |
| 200D skew-t: random-walk baselines | Equal sampling time | `VsRWM_NUTS` |
| Eight Schools | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| GARCH(1,1) | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| Logistic regression: both priors | Equal sampling time, independent reference | `QQ-regresssion` |

Both current comparisons use 2 chains of 0.05 seconds for Quick check, 20 chains of
2 seconds for Smaller reproduction, and 50 chains of 20 seconds for Full paper
settings, for every target and VI choice. With VI, both map families receive
equal fresh-batch budgets. Without VI, both maps use R=1 with zero centre/deformation,
and no fitting draws. Ordinary methods use acceptance-target 0.234 calibration;
reflected BW uses ball-movement pilots within the same transition budget.
All steps freeze before timed sampling. Numba DCS starts from the same JAX
pilot state and selected step as its JAX counterpart.

The CLI identifiers `skew-t-100d-fixed` and `skew-t-100d-boundary` and their
scripts remain available for historical protocols, as do report regeneration
and the [three-method](SCS_100D.md) and [boundary](BOUNDARY_100D.md) descriptions.
They do not launch the new configurable suite and are no longer menu choices.

The super-heavy-tail experiment retains equal iteration counts. Its mean norm
does not exist at ν=0.5, so use its tail probability, finite mean log norm and
marginal quantiles. These plots cannot be reconstructed by rescaling timed MSE.

**All seven experiments + posterior table** runs the four-method JAX comparison
on the 100D skewed-t target and the other six experiments,
and prepares independent logistic references. Its VI choice controls both maps
in the comparison; the other experiments retain their existing map options.
The optimized Numba and 500D comparisons are separate selections.
Quick check only verifies that
execution works. Use Smaller reproduction first to estimate resource needs;
use Full paper settings for manuscript results. Keep worker counts and other
computer activity comparable across methods. The Ryzen 16-core/32-thread CPU
can use 32 workers where enough independent chains are available. See
[parallel execution](PARALLEL.md).

To compare fitted and untuned maps in other experiments, choose **Möbius map**
or **Vanilla map** and start a fresh run for each. Vanilla fixes R=1, centre=0
and Möbius=0 while preserving β and separate proposal tuning. This older
DCS-only selector does not control SCS; use the new comparison's **With VI /
Without VI** selector to switch both map families. Eight Schools and GARCH
retain their identity maps.

## Optional command-line equivalents

```bash
python scripts/reproduce_all.py --preset paper --workers 32 --vi on --output-root results/paper-new
python run_experiment.py sampler-comparison --target skew-t-100d --vi on --preset paper --workers 32 --output-dir results/comparison-100d-vi
python run_experiment.py sampler-comparison-numba --target skew-t-100d --vi off --preset paper --workers 32 --output-dir results/numba-100d-no-vi
python run_experiment.py sampler-comparison --target skew-t-500d --vi off --preset paper --workers 32 --output-dir results/comparison-500d-no-vi
python run_experiment.py skew-t-50d-nu-0.5 --preset paper --workers 32 --output-dir results/superheavy-new
python run_experiment.py --plot-only results/comparison-100d-vi
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
