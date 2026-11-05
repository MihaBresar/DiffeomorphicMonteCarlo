# Running the experiments

Current version-10 comparisons contain DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and DCS-Billiard
in both JAX and optimized Numba modes, with fitted maps and 100D/500D targets.
VI is mandatory for new comparisons; the launcher has no VI-off option.
New reports contain only radial estimates/MSE/counts/time plus functional ESS
and ESS/s. See [the current protocol](COMPARISONS.md).

On Windows, download the updated repository ZIP, extract it into a new folder,
then double-click **Start.bat**. On macOS, use **Start.command**. Click
**Set up environment**, select the experiment and run size, select CPU workers,
and click **Start run**. The launcher starts fresh runs automatically.

All seven experiments now discard the first third of their actual generated
sampling iterations **after a separate pilot**. Pilots never count
inside that third. Sampling scales are frozen. Timed runs apply this rule to
the final measured iteration count; thinning keeps its original iteration grid.
For 500,000 iterations, discard 166,667 and use the remaining 333,333 before
thinning. At thinning 10 this stores 33,334 post-burn-in states.

## Version-10 fitting update

Rerun the **100D and 500D five-method comparisons**, in either implementation
mode, in fresh folders. Version 10 makes VI mandatory and matches the map
fitting resources: both families use 1,000 updates on their own fixed
2,000-point cloud for Full paper settings, 400 × 500 for Smaller reproduction,
and 20 × 128 for Quick check. DCS now reuses its cloud instead of drawing a
fresh batch at each update, so **all preset sizes need new runs**. Run all five
methods together under the same execution conditions.

SCS's original author fit is unchanged: seed 0, unit initialization and its
recorded fitting-only clipping. DCS retains its own parameterization,
initialization, seed and gradient clipping. Equal update and cloud counts do
not imply identical fitting algorithms or equal preparation time.

The paired logistic and 50D super-heavy-tail experiments follow the same
fixed-cloud budget policy. Their production kernels and step-tuning rules
are unchanged, but DCS fitting changes. Other experiments without SCS keep
their recorded fitting policy. Independent logistic reference draws need no
new sampling solely because map fitting changed, provided they remain valid
for the recorded target and reference protocol. Replotting old chains cannot
apply a new fit.

## Current sampler comparisons

Choose **Five samplers: DCS-RWM, DCS-BW, SCS, reflected BW and DCS-Billiard (JAX)** or
**Five samplers: optimized Numba DCS versus SCS**. Both version-10 suites
contain the same methods, with 100D/500D and VI fitted for both map families. Numba production differs
in implementation and recording costs; author SCS remains in JAX.

Run every method together in a fresh folder. New runs contain radial-tail
probability and mean norm, their MSEs, counts and times, plus functional ESS
and ESS/s. Coordinate statistics and rotation methods
are omitted. Both radial references are analytic. Keep the configuration,
manifest and saved data with the report to identify the experiment.

Billiards retain jointly fitted centre/R/β after Möbius with effective δ=0.
Billiards tune their own production kernel with a bounded physical
radial-movement score. SCS defaults to the authors' fixed h=0.1 and also
supports acceptance-target 0.234 tuning, with a discarded pilot under either policy.
RWM/BW retain acceptance-target 0.234 pilots; reflected BW retains its
ball-movement pilot in the shared fitted ball map with no geometric ceiling
of 2. All steps freeze and the first third of production is discarded.
Every method receives the same number of pilot transitions. Use
`--scs-step-rule acceptance` or `--scs-step-rule fixed` to select the SCS
policy explicitly; the [matched-map sensitivity helper](AUTHOR_SCS_CALIBRATION.md#reproduction)
runs the opposite policy from a completed primary without refitting.
Use fresh sampling batches to assess variability conditional on the fitted
maps. Keep worker count and machine conditions consistent. See
[protocol](COMPARISONS.md) and [ESS](ESS.md).

Other experiment ESS reports describe their recorded retained-draw basis;
check the saved stride.

## Experiment choices

Use a fresh output directory for each new run. Completed figures can be rebuilt
from saved results without sampling or changing their statistics. MSE labels
use two decimal places in scientific notation, sample counts remain exact
integers, and CSV/JSON retain full precision.

| Launcher experiment | Current comparison | Outputs |
|---|---|---|
| Five samplers: DCS-RWM, DCS-BW, SCS, reflected BW and DCS-Billiard (JAX) | Selected target, fitted maps; equal time, JAX | `VsSCS`, radial estimates, counts, runtime, bias/variance, functional ESS |
| Five samplers: optimized Numba DCS versus SCS | Selected target, fitted maps; equal time, mixed implementations | Same comparison figure families and exact-value tables |
| 50D t: infinite first moment, equal samples | 50 chains/method, 500,000 iterations/chain | `SuperInfiniteFirstTest`, tail/log-norm and quantiles |
| 200D skew-t: DCS versus NUTS | Equal sampling time | `NUTS_PBW_vs` |
| 200D skew-t: random-walk baselines | Equal sampling time | `VsRWM_NUTS` |
| Eight Schools | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| GARCH(1,1) | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| Logistic regression: both priors | Equal sampling time, independent reference | `QQ-regresssion` |

Both current comparisons use 2 chains of 0.05 seconds for Quick check, 20 chains of
2 seconds for Smaller reproduction, and 50 chains of 20 seconds for Full paper
settings, for either target. SCS uses the pinned
authors' fixed-cloud optimizer, seed 0, and full budget of 1,000 updates on
2,000 reference draws; DCS uses seed 42 with the same 1,000 updates on
its own fixed 2,000-point cloud. Their clipping rules also differ, as recorded in the
[protocol](COMPARISONS.md). Quick and Smaller settings reduce both fitting
budgets; full fitting may change rankings. The same separate
transition budget applies to every method and both SCS step policies.
All steps freeze before timed sampling. Numba DCS starts from the same JAX
pilot state and selected step as its JAX counterpart.

The super-heavy-tail experiment retains equal iteration counts. Its mean norm
does not exist at ν=0.5, so use its tail probability, finite mean log norm and
marginal quantiles. These plots cannot be reconstructed by rescaling timed MSE.

**All seven experiments + posterior table** runs the five-method JAX comparison
on the 100D skewed-t target and the other six experiments,
and prepares independent logistic references. It fits both comparison maps
by VI and uses the specified maps for the other experiments.
The optimized Numba and 500D comparisons are separate selections.
Quick check only verifies that
execution works. Use Smaller reproduction first to estimate resource needs;
use Full paper settings for the recorded full benchmark budgets. This does not
guarantee the values in any previous manuscript figure. Keep worker counts and
other computer activity comparable across methods; concurrency is limited by
the detected allocation, available chains and memory. See
[parallel execution](PARALLEL.md).

The launcher uses fitted maps for the synthetic and logistic experiments.
Eight Schools and GARCH retain their specified identity maps. Previously saved
untuned-map reports remain readable; they are not relabelled as fitted runs.

## Optional command-line equivalents

```bash
python scripts/reproduce_all.py --preset paper --workers 2 --output-root results/paper-new
python run_experiment.py sampler-comparison --target skew-t-100d --preset paper --workers 2 --output-dir results/comparison-100d-vi
python run_experiment.py sampler-comparison-numba --target skew-t-100d --preset paper --workers 2 --output-dir results/numba-100d-vi
python run_experiment.py sampler-comparison --target skew-t-500d --preset paper --workers 2 --output-dir results/comparison-500d-vi
python run_experiment.py skew-t-50d-nu-0.5 --preset paper --workers 2 --output-dir results/superheavy-new
python run_experiment.py --plot-only results/comparison-100d-vi
```

These examples use two workers; choose a worker count appropriate to the
machine and keep it fixed across comparisons.

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

Manuscript plot filenames are preserved in each run's `figures` folder.
Inspect diagnostics and the resolved protocol before using the generated
figures or posterior-table rows in a publication.
