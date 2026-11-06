# Running the experiments

To run the reported results' configuration, choose **DCS vs original SCS**. Its bundled maps and parameters are fixed, with no
manual step choices or refitting. Use **Smaller reproduction** for one reported
20-chain × 3-second batch, or **Full paper settings** for 50 chains × 20 seconds.
See [the study instructions](REPORTED_STUDY.md), including the recorded seed batches.

The previous version-11 comparisons contain DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and DCS-Billiard
in both JAX and optimized Numba modes, with fitted maps and 100D/500D targets.
VI is mandatory when running that earlier recipe; the reported study loads its bundled maps. The launcher has no VI-off option.
New reports contain only radial estimates/MSE/counts/time plus functional ESS
and ESS/s. See [the previous fitted-map protocol](COMPARISONS.md).

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

## Earlier fitted-map version-11 sampling update

For the earlier fitted-map recipe, rerun all five methods together for each 100D/500D target and implementation
mode used in a new comparison. All four DCS methods share h=0.07:
BW/reflected-BW/billiard use that increment radius; Gaussian RWM uses
coordinate standard deviation h/√(d+2). Each uses its own kernel during
discarded warmup without step adaptation. SCS retains author h=0.1. All methods still receive the same
warmup-transition budget and first-third production burn-in.

The version-10 fixed-cloud VI settings and fitted maps are unchanged. New v11
results require fresh sampling; replotting old chains cannot apply a new step
policy. Its public entry points use the version-11 preset steps automatically;
manual step-size and calibration-policy flags are not accepted. The preset
is not claimed optimal. Fresh-cloud VI studies remain experimental.
The other six experiments do not change under this sampling update.

## Earlier version-10 fitting update

Rerun the **100D and 500D five-method comparisons**, in either implementation
mode, in fresh folders if upgrading from version 9 or earlier. Version 10 made VI mandatory and matched the map
fitting resources: both families use 1,000 updates on their own fixed
2,000-point cloud for Full paper settings, 400 × 500 for Smaller reproduction,
and 20 × 128 for Quick check. DCS began reusing its cloud instead of drawing a
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

## Earlier fitted-map sampler comparisons

The compatibility commands `sampler-comparison` and
`sampler-comparison-numba` retain both version-11 suites. These options are
no longer present in the launcher. Both suites
contain the same methods, with 100D/500D and VI fitted for both map families. Numba production differs
in implementation and recording costs; author SCS remains in JAX.

Run every method together in a fresh folder. New runs contain radial-tail
probability and mean norm, their MSEs, counts and times, plus functional ESS
and ESS/s. Coordinate statistics and rotation methods
are omitted. Both radial references are analytic. Keep the configuration,
manifest and saved data with the report to identify the experiment.

Billiards retain jointly fitted centre/R/β after Möbius with effective δ=0.
All four DCS methods share fixed h=0.07, with Gaussian RWM converted to
coordinate standard deviation h/√(d+2). The single SCS baseline uses the authors' fixed h=0.1
throughout discarded warmup and production; alternative SCS step tuning is not offered.
BW, reflected BW and billiard use identical uniform-ball increment scales.
Each DCS method uses its own production kernel during fixed-step discarded
equilibration; reflection and billiard receive no separate step optimization. All steps freeze and the first third of production is discarded.
Every method receives the same number of discarded warmup transitions.
The [earlier SCS sensitivity study](AUTHOR_SCS_CALIBRATION.md#reproduction)
remains reproducible through its archived source revision.
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

| Launcher experiment | Comparison recipe | Outputs |
|---|---|---|
| DCS vs original SCS | Default: bundled maps, recorded zero-centre DCS settings, original author SCS; equal sampling time, extra prior DCS calibration | `VsSCS`, radial estimates, counts, runtime, bias/variance, functional ESS |
| Numba extension | Same bundled settings; Numba DCS versus author JAX SCS; implementation extension | Same comparison figure families and exact-value tables |
| 50D t: infinite first moment, equal samples | 50 chains/method, 500,000 iterations/chain | `SuperInfiniteFirstTest`, tail/log-norm and quantiles |
| 200D skew-t: DCS versus NUTS | Equal sampling time | `NUTS_PBW_vs` |
| 200D skew-t: random-walk baselines | Equal sampling time | `VsRWM_NUTS` |
| Eight Schools | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| GARCH(1,1) | HnR time matched to mean NUTS sampling time | Posterior table and diagnostics |
| Logistic regression: both priors | Equal sampling time, independent reference | `QQ-regresssion` |

Both reported-study options use 2 chains of 0.05 seconds for Quick check,
20 chains of 3 seconds for Smaller reproduction, and 50 chains of 20 seconds
for Full paper settings. All presets load the same frozen full-budget maps;
no fitting or parameter recalibration occurs.

Both earlier fitted-map comparisons use 2 chains of 0.05 seconds for Quick check, 20 chains of
2 seconds for Smaller reproduction, and 50 chains of 20 seconds for Full paper
settings, for either target. SCS uses the pinned
authors' fixed-cloud optimizer, seed 0, and full budget of 1,000 updates on
2,000 reference draws; DCS uses seed 42 with the same 1,000 updates on
its own fixed 2,000-point cloud. Their clipping rules also differ, as recorded in the
[protocol](COMPARISONS.md). Quick and Smaller settings reduce both fitting
budgets; full fitting may change rankings. The same separate
transition budget applies to every method; SCS uses only fixed h=0.1.
All steps freeze before timed sampling. Numba DCS starts from the same JAX
pilot state and selected step as its JAX counterpart.

The super-heavy-tail experiment retains equal iteration counts. Its mean norm
does not exist at ν=0.5, so use its tail probability, finite mean log norm and
marginal quantiles. These plots cannot be reconstructed by rescaling timed MSE.

**All seven experiments + posterior table** runs **DCS vs original SCS**
on the selected 100D/500D target and the other six experiments,
and prepares independent logistic references. It loads the bundled comparison
maps and uses the specified maps for the other experiments.
The current JAX and Numba comparisons can also be run separately for either dimension.
Quick check only verifies that
execution works. Use Smaller reproduction first to estimate resource needs;
use Full paper settings for the recorded full benchmark budgets. This does not
guarantee the values in any previous manuscript figure. Keep worker counts and
other computer activity comparable across methods; concurrency is limited by
the detected allocation, available chains and memory. See
[parallel execution](PARALLEL.md).

The earlier recipes use fitted maps for the synthetic and logistic experiments;
the reported-study entries load their recorded effective maps without refitting.
Eight Schools and GARCH retain their specified identity maps. Previously saved
untuned-map reports remain readable; they are not relabelled as fitted runs.

## Optional command-line equivalents

```bash
python run_experiment.py reported-zero-centre --target skew-t-100d --preset paper --workers 2 --output-dir results/reported-100d
python run_experiment.py reported-zero-centre --target skew-t-500d --preset paper --workers 2 --output-dir results/reported-500d
python run_experiment.py reported-zero-centre-numba --target skew-t-100d --preset paper --workers 2 --output-dir results/reported-numba-100d
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
