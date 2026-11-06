# Reported zero-centre study

This comparison extends the reported **zero-centre DCS versus original SCS**
experiment with a second SCS variant using the same radius/step pilot budget
as DCS. Its six methods are DCS-RWM, DCS-BW, DCS-BW-reflect, DCS-Billiard,
original author SCS, and **SCS calibrated**. The original five-method results
used JAX on 100D and 500D skewed Student-t targets with ν=3. Current runs keep
those five methods' parameters and add the independently pilot-selected SCS
control. The launcher selects this comparison by default.

The previous fitted-centre comparison remains available under its own name.
It uses different DCS maps and h=0.07; its results are not replications of this study.

## Run without commands

1. Download and extract the updated repository. Open **Start.bat** on Windows,
   **Start.command** on macOS, or open **START_HERE.py** in IDLE and choose
   **Run → Run Module**.
2. Choose **DCS vs original SCS**, then select
   **100D** or **500D**. There are no map, radius or step-size choices to make.
3. Run **Quick check** first. Then choose **Smaller reproduction** or
   **Full paper settings**, select the worker count, and click **Start run**.
4. Open the generated **paper_values.html** for figures, MSE, exact counts,
   measured time and functional ESS. CSV and JSON retain full precision.

| Preset | Chains per method | Production seconds per chain | Separate warmup |
|---|---:|---:|---:|
| Quick check | 2 | 0.05 | 1,500 |
| Smaller reproduction | 20 | 3.00 | 1,500 |
| Full paper settings | 50 | 20.00 | 1,500 |

Every preset loads the same frozen full-budget maps. Quick check tests execution
only. Smaller reproduction matches one reported batch's sampling budget; the
reported pooled result combined two such batches. Full paper settings extends
the sampling budget while retaining the study's maps and proposal parameters.
All runs discard the first third of production iterations after the separate warmup.

## Exact frozen settings

| Parameter | 100D | 500D |
|---|---:|---:|
| DCS physical centre μ | zero vector | zero vector |
| DCS deformation δ | zero vector | zero vector |
| DCS β | 2 | 2 |
| DCS radius R | 0.6941299233184104 | 1.3882232419648068 |
| Shared DCS h | 0.025 | 0.044721359549995794 |
| SCS h | 0.1 | 0.1 |
| SCS latitude | 1.1 | 1.1 |
| SCS fitted radius | 1.1048295589441204 | 2.2313627251649697 |
| SCS calibrated h | 0.05 | 0.025 |
| SCS calibrated radius | 0.5524147794720602 | 1.1156813625824848 |

BW, reflected BW and billiard use uniform-ball increments of radius h.
Gaussian RWM uses coordinate standard deviation h/√(d+2). All four DCS
variants use the same production map. Original SCS retains its original fitted
centre, observer and radius with h=0.1. SCS calibrated copies that same author
centre and observer, changing only radius and h to the recorded pilot choices.
Both SCS variants execute the unchanged author transition in JAX, including
in the Numba extension. Neither adapts during warmup or production.

The map assets and checksums are in [studies/zero_centre/maps](../studies/zero_centre/maps).
They preserve the original VI maps and fitting histories separately from the
actual production-map overrides. Each original fit used 1,000 updates and a
fixed reference cloud of 2,000 points. New runs load these assets and do not
refit VI or repeat parameter selection. The SCS fit came from the pinned authors'
original optimizer. The separate calibration asset records the selected SCS
parameters and shared pilot resources. Missing or modified assets fail validation.

## Which preparation budgets match?

| Preparation | DCS | SCS calibrated |
|---|---:|---:|
| Original VI updates | 1,000 | 1,000 |
| Fixed reference-cloud points | 2,000 | 2,000 |
| VI objective/gradient point evaluations | 2,000,000 | 2,000,000 |
| Radius/step candidates | 16 | 16 |
| Exact target starts shared across candidates | 4,096 | 4,096 |
| Transitions per start and candidate | 8 | 8 |
| Pilot transitions per dimension | 524,288 | 524,288 |
| New VI or pilot work in a frozen-map run | none | none |

The joint pilot used the same radial movement objective and equal transition
counts, with method-specific radius and step grids. Original SCS remains a
separate author-default baseline; its production parameters do not use the
joint pilot's selected R and h. The later centre diagnostic examined both DCS
and SCS variants. DCS adopts a zero-centre/zero-deformation production map;
SCS calibrated retains the author-fitted centre and observer. Equal budgets
do not mean identical fitting objectives, transformations or retained choices.

Equal evaluation and transition counts also do not mean equal elapsed time.
The archived pilot timing measures batched transition evaluation and excludes
compilation, inverse mapping and scoring; it is not full preparation time.
Current runs record map-loading and warmup costs separately and reuse the
previous fits and calibration. Production-time MSE and ESS/s measure sampling
after preparation, not total time from an unprepared target. Excluding
preparation consistently is not itself evidence of a bias towards DCS.

## Recorded results and replication seeds

The [reported results](../studies/zero_centre/results/index.html) include both
20-chain batches and their predeclared 40-chain pooled analysis, with full-precision
chain estimates, counts, timing and functional ESS. All chains for the five
displayed methods are retained. The archive also contains the **SCS jointly
calibrated** control, now used as **SCS calibrated** in fresh six-method runs.
Historical reports remain five-method reports; adding a bar requires a fresh
complete comparison, not relabelling or appending unrelated timed results.

The original JAX runs used **two workers**, 1,500 warmup transitions and
3.00 seconds per chain. Batch 20 used seeds 20,901,001–20,901,020; batch 21
used 21,901,001–21,901,020. Fresh study runs default to batch 22. Recorded
batches are explicitly labelled in the launcher. A stronger computer can
generate more iterations under the same time budget, so matching the seeds
does not promise identical sample counts or MSE.

Command-line equivalents, from the repository root:

```sh
# Fresh sampling with the reported maps and one reported batch's budget.
python run_experiment.py reported-zero-centre --target skew-t-100d --preset reproduce --workers 2 --output-dir results/reported-100d

# Match the first reported batch's seed inputs and sampling budget.
python run_experiment.py reported-zero-centre --target skew-t-100d --preset reproduce --workers 2 --seed-batch 20 --output-dir results/reported-100d-batch20

# Longer assessment on the same frozen 500D study configuration.
python run_experiment.py reported-zero-centre --target skew-t-500d --preset paper --workers 2 --output-dir results/reported-500d-paper
```

The separately labelled **Numba extension** uses the same maps and steps with
optimized DCS production and original JAX SCS. It is an implementation extension,
not the implementation that produced the reported JAX results.

## What this comparison establishes

Target, initial physical point, production budgets, concurrency, burn-in and
radial estimators follow common rules. Every rejected state remains in the
estimates and unthinned ESS. Compilation, asset loading, warmup, ESS computation
and file/report generation are outside the production clock and recorded separately.
Original fitting costs are retained as provenance; they are not charged again
when frozen maps are loaded.

DCS and SCS calibrated use settings selected under equal VI evaluation counts
and equal joint-pilot transition budgets. The production settings were frozen
before the reported assessment batches; no assessment-MSE reselection is used.
Original SCS provides a separate baseline with the authors' fixed h=0.1. The
compared maps and preparation rules differ, so this is not a change of sampling
kernel alone or a claim of identical total development effort. The bundled fit
and calibration provenance records these distinctions. There is no guarantee
that DCS wins on another run.

Both reported statistics are radial: mean norm and radial-tail probability.
Their MSE and ESS do not establish angular or marginal mixing. The 40-chain
results repeat sampling conditional on one frozen fit per family and dimension;
they do not assess variation across independently fitted maps. Small observed
MSE differences should not be described as decisive, especially with ν=3 norm errors.

## Calibration sensitivity and remaining limitations

The original-SCS-only comparison uses calibrated DCS against the authors'
original SCS settings. The added SCS calibrated variant addresses that radius/step
calibration imbalance. Exact target starts are available for this synthetic
example; neither method's oracle-assisted pilot is a general-purpose tuning
assumption for targets without exact draws.

The archive contains an **SCS jointly calibrated** control that used the same
524,288-transition pilot budget and retained its author-fitted centre and
observer while selecting R and h. Its settings were frozen before assessment.
In the recorded pooled 100D results:

| Method | Radial-tail MSE | Mean-norm MSE |
|---|---:|---:|
| Original SCS | 2.27e-7 | 5.21e-3 |
| SCS jointly calibrated | 1.60e-7 | 2.80e-3 |
| Ordinary DCS-BW | 2.03e-7 | 1.59e-3 |

The calibrated SCS control has 29.84% lower tail MSE and 46.14% lower norm MSE
than original SCS. Its tail-MSE ordering against ordinary BW reverses, while
BW retains lower norm MSE. These are recorded sensitivity results, not new
confirmatory runs or a claim that this is the best possible SCS tuning. The
[complete calibration archive](../studies/zero_centre/results/historical_calibration_archive.json)
retains both batches, all controls and full precision. Both original and calibrated
SCS are included in current runs, without selecting a baseline by its resulting MSE.

The pilot's radial objective is aligned with the reported radial functions.
For this sign-skewed target, radial functions have the same distribution as
under symmetric Student-t; these measurements do not expose asymmetric
marginal or angular mixing. They therefore support radial-efficiency claims
on this target, not overall dominance. Production time also excludes
preparation, so it is not total time to obtain a tuned sampler. Finally, the
frozen maps assess conditional sampling variability, and the Numba extension
includes framework differences relative to author JAX SCS. Neither establishes
universal sampler superiority.
