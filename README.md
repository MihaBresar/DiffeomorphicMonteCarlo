# Diffeomorphic Markov chain Monte Carlo

Research software accompanying *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

The repository provides sampling kernels, explicit experiment configurations, independent reference values, and scripts that generate figures and tables from saved runs. **The default launcher experiment now reproduces the reported zero-centre DCS study**, using its bundled maps and frozen parameters, with an additional SCS control using zero centre/observer and its prior pilot-selected radius and step. The previous fitted-centre version-11 recipe remains available only through its compatibility commands and saved reports. Neither reproduces the original paper figures unchanged; billiards and reflected BW are later extensions.

Start with the [reported study instructions](docs/REPORTED_STUDY.md), [reviewer guide](docs/REVIEW_GUIDE.md), or [experiment catalogue](docs/EXPERIMENTS.md). No general performance ranking is assumed; comparisons must be assessed for the functions and budgets measured.

## Install and run

Use **64-bit Python 3.12** and the pinned CPU dependencies in `requirements-cpu.txt`. Extract the repository ZIP before running it. Initial setup requires internet access; benchmark data are bundled. A GPU is not required.

### Desktop launcher

| System | Open the launcher |
|---|---|
| Windows | Double-click `Start.bat`. |
| macOS | Double-click `Start.command`. If the downloaded script cannot open, use IDLE as below. |
| Linux desktop, Windows or macOS with IDLE | In IDLE, open `START_HERE.py`, then select **Run → Run Module**. |

Click **Set up environment**, then leave **DCS vs original SCS** and **Quick check** selected. Start with one or two CPU workers. After it completes, the launcher links to the report and figures. Keep its console or IDLE window open while it runs. Open the Python launcher, rather than opening `launcher/index.html` directly. See [click-to-run instructions](docs/CLICK_TO_RUN.md) for details.

### Terminal

From the extracted repository directory, on macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

On Windows, in Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements-cpu.txt
```

Then run the smallest comparison:

```sh
python run_experiment.py reported-zero-centre --preset quick --workers 1 --output-dir results/first-check
```

This checks execution and report generation. Its short chains are not evidence of sampler performance. Python 3.12 is the supported environment; Linux and Windows checks are defined in [CI](.github/workflows/ci.yml), and macOS has local validation. Inspect the [Actions results](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/actions) for the status of a particular commit.

## Reported zero-centre study

Choose **100D** or **500D**, then **Smaller reproduction** (20 chains × 3 seconds)
or **Full paper settings** (50 chains × 20 seconds). Both retain the same frozen
full-budget maps, 1,500 separate warmup transitions and first-third production
discard. No manual radius, step or fitting choices are needed. Original SCS
retains its author-fitted map and h=0.1. All four DCS variants use zero centre
and deformation, with the study's fixed radius and step for each dimension.
Current runs use **SCS calibrated (zero)**, which sets physical centre and
observer displacement to zero while retaining its previously selected radius
and step. The R/h pilot was performed before zeroing these parameters; no
recalibration is claimed. Latitude remains 1.1. Both SCS
variants appear in the same six-method report; there is no extra control to select.

The original results used JAX and two workers. The default new seed batch is 22;
recorded batches 20 and 21 are available explicitly. The separately named Numba
option is an implementation extension. DCS and SCS calibrated used equal VI
evaluation budgets and equal radius/step pilot budgets: 524,288 transitions
per dimension from the same 4,096 exact target starts. The original SCS baseline
does not apply that pilot. Production timing excludes historical preparation
for both; current runs reuse the frozen settings with the declared SCS map override.
Equal historical counts do not imply equal elapsed preparation time or optimal
settings after the map change. Archived fitted-observer results are not results
for the new zero-observer control.

See the [full protocol and run instructions](docs/REPORTED_STUDY.md) and
[reported results and provenance](studies/zero_centre/results/index.html).
The original 40-chain pooled result combines two 20-chain batches; a fresh
Smaller run is one batch. Time budgets do not imply identical sample counts
or outcomes on another computer.

## Earlier recipe compatibility

The earlier fitted-map comparisons have been removed from the launcher. Their
`sampler-comparison` and `sampler-comparison-numba` command identifiers and
saved-report readers remain available for reproducibility. They use different
maps and h=0.07, and are not replications of the current zero-centre study.
See the [archived recipe specification](docs/COMPARISONS.md).

## Measurements and saved outputs

Current comparisons measure **radial-tail probability and mean norm**, with analytic reference values. Reports include per-chain estimates, MSE, exact generated and post-burn-in counts, production time, functional ESS and ESS/s. Rejected repeats contribute to every estimator and diagnostic. These radial functions do not establish marginal or overall mixing.

| Artifact | Contents |
|---|---|
| `configuration.json`, `manifest.json` | Resolved settings, source hashes, environment, CPU policy, run status and elapsed time |
| `maps.json`, `method_maps.json` | Fitted maps, fitting history and method-specific effective maps |
| `summary.json`, `paper_values.csv`, `paper_values.html` | Per-chain and aggregated results; machine-readable values retain full precision |
| `functional_ess.csv`, tuning CSV files | Functional ESS, unavailable reasons, and recorded pilot choices |
| `chains/`, `figures/` | Saved scalar traces and metadata, plus PNG/PDF figures |

Comparison ESS uses every post-burn-in scalar observation. Large unthinned temporary histories are removed after diagnostics; small saved traces can be thinned. Other experiments save retained physical draws and use their recorded diagnostic stride. See [ESS definitions](docs/ESS.md) and [output processing](docs/RESULT_PROCESSING.md).

Displayed estimates use two decimals or scientific notation, while integer sample counts stay exact. Empirical MSE variability is conditional on the fitted maps and available chains. The t(3) norm has no fourth moment, so rare errors can dominate MSE and its estimated uncertainty. ESS is not a convergence guarantee.

## Other experiments

The repository also includes 200D skew-t comparisons with NUTS and random-walk baselines, a fixed-iteration 50D t(0.5) experiment, logistic regression, Eight Schools and GARCH. Only the two posterior examples use PosteriorDB. Empirical targets need not satisfy the manuscript's sufficient theorem assumptions. See [target definitions](docs/TARGETS.md) and the [figure-by-figure rerun guide](docs/RERUN.md).

Every VI-enabled SCS fit uses the same pinned author routine, including logistic regression and the 50D t(0.5) experiment. Their SCS production kernels remain the checked Numba implementation with acceptance-target 0.234 step tuning. SCS fitting has its own recorded budget: 1,000 updates on 2,000 fixed draws for Full paper settings, 400 on 500 for Smaller reproduction, and 20 on 128 for Quick check. Experiments containing SCS use fixed clouds and matching per-family evaluation budgets. Version 10 established that fitting policy, including the full logistic DCS budget of 1,000 updates; version 11 leaves it unchanged. The v11 shared fixed DCS move applies only to the earlier fitted-map compatibility comparisons, not the other six experiments. Current frozen-map comparisons use their separately documented parameters and include both SCS variants.

The all-experiments runner includes the current JAX comparison (100D by default, or the selected 500D target) and those six experiments; it does not expand every target/backend combination. The [manuscript relationship](docs/MANUSCRIPT_CHANGES.md) explains the corrections and scope of current experiments. The current billiard variant is distinct from the original paper's BW sampler. New results cannot retrospectively validate the original figures.

## Citation, attribution and contributions

Cite the accompanying manuscript and record the exact software commit or archived release used. [CITATION.cff](CITATION.cff) contains the manuscript title and authors and the software metadata. No archive DOI is currently recorded; an archived publication release should add its verified DOI.

Project code is [MIT licensed](LICENSE). The unchanged SCS source is pinned to [`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565), with its original license and checksums preserved. PosteriorDB and other dependencies retain their own terms; see [third-party attribution](THIRD_PARTY.md).

For a new target, start with [the basic sampler example](examples/basic_sampler.py). For review or contributions, see [the reviewer guide](docs/REVIEW_GUIDE.md), [validation](docs/VALIDATION.md), [contributing](CONTRIBUTING.md) and [security policy](SECURITY.md).
