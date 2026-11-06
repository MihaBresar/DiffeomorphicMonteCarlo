# Diffeomorphic Markov chain Monte Carlo

Research software accompanying *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

The repository provides sampling kernels, explicit experiment configurations, independent reference values, and scripts that generate figures and tables from saved runs. It includes corrected manuscript experiments and subsequent sampler comparisons. **The current comparison is protocol version 11; it does not reproduce the original paper figures unchanged.** Billiards and reflected BW are later extensions. The documentation describes the current experiments so they can be independently run and inspected.

Start with the [reviewer guide](docs/REVIEW_GUIDE.md), [comparison protocol](docs/COMPARISONS.md), or [experiment catalogue](docs/EXPERIMENTS.md). No general performance ranking is assumed; comparisons must be assessed for the functions and budgets measured.

## Install and run

Use **64-bit Python 3.12** and the pinned CPU dependencies in `requirements-cpu.txt`. Extract the repository ZIP before running it. Initial setup requires internet access; benchmark data are bundled. A GPU is not required.

### Desktop launcher

| System | Open the launcher |
|---|---|
| Windows | Double-click `Start.bat`. |
| macOS | Double-click `Start.command`. If the downloaded script cannot open, use IDLE as below. |
| Linux desktop, Windows or macOS with IDLE | In IDLE, open `START_HERE.py`, then select **Run → Run Module**. |

Click **Set up environment**, then choose the JAX five-sampler comparison and **Quick check**. Start with one or two CPU workers. After it completes, the launcher links to the report and figures. Keep its console or IDLE window open while it runs. Open the Python launcher, rather than opening `launcher/index.html` directly. See [click-to-run instructions](docs/CLICK_TO_RUN.md) for details.

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
python run_experiment.py sampler-comparison --preset quick --workers 1 --output-dir results/first-check
```

This checks execution and report generation. Its short chains are not evidence of sampler performance. Python 3.12 is the supported environment; Linux and Windows checks are defined in [CI](.github/workflows/ci.yml), and macOS has local validation. Inspect the [Actions results](https://github.com/MihaBresar/DiffeomorphicMonteCarlo/actions) for the status of a particular commit.

## Current comparison

Both modes contain **DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and DCS-Billiard**.

| Command identifier | Production implementation |
|---|---|
| `sampler-comparison` | All five methods in the same JAX timing and recording framework |
| `sampler-comparison-numba` | Optimized Numba DCS kernels versus the unchanged author SCS transition in JAX |

The second mode measures implementation efficiency, including different framework and output costs. It does not isolate transition geometry.

Both support the normal-CDF-skewed Student-t target with ν=3, α=(100,−100,0,…), dimension **100 or 500**, and DCS exponent β=2. Select `--target skew-t-100d` or `--target skew-t-500d`. Both map families are fitted by VI in every new comparison. The 500D target is an extension. These targets differ from the Student-t-CDF-skewed target in the SCS paper.

SCS uses its authors' unchanged VI routine; DCS retains its own VI fit. Their fitting recipes differ; both receive the same number of objective/gradient evaluations, with full budgets of 1,000 updates on 2,000 points per update. This compares the fitted sampling procedures; it does not isolate proposal geometry from map fitting. Billiards retain the ball fit's centre, scale and β in post-Möbius coordinates, with zero effective deformation; the other DCS methods retain the original ball map.

All four DCS methods share one fixed move h=0.07: BW, reflected BW and billiard use a uniform-ball increment of radius h; RWM uses Gaussian coordinate standard deviation h/√(d+2). This matches the increments’ coordinate covariances, not their physical endpoints or acceptance rates. Each method uses its own kernel during discarded equilibration, with no separate step adaptation. The single SCS baseline uses the authors' fixed **h=0.1** and unchanged author VI/transition; alternative SCS calibration is not offered in new five-method comparisons. Every method receives the same discarded warmup-transition budget. All production steps are frozen, and the first third of generated production iterations is discarded.

The launcher and public commands use the protocol's preset DCS h=0.07 automatically; there is no manual step-size or calibration-policy choice. This preset is not a claim of optimality or guaranteed superiority. Internal configuration helpers retain historical settings for study reproduction. Version 11 preserves the version-10 fixed-cloud VI maps and changes all four DCS proposal-step policies, so new sampling is required for a v11 comparison. Fresh-cloud VI and zero-centre studies remain experimental and do not change the public fitting default. Public runs keep the fitted physical centres; they do not apply a zero-centre override or post-VI radius calibration.

The current suite retains only this original SCS baseline. Historical alternative-step results remain recorded for reproducibility. The fixed SCS choice followed a documented 100D development study; it was frozen before the planned full-paper seed-batch-8 assessment. It is not a demonstrated optimum for 500D. See [all sensitivity results and the selection rule](docs/AUTHOR_SCS_CALIBRATION.md).

| Preset | Chains per method | Production seconds per chain | Separate pilot transitions |
|---|---:|---:|---:|
| `quick` | 2 | 0.05 | 100 |
| `reproduce` | 20 | 2 | 1,000 |
| `paper` | 50 | 20 | 1,500 |

Smaller presets also reduce VI training. Full VI uses 1,000 updates and 2,000 objective/gradient evaluation points per update for each family. Each family reuses its own fixed reference cloud throughout fitting. This matches update counts, cloud size and objective/gradient evaluation counts; the base distributions, parameterizations, seeds and clipping rules remain method-specific. Fitting time need not match. The [protocol](docs/COMPARISONS.md) specifies the complete budgets and [reproducibility guide](docs/REPRODUCIBILITY.md) defines costs, seeds and diagnostics.

```sh
# Inspect the resolved configuration without sampling.
python run_experiment.py sampler-comparison --preset paper --dry-run

# Smaller 100D comparison with fitted maps.
python run_experiment.py sampler-comparison --target skew-t-100d --preset reproduce --workers 2 --output-dir results/jax-100d-vi

# Full 500D comparison with the frozen default and a fresh sampling batch.
python run_experiment.py sampler-comparison --target skew-t-500d --preset paper --workers 2 --seed-batch 8 --output-dir results/jax-500d-batch-8

# Regenerate a completed report without sampling.
python run_experiment.py --plot-only results/jax-100d-vi
```

Choose a fresh output directory for each comparison. Set the worker count to the available CPU allocation and keep it fixed across comparisons. Timed counts vary with hardware and system load even when seeds agree. Run the complete method set together; do not mix timing results from different code or execution policies.

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

Every VI-enabled SCS fit uses the same pinned author routine, including logistic regression and the 50D t(0.5) experiment. Their SCS production kernels remain the checked Numba implementation with acceptance-target 0.234 step tuning. SCS fitting has its own recorded budget: 1,000 updates on 2,000 fixed draws for Full paper settings, 400 on 500 for Smaller reproduction, and 20 on 128 for Quick check. Experiments containing SCS use fixed clouds and matching per-family evaluation budgets. Version 10 established that fitting policy, including the full logistic DCS budget of 1,000 updates; version 11 leaves it unchanged. The v11 shared fixed DCS move applies to the two five-method comparisons, not the other six experiments. Rerun all five compared methods together to measure the new step policy.

The all-experiments runner includes the JAX 100D comparison and those six experiments; it does not expand every target/backend combination. The [manuscript relationship](docs/MANUSCRIPT_CHANGES.md) explains the corrections and scope of current experiments. The current billiard variant is distinct from the original paper's BW sampler. New results cannot retrospectively validate the original figures.

## Citation, attribution and contributions

Cite the accompanying manuscript and record the exact software commit or archived release used. [CITATION.cff](CITATION.cff) contains the manuscript title and authors and the software metadata. No archive DOI is currently recorded; an archived publication release should add its verified DOI.

Project code is [MIT licensed](LICENSE). The unchanged SCS source is pinned to [`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565), with its original license and checksums preserved. PosteriorDB and other dependencies retain their own terms; see [third-party attribution](THIRD_PARTY.md).

For a new target, start with [the basic sampler example](examples/basic_sampler.py). For review or contributions, see [the reviewer guide](docs/REVIEW_GUIDE.md), [validation](docs/VALIDATION.md), [contributing](CONTRIBUTING.md) and [security policy](SECURITY.md).
