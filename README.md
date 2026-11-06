# Diffeomorphic Markov chain Monte Carlo

Research software accompanying *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

Run reproducible comparisons of DCS, the Sub-Cauchy Sampler (SCS), random-walk baselines and NUTS. Each run records its settings, maps, reference values and numerical outputs. The repository provides the current tests without an assumed performance ranking.

Start with the [launcher instructions](docs/CLICK_TO_RUN.md), [comparison protocol](docs/REPORTED_STUDY.md) or [experiment catalogue](docs/EXPERIMENTS.md).

## Install and run

Use **64-bit Python 3.12** and the pinned CPU dependencies in `requirements-cpu.txt`. Extract the repository ZIP before running it. Initial setup requires internet access. A GPU is not required.

| System | Open the launcher |
|---|---|
| Windows | Double-click `Start.bat`. |
| macOS | Double-click `Start.command`. |
| Any system with Python IDLE | Open `START_HERE.py` and select **Run → Run Module**. |

Click **Set up environment**, choose **DCS vs original SCS**, **Quick check**, and one or two CPU workers, then **Start run**. Keep the console or IDLE window open. After the check succeeds, choose Smaller reproduction or Full paper settings.

For terminal use on macOS or Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python run_experiment.py reported-zero-centre --preset quick --workers 1 --output-dir results/first-check
```

On Windows, create and activate the environment with `py -3.12 -m venv .venv` and `.venv\Scripts\activate.bat`, then use the same installation and experiment commands. Quick checks verify execution, not sampler performance.

## DCS and SCS comparison

The six methods are **DCS-RWM, DCS-BW, original SCS, SCS calibrated (zero), DCS-BW-reflect and DCS-Billiard**. Choose a 100D or 500D normal-CDF-skewed Student-t target with ν=3.

- **DCS vs original SCS** uses JAX for every method.
- **Numba extension** uses optimized Numba DCS production and unchanged author JAX SCS.
- Smaller uses 20 chains × 3 seconds per method; Full uses 50 chains × 20 seconds.
- Every preset loads the same maps, uses 1,500 separate warmup transitions, and discards the first third of production iterations.
- **Seed 1–Seed 8** select repeatable sampling streams without changing maps or steps.

Original SCS retains its author-fitted map and h=0.1. All DCS methods share a zero-centre, zero-deformation map and dimension-specific radius and step. The additional SCS control uses zero physical centre and observer, with R/h selected **before** those coordinates were zeroed. Its R/h have not been recalibrated for the zero map and are not claimed optimal.

The [protocol](docs/REPORTED_STUDY.md) gives exact settings and preparation budgets. Equal pilot counts do not make the map choices or complete preparation procedures identical.

## Other experiments

Six additional experiments cover 200D skew-t comparisons with NUTS and random-walk baselines, a fixed-iteration 50D t(0.5) target, logistic regression, Eight Schools and GARCH. Their DCS and ordinary random-walk production kernels use Numba; NUTS retains its compiled implementation. Logistic and super-heavy-tail SCS use the checked Numba implementation with original author VI.

**All seven experiments + posterior table** runs the JAX DCS/SCS comparison on the selected target and these six experiments. See [experiment settings](docs/EXPERIMENTS.md), [targets](docs/TARGETS.md) and [running instructions](docs/RERUN.md).

## Results

| Artifact | Contents |
|---|---|
| `configuration.json`, `manifest.json` | Settings, source hashes, environment, CPU allocation and run status |
| Map and fitting records | Source fits, effective maps and preparation |
| `summary.json`, `paper_values.csv`, `paper_values.html` | Results and full-precision exports |
| `functional_ess.csv` | Functional ESS, draw stride and unavailable reasons |
| `chains/`, `figures/` | Saved traces and metadata, PNG/PDF figures |

Reports include estimates, MSE, exact generated and post-burn-in counts, production time, ESS and ESS/s. Rejected repeats remain observations. Displays use two decimal places or scientific notation; integer counts remain exact. CSV/JSON retain full precision. Timed counts depend on hardware.

The main comparison measures radial mean norm and tail probability, not general or marginal superiority. The t(3) norm has no fourth moment, so empirical MSE uncertainty can be unstable. See [ESS](docs/ESS.md) and the [reviewer guide](docs/REVIEW_GUIDE.md).

## Citation and contributions

Cite the manuscript and record the exact software commit; see [CITATION.cff](CITATION.cff). Project code is [MIT licensed](LICENSE). SCS source, licenses and checksums are preserved in `third_party/scs_paper/`. PosteriorDB retains its own terms; see [third-party attribution](THIRD_PARTY.md).

See [validation](docs/VALIDATION.md), [contributing](CONTRIBUTING.md), [security](SECURITY.md) and the [basic sampler example](examples/basic_sampler.py).
