# Diffeomorphic Markov chain Monte Carlo

Reproducible experiments for *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

This corrected implementation centralizes target densities and transformations, saves individual chains, and regenerates figures and tables from saved results. The default 100D comparison uses original DCS fitting and separate step tuning and uses the SCS authors’ original implementation. All experiments freeze proposal tuning before sampling, then discard the first third of generated sampling iterations. The synthetic and logistic targets are local constructions; only Eight Schools and GARCH use PosteriorDB. Empirical examples are not required to satisfy the paper's sufficient theorem assumptions.

## Run with clicks — no commands needed

Download this repository using **Code → Download ZIP**, extract it, and install
**Python 3.12 (64-bit)**. On Windows double-click **Start.bat**; on macOS open
**Start.command**. Alternatively, open **START_HERE.py** in Python's IDLE editor
and choose **Run → Run Module** (F5).

In the browser, click **Set up environment**, then **Start run** with
**Quick check** selected. Choose **CPU workers** to run independent chains in
parallel (start with 2–4, or click **Use all CPU threads** for the maximum).
The Ryzen 9 9950X3D can use 32 workers when all its threads are available.
Once that succeeds, choose **Full paper settings**
for the publication runs. Links to the figures and tables appear automatically
when the run finishes. See the [step-by-step click-to-run guide](docs/CLICK_TO_RUN.md)
for installation, moving to another computer, and troubleshooting. These
experiments run locally on your computer.

## Install and check from a terminal

Use **Python 3.12**, a CPU, and a clean environment. The complete dependency closure is pinned in `requirements-cpu.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python -m examples.basic_sampler
python run_experiment.py skew-t-50d-nu-0.5 --preset quick
```

On Windows activate with `.venv\Scripts\activate`. Local validation uses macOS ARM64. CI checks CPU workers on Linux and Windows; consult the repository's Actions tab for the latest results. Other Python/JAX versions are not part of the tested environment.

## Regenerate results

```bash
# All short execution checks, including independent logistic references:
python scripts/reproduce_all.py --preset quick --output-root results/quick

# Smaller reproductions (20 chains for the default SCS comparison, otherwise 8):
python scripts/reproduce_all.py --preset reproduce --output-root results/reproduce

# All full paper experiments and the combined posterior table:
python scripts/reproduce_all.py --preset paper --output-root results/paper

# The same pipeline with four parallel CPU workers:
python scripts/reproduce_all.py --preset paper --workers 4 --output-root results/paper-4-workers

# Continue interrupted work; completed chains are checked and reused:
python scripts/reproduce_all.py --preset paper --output-root results/paper --resume

# Rebuild figures/tables without running samplers:
python scripts/reproduce_all.py --preset paper --output-root results/paper --plot-only
```

The full run is substantial. Timed synthetic/regression comparisons require thousands of seconds of aggregate per-chain production sampling. Parallel workers overlap that work; speedup depends on CPU and memory limits. The fixed-iteration super-heavy-tail comparison, both posterior comparisons, warmup, fitting and reference generation add to this. Actual CPU time and disk use depend on the machine. First measure a `reproduce` run; do not extrapolate performance from `quick` plots. See [the rerun guide](docs/RERUN.md) for individual commands and exact manuscript filenames, and [parallel execution](docs/PARALLEL.md) for CPU allocation, timing and resume rules.

## Experiments

| Identifier | Paper target | Production design |
|---|---|---|
| `skew-t-200d-nuts` | d=200, ν=3, α=(20,-30,0,…) | 50 chains/method, 20 s each |
| `skew-t-200d-baselines` | Same target | 50 chains/method, 10 s each |
| `skew-t-100d-fixed` | d=100, ν=3, α=(100,-100,0,…) | 50 chains/method, 20 seconds each; original DCS-RWM/BW and author SCS |
| `skew-t-50d-nu-0.5` | Symmetric t, d=50, ν=0.5 | 50 chains/method, 500,000 steps each; discard 166,667 |
| `logistic-regression` | d=20, independent t priors, ν=4 and 6 | 20 chains/method/ν, 30 s each |
| `eight-schools` | Centered hierarchy, log τ coordinate | 50 chains, 150,000 NUTS steps; DCS matched to mean NUTS production time |
| `garch11` | Four-parameter PosteriorDB GARCH | Same posterior timing design |

The legacy name `skew-t-100d-baselines` forwards to the corrected **200-dimensional** configuration with a message. The default comparison’s presets are explicit in `scripts/reproduce_scs_100d.py`; other experiment settings are in `configs/<identifier>.json`. `--dry-run` displays the resolved configuration.

The default **100D skewed t (ν=3): DCS-RWM, DCS-BW and SCS** is one equal-time comparison,
with equal sampling time and measured iteration counts. The first third of each sampling run is discarded after separate tuning. DCS uses its original
VI and separate smoothed step tuning toward 0.234, frozen before timed sampling; SCS uses the author’s original
VI and fixed step 0.1. Both implementations compile their production loops before
timing. The shared target is the original DCS normal-CDF skew-t; the SCS paper’s
Student-t-CDF target is different. See [the exact protocol](docs/SCS_100D.md).
The command-line identifier routes to the same runner as the launcher.

Earlier fitting, symmetric, equal-time and joint-radius audits remain in `scripts`,
`validation` and their protocol documents for provenance. They are no longer
launcher choices.

All canonical benchmark kernels now use compiled sampling loops, including hit-and-run and the ordinary/spherical baselines. See [native implementation checks and measured speedups](docs/NATIVE_KERNELS.md) and [the HnR/NUTS audit](docs/HNR_NUTS.md).

## Saved artifacts

The other six experiments save `configuration.json`, `manifest.json`, `summary.json`, `chain_metrics.csv`, and PNG/PDF figures. Each scenario contains map parameters and optimization history, its independent reference and provenance, and one lossless NPZ checkpoint per method/chain. Checkpoints contain all retained physical-coordinate draws and metadata; NumPy loads them with `allow_pickle=False`. Reference preparation has its own checkpoints. Failed or empty chains raise an error and leave the run marked failed; they are never replaced by zeros.

Workers stream draws through bounded buffers, preventing a second full-chain allocation at the end of sampling. Large reports process one chain and one cross-chain diagnostic quantity at a time, with progress messages and temporary disk storage. Saving chooses compression only when a bounded sample indicates useful space savings. Completed summaries can regenerate figures without repeating diagnostics. See [result processing](docs/RESULT_PROCESSING.md) for timing and storage details. Kernels, burn-in and retained-sample statistics are unchanged; new timed comparisons should rerun every method together because output streaming now occurs inside the production clock.

Summary files report chain estimates, MSE/RMSE and descriptive bootstrap intervals, functional MCSE, bulk/tail ESS, rank R-hat, NUTS divergences, tree saturation indicators, work counts, frozen tuning and timings. Unavailable diagnostics are null with an explanation. MSE is averaged over independent-chain squared errors. Posterior references remain finite Monte Carlo estimates, with their uncertainty reported.

Each of the other skew-t runs also creates a supplementary `*_directional` figure. Radial functions cannot assess whether skewness is sampled correctly, so these check a bounded directional probability and a bounded directional mean against independent references.

The original Q–Q, radial boxplot and linear MSE figure families are restored,
including both fixed-time and fixed-iteration comparisons. Additional directional
and MSE-uncertainty plots live in `figures/supplementary`. Plot labels and HTML
tables use two decimals, with scientific notation for MSE and small values.
Integer sample counts stay exact; CSV and JSON files retain full numerical
precision. See [the figure guide](docs/FIGURES.md).

For experiments other than the default 100D comparison, the **DCS map** selector offers **Möbius map** (the default VI fit of R, centre
and Möbius parameters) and **Vanilla map** (fully untuned R=1, centre=0 and
Möbius=0). Vanilla retains the experiment's β and skips map fitting entirely.
Both choices keep proposal-scale warmup and freezing unchanged; SCS keeps its
existing fit. They create separate experimental configurations, labelled in the
figures and exact-value tables. The two PosteriorDB experiments keep their
identity maps. For command-line runs use `--ball-map-policy mobius` or
`--ball-map-policy vanilla`. Fixed R=√d has been removed from new runs; saved
results can still be replotted. See [the implementation audit](docs/IMPLEMENTATION_AUDIT.md)
for earlier tuning comparisons and their limits.

## Use the sampler on another target

```python
import numpy as np
from dcs_bench.api import sample

log_density = lambda x: -2.5 * np.log1p(np.dot(x, x) / 3)
x, info = sample(log_density, initial=np.zeros(2), draws=10000,
                 warmup=1500, beta=2, seed=81)
```

`sample` needs no gradients; it accepts DCS-BW, DCS-RWM and DCS-HnR. Benchmark map fitting uses JAX gradients. [Target and coordinate definitions](docs/TARGETS.md), [timing and reproducibility](docs/REPRODUCIBILITY.md), [validation](docs/VALIDATION.md), and [manuscript changes](docs/MANUSCRIPT_CHANGES.md) explain the scientific choices.

Code is [MIT licensed](LICENSE). Bundled third-party materials retain their upstream license; see [THIRD_PARTY.md](THIRD_PARTY.md). Citation metadata are in `CITATION.cff`. A publication release should archive the exact code and successful paper result directories together; add the archive DOI to the citation when it exists.
