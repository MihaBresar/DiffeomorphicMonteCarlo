# Diffeomorphic Markov chain Monte Carlo

Reproducible experiments for *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

This corrected implementation centralizes target densities and transformations, freezes tuning before retained sampling, saves individual chains, and regenerates figures and tables from saved results. The synthetic and logistic targets are local constructions; only Eight Schools and GARCH use PosteriorDB. Empirical examples are not required to satisfy the paper's sufficient theorem assumptions.

## Run with clicks — no commands needed

Download this repository using **Code → Download ZIP**, extract it, and install
**Python 3.12 (64-bit)**. On Windows double-click **Start.bat**; on macOS open
**Start.command**. Alternatively, open **START_HERE.py** in Python's IDLE editor
and choose **Run → Run Module** (F5).

In the browser, click **Set up environment**, then **Start new run** with
**Quick check** selected. Once that succeeds, choose **Full paper settings**
for the publication runs. Buttons also resume saved runs, rebuild figures,
and open results. See the [step-by-step click-to-run guide](docs/CLICK_TO_RUN.md)
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

On Windows activate with `.venv\Scripts\activate`. The local validation was on macOS ARM64; Linux is covered by the supplied CI configuration, not a claim of an already completed remote CI run. Other Python/JAX versions are not part of the tested environment.

## Regenerate results

```bash
# All short execution checks, including independent logistic references:
python scripts/reproduce_all.py --preset quick --output-root results/quick

# A smaller scientific reproduction (8 chains; not the paper configuration):
python scripts/reproduce_all.py --preset reproduce --output-root results/reproduce

# All full paper experiments and the combined posterior table:
python scripts/reproduce_all.py --preset paper --output-root results/paper

# Continue interrupted work; completed chains are checked and reused:
python scripts/reproduce_all.py --preset paper --output-root results/paper --resume

# Rebuild figures/tables without running samplers:
python scripts/reproduce_all.py --preset paper --output-root results/paper --plot-only
```

The full run is substantial. The four timed synthetic/regression comparisons alone require at least 7,900 seconds of sequential production sampling at the paper settings. Fixed-iteration SCS, both posterior comparisons, warmup, fitting and reference generation add to this. Actual CPU time and disk use depend on the machine. First measure a `reproduce` run; do not extrapolate performance from `quick` plots. See [the rerun guide](docs/RERUN.md) for individual commands and exact manuscript filenames.

## Experiments

| Identifier | Paper target | Production design |
|---|---|---|
| `skew-t-200d-nuts` | d=200, ν=3, α=(20,-30,0,…) | 50 chains/method, 20 s each |
| `skew-t-200d-baselines` | Same target | 50 chains/method, 10 s each |
| `skew-t-100d-fixed` | d=100, ν=3, α=(100,-100,0,…) | 20 chains/method, 500,000 steps each |
| `skew-t-50d-nu-0.5` | Symmetric t, d=50, ν=0.5 | 50 chains/method, 10 s each |
| `logistic-regression` | d=20, independent t priors, ν=4 and 6 | 20 chains/method/ν, 30 s each |
| `eight-schools` | Centered hierarchy, log τ coordinate | 50 chains, 150,000 NUTS steps; DCS matched to mean NUTS production time |
| `garch11` | Four-parameter PosteriorDB GARCH | Same posterior timing design |

The legacy name `skew-t-100d-baselines` forwards to the corrected **200-dimensional** configuration with a message. Every setting is explicit in `configs/<identifier>.json`; the paper profile no longer inherits script constants. `--dry-run` displays the resolved configuration.

## Saved artifacts

Each run saves `configuration.json`, `manifest.json`, `summary.json`, `chain_metrics.csv`, and PNG/PDF figures. Each scenario contains map parameters and optimization history, its independent reference and provenance, and one compressed NPZ checkpoint per method/chain. Checkpoints contain all retained physical-coordinate draws and metadata; NumPy loads them with `allow_pickle=False`. Reference preparation has its own checkpoints. Failed or empty chains raise an error and leave the run marked failed; they are never replaced by zeros.

Summary files report chain estimates, MSE/RMSE and descriptive bootstrap intervals, functional MCSE, bulk/tail ESS, rank R-hat, NUTS divergences, tree saturation indicators, work counts, frozen tuning and timings. Unavailable diagnostics are null with an explanation. MSE is averaged over independent-chain squared errors. Posterior references remain finite Monte Carlo estimates, with their uncertainty reported.

Each skew-t run also creates a supplementary `*_directional` figure. Radial functions cannot assess whether skewness is sampled correctly, so these check a bounded directional probability and a bounded directional mean against independent references.

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
