# Diffeomorphic Markov chain Monte Carlo

Reproducible experiments for *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

This corrected implementation centralizes target densities and transformations, saves individual chains, and regenerates figures and tables from saved results. The current comparison suite offers a shared-JAX comparison and a separate optimized-Numba DCS comparison, each with two targets and VI on/off for both map families. SCS calls the unmodified author transition directly. All experiments freeze proposal tuning before sampling, then discard the first third of generated sampling iterations. The synthetic and logistic targets are local constructions; only Eight Schools and GARCH use PosteriorDB. Empirical examples are not required to satisfy the paper's sufficient theorem assumptions.

## Run with clicks — no commands needed

Choose one of two comparisons:

- **Sampler comparison (JAX):** DCS-RWM, DCS-BW, author SCS, reflected DCS-BW, DCS-BW-rotate and DCS-RWM-rotate, all in the same JAX timing/recording framework.
- **Optimized Numba comparison:** DCS-BW, reflected DCS-BW, DCS-BW-rotate and DCS-RWM-rotate with fused native loops, versus the unmodified author SCS in JAX. This measures implementation efficiency; it does not equalize framework and output-processing costs.

Each offers **100D skewed t(3)** or **500D skewed t(3)**, both with DCS β=2, and **VI on/off for both map families**. Step sizes are always pretuned separately and frozen before sampling. Reflected BW uses ball-movement calibration without a hard ceiling of 2; ordinary BW, RWM and SCS target acceptance 0.234. DCS-BW-rotate chooses reflected BW or a fixed-radius angular move with equal probability; it splits the same tuning budget between the two proposal scales and freezes both. DCS-RWM-rotate uses the same mixture with reflected Gaussian increments, whose coordinate scale starts at `0.1/sqrt(d+2)` before separate tuning. All earlier controls remain. Reports preserve all radial plots and add means and prespecified tails for X₁, X₂ and X₃, with analytic or independent references and their precision shown. See [the comparison protocol](docs/COMPARISONS.md).

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

On Windows, use the latest ZIP, open **Start.bat**, choose a comparison, target and VI setting, then **Smaller reproduction** or **Full paper settings**. On the Ryzen 9 9950X3D you can choose **32 CPU workers**. Both comparisons offer original sampling seeds (batch 0) and fresh batches 1–5; the map-fitting seed remains 42. Repeating a batch reuses its random streams, but wall-clock stopping can change sample counts and MSE. Compare independent batches to assess variability conditional on the fitted maps.

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
| `sampler-comparison` | 100D/500D skew-t(3), DCS β=2 | 50 chains/method, 20 s each; DCS-RWM/BW/reflect/BW-rotate/RWM-rotate and SCS in JAX |
| `sampler-comparison-numba` | Same two selectable targets | Same budgets; optimized Numba DCS-BW/reflect/BW-rotate/RWM-rotate versus author JAX SCS |
| `skew-t-50d-nu-0.5` | Symmetric t, d=50, ν=0.5 | 50 chains/method, 500,000 steps each; discard 166,667 |
| `logistic-regression` | d=20, independent t priors, ν=4 and 6 | 20 chains/method/ν, 30 s each |
| `eight-schools` | Centered hierarchy, log τ coordinate | 50 chains, 150,000 NUTS steps; DCS matched to mean NUTS production time |
| `garch11` | Four-parameter PosteriorDB GARCH | Same posterior timing design |

**All seven experiments** includes the JAX comparison on the original 100D target, plus the other six experiments; it does not expand into all eight target/VI/backend combinations. Run the two comparisons individually to select the 500D target. Their presets are defined in `scripts/reproduce_scs_100d.py` and resolved by `scripts/reproduce_comparison.py`; the other six use `configs/<identifier>.json`. `--dry-run` shows the exact configuration.

```bash
python run_experiment.py sampler-comparison --target skew-t-100d --vi on --preset reproduce --workers 4 --output-dir results/jax-100d-vi
python run_experiment.py sampler-comparison-numba --target skew-t-100d --vi off --preset reproduce --workers 4 --output-dir results/numba-100d-no-vi
python run_experiment.py sampler-comparison --target skew-t-500d --vi on --preset paper --workers 32 --seed-batch 1 --output-dir results/jax-500d-batch-1
```

`--seed-batch b` accepts 0–999 and sets sampling seed 901001 + 1,000,000 × b, leaving fit seed 42 unchanged. The comparison uses the manuscript's normal-CDF-skewed t density; it is not the SCS paper's Student-t-CDF example. The two map families have equal VI budgets, and the two implementation modes reuse the same JAX step calibration. The [protocol](docs/COMPARISONS.md) lists the remaining method-specific choices and clock scope.

Old SCS and boundary comparisons are removed from the launcher. Their command-line identifiers, archived results and replot support remain for provenance. The legacy `skew-t-100d-baselines` alias still forwards to the corrected 200D configuration.

All canonical benchmark kernels now use compiled sampling loops, including hit-and-run and the ordinary/spherical baselines. See [native implementation checks and measured speedups](docs/NATIVE_KERNELS.md) and [the HnR/NUTS audit](docs/HNR_NUTS.md).

The [4 November laptop results](docs/COMPARISON_RESULTS.md) cover all eight original-target/VI/backend combinations under version 2, with its four measurements (radial quantities and X₁): 560 chains, with full-precision errors, counts and timings. Coordinate and radial rankings differ. The earlier 3 November radial-only results remain documented separately; these smaller runs do not establish universal rankings. Version 3 added X₂/X₃ measurements; version 4 adds the BW/rotation mixture. Neither addition is present in those archived results. Version 5 adds reflected Gaussian RWM plus rotation. Run every method together in a fresh version-5 comparison.

## Saved artifacts

The other six experiments save `configuration.json`, `manifest.json`, `summary.json`, `chain_metrics.csv`, and PNG/PDF figures. Each scenario contains map parameters and optimization history, its independent reference and provenance, and one lossless NPZ checkpoint per method/chain. Checkpoints contain all retained physical-coordinate draws and metadata; NumPy loads them with `allow_pickle=False`. Reference preparation has its own checkpoints. Failed or empty chains raise an error and leave the run marked failed; they are never replaced by zeros.

**200D random-walk baselines** uses a compact report for all presets: mean norm,
tail probability, their MSEs, generated and retained sample counts, and timing.
Each worker saves its two estimates after the one-third burn-in; the report
reads these small summaries. References come directly from the analytic radial
law. This experiment skips ESS, R-hat, coordinate quantiles and directional
checks, keeping the radial boxplots, MSE bars and sample-count/runtime figures.

Workers stream draws through bounded buffers, preventing a second full-chain allocation at the end of sampling. Large reports process one chain and one cross-chain diagnostic quantity at a time, with progress messages and temporary disk storage. Saving chooses compression only when a bounded sample indicates useful space savings. Completed summaries can regenerate figures without repeating diagnostics. See [result processing](docs/RESULT_PROCESSING.md) for timing and storage details. Kernels, burn-in and retained-sample statistics are unchanged; new timed comparisons should rerun every method together because output streaming now occurs inside the production clock.

The general experiment reports, excluding the current comparison suite and the compact 200D baseline report, include chain estimates, MSE/RMSE and descriptive bootstrap intervals, functional MCSE, bulk/tail ESS, rank R-hat, NUTS divergences, tree saturation indicators, work counts, frozen tuning and timings. Unavailable diagnostics are null with an explanation. MSE is averaged over independent-chain squared errors. Posterior references remain finite Monte Carlo estimates, with their uncertainty reported.

The general skew-t runs, excluding the current comparison suite and the compact 200D baselines, also create a supplementary `*_directional` figure. Radial functions cannot assess whether skewness is sampled correctly, so these check a bounded directional probability and a bounded directional mean against independent references. The current comparison suite keeps those radial metrics and adds E[X₁], E[X₂], E[X₃] and P(X₁ > c), P(X₂ < −c), P(X₃ > c). The mirrored X₂ check assesses the other skewed coordinate, while X₃ tests a symmetric Student-t marginal. Thresholds are fixed before sampling. Independent reference estimates include MCSE, and no set of these functionals establishes overall mixing.

The original Q–Q, radial boxplot and linear MSE figure families are restored,
including both fixed-time and fixed-iteration comparisons. Additional directional
and MSE-uncertainty plots live in `figures/supplementary`. Plot labels and HTML
tables use two decimals, with scientific notation for MSE and small values.
Integer sample counts stay exact; CSV and JSON files retain full numerical
precision. See [the figure guide](docs/FIGURES.md).

The current comparisons also export inexpensive empirical MSE
uncertainty summaries computed from per-chain squared errors. These describe
the saved run; they are not formal 95% confidence intervals or uncertainty
over refitting the maps. The t(3) norm has an infinite fourth moment, so its
squared-error variance need not exist and empirical error bars can miss rare
large errors. Keep this limitation in mind when comparing MSE across batches.

For the six other experiments, the **DCS map** selector offers **Möbius map** (the default VI fit of R, centre
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
