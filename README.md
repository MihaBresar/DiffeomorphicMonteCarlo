# Diffeomorphic Markov chain Monte Carlo

Reproducible experiments for *Diffeomorphic Markov chain Monte Carlo: fast mixing for heavy-tailed distributions*, by Miha Brešar and Aleksandar Mijatović.

This corrected implementation centralizes target densities and transformations, saves individual chains, and regenerates figures and tables from saved results. The current comparison suite offers a shared-JAX comparison and a separate optimized-Numba DCS comparison, each with two targets and VI on/off for both map families. SCS calls the unmodified author transition directly. All experiments freeze proposal tuning before sampling, then discard the first third of generated sampling iterations. The synthetic and logistic targets are local constructions; only Eight Schools and GARCH use PosteriorDB. Empirical examples are not required to satisfy the paper's sufficient theorem assumptions.

## Run with clicks — no commands needed

Choose one of two comparisons:

- **Sampler comparison (JAX):** DCS-RWM, DCS-BW, author SCS, DCS-BW-reflect and optimized **DCS-Billiard**, all in the same JAX timing/recording framework.
- **Optimized Numba comparison:** the same four DCS methods in fused native loops, versus the unchanged author SCS in JAX. This compares implementation efficiency, including different framework costs.

Both offer **100D or 500D skewed t(3)**, DCS β=2, and **VI on/off**. The current version 9 uses the pinned authors' VI routine and fixed h=0.1 for SCS, with acceptance tuning available as a sensitivity option. DCS fitting is unchanged. Ordinary RWM/BW tune toward acceptance 0.234; billiards tune their own production kernel by bounded physical radial movement, and reflected BW retains its ball-movement pilot. Every method receives the same separate pilot transition budget, and all production steps are frozen. Billiards retain the jointly fitted centre/R/β with zero effective deformation. The fitting recipes differ, so this is an author-baseline comparison rather than identical VI optimization. See [the proposal](docs/BILLIARD.md), [calibration protocol](docs/COMPARISONS.md), and [SCS fixed-step sensitivity](docs/AUTHOR_SCS_CALIBRATION.md).

New reports contain mean norm, radial-tail probability, MSE, exact sample counts, time, **functional ESS and ESS per second**. Coordinate statistics and the rotation comparison methods are removed. The original radial plots remain. See [the protocol](docs/COMPARISONS.md) and [ESS definitions](docs/ESS.md).

On Windows, extract the downloaded ZIP into a fresh folder and double-click
**Start.bat**. On macOS, double-click **Start.command**. Click **Set up
environment**, then run **Quick check**. After it succeeds, choose **Full paper
settings**, 100D or 500D, and **With VI** or **Without VI**. Select the JAX or
Numba five-sampler comparison and click **Start run**. Repeat the target/VI
combinations separately. Full settings use 50 chains per method at 20.00
seconds per chain, after separate tuning. On the Ryzen 9 9950X3D, **Use all CPU
threads** can select up to 32 workers. Results appear under the launcher.
See [the click-to-run guide](docs/CLICK_TO_RUN.md).

## Install and check from a terminal

Use **Python 3.12**, a CPU and a clean environment. Dependencies are pinned.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python -m examples.basic_sampler
python run_experiment.py sampler-comparison --preset quick
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
| `sampler-comparison` | 100D/500D skew-t(3), DCS β=2 | 50 chains/method, 20 s each; DCS-RWM/BW/BW-reflect/Billiard and SCS in JAX |
| `sampler-comparison-numba` | Same two selectable targets | Same budgets; optimized Numba DCS-RWM/BW/BW-reflect/Billiard versus author JAX SCS |
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

`--seed-batch b` accepts 0–999 and sets sampling seed 901001 + 1,000,000 × b, leaving DCS fit seed 42 and author SCS fit seed 0 unchanged. The comparison uses the manuscript's normal-CDF-skewed t density; it is not the SCS paper's Student-t-CDF example. SCS uses the authors' fixed-cloud VI and DCS retains fresh-batch VI; their optimizer settings and full fitting budgets differ and are recorded. Both execution modes reuse the same JAX step calibration. The [protocol](docs/COMPARISONS.md) lists these choices and clock scope.

Old SCS and boundary comparisons are removed from the launcher. Their command-line identifiers, archived results and replot support remain for provenance. The legacy `skew-t-100d-baselines` alias still forwards to the corrected 200D configuration.

All canonical benchmark kernels now use compiled sampling loops, including hit-and-run and the ordinary/spherical baselines. See [native implementation checks and measured speedups](docs/NATIVE_KERNELS.md) and [the HnR/NUTS audit](docs/HNR_NUTS.md).

The [4 November laptop results](docs/COMPARISON_RESULTS.md) cover all eight original-target/VI/backend combinations under version 2, with its four measurements (radial quantities and X₁): 560 chains, with full-precision errors, counts and timings. Coordinate and radial rankings differ. The earlier 3 November radial-only results remain documented separately; these smaller runs do not establish universal rankings. Version 3 added X₂/X₃ measurements; version 4 added the BW/rotation mixture. Neither addition is present in those archived results. Version 5 added reflected Gaussian RWM plus rotation. **Version 7 uses RWM, BW, reflected BW, billiards and SCS in both modes**, removes coordinate statistics and adds functional ESS. **Version 8 changed only the billiard/SCS calibration to equal-budget own-kernel radial pilots. Version 9 uses SCS author VI and defaults to fixed h=0.1, with acceptance adaptation available; DCS remains unchanged.** The older version 6 introduced JAX billiards. Run every method together in a fresh comparison for current figures. Archived results do not establish performance of later versions. See [the billiard construction and interpretation](docs/BILLIARD.md).

## Saved artifacts

The other six experiments save `configuration.json`, `manifest.json`, `summary.json`, `chain_metrics.csv`, and PNG/PDF figures. Each scenario contains map parameters and optimization history, its independent reference and provenance, and one lossless NPZ checkpoint per method/chain. Checkpoints contain all retained physical-coordinate draws and metadata; NumPy loads them with `allow_pickle=False`. Reference preparation has its own checkpoints. Failed or empty chains raise an error and leave the run marked failed; they are never replaced by zeros.

**200D random-walk baselines** uses a compact report for all presets: mean norm,
tail probability, their MSEs, generated and retained sample counts, and timing.
Each worker saves its two estimates after the one-third burn-in; the report
reads these small summaries. References come directly from the analytic radial
law. Functional ESS is recorded alongside these estimates; it skips R-hat, coordinate quantiles and directional
checks, keeping the radial boxplots, MSE bars and sample-count/runtime figures.

Workers stream draws through bounded buffers, preventing a second full-chain allocation at the end of sampling. Large reports process one chain and one cross-chain diagnostic quantity at a time, with progress messages and temporary disk storage. Saving chooses compression only when a bounded sample indicates useful space savings. Completed summaries can regenerate figures without repeating diagnostics. See [result processing](docs/RESULT_PROCESSING.md) for timing and storage details. Kernels, burn-in and retained-sample statistics are unchanged; new timed comparisons should rerun every method together because output streaming now occurs inside the production clock.

The general experiment reports, excluding the current comparison suite and the compact 200D baseline report, include chain estimates, MSE/RMSE and descriptive bootstrap intervals, functional MCSE, bulk/tail ESS, rank R-hat, NUTS divergences, tree saturation indicators, work counts, frozen tuning and timings. Unavailable diagnostics are null with an explanation. MSE is averaged over independent-chain squared errors. Posterior references remain finite Monte Carlo estimates, with their uncertainty reported.

The general skew-t runs, excluding the current comparison suite and the compact 200D baselines, also create a supplementary `*_directional` figure. Radial functions cannot assess whether skewness is sampled correctly, so these check a bounded directional probability and a bounded directional mean against independent references. The current comparison suite reports only the two original radial functions, with analytic references, ESS and ESS/s. Thresholds are fixed before sampling. This restricted set of functions does not establish overall mixing; archived coordinate reports retain their original scope.

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

The historical [own-kernel calibration results](docs/CALIBRATION_RESULTS.md) retain version-8 smaller JAX runs with VI on/off and matched historical-pilot controls. The current [author-SCS calibration study](docs/AUTHOR_SCS_CALIBRATION.md) separates map fitting from proposal-step sensitivity. The current billiard variant is distinct from the original paper's BW sampler; these comparisons do not reproduce or validate the original figures.
