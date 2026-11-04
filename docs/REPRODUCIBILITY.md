# Reproducibility and timing policy

The [current selectable-target comparisons](COMPARISONS.md) record protocols
`timed-sampler-comparison-v8` and `timed-numba-comparison-v8`. Both include
DCS-RWM, DCS-BW, author SCS, DCS-BW-reflect and DCS-Billiard, 100D/500D targets, and VI on/off.
The same JAX pilot initializes each DCS method in either implementation mode;
Numba production uses an independent PCG64 stream and fused native summaries.
SCS calls the unmodified author transition. Billiards retain the jointly fitted
centre/R/β with zero production deformation after Möbius; this is not a refit.
Billiard and SCS use their own production kernels in equal-budget radial pilots;
actual pilot seconds are recorded separately and need not match. See
[the calibration protocol](COMPARISONS.md) and [construction](BILLIARD.md).
Reflected BW keeps its shared ball map
and ball-movement pilot with no geometric step ceiling of 2. All steps freeze and one-third of actual
production iterations is discarded.

Only mean norm and radial-tail probability are measured in the comparisons.
Their references are analytic. Functional ESS uses unthinned scalar histories
including rejections, captured inside each sampling clock; ESS computation is
outside the clock and serialized to bound workspace memory. ESS/s uses summed
per-chain ESS divided by summed full production seconds. It is a descriptive
rate, not the ESS of a concatenated unequal-length pooled chain. Unavailable
chains remain visible. See [the diagnostic definition](ESS.md). Historical
coordinate/rotation reports preserve their recorded scope.

All new runs use CPU policy **`pjrt-single-compute-thread-v1`**, setting
`PJRT_NPROC=1` before JAX backend creation for both DCS and SCS. Earlier
Eigen-only limits did not constrain every generated CPU kernel: the laptop
audit found dimension-dependent extra CPU consumption and distorted sampling
rates. The common correction changes execution resources, not proposal or
tuning rules. Runtime helper threads can remain. Manifests record the policy
and complete thread settings; generic chain records do too. **Rerun the whole
method comparison for corrected timed results**, since changed throughput also
changes retained counts and MSE. Do not combine old and new timing cohorts.
The CPU correction itself preserved version-5 sampling rules. Version 7
simplified both suites and added native billiards and functional ESS. Version 8
changes billiard/SCS calibration to equal-budget own-kernel radial pilots. Source hashes
and the runtime policy distinguish the CPU correction. See [parallel execution](PARALLEL.md).

The policies below describe the six other main experiments.

All paper inputs are explicit JSON files. The manifest records their resolved values, scientific source hashes, Git commit/status when available, Python and package versions, OS/CPU count, JAX devices, float64/CPU policy and numerical-library thread settings. A run made from a dirty or uncommitted checkout records that fact; use a tagged clean checkout for the publication archive.

Benchmarks support bounded parallel chains through `--workers N` and the launcher's CPU-worker selector. Methods run one at a time. Workers use fresh Python processes, single-thread numerical-library settings, and distinct allowed CPU threads on Linux/Windows, filling physical cores before SMT siblings; macOS scheduling remains OS-managed. Worker allocation, actual thread settings and total invocation wall time are recorded. Hardware sharing, thermal throttling and other active processes still affect wall-time experiments. Keep the same worker count for comparisons and resumed runs; see [parallel execution](PARALLEL.md). The common log-density work counts are complementary to timing; a target evaluation is not assumed to cost the same as a gradient evaluation.

Map fitting, NUTS compilation and warmup happen before the production clock. JAX batches are synchronized at each timing boundary. NumPyro warmup and production loops are compiled separately, without discarding an entire extra production run. BlackJAX timed sampling uses batches of 16 iterations at paper scale; it completes the last batch and reports the resulting overshoot. Metropolis/slice timing can overrun by its last transition. For a paper comparison, inspect overshoots relative to the stated budgets and report actual measured sampling time.

Generic experiment workers collect output with an 8 MiB buffered temporary
file instead of keeping all batches and concatenating a second full array.
Thinning, physical conversion, scalar NUTS diagnostic collection, draw writes
and the final buffer flush are inside the production clock. Final mapping,
NPZ serialization and report construction are outside it. NumPyro fixed-count
production uses at most 1,024 transitions per compiled batch and an exact
remainder, with the random state carried across batches. When comparing timing
results, rerun the complete method set with this collection policy; do not mix
previous in-memory timings with new streamed timings. Sample selection and
the fixed-step transition sequences are unchanged.

All proposal adaptation stops at the end of warmup. The fixed-iteration experiment retains samples from 500,000 subsequent steps with thinning 50. Other paper runs thin by 10; every retained physical-coordinate draw is saved. Thinning is storage policy, not an efficiency claim. ESS and MCSE refer to retained draws. NUTS uses target acceptance 0.8 and maximum depth 10 in the benchmark; reference NUTS uses stricter settings. These replace inconsistent hidden caps/defaults in the older scripts.

Synthetic map fitting is shared across chains and the resulting results are conditional on that fit. Optimizer seeds, settings, all training losses and fitted map parameters are saved. For per-method total-cost reporting, the full corresponding fit/compilation cost is amortized over that method's chains. Fitting once is not equivalent to independently rerunning the entire fitting pipeline for every replicate.

Each new configuration records `ball_map_policy`: `mobius` fits DCS map parameters
by VI; `vanilla` skips map training and uses R=1, centre=0 and Möbius=0. Vanilla
retains the configured β and records zero optimizer steps and training draws.
The proposal-scale warmup procedure and the SCS comparator are the same for both
choices. The two PosteriorDB experiments retain their identity maps regardless
of this selector. Use separate run folders for map ablations; the figures and
exact-value tables identify the selected map.

Posterior DCS receives the mean **production** time of the fixed-count NUTS runs; setup is reported separately for both. The quick profile alone imposes a one-second minimum DCS budget to ensure a useful execution check. Quick posterior comparisons therefore do not represent an equal-time experiment.

`summary.json` and `chain_metrics.csv` report production cost and amortized complete cost. NUTS work counters count integrator steps (gradient evaluations inside trajectories); initialization, warmup and compilation are reported separately and are not silently included in that count. NumPyro's tree-saturation flag detects a full tree through its recorded step count; it is not an exact count of every trajectory that entered the final doubling level.

The mean estimator is formed within each chain first. MSE is the mean of those chains' squared errors, and RMSE its square root. Posterior table RMSE additionally averages squared errors over physical coordinates before taking the square root. The coordinate-wise values are also saved because parameters have different units. PosteriorDB and logistic references have finite Monte Carlo uncertainty and are not labelled exact. Approximate MSE bootstrap intervals describe variability across the available runs; they do not establish reliable heavy-tail asymptotics, especially for unbounded functions without fourth moments.

The 200D random-walk baseline experiment reports the two radial estimates,
their MSEs, sample counts, timing, functional ESS and ESS per second. It uses
analytic radial references and skips R-hat, coordinate quantiles and directional
checks. Workers cache estimates and functional ESS from the retained draws
after burn-in; reporting reads these small records. ESS calculation follows
production and has its own recorded cost. Historical records without cached
ESS require one pass over the saved chain. This does not change sample
selection, the sampler or tuning. See [the ESS definitions](ESS.md); the broader
diagnostic rules below apply to the other generic experiments.

Multichain diagnostics use a common prefix of length equal to the shortest chain and record that length. Estimator means and per-chain diagnostics use all retained draws. Reporting streams one chain at a time and stages functional values on disk; cross-chain diagnostics process one functional at a time. This changes memory use, not the diagnostic definitions or sample selection. A constant functional within a chain yields unavailable MCSE/ESS rather than a reassuring zero. Rank R-hat is not computed from a single chain.

Atomic per-chain NPZ files allow restart after process interruption. Resume validates the source/configuration fingerprint and every checkpoint; an incomplete chain is restarted from its seed. Plot-only uses the saved run's completed summary when available, with no simulation or diagnostic recomputation. Add `--recompute-diagnostics` to `--plot-only SAVED_RUN` to rebuild all statistics from the original retained chains. A missing summary is rebuilt automatically. A wall-time run is not bitwise deterministic across machines; fixed-step runs with the same environment and seeds are more strictly reproducible.

Before release, archive source plus the successful paper result directories, tag the version, and record the archive DOI. Do not commit `.venv`, temporary execution results, or a duplicate `github_release` repository tree. None of the supplied commands publishes to GitHub or Zenodo.

Storage grows with retained draws: an n-by-d float64 chain uses 8nd bytes before compression. Each concurrent worker streams its chain through a bounded buffer; its numerical runtime has additional memory costs. Reporting holds one physical chain and its functionals at a time, or one functional across chains plus the diagnostic library's working arrays. Temporary diagnostic files live under the saved run and are removed when processing finishes, including on handled errors. Large reports still require disk space and sorting/FFT work proportional to their retained samples. See [result processing](RESULT_PROCESSING.md). The affordable profile reduces chain counts, budgets and reference sizes. Resume also verifies the runtime environment to prevent mixing hardware or dependency conditions within a wall-time comparison.
