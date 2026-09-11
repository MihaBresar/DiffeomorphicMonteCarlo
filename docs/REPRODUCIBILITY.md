# Reproducibility and timing policy

All paper inputs are explicit JSON files. The manifest records their resolved values, scientific source hashes, Git commit/status when available, Python and package versions, OS/CPU count, JAX devices, float64/CPU policy and numerical-library thread settings. A run made from a dirty or uncommitted checkout records that fact; use a tagged clean checkout for the publication archive.

Benchmarks support bounded parallel chains through `--workers N` and the launcher's CPU-worker selector. Methods run one at a time. Workers use fresh Python processes, single-thread numerical-library settings, and distinct allowed CPU threads on Linux/Windows, filling physical cores before SMT siblings; macOS scheduling remains OS-managed. Worker allocation, actual thread settings and total invocation wall time are recorded. Hardware sharing, thermal throttling and other active processes still affect wall-time experiments. Keep the same worker count for comparisons and resumed runs; see [parallel execution](PARALLEL.md). The common log-density work counts are complementary to timing; a target evaluation is not assumed to cost the same as a gradient evaluation.

Map fitting, NUTS compilation and warmup happen before the production clock. JAX batches are synchronized at each timing boundary. NumPyro warmup and production loops are compiled separately, without discarding an entire extra production run. BlackJAX timed sampling uses batches of 16 iterations at paper scale; it completes the last batch and reports the resulting overshoot. Metropolis/slice timing can overrun by its last transition. For a paper comparison, inspect overshoots relative to the stated budgets and report actual measured sampling time.

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

Multichain diagnostics use a common prefix of length equal to the shortest chain and record that length. Estimator means and per-chain diagnostics use all retained draws. A constant functional within a chain yields unavailable MCSE/ESS rather than a reassuring zero. Rank R-hat is not computed from a single chain.

Atomic per-chain NPZ files allow restart after process interruption. Resume validates the source/configuration fingerprint and every checkpoint; an incomplete chain is restarted from its seed. Plot-only always reads the saved run's configuration/reference, with no simulation. A wall-time run is not bitwise deterministic across machines; fixed-step runs with the same environment and seeds are more strictly reproducible.

Before release, archive source plus the successful paper result directories, tag the version, and record the archive DOI. Do not commit `.venv`, temporary execution results, or a duplicate `github_release` repository tree. None of the supplied commands publishes to GitHub or Zenodo.

Storage grows with retained draws: an n-by-d float64 chain uses 8nd bytes before compression. The largest paper runs can produce several gigabytes in total, although only one chain needs to be sampled in memory at a time. The affordable profile reduces chain counts, budgets and reference sizes. Resume also verifies the runtime environment to prevent mixing hardware or dependency conditions within a wall-time comparison.
