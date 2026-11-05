# Reproducibility and timing

This page describes the current comparison and then the six other manuscript experiments. The resolved `configuration.json` and the source/runtime record in `manifest.json` define an individual run.

## Current comparison: version 11

`sampler-comparison` records `timed-sampler-comparison-v11`; `sampler-comparison-numba` records `timed-numba-comparison-v11`. Both contain DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and DCS-Billiard, with 100D/500D skewed t(3) targets and VI fitted for both map families. New comparison requests cannot disable VI. See the [complete protocol](COMPARISONS.md) and [reviewer checklist](REVIEW_GUIDE.md).

The JAX mode uses the same full-state scan, projection and scalar recording framework for all methods. The Numba mode uses fused DCS production loops with density/radius caching, while SCS calls its unchanged author JAX transition. Each DCS method uses the same JAX pilot in either mode; Numba production uses a separate PCG64 stream. This preserves proposal laws, not bitwise trajectories across backends.

### Target, maps and independent pilots

All methods receive the same physical target, analytic references, all-ones physical initial state, chain count and production-time budget. Only radial-tail probability and mean norm are measured. The normal-CDF skew leaves the Student-t radial law unchanged. The radial thresholds are fixed before sampling: 50 in 100D and 50√5 in 500D.

VI is fitted once per map family and reused across chains. Results therefore condition on the saved maps; changing sampling seed batches does not measure refitting uncertainty.

| Setting | DCS | SCS |
|---|---|---|
| Fitting routine | Fixed-cloud reverse KL, seed 42, Adam 0.01, gradient-norm cap 10, no coordinate/CDF clipping | Unchanged pinned author VI, one fixed reference cloud, seed 0, Adam 0.01, coordinate clip ±1,000 in the target and Jacobian arguments, no gradient clipping |
| Full VI budget | 1,000 updates × 2,000 fixed reference draws | 1,000 updates × 2,000 fixed reference draws |

Each family reuses its own fixed cloud at every update: 2,000 distinct reference draws and 2,000,000 objective/gradient point evaluations over the full optimization. The cloud sizes and update counts match; the reference distributions, actual draws, parameterizations, fitting seeds and clipping rules still differ. Their actual fitting times, parameters and losses are recorded separately. The VI clipping is confined to the author's optimizer; production densities remain unclipped. Losses across map families are not normalized KL values that can be compared directly.

DCS-RWM/BW/reflected BW use the ball map. Billiards retain its jointly fitted centre/R/β in post-Möbius coordinates, with effective deformation zero. This is not a second VI fit constrained to zero deformation. `method_maps.json` and `effective_maps.csv` record the maps actually used.

Every chain has separate discarded warmup before production. All four DCS methods share one fixed move h=0.07: BW, reflected BW and billiard use a uniform-ball increment of radius h; RWM uses Gaussian coordinate standard deviation h/√(d+2). This matches the increments’ coordinate covariances, not their physical endpoints or acceptance rates. Each method uses its own kernel during discarded equilibration, with no separate step adaptation. SCS uses fixed h=0.1 throughout warmup and production by default. All methods receive equal discarded transition counts, not necessarily equal warmup seconds or equally optimal settings. The final warmup state initializes production and every production step is fixed.

Use `--dcs-fixed-move 0.15` for the declared shared-DCS sensitivity setting, or `--dcs-step-rule acceptance` for historical method-specific calibration (BW/RWM acceptance target 0.234, reflected-BW ball movement and billiard radial movement). SCS's separate sensitivity option remains `--scs-step-rule acceptance`. These choices are recorded. The DCS move applies to all four DCS variants; SCS's policy is selected separately. VI is unchanged. Neither fixed DCS setting is claimed optimal. Version 11 requires fresh sampling but retains the version-10 fixed-cloud maps; experimental fresh-cloud VI studies are not public defaults.

The fixed SCS default was selected using the disclosed 100D seed-batch-7 development study and its full-author-VI sensitivity pair. Its rule did not select by DCS ranking. That study used version 9. Evaluate version 11 with fresh, previously uninspected full-paper sampling seeds, separately from development evidence. The 500D setting extends the SCS choice without a separate tuning study. [Both SCS policies and their historical outcomes](AUTHOR_SCS_CALIBRATION.md) remain available.

### Sampling clock and CPU allocation

Methods run in synchronized waves at the same concurrency and ordered CPU allocation, including partial waves. Method order rotates between waves. Each production clock includes every generated transition, full-state projection and radial bookkeeping in JAX, scalar capture for ESS, and the final buffer flush. Numba performs its recorded fused equivalent. It excludes fitting, compilation, pilots, synchronization waits, ESS computation, file compression and report generation. These costs are recorded separately; a production-time ranking is not an end-to-end cost ranking.

Sampling completes its final 300-transition batch. Actual elapsed seconds, overshoot and counts are saved. Exactly the first third of the actual generated iterations is discarded; repeated states after rejection enter both estimates and ESS. Saved trace thinning changes neither radial estimates nor comparison ESS.

The common CPU policy is `pjrt-single-compute-thread-v1`: `PJRT_NPROC=1` is set before JAX initialization for DCS and SCS, alongside numerical-library thread limits. Runtime helper threads may still exist. Linux/Windows workers use allowed CPU threads, filling physical cores before SMT siblings; macOS allocation is OS-managed. [Parallel execution](PARALLEL.md) describes detection and affinity. Keep hardware, worker count and system load comparable, and avoid concurrent unrelated benchmarks.

Do not combine rates or counts from different execution policies. Changing execution resources or output collection can change timed counts and therefore MSE; rerun the complete method set after such changes.

### Estimates, uncertainty and ESS

Each chain first estimates the same function from all of its post-burn-in iterations. MSE is the average squared error across independent chains, giving each chain equal weight even when timed counts differ. Its empirical bias-squared and variance decomposition is saved. Analytic references have no Monte Carlo reference error for these radial functions.

Functional ESS uses each chain's unthinned post-burn-in scalar history, including repeats. It is computed separately for the norm and tail indicator, after sampling. FFT work is serialized to bound memory; diagnostic computation and waiting times are separate. ESS/s is the sum of per-chain ESS divided by the sum of full production seconds, including production burn-in. It is a descriptive rate, not the ESS of a concatenated unequal-length chain. Unavailable chain diagnostics remain visible; an incomplete subset is not reported as a complete method total. See [the precise ESS definition](ESS.md).

ESS does not establish convergence or absence of bias. Radial results do not establish coordinate mixing. Empirical MSE variability conditions on fixed maps and finite replicates. Because the t(3) norm lacks a fourth moment, its squared-error variance need not exist; empirical uncertainty summaries can miss rare large errors. Report all prespecified functions and cases, including outcomes favoring a comparator.

### Seeds, files and replay

Sampling batch b, for b from 0 to 999, begins at `901001 + 1000000*b`; chain i uses that seed plus i. DCS fit seed remains 42 and author SCS fit seed remains 0. A repeated batch reuses the pseudorandom streams, but wall-time stopping can produce different counts even on the same machine. Fixed-step choices do not make timed runs bitwise reproducible across machines.

Keep `configuration.json`, `manifest.json`, maps, chain files, `summary.json` and the generated tables together. The manifest records scientific source hashes, runtime environment, CPU settings, status and invocation wall time. Check that status is `complete` before using a report. Failed or incomplete chains are not replaced by zeros.

Current comparisons start in fresh folders and do not resume partial production. Rebuild a completed report with:

```sh
python run_experiment.py --plot-only results/SAVED_RUN
```

Replotting uses the recorded summary; it cannot recover measurements or unthinned histories that were not saved. The [SCS step-control runner](AUTHOR_SCS_CALIBRATION.md#reproduction) reuses a validated primary's exact maps and changes only the SCS step policy, with equal discarded transition budgets. It records original fitting time separately from zero new fitting work. That zero is a reuse cost, not the independent cost of fitting SCS.

## Other manuscript experiments

The six generic experiments use explicit JSON files in `configs/`, resolved preset overrides and the same first-third production burn-in after separate tuning. They include the fixed-iteration super-heavy-tail comparison, 200D baselines, a NUTS comparison, logistic regression, Eight Schools and GARCH. See [target definitions](TARGETS.md), [rerun commands](RERUN.md) and [experiment specifications](EXPERIMENTS.md).

These experiments save retained physical-coordinate draws. Their thinning, diagnostic stride and reference construction differ from the compact current comparison. Their manifests record source fingerprints, dependencies, hardware and resolved inputs. Posterior and logistic references are finite Monte Carlo estimates with reported uncertainty, not exact truth. The synthetic radial references are analytic.

Generic workers stream draws through bounded buffers. Thinning, physical conversion, draw collection, the final buffer flush and NUTS scalar diagnostic collection are inside production time. Mapping finalization, NPZ serialization and reports are outside it. NUTS compilation and warmup precede production; timed BlackJAX runs finish the final batch and report its overshoot. NumPyro fixed-count runs carry the state and random stream through bounded batches and an exact remainder.

Posterior DCS receives the mean production time of fixed-count NUTS runs; setup is reported separately. The quick posterior preset alone imposes a one-second minimum DCS budget as an execution check, so its plots are not an equal-time performance result. Benchmark NUTS uses target acceptance 0.8 and maximum depth 10; reference runs use their independently recorded settings.

New public runs use the fitted DCS map (`ball_map_policy=mobius`) for synthetic and logistic targets; the untuned `vanilla` choice is no longer exposed by the launcher or public command line. Internal helpers and saved reports retain compatibility with those earlier parameters. PosteriorDB examples retain their specified identity maps.

Every VI-enabled SCS fit, including logistic and the ν=0.5 experiment, calls the same pinned author routine: seed 0, a fixed reference cloud, Adam 0.01, physical-coordinate clipping at ±1,000 inside VI, and no gradient clipping. The separate `scs_vi` budget is 1,000 updates × 2,000 fixed draws for paper, 400 × 500 for reproduce, and 20 × 128 for quick. It does not inherit the DCS optimizer budget. SCS keeps the author's unit-scale initialization and unconstrained centre/observer fit; the DCS-only radial-median initialization and symmetry constraints do not apply to SCS.

Generic SCS production still uses the checked Numba implementation with acceptance-target 0.234 step tuning. Changing its VI implementation does not turn it into the unchanged author JAX production implementation or the fixed-h=0.1 comparison. Version 10 established matched fixed-cloud fitting: DCS receives 1,000 full updates, whereas version 9 used 1,500 full updates with fresh draws. The full logistic DCS budget became 1,000 and the 50D DCS cloud became fixed. Those fitting changes require fresh generic results if upgrading from the earlier policy. Version 11 leaves those fits and all six generic experiments' step policies unchanged; only the two current five-method comparisons receive the shared fixed DCS move default. Experiments without SCS retain their earlier fitting policy.

Generic ESS/MCSE describes the retained series used by the estimator; it is not multiplied by the thinning stride. Multichain diagnostics use a recorded common prefix of the shortest chain, while means and per-chain diagnostics use all retained draws. A constant functional has unavailable ESS/MCSE rather than an artificial zero; rank R-hat requires multiple chains. The compact 200D baseline report omits coordinate quantiles and cross-chain diagnostics. [Result processing](RESULT_PROCESSING.md) explains storage and progress reporting.

Generic checkpoints support resume after source, configuration and environment validation. An incomplete chain restarts from its seed. Completed summaries can be replayed without recalculating diagnostics; `--recompute-diagnostics` requests rebuilding from saved retained chains. These resume rules do not apply to the current fresh-folder comparison runner.

## Publication archive

Archive the exact clean source snapshot, pinned requirements, resolved inputs and complete successful result directories used for the manuscript. Record the commit or release and an archive DOI when one exists. Keep execution checks, development studies and independent full runs clearly distinguished.

Storage and setup costs depend on dimension, counts, worker concurrency and diagnostics. Measure a smaller run before scheduling a full study. Preserve exact numbers and metadata alongside rendered figures; two-decimal presentation must not replace full-precision exports.
