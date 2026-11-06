# Reproducibility and timing

Preserve a run's configuration, manifest, maps, summaries and chain metadata together. Settings, source hashes and environment identify the result. Use a fresh directory for each independent run.

## Main comparison

The [protocol](REPORTED_STUDY.md) fixes targets, maps and steps. Both implementation modes load the same map assets; SCS source is pinned by revision and checksum.

This package supports sampling reproduction from the supplied frozen inputs.
It does not yet provide a standalone replay of all preparation: the joint
radius/step pilot and centre-choice diagnostic are recorded by hashes and
selected values, but their complete executable recipes and pilot seed are not
bundled. Relative `review/...` paths in the immutable manifest identify source
provenance; they are not required files for running the current comparisons.

**Seed 1–Seed 8** change sampling streams, not fitting. Exact internal seeds remain in saved configurations and chain metadata. Repeating a seed on another machine need not reproduce timed counts or long floating-point trajectories.

DCS and calibrated SCS received equal VI evaluation and R/h selection-transition budgets. SCS's selected R/h preceded zeroing its centre and observer. Original SCS keeps its author fit and h=0.1. Original fits and production overrides remain distinct; equal budgets do not imply identical map-preparation procedures.

## Clocks and allocation

Compared methods receive the same requested production seconds and worker allocation. Production includes transitions, rejected repeats, radial recording and scalar capture for ESS. Compilation, warmup, synchronization waits, fitting, ESS calculation and report generation have separate clocks.

The main comparison completes its final 300-transition block; general timed kernels use bounded batches. Actual time, overshoot and counts are recorded. Production includes the first third subsequently discarded.

Generic workers stream retained physical draws and flush buffers within production time. Final archive serialization and reporting are outside it. Posterior HnR receives mean NUTS production time. Total invocation wall time is reported separately.

Choose workers within CPU and memory limits and keep the choice fixed across methods. Main-comparison methods run in synchronized waves with rotating order; other experiments distribute independent chains. Avoid concurrent intensive activity.

Order changes between waves, starting with the same order for each seed choice.
With at least as many workers as chains there is only one wave, so no order
rotation occurs. Few-wave runs do not fully counterbalance thermal or load
effects; synchronized allocation alone cannot eliminate these effects.

## Estimates and diagnostics

Main-comparison estimates and functional ESS use all post-burn-in scalar observations; saved-trace thinning changes neither. Other experiments record the actual retained-draw basis.

MSE averages individual chains' squared errors against the reference. Empirical MSE variability and paired differences use the available chains and can be unstable under heavy tails. Main radial references are analytic.

ESS/s is summed per-chain functional ESS divided by summed full production seconds. It is not cross-chain convergence evidence. See [ESS](ESS.md).

## Preparation and publication

Fitting, reference generation and sampling have separate recorded seeds. Every SCS fit uses the original author optimizer. Logistic references use independent NUTS chains with quality checks; PosteriorDB references retain their source processing.

Numba DCS shares JAX warmup states and fixed steps, then uses a separate production generator. Matching laws does not imply bitwise trajectories. Use Python 3.12 and pinned CPU dependencies and record platform details.

Using JAX for both families supplies a common execution framework, not a
guarantee of equal implementation optimization. The DCS implementation includes
specialized optimizations while SCS retains the authors' transition code.
Speed results describe these implementations; the Numba extension makes that
implementation difference explicit.

Publish complete method sets and all chains under one protocol. Identify target, estimand, preset, seeds, workers and timing scope. Report preparation cost separately when it is excluded from sampling time.
