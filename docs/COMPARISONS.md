# Current sampler comparisons

The launcher offers two fresh-run comparisons with the same target selector and
VI on/off switch. Historical SCS/boundary choices are no longer visible, but
saved reports and their command-line identifiers remain supported.

| Identifier | Production implementations |
|---|---|
| `sampler-comparison` | DCS-RWM, DCS-BW, SCS, DCS-BW-reflect, all JAX |
| `sampler-comparison-numba` | DCS-BW and DCS-BW-reflect in optimized Numba; author SCS in JAX |

RWM here means **DCS-RWM in the transformed ball**, not an ordinary random walk
on physical coordinates. Reflected BW folds a proposed displacement along its
chord through the ball; it is not the spherical SCS step-out proposal. See the
[reflection argument](BOUNDARY_100D.md).

## Targets and references

| Selector | Density and map | Radial threshold | Radial tail reference | Mean-norm reference |
|---|---|---:|---:|---:|
| `skew-t-100d` | Normal-CDF-skewed t, d=100, ν=3, α=(100,−100,0,…), DCS β=2 | 50.00 | 0.0107346153 | 13.7852602890 |
| `skew-t-500d` | Same skew and ν, d=500, DCS β=2 | 50√5 ≈ 111.80 | 0.0106799328 | 30.8864890662 |

Thresholds are fixed before sampling, independent of method, VI and pilot
outcomes. JSON/CSV retain full precision. Skewing the t target by its sign
leaves the radial law unchanged for both targets: ||X||²/d has the F(d,ν) law.

Both targets also report **E[X₁]** and **P(X₁ > 3)**. These measurements assess
the asymmetric part of the distribution; radial measurements alone cannot do
that. The coordinate threshold is fixed before running the new comparisons.
The existing target distributions and physical initial points (all ones) are
unchanged. The usual third-moment skewness coefficient is not defined at ν=3
and is not reported.

Independent references use symmetric Student-t draws, integrate the sign
selection exactly conditional on each draw, and apply fixed known-mean control
variates for the first-coordinate expectations. These variance reductions
do not use competing sampler output, fit results or adaptive choices. Reference estimates and their Monte Carlo
standard errors are saved separately from sampling errors. Reference budgets
are 50,000 draws for Quick check, 1,000,000 for Smaller reproduction and
5,000,000 for Full paper settings, using a fixed seed independent of fitting
and sampler streams. The report distinguishes analytic values from estimated
references and flags reference MCSE above one tenth of observed sampler RMSE.
An estimated reference is not relabeled an exact truth; its precision must be
considered when interpreting small MSE differences.

## What is matched

- Same physical target, four functionals, independent references and physical
  initialization (all coordinates equal to one).
- Same requested per-chain sampling time, number of independent chains,
  warmup-transition budget, 300-transition clock checks and exact first-third
  burn-in. Repeated states after rejection enter every estimate.
- Same fitting budget and optimizer settings when VI is enabled: fresh
  independent base draws, reverse-KL objective, Adam learning rate 0.01,
  gradient-norm cap 10 and fit seed 42. Initial map scales are one. No target,
  CDF or physical-coordinate clipping is used.
- Same ordered CPU subset and simultaneous-chain count per wave across
  methods, including partial waves. Method order rotates between waves.
  macOS scheduling is explicitly OS-managed; Linux/Windows pinning is recorded.
- The same JAX calibration code, seed, final state and frozen step for a DCS
  method in either implementation mode. Production in Numba uses a separate,
  domain-separated NumPy PCG64 stream, so cross-backend trajectories are not
  bitwise copies.

VI **off** skips fitting for both the ball and spherical maps: scale R=1,
centre zero and Möbius/observer deformation zero. DCS β remains 2;
SCS latitude stays 1.1. Step pretuning still runs. VI **on** fits both map
families, with one shared ball fit for all DCS methods. VI losses omit
different base-normalizing constants and are not
comparable normalized KL values across map families.

## Method-specific choices

DCS-RWM, ordinary DCS-BW and SCS tune their proposal steps toward acceptance
0.234 using the same observed-acceptance dual-averaging rule. Reflected BW
uses ball squared-displacement pilots within the same number of warmup
transitions: rotating h/2, h and 2h windows, including zero movement on
rejection. Its step has no special ceiling of 2. Both rules retain broad
finite-precision scale guards; neither guarantees optimal radial efficiency.
All steps freeze before sampling. Candidate scores, selected steps and guard
flags are saved; they must be inspected rather than treating equal resources
as proof that every method is optimally tuned.

In particular, a target acceptance may be unreachable for a step-out proposal
whose large-step distribution saturates. In the historical version-1 100D no-VI smaller run, SCS
reached its numerical upper guard in 12 of 20 chains. This is a tuning
limitation to disclose; neither the guard flag nor equal warmup budgets
establishes optimal tuning.

The JAX comparison uses equivalent full-state scans, physical projection and
four-functional reduction for all methods. The Numba comparison deliberately
changes this implementation work: DCS caches current density, radius and first
coordinate, skips recomputing
unchanged rejected-state values and fuses all-iteration accumulation without
returning every full state. SCS retains the unmodified author transition and
JAX full-state processing. Thus the second comparison measures these concrete
implementations; its timing does not isolate a purely geometric algorithm
advantage. The pinned SCS source remains byte-for-byte unchanged.

## Budgets, clocks and outputs

| Preset | Chains per method | Seconds per chain | Separate warmup | VI updates × draws/update |
|---|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 20 × 128 |
| Smaller reproduction | 20 | 2 | 1,000 | 400 × 500 |
| Full paper settings | 50 | 20 | 1,500 | 1,500 × 2,000 |

VI off uses zero fitting updates regardless of preset. Independent reference
preparation, compilation, fitting, warmup, synchronization waits and file/report
generation are outside the
sampling clock and recorded separately. Native DCS setup includes the shared
JAX calibration/scan compilation plus Numba compilation. Actual sampling
seconds and overshoot are saved; short runs can exceed the requested budget
by a larger relative amount. All 300 states in each completed block count.

The first third of the actual timed iterations is discarded. Prefix sums at
100-step boundaries implement this exactly; thinning affects only saved
traces, not estimates or MSE. All four functionals, including the two new
coordinate quantities, are accumulated inside every sampling clock. Reports
retain the original radial boxplots, MSE and count/time figure families and add
separate coordinate mean and tail-probability figures. They use two decimal
places where appropriate and export exact integer counts and full-precision
values. `reference_values.csv` includes reference values, MCSE and exact flags.

Both current modes use protocol version 2 and report radial and coordinate
quantities. Historical version-1 reports remain radial-only and keep their
recorded numbers. These measurements do not establish an overall mixing
advantage. MSE is the average squared error across independent chains; the fitted map is shared,
so uncertainty is conditional on that fit. Empirical MSE error summaries are
descriptive, not formal confidence intervals: the supplied t(3) targets
have infinite fourth radial moment, making rare large errors
particularly important for norm and coordinate-mean errors. MSE against an
estimated reference also carries reference uncertainty. Smaller reproduction
results are exploratory; use longer runs and fresh seed batches before publication claims.

See [click-to-run instructions](CLICK_TO_RUN.md), [rerun commands](RERUN.md)
and [native implementation checks](NATIVE_KERNELS.md).
