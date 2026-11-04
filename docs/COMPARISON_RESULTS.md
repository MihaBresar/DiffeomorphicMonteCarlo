# Laptop comparison results

## Current coordinate checks, 4 November 2026

The recorded version-2 runs add **E[X₁]** and **P(X₁ > 3)** to the existing
100D and 500D targets. No additional distribution was included. All eight
target/VI/backend combinations completed: 20 chains per method, 2 seconds of
production per chain, four workers, 1,000 separate warmup transitions, and
400 VI updates with 500 fresh draws when enabled. The first third of actual
production iterations is discarded. These 560 chains use sampling seed batch
1 and fit seed 42; they are fresh fixed-time runs, not reprocessed older draws.

| Comparison | Target | VI | Lowest mean-X₁ MSE | Lowest X₁-tail MSE |
|---|---|---|---|---|
| Shared JAX | 100D | on | SCS | SCS |
| Shared JAX | 100D | off | SCS | SCS |
| Shared JAX | 500D | on | SCS | SCS |
| Shared JAX | 500D | off | SCS | SCS |
| Optimized Numba DCS / author JAX SCS | 100D | on | DCS-BW-reflect | DCS-BW-reflect |
| Optimized Numba DCS / author JAX SCS | 100D | off | DCS-BW-reflect | DCS-BW-reflect |
| Optimized Numba DCS / author JAX SCS | 500D | on | SCS | SCS |
| Optimized Numba DCS / author JAX SCS | 500D | off | SCS | SCS |

These are observed minima from smaller runs, not universal or statistically
established rankings. Reflected BW had the lowest radial-tail MSE in every
case. For mean norm, reflected BW had the lowest MSE except in the two 500D
shared-JAX cases, where DCS-RWM did. The coordinate measurements therefore
show a different pattern from the radial measurements. The optimized mode
compares concrete implementations with different frameworks and output work.

[Full-precision results](../validation/asymmetry_smaller_results.csv) retain
all 28 method/case rows, four MSEs, bias/variance components, counts, timings,
steps, guard flags and reference uncertainties. Coordinate reference values
use one million independent draws with conditional sign integration and fixed
known-mean controls. Their MCSE was below 10% of each observed sampler RMSE in
this matrix. MSE variability is conditional on these references and the shared
fit, and is descriptive; heavy tails and only 20 chains still limit precision.

The [validation record](../validation/asymmetry_checks.json) records 246
passing automated tests, eight quick configurations/56 chains, all 560 smaller
chains and source hashes. Target densities, map formulas and map-fitting code
are byte-for-byte unchanged from the preceding release, as is the pinned
author SCS source. Both new statistics include every retained iteration and
rejection repeat, with their accumulation inside the sampling clock.

## Historical radial-only results, 3 November 2026

These are **historical version-1, radial-only results**. They predate the
first-coordinate measurements in the version-2 suite. Replotting preserves these recorded estimates; it cannot add
unsaved all-iteration coordinate statistics or measure the new processing cost.

These are Smaller reproduction results from an Apple M4 Pro laptop with four workers: 20 chains per method, 2 seconds of production per chain, 1,000 separate warmup transitions, and 400 VI updates with 500 fresh draws per update when enabled. One third of actual production iterations is discarded. All eight Student-t comparisons and 560 chains completed on their first attempt. Both dimensions use ν=3 and DCS β=2. These results are conditional on fit seed 42 and sampling seed batch 0.

### Eight Student-t comparisons

| Target | VI | Lowest tail MSE, JAX | Lowest norm MSE, JAX | Lowest tail MSE, Numba comparison | Lowest norm MSE, Numba comparison |
|---|---|---|---|---|---|
| skew-t-100d | on | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect |
| skew-t-100d | off | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect |
| skew-t-500d | on | DCS-BW-reflect | DCS-RWM | DCS-BW-reflect | DCS-BW-reflect |
| skew-t-500d | off | DCS-BW-reflect | SCS | DCS-BW-reflect | DCS-BW-reflect |

These are observed minima, not significance statements or universal rankings. The optimized Numba comparison deliberately changes implementation and output-processing costs, while the four-method JAX comparison shares that framework. Native reflected BW had the lowest observed MSE for both quantities in each native case. The JAX results were mixed: reflected BW had the lowest tail MSE in all four cases, while DCS-RWM had the lowest 500D mean-norm MSE with VI and SCS had the lowest 500D mean-norm MSE without VI.

Complete method-by-method values, exact counts, setup and sampling times, acceptance, frozen steps and guard flags are in [the full-precision results](../validation/comparison_smaller_results.csv); [fitted parameters](../validation/comparison_smaller_results_fitting.csv) are separate. Each timing includes all generated transitions, including rejected-state repeats, and every method gets the same requested production budget.

### Interpretation and validation

SCS reached its existing numerical step safeguard in 12 of 20 chains in each 100D no-VI comparison. These flags are preserved: equal tuning resources do not establish optimal tuning. Reflected BW has no special ceiling of 2, and some 500D no-VI pilots selected 3.2. No post-hoc truncation or retuning was applied.

The two reported functionals are radial and do not assess sampling of the skewed direction. Heavy tails and only 20 chains limit MSE precision; empirical uncertainty summaries are descriptive, not formal confidence intervals. New seed batches and longer runs are needed before publication-level claims.

The associated executable checks covered target densities, transitions, fitting and calibration, exact burn-in, reports, and partial-worker scheduling. The pinned author SCS files remained unchanged. See [the historical validation record](../validation/comparison_suite_checks.json) and [current protocol](COMPARISONS.md). Remote Windows/Linux checks require successful GitHub Actions execution.
