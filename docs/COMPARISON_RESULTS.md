# Laptop comparison results, 3 November 2026

These are Smaller reproduction results from an Apple M4 Pro laptop with four workers: 20 chains per method, 2 seconds of production per chain, 1,000 separate warmup transitions, and 400 VI updates with 500 fresh draws per update when enabled. One third of actual production iterations is discarded. All eight Student-t comparisons and 560 chains completed on their first attempt. Both dimensions use ν=3 and DCS β=2. These results are conditional on fit seed 42 and sampling seed batch 0.

## Eight Student-t comparisons

| Target | VI | Lowest tail MSE, JAX | Lowest norm MSE, JAX | Lowest tail MSE, Numba comparison | Lowest norm MSE, Numba comparison |
|---|---|---|---|---|---|
| skew-t-100d | on | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect |
| skew-t-100d | off | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect | DCS-BW-reflect |
| skew-t-500d | on | DCS-BW-reflect | DCS-RWM | DCS-BW-reflect | DCS-BW-reflect |
| skew-t-500d | off | DCS-BW-reflect | SCS | DCS-BW-reflect | DCS-BW-reflect |

These are observed minima, not significance statements or universal rankings. The optimized Numba comparison deliberately changes implementation and output-processing costs, while the four-method JAX comparison shares that framework. Native reflected BW had the lowest observed MSE for both quantities in each native case. The JAX results were mixed: reflected BW had the lowest tail MSE in all four cases, while DCS-RWM had the lowest 500D mean-norm MSE with VI and SCS had the lowest 500D mean-norm MSE without VI.

Complete method-by-method values, exact counts, setup and sampling times, acceptance, frozen steps and guard flags are in [the full-precision results](../validation/comparison_smaller_results.csv); [fitted parameters](../validation/comparison_smaller_results_fitting.csv) are separate. Each timing includes all generated transitions, including rejected-state repeats, and every method gets the same requested production budget.

## Interpretation and validation

SCS reached its existing numerical step safeguard in 12 of 20 chains in each 100D no-VI comparison. These flags are preserved: equal tuning resources do not establish optimal tuning. Reflected BW has no special ceiling of 2, and some 500D no-VI pilots selected 3.2. No post-hoc truncation or retuning was applied.

The two reported functionals are radial and do not assess sampling of the skewed direction. Heavy tails and only 20 chains limit MSE precision; empirical uncertainty summaries are descriptive, not formal confidence intervals. New seed batches and longer runs are needed before publication-level claims.

The executable checks cover target densities, transitions, fitting and calibration, exact burn-in, reports, and partial-worker scheduling. The pinned author SCS files remain unchanged. See [the validation record](../validation/comparison_suite_checks.json) and [protocol](COMPARISONS.md) for the current check results. Remote Windows/Linux checks require successful GitHub Actions execution.
