# DCS/SCS implementation and tuning audit

This is a historical audit. The [current comparison suite](COMPARISONS.md) offers a four-method JAX comparison and a separate optimized-Numba comparison, each with target and shared VI switches.
The frozen-adaptation and launcher descriptions below describe the earlier release.

The September 11 update restores the original figures, removes the physical-core
worker cap, improves scheduling and density-evaluation speed, and restores fresh
VI batches. It does not hard-code a winning sampler or reuse old plotted values.

## Measured implementation overhead

The rewritten DCS code recalculated target normalizers, map invariants and
scalar reductions inside the sampling loop. Frozen map/target constants are now
cached once per chain. The same optimization is applied to SCS and ordinary RWM
targets. The ball-map density retains algebraic cancellation near the boundary;
no interior cutoff or CDF clipping is introduced. Regression tests compare
cached evaluations against the canonical formulas and compare acceptance
decisions and states with the uncached kernels at fixed proposal scales.

The following local macOS CPU audit held saved maps, initial states, seeds,
warmup (1,500 steps), production (60,000 steps), and thinning (10) constant.
It alternated method order across three repeats. Values are median iterations/s,
rounded here for readability; the full records and precision are in
[`validation/implementation_audit.json`](../validation/implementation_audit.json).

| Target | Method | Before | After | Speed factor |
|---|---|---:|---:|---:|
| 100D skew-t | DCS-RWM | 79,574 | 106,516 | 1.34 |
| 100D skew-t | DCS-BW | 72,222 | 93,519 | 1.29 |
| 100D skew-t | SCS | 63,153 | 81,485 | 1.29 |
| 50D t, ν=0.5 | DCS-RWM | 68,896 | 122,422 | 1.78 |
| 50D t, ν=0.5 | DCS-BW | 60,759 | 102,729 | 1.69 |
| 50D t, ν=0.5 | SCS | 55,155 | 77,671 | 1.41 |
| Logistic, both priors | DCS-BW | 45,642 | 76,283 | 1.67 |
| Logistic, both priors | SCS | 46,441 | 62,192 | 1.34 |

These measure kernel overhead with identical saved fits, not the expected
throughput of a full run on another machine. Floating-point evaluation order
changes slightly; adaptive trajectories are not guaranteed bitwise identical
between releases. The target density and proposal definitions remain the same.
To repeat the overhead audit, run `scripts/benchmark_kernels.py` against the same
saved result directory under each code version. It saves fitted-map checksums,
seeds, draw hashes, full records, source hashes and environment information.

## VI batch reuse changed the fitted maps

The original optimizers used new random base draws at each step. The rewrite
used one fixed sample throughout training. A finite-sample VI objective is a
legitimate approximation, but it can overfit, and it was not the original
procedure. Fresh batches are now restored for **both** ball and spherical-cap
maps. All fitted parameters remain frozen during retained sampling.

On the 100D fixed-step target with 400 optimization steps and 500 draws/batch,
the old ball-map training loss was −88.60, whereas its loss on 20,000 independent
base draws was −77.59 (MCSE 1.71). With fresh training batches the independent
loss improved to −86.16. The norm of the fitted centre fell from 1.38 to 0.27.
This supports a fitting-quality issue, not a claim that VI minimizes MCMC error.

A paired audit using seeds 801–808, 1,000 warmup steps and 25,000 production steps
per chain found the following MSEs. These are exploratory eight-chain results,
not substitutes for full paper runs; the full precision and estimates are saved.

| VI batches | Method | Tail-probability MSE | Mean-norm MSE |
|---|---|---:|---:|
| Fixed | DCS-RWM | 9.35e-6 | 0.3630 |
| Fixed | DCS-BW | 3.11e-6 | 0.2491 |
| Fixed | SCS | 2.10e-6 | 0.0352 |
| Fresh | DCS-RWM | 4.35e-6 | 0.0871 |
| Fresh | DCS-BW | 1.30e-5 | 0.1150 |
| Fresh | SCS | 3.91e-6 | 0.1463 |

DCS had lower mean-norm MSE with fresh VI in this check. SCS retained the lowest
tail-probability MSE. This is why restoring the implementation cannot establish
that DCS wins every metric; all outcomes must remain visible.

## Historical fixed R=√d comparison

This earlier audit held the ball scale at R=√d while VI fitted the centre and
Möbius parameters. SCS continued its own map fit, and the identity maps in the
two PosteriorDB experiments were unchanged. This option has since been removed
from new runs at the author's request. The launcher now offers a joint Möbius VI
fit or a fully untuned vanilla map (R=1, centre=0 and Möbius=0). Historical
fixed-R results and their labels remain readable through plot-only.

With fresh VI and the same eight seeds and 25,000-step budgets, the check gave:

| Target | Scale policy | Tail MSE | Second radial-functional MSE |
|---|---|---:|---:|
| 100D, ν=3 | Fitted R | 1.30e-5 | 0.1150 |
| 100D, ν=3 | R=√d | 4.95e-5 | 3.3941 |
| 50D, ν=0.5 | Fitted R | 8.60e-5 | 0.000291 |
| 50D, ν=0.5 | R=√d | 0.074617 | 0.977825 |

The second functional is mean norm for ν=3 and mean log₁₀(1+norm) for ν=0.5.
These short checks argue against making fixed R=√d the default for these targets.
Fixing R is a valid tuning experiment, not a change in the target distribution.
In particular, R=√d can be badly mismatched to the β=0.2 map's extreme radial
expansion. These results are historical comparisons, not runs of the new vanilla option.

## Other differences from the old scripts

The old 100D fixed-step script continually adapted DCS's proposal scale during
retained sampling, whereas SCS used a fixed step of 0.1. The corrected pipeline
adapts both in separate warmup and then freezes both. Target settings and
references also received the corrections listed in [the rerun guide](RERUN.md).
Those differences can change relative MSE even with correct kernels. They are
not reversed merely to reproduce an old ranking.

For publication, rerun full paper settings under a recorded worker allocation
and inspect the independent-chain errors, acceptance rates and fitted maps.
Graph-only updates need only **Rebuild figures**; performance or tuning changes
need **Start new run**. Existing checkpoints remain available for replotting.
