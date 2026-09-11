# One 100D skewed-t comparison

The default launcher experiment compares **DCS-RWM, DCS-BW and SCS at the same
sampling time per chain** on the existing 100-dimensional normal-CDF-skewed
Student-t target, with ν=3 and α=(100,−100,0,…). It estimates P(‖X‖≥50) and E‖X‖
against exact radial truth. All three methods start at the physical vector of
ones, using the correct inverse map. The protocol identifier is
`timed-dcs-author-scs-v2`.

| Preset | Independent chains per method | Timed sampling seconds per chain | Separate warmup iterations | Stored radial trace interval |
|---|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 1 |
| Smaller reproduction | 20 | 2.00 | 1,000 | 10 |
| Full paper settings | 50 | 20.00 | 1,500 | 50 |

After separate fitting and warmup, each method samples for its time budget.
The first third of **its actual timed iterations** is discarded. Every remaining
iteration contributes to the estimates, including repeated states after
rejection. The stored trace interval only reduces saved instantaneous radial
traces; it does not change the estimates or sample denominator. Different
methods and chains can produce different iteration counts.

Quick check uses abbreviated fitting and cannot support a performance ranking.
Fifty independent chains improve precision of the empirical error assessment;
they do not remove bias caused by insufficient exploration within a chain.

## DCS fitting and separate step calibration

The original DCS VI function is extracted from the byte-preserved
`validation/legacy_fixed_step.py`, with ν explicitly set to 3. The full fit uses
1,501 optimizer updates, 2,000 fresh base draws per update, seed 42, Adam with
learning rate 0.01, and β=2. The fitted R, centre and Möbius vector are shared by
DCS-RWM and DCS-BW and fixed during sampling. Quick/reproduction budgets use
21/401 updates and 128/500 draws. Original clipping in the VI objective is
preserved and recorded; the production target has no CDF clipping or artificial
interior-ball cutoff.

DCS-RWM proposes isotropic Gaussian increments. DCS-BW proposes uniformly in a
Euclidean ball. Both reject proposals outside the open unit ball. Each chain
starts with proposal scale 0.10 and calibrates it using the original smoothed
dual-averaging update toward acceptance 0.234 during the separate warmup listed
above. **The resulting scale is frozen before timed sampling starts.** The map
and step therefore remain fixed during the timed first-third burn-in and all
subsequent estimation. This differs from the historical fixed-count comparison,
which continued step adaptation throughout production.

`dcs_bench/fast_dcs.py` compiles the transition using Numba without fast-math;
`dcs_bench/timed_radial.py` runs the timed frozen-step batches. Map and target
constants are cached. Density evaluation also computes the physical radius;
rejection reuses the current radius. Normal log-CDF values come from the actual
SciPy double-precision routine. Tests compare the frozen transitions and random
state with the reference transition, including warmup and rejection repeats.

## SCS: original author implementation

The MIT-licensed source is vendored without modification at
[`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565),
the paper-date revision. The adapter invokes its tangent-Gaussian proposal,
normalization, cap step-out, projection, Jacobian and Metropolis transition.
Source hashes and licence are in `third_party/scs_paper`.

The full fit uses the original fixed batch of 2,000 base draws, seed 0, 1,000
Adam updates with learning rate 0.01 and its original objective clipping.
Latitude is 1.10. The step stays fixed at 0.10 during both separate warmup and
timed sampling, as in the authors'
[`run_skewt.py`](https://github.com/liusf15/sub-cauchy-sampler/blob/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565/experiments/run_skewt.py).
SCS receives the same number of separate warmup transitions as DCS, with no
step adaptation. The original fitting budget differs from DCS: this comparison
uses their original VI recipes. Setup time is reported separately. Quick and
smaller settings shorten SCS fitting to 20/400 updates and 128/500 fixed draws.

The author transition runs inside compiled JAX scans. Progress callbacks are
suppressed outside the vendored code. Independent keys per timed batch preserve
the transition law, but do not reproduce the author's bitwise random sequence.
Tests compare the timed adapter with full-state executions of the original
transition and radial accumulation.

**Target distinction:** the [SCS paper §4.1](https://arxiv.org/html/2601.11066v1#S4.SS1)
uses a Student-t CDF skewing factor and ν=1 or 2. This comparison applies its
original algorithm and VI to our existing normal-CDF-skewed ν=3 target. It does
not reproduce the SCS paper's target or plotted values. All three methods here
evaluate the same normal-CDF target; its radial truth is unchanged by skewing.

## Timing, burn-in and outputs

Fitting, compilation and separate warmup finish before the sampling clock
starts. The timed phase includes its first-third burn-in, transitions, radial
accumulation and compact in-memory recording. JAX results are synchronized
before reading the clock. Setup, inter-worker synchronization waits, disk
writes and plotting are outside that clock; setup costs and total invocation
wall time are recorded separately.

All methods check elapsed time after the same **300-transition batches**. A
chain stops at the first completed batch meeting its budget; actual elapsed
time and overshoot are recorded. Its iteration count N is a multiple of 300.
Cumulative tail counts and norm sums are stored every 100 transitions, so N/3
is always an available boundary. Subtracting that boundary's cumulative values
gives the estimates over **all 2N/3 post-burn-in iterations**, without storing
every 100-dimensional state. Tests cover a burn boundary inside a timed batch,
variable counts and repeated states. A timing batch can be noticeable relative
to a 0.05-second smoke check; use the longer presets for comparisons.

Parallel chains run in synchronized waves, with only one method sampling at a
time. Method order rotates between waves. Each method receives identical wave
sizes and CPU allocation; Windows/Linux affinity fills physical cores before
SMT siblings. A 50-chain run can use 32 workers on the Ryzen 9 9950X3D. Sequential
runs provide the clearest isolated timing comparison; parallel timings describe
the recorded concurrent CPU workload. Keep the machine otherwise idle when
measuring speed.

`VsSCS.png`/PDF preserve the manuscript's three linear panels: tail MSE, norm
MSE and average sampling time. `scs_100d_estimates` adds radial estimate boxplots
and MSE panels. Every chain estimate is shown, so a small spread around an
incorrect value remains visible. Other plots show timing, actual iteration
counts, throughput and empirical bias squared plus variance. The timed protocol
uses the final post-burn estimates; it does not infer equal-time MSE by scaling
errors from fixed-count runs.

`paper_values.html` displays readable values, while CSV and JSON retain full
stored precision. The reports include actual total iterations, discarded and
estimation counts, per-chain errors, elapsed times and tuned parameters. Each
independent chain has equal weight in MSE, even when iteration counts differ.
Variance uses divisor n, so **MSE = empirical bias² + variance**, up to
floating-point rounding.

Use **Start.bat → Set up environment → 100D skewed t (ν=3) → Start run** on
Windows, or **Start.command** on macOS. Run environment setup again after an
update. A command-line run must use a fresh output folder:

```bash
python run_experiment.py skew-t-100d-fixed --preset paper --workers 32 --output-dir results/scs-100d-timed-paper
```

The historical `skew-t-100d-fixed` identifier is retained for compatibility;
its default protocol now fixes **time**. `--time-budget` overrides seconds per
chain. The launcher and compatibility entry points route this identifier to
the same runner. Historical plots remain available. New fixed-time plots
require rerunning this one experiment; old fixed-count runs cannot be converted
into this protocol because the adaptation and burn-in rules have also changed.

## Historical fixed-count evidence, 11 September 2026

The following archived results predate the timed protocol. They used one worker
on macOS ARM64, 50 chains per method and **500,000 production iterations per
chain**, after 100 warmup iterations. DCS continued adapting its step during
production. These values describe those recorded maps, settings and hardware;
they are not the results of the current 20-second preset.

| Method | Mean sampling seconds per chain | Iterations/s | Tail MSE | Norm MSE |
|---|---:|---:|---:|---:|
| DCS-RWM | 0.31 | 1,595,420.79 | 2.81e-07 | 2.00e-03 |
| DCS-BW | 0.34 | 1,483,033.64 | 2.10e-07 | 2.02e-03 |
| SCS | 5.34 | 93,697.44 | 8.92e-08 | 1.27e-03 |

DCS-RWM and DCS-BW achieved 17.03× and 15.83× the SCS iteration throughput.
SCS had lower error per fixed number of iterations for both functions. No
equal-time MSE is inferred from these ratios. Squared empirical norm bias was
less than 1% of MSE for every method; the recorded error mostly reflected
variation across chains. More chains alone cannot establish that an earlier
biased run was repaired when fitting, adaptation or run length also changed.

Full precision and individual chain results are archived in
`validation/scs_100d_comparison.json`, with environment/source hashes in
`validation/scs_100d_protocol.json`. Every method contributed exactly
25,000,000 production iterations to that historical comparison.

Historical implementation checks also remain available:

- Three 25,000-iteration repeats with the same continuous adaptation found
  compiled DCS-RWM 24.00–24.70× and DCS-BW 25.60–25.90× faster than the NumPy
  reference on macOS ARM64. These are implementation speedups, with no promised
  Ryzen speedup. See `validation/dcs_compilation_benchmark.json` and
  `scripts/benchmark_dcs_compilation.py`.
- Three alternating 50,000-iteration repeats gave median 0.41 seconds for the
  fixed-count SCS adapter and 0.41 seconds for a direct author scan, including
  the same all-iteration radial reporting. The unrounded timing difference was
  0.26%. See `validation/scs_adapter_timing.json` and
  `scripts/benchmark_scs_adapter.py`. This checks the older fixed-count adapter;
  it is not a measurement of the current 300-transition timed batches.
- An earlier constant-caching check improved BW by 1.30× and SCS by 1.29×.
  That change alone cannot explain a large change in their speed ratio.
  Proposal scale, rejection frequency, fitted geometry, implementation and
  timing boundaries all affect transition cost.
