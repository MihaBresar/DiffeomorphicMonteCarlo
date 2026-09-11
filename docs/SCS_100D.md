# One 100D skewed-t comparison

The default launcher experiment compares **DCS-RWM, DCS-BW and SCS** on the
existing 100-dimensional normal-CDF-skewed Student-t target, with ν=3 and
α=(100,−100,0,…). It estimates P(‖X‖≥50) and E‖X‖ against exact radial truth.
All three methods start at the physical vector of ones, using the correct
inverse map. The historical DCS initialization error is not restored.

| Preset | Independent chains per method | Production iterations per chain | Burn-in | Stored radial trace interval |
|---|---:|---:|---:|---:|
| Quick check | 2 | 500 | 100 | 1 |
| Smaller reproduction | 20 | 25,000 | 100 | 10 |
| Full paper settings | 50 | 500,000 | 100 | 50 |

Every production iteration contributes to estimates, including repeated states
after rejection. Thinning only reduces saved instantaneous radial traces.
Quick check uses abbreviated fitting and cannot support a performance ranking.
Fifty chains improve precision of the empirical error assessment; they do not
remove bias caused by insufficient exploration within individual chains.

## DCS: original fitting and adaptation, compiled implementation

The original DCS VI function is extracted from the byte-preserved
`validation/legacy_fixed_step.py`, with ν explicitly set to 3. The full run uses
1,501 optimizer updates, 2,000 fresh base draws per update, seed 42, Adam with
learning rate 0.01, and β=2. The fitted R, centre and Möbius vector are shared by
DCS-RWM and DCS-BW and fixed during sampling. Quick/reproduction budgets use
21/401 updates and 128/500 draws. Original clipping in the VI objective is
preserved and recorded; the production target has no CDF clipping or artificial
interior-ball cutoff.

DCS-RWM proposes isotropic Gaussian increments. DCS-BW proposes uniformly in a
Euclidean ball. Both reject proposals outside the open unit ball. Both start
with proposal scale 0.1 and use the original smoothed dual-averaging update
toward acceptance 0.234, continuously through burn-in **and production**. This
is the historical adaptive experiment, not a frozen-kernel comparison or a
claim that 0.234 optimizes radial MSE for every target.

`dcs_bench/fast_dcs.py` compiles the full loop using Numba without fast-math.
Map and target constants are cached. Density evaluation also computes the
physical radius; rejection reuses the current radius. Normal log-CDF values
come from the actual SciPy double-precision routine, not an approximation.
Tests compare direct densities, proposed-state acceptance, random-generator
state, adaptive trajectories and all-iteration estimates with NumPy references.
Floating-point reductions need not yield bitwise-identical long adaptive runs.

The controlled implementation check (three 25,000-iteration repeats per method,
one fixed recorded map, same continuous adaptation) found RWM 24.0–24.7× and BW
25.6–25.9× faster on macOS ARM64 than the NumPy reference. These are implementation
speedups, not MSE improvements or promised Ryzen speedups. Raw measurements are
in `validation/dcs_compilation_benchmark.json`; repeat them with
`python scripts/benchmark_dcs_compilation.py`.

## SCS: original author implementation

The MIT-licensed source is vendored without modification at
[`ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565),
the paper-date revision. The adapter invokes its tangent-Gaussian proposal,
normalization, cap step-out, projection, Jacobian and Metropolis transition.
Source hashes and licence are in `third_party/scs_paper`.

The full fit uses the original fixed batch of 2,000 base draws, seed 0, 1,000
Adam updates with learning rate 0.01 and its original objective clipping. Latitude
is 1.1. The step is fixed at 0.1, as in the authors’
[`run_skewt.py`](https://github.com/liusf15/sub-cauchy-sampler/blob/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565/experiments/run_skewt.py).
The original fitting budget differs from DCS: this compares their original
recipes, not equal VI resources. Setup time is reported separately. Quick and
smaller settings shorten SCS fitting to 20/400 updates and 128/500 fixed draws.

The author transition runs inside compiled JAX scans with online radial
accumulators. Progress callbacks are suppressed outside the vendored code.
Independent keys per storage block preserve the transition law, but do not
reproduce the author's bitwise random sequence. Tests compare the adapter's
estimates with full-state executions of the original transition, including
rejections and a final incomplete storage block.

**Target distinction:** the [SCS paper §4.1](https://arxiv.org/html/2601.11066v1#S4.SS1)
uses a Student-t CDF skewing factor and ν=1 or 2. This comparison applies its
original algorithm and settings to our existing normal-CDF-skewed ν=3 target.
It is not a literal reproduction of the SCS paper's target or plotted values.
All three methods here evaluate the same normal-CDF target; its radial truth
is unchanged by skewing.

## Timing and outputs

This experiment fixes iteration counts and measures actual elapsed sampling
time. It does not estimate equal-time MSE by rescaling fixed-count errors.
Both compiled loops finish compilation and burn-in before their production
clocks start. Production timing includes adaptation and radial estimation;
it excludes fitting, compilation, synchronization waits, disk writes and plots.
Those setup costs and total invocation wall time are recorded separately.

Parallel chains run in synchronized waves, with only one method sampling at a
time. Method order rotates between waves. Each method receives identical wave
sizes and CPU allocation; Windows/Linux affinity fills physical cores before
SMT siblings. A 50-chain run can use 32 workers on the Ryzen 9 9950X3D. Sequential
runs provide the clearest isolated timing comparison; timings from parallel
runs describe that recorded concurrent CPU workload. Keep the machine otherwise
idle when measuring speed.

`VsSCS.png`/PDF preserve the manuscript’s three linear panels: tail MSE, norm
MSE and average sampling time. `scs_100d_estimates` adds radial estimate boxplots
and MSE panels.
Every chain estimate is shown, so a small spread around an incorrect value is
visible. Additional plots show runtime, iterations per second, empirical
bias squared plus variance, and MSE at saved prefixes of these same chains.
`paper_values.html` and CSV files contain full stored precision, exact counts,
individual chain errors and tuned parameters. The variance uses divisor n,
so **MSE = empirical bias² + variance** exactly, up to rounding.

Use **Start.bat → Set up environment → 100D skewed t (ν=3) → Start run** on
Windows, or **Start.command** on macOS. Run environment setup again after an
update: this version adds pinned Numba and llvmlite. Choose a new output folder
for a command-line run:

```bash
python run_experiment.py skew-t-100d-fixed --preset paper --workers 32 --output-dir results/scs-100d-paper
```

The launcher, compatibility entry points and All seven experiments route this
identifier to the same runner. Earlier audits remain archived in the repository
but are absent from the launcher. Old plots remain available; new speed and
MSE results require rerunning this one experiment.

## Recorded full run, 11 September 2026

One worker on macOS ARM64; 50 independent chains per method, each with 500,000
production iterations. These values are conditional on the recorded fitted
maps and hardware; they do not establish a universal ranking.

| Method | Mean sampling seconds per chain | Iterations/s | Tail MSE | Norm MSE |
|---|---:|---:|---:|---:|
| DCS-RWM | 0.3133969441795489 | 1595420.7891495712 | 2.810255203374965e-07 | 0.0019954432327649384 |
| DCS-BW | 0.3371467700600624 | 1483033.6352055974 | 2.0956641753087117e-07 | 0.0020199195066246223 |
| SCS | 5.336324878320011 | 93697.44372786589 | 8.922096831685801e-08 | 0.0012744363219996985 |

DCS-RWM and DCS-BW achieved 17.03× and 15.83× the SCS iteration throughput, respectively. SCS had lower error per fixed number of iterations for both functions. No equal-time MSE is inferred by multiplying these errors by a timing ratio.

The squared empirical norm bias was less than 1% of MSE for every method.
The recorded norm error therefore mostly reflects variation across chains.
This does not show that more chains alone repaired an earlier biased run;
that comparison also differed in fitting, adaptation or run length.

Full precision, individual chain results and setup parameters are archived in
`validation/scs_100d_comparison.json`, with environment/source hashes in
`validation/scs_100d_protocol.json`. Every method contributes exactly 25,000,000
production iterations to this comparison.

The old approximate threefold speed ratio is not fully attributable from the
archived plots alone. The earlier controlled constant-caching check improved
100D BW by 1.295× and SCS by 1.290×, so that change alone cannot explain a large
change in their ratio. Proposal scale, rejection frequency, map geometry,
implementation and timing boundaries all affect transition cost. The new
compiled-loop benchmark isolates an actual DCS implementation speedup.

A separate check verified that the SCS adapter does not create a large
comparison overhead: three alternating 50,000-iteration runs gave median
0.4086899159956374 s for the adapter and 0.40761949999432545 s for one direct
author scan, with identical all-iteration radial reporting costs included.
The measured difference was 0.26%. Raw records are in
`validation/scs_adapter_timing.json`; repeat using
`python scripts/benchmark_scs_adapter.py`. This is an implementation timing
check, not another target or sampler-tuning experiment.
