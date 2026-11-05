# Native sampling kernels

All seven launcher experiments use compiled loops for their canonical sampling
methods. `compiled_sampling.py` covers DCS-RWM, DCS-BW, DCS-HnR, ordinary Gaussian
and stable random walks, and SCS. It supports the skew-t, logistic, Eight Schools
(centered and noncentered), and GARCH densities. The current `sampler-comparison` runs DCS-RWM, DCS-BW, reflected BW, DCS-Billiard and author SCS in a shared compiled JAX scan framework. The separate `sampler-comparison-numba` runs the same four DCS methods in optimized native loops against unchanged author JAX SCS; both targets are available and VI is mandatory for both map families. Current reports accumulate radial quantities and functional ESS, caching unchanged values after rejection. See [the exact comparison scope](COMPARISONS.md).
NumPyro/BlackJAX NUTS retain their compiled implementations.

Each worker compiles before the sampling start barrier. Warmup is separate;
production steps are frozen. Native kernels use float64 without fast-math,
cache target/map constants, and call the actual SciPy normal log-CDF routine.
SCS and ordinary random-walk baselines receive the same compilation treatment.
The first third of actual generated sampling iterations is discarded for every
current experiment. In the version-11 100D/500D comparisons, all four DCS variants share fixed h=0.07: BW/reflected-BW/billiard use a uniform-ball increment of radius h and Gaussian RWM uses coordinate standard deviation h/√(d+2). Their discarded warmup uses each actual kernel without step adaptation. Author SCS remains fixed at h=0.1 by default, with acceptance-target 0.234 tuning available as a sensitivity option. The shared DCS h=0.15 alternative and historical acceptance-tuned control are recorded explicitly when selected. The generic experiments retain their recorded step policies, including acceptance-target 0.234 for logistic and ν=0.5 SCS. Every public VI-enabled SCS fit uses the pinned author optimizer; v11 retains v10 fixed-cloud maps and budgets. All production steps remain fixed.

Fixed-count runs use bounded chunks. Timed general kernels check the clock
after batches of 256 transitions, and record elapsed time and overshoot. The
specialized default radial comparison uses 300 for every method, with prefix
sums every 100 to remove exactly its first third. Quick runs verify execution;
their short timings and abbreviated tuning cannot establish rankings.

The public `sample(log_density, ...)` interface accepts arbitrary Python
callables. These use the explicit NumPy fallback; arbitrary Python cannot be
promised native compilation. It still returns the requested number of thinned,
post-burn-in draws and reports the implementation. Built-in benchmark targets
always use compiled paths for the methods listed above.

## Optimized radial BW loops

`numba_radial.py` fuses transformed density, physical radius and coordinate summaries for BW/reflect on skew-t targets. It caches the current density, radius and first three coordinates through rejections, reuses scratch buffers, and stores only cumulative sums plus thinned functionals. Every repeated post-Metropolis state still enters the estimates. It uses float64, no fast-math and uniform-ball displacements from d+2 Gaussian variables. Arbitrary β remains supported; β=2 uses the exact boundary-gap simplification.

Native production starts at the identical final JAX warmup state and fixed production step used in the shared-framework mode. PCG64 production streams are independent of the calibration stream, so parity means matching proposal and target laws, not identical seeded trajectories. Controlled-increment tests compare Metropolis decisions, density/Jacobian formulas, reflected chords and all-state cumulative statistics, including d=500.

`numba_rwm.py` and `numba_billiard.py` provide the current native ordinary RWM
and billiard kernels, with corresponding JAX warmup starts and fixed
production steps. The rotation-mixture modules remain for compatibility and regression
checks; they are not part of the current five-method comparison. The
implementation measurements below concern their recorded kernels and settings.

A local macOS ARM64 throughput check used 900,000 transitions per case, three rotated repeats, and the same saved 100D map and method-specific calibrated start/step. Median paired JAX/Numba time ratios were **10.73× for BW** and **10.27× for reflected BW**. Median rates were 161,765 versus 1,734,092 iterations/s for BW, and 129,286 versus 1,334,991 for reflected BW. Compilation and calibration were excluded. These measurements compare DCS implementations, not DCS against SCS, and do not establish MSE improvements or predictions for other hardware. The complete map, configuration and measurements are in [the validation record](../validation/radial_numba_throughput.json). Reproduce them using `python scripts/benchmark_radial_numba.py` while no other CPU-heavy jobs run. Source hashes identify the measured revision; the subsequently corrected RNG-description string does not alter sampling.

## Paired implementation measurements

Measured on macOS ARM64, 11 September 2026, using three sequential paired runs
of 10,000 iterations each. Every pair uses the same target, fixed map, raw
step-size warmup, frozen kernel and storage interval. Speedups below are the
median ratio of NumPy sampling seconds to native sampling seconds; compilation
and tuning are excluded and separately recorded.

| Target | Method | Native speedup |
|---|---|---:|
| 200D skewed t | DCS-RWM | 7.98× |
| 200D skewed t | DCS-BW | 8.76× |
| 200D skewed t | RWM-Gaussian | 6.34× |
| 200D skewed t | RWM-stable | 3.88× |
| 50D t, ν=0.5 | DCS-RWM | 20.26× |
| 50D t, ν=0.5 | DCS-BW | 20.69× |
| 50D t, ν=0.5 | SCS | 15.53× |
| Logistic regression | DCS-BW | 15.98× |
| Eight Schools | DCS-HnR | 91.73× |
| GARCH | DCS-HnR | 90.80× |

These are implementation speedups, not MSE improvements or predictions for other
hardware. Long floating-point trajectories can diverge after different reduction
rounding; proposal/density parity is verified separately. Repeat the measurements
with `python scripts/benchmark_canonical_compilation.py`. Full-precision records,
configs, maps and source hashes are in
`validation/canonical_compilation_benchmark.json`.

Regression tests compare independent NumPy target and transformation formulas,
432 proposal trials (including acceptance, random-generator state and slice
work), and 36 short warmup/frozen trajectories across all target families and
six algorithms. Super-heavy-tail boundary tests exercise β=0.2 and ν=0.5.

For the other changes affecting HnR versus NUTS, see [the posterior audit](HNR_NUTS.md).
Fresh timings and the new burn-in policy require new runs; see [the rerun guide](RERUN.md).
