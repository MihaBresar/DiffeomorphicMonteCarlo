# Native sampling kernels

All seven launcher experiments use compiled loops for their canonical sampling
methods. `compiled_sampling.py` covers DCS-RWM, DCS-BW, DCS-HnR, ordinary Gaussian
and stable random walks, and SCS. It supports the skew-t, logistic, Eight Schools
(centered and noncentered), and GARCH densities. The current `sampler-comparison` runs DCS-RWM/BW/reflect/rotate and author SCS in a shared compiled JAX scan framework. The separate `sampler-comparison-numba` runs optimized native BW/reflect/rotate against unchanged author JAX SCS; both targets and both VI settings are available. The native loops accumulate radial and first-three-coordinate quantities, caching unchanged values after rejection. See [the exact comparison scope](COMPARISONS.md).
NumPyro/BlackJAX NUTS retain their compiled implementations.

Each worker compiles before the sampling start barrier. Warmup is separate;
production steps are frozen. Native kernels use float64 without fast-math,
cache target/map constants, and call the actual SciPy normal log-CDF routine.
SCS and ordinary random-walk baselines receive the same compilation treatment.
The first third of actual generated sampling iterations is discarded for every
current experiment. Ordinary BW, DCS-RWM and SCS use separate per-chain acceptance-target 0.234 calibration. Reflected BW uses squared ball movement within the same warmup budget, without a special bound of 2. All production steps freeze. Fixed SCS h=0.10 describes an archived recipe, not either current comparison.

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

Native production starts at the identical JAX-calibrated state and frozen step used in the shared-framework mode. PCG64 production streams are independent of the calibration stream, so parity means matching proposal and target laws, not identical seeded trajectories. Controlled-increment tests compare Metropolis decisions, density/Jacobian formulas, reflected chords and all-state cumulative statistics, including d=500.

`numba_rotation.py` adds DCS-BW-rotate: a fixed 50/50 mixture of one reflected BW move or one angular move per iteration. It uses the same JAX pilot calibration and both frozen scales as the JAX mixture. Its fused loop records the same eight all-iteration quantities and burn-in prefix sums as native BW/reflect. Angular proposals preserve physical radius about the fitted centre by rotating after the Möbius map; only the chosen mixture component runs. The separate pilot phases share one warmup-transition budget. This implementation does not by itself establish a speed or MSE advantage over SCS; the throughput measurements below concern the earlier BW/reflect kernels only.

A local macOS ARM64 throughput check used 900,000 transitions per case, three rotated repeats, and the same saved 100D map and method-specific calibrated start/step. Median paired JAX/Numba time ratios were **10.73× for BW** and **10.27× for reflected BW**. Median rates were 161,765 versus 1,734,092 iterations/s for BW, and 129,286 versus 1,334,991 for reflected BW. Compilation and calibration were excluded. These measurements compare DCS implementations, not DCS against SCS, and do not establish MSE improvements or promised Ryzen speedups. The complete map, configuration and measurements are in [the validation record](../validation/radial_numba_throughput.json). Reproduce them using `python scripts/benchmark_radial_numba.py` while no other CPU-heavy jobs run. Source hashes identify the measured revision; the subsequently corrected RNG-description string does not alter sampling.

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

These are implementation speedups, not MSE improvements or promised Ryzen
speedups. Long floating-point trajectories can diverge after different reduction
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
