# Native sampling kernels

All seven launcher experiments use compiled loops for their canonical sampling
methods. `compiled_sampling.py` covers DCS-RWM, DCS-BW, DCS-HnR, ordinary Gaussian
and stable random walks, and SCS. It supports the skew-t, logistic, Eight Schools
(centered and noncentered), and GARCH densities. The default 100D comparison
uses its specialized native radial accumulator and the SCS authors' JAX kernel.
NumPyro/BlackJAX NUTS retain their compiled implementations.

Each worker compiles before the sampling start barrier. Warmup is separate;
production steps are frozen. Native kernels use float64 without fast-math,
cache target/map constants, and call the actual SciPy normal log-CDF routine.
SCS and ordinary random-walk baselines receive the same compilation treatment.
The first third of actual generated sampling iterations is discarded for every
current experiment. The default author-SCS recipe retains fixed h=0.10; its
untimed equilibration is reported separately from DCS step calibration.

Fixed-count runs use bounded chunks. Timed general kernels check the clock
after batches of 256 transitions, and record elapsed time and overshoot. The
specialized default radial comparison uses 300 for every method, with prefix
sums every 100 to remove exactly its first third. Quick runs verify execution;
their short timings and abbreviated tuning cannot establish rankings.

The public `sample(log_density, ...)` interface accepts arbitrary Python
callables. These use the explicit NumPy fallback; arbitrary Python cannot be
promised native compilation. It still returns the requested number of thinned,
post-burn-in draws and reports the implementation. Built-in benchmark targets
always use the native path for the methods listed above.

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
