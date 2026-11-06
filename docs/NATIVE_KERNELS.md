# Native sampling kernels

The main [DCS/SCS comparison](REPORTED_STUDY.md) has two implementation modes. JAX uses a shared compiled scan and recording framework for all six methods. Numba extension replaces DCS production with fused native loops while both SCS variants retain unchanged author JAX transitions.

The other six experiments use `compiled_sampling.py` for Numba DCS-RWM, DCS-BW, DCS-HnR, Gaussian/stable random walks and SCS on supported canonical targets. NumPyro/BlackJAX NUTS retain their own compiled implementations. SCS fitting always uses the pinned author optimizer.

Native kernels use float64 without fast-math, cache map/target constants and preserve rejected states. Compilation and separate warmup occur outside sampling time; production parameters are frozen. Exact first-third production discard applies to every experiment.

Main radial comparisons check time after 300-transition blocks. General timed kernels use bounded blocks and record elapsed time and overshoot. Recording and required buffer flushes belong inside the production clock.

## Specialized DCS loops

`numba_radial.py` fuses transformed density and radial recording for BW and reflected BW. It caches unchanged values after rejection and reuses scratch buffers. Uniform-ball increments use d+2 Gaussian variables. General β is supported, with an equivalent simplified density for β=2.

`numba_rwm.py` and `numba_billiard.py` provide native RWM and billiard production. These kernels share their corresponding JAX warmup state and fixed step. Their independent PCG64 production streams preserve proposal laws rather than bitwise JAX trajectories.

Tests compare map/density formulas, Metropolis decisions, boundary handling and all-state statistics under controlled inputs. See [validation](VALIDATION.md) and [billiard geometry](BILLIARD.md).

## Custom densities

The public `sample(log_density, ...)` interface accepts arbitrary Python callables through its explicit NumPy fallback. Arbitrary user functions are not promised native compilation. Built-in benchmark production uses its declared compiled route and records the implementation.

For posterior comparisons, see [HnR and NUTS](HNR_NUTS.md). Timing claims must use the same target, hardware allocation, recording and clock scope.
