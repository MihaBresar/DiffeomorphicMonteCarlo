# Release notes

## Current public experiment suite

- Six-method DCS/SCS comparisons in JAX, with a separate Numba DCS extension.
- Frozen 100D/500D maps and preset steps, including original author SCS and the zero-centre/zero-observer SCS control.
- Eight numbered sampling-seed choices, with exact seeds recorded.
- Numba production kernels for DCS and random-walk methods in the other experiments.
- Exact sample counts, two-decimal displays, full-precision exports, functional ESS and separate preparation/production timing.

The [comparison protocol](docs/REPORTED_STUDY.md) specifies settings and limitations. Zero-map SCS retains R/h selected on its author-fitted map before zeroing; this release does not recalibrate those parameters.
