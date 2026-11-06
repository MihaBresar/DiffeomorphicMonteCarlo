# Release notes

## Current public experiment suite

- Six-method DCS/SCS comparisons in JAX, with a separate Numba DCS extension.
- Frozen 100D/500D maps and steps selected by a matched radius/step pilot on the final zero maps for DCS and calibrated SCS; original author SCS remains unchanged.
- A portable preparation driver reproduces full-budget VI, all calibration candidates and frozen selections with explicit seeds, hashes and costs.
- Eight numbered sampling-seed choices, with exact seeds recorded.
- Numba production kernels for DCS and random-walk methods in the other experiments.
- Exact sample counts, two-decimal displays, full-precision exports and separate preparation/production timing.
- Functional ESS only in the main DCS–SCS comparison and its Numba extension; other experiments skip chain diagnostics and NUTS diagnostic trace files.

The [comparison protocol](docs/REPORTED_STUDY.md) specifies settings and limitations. Both optimized families calibrate on their effective production maps. Equal transition budgets are distinct from equal elapsed preparation time.
