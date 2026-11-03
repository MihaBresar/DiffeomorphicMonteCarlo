# Changes

## 2026-11-03 — matched 100D comparison using author SCS

- Retain the byte-for-byte author SCS transition and implement DCS-RWM/BW in the same JAX scan framework, with equivalent full-state recording, projection and radial reduction.
- Give both map families the same fresh-batch VI resources, Adam settings and gradient-norm clipping; remove separate SCS fitting overrides.
- Tune every method on separate warmup draws toward acceptance 0.234 using the same observed-acceptance dual averaging, then freeze the scale.
- Preserve equal sampling time, physical initialization, first-third burn-in, exact all-iteration estimates, integer sample counts and existing figure families.
- Save the protocol as `timed-matched-dcs-scs-v3`; keep historical reports plot-compatible. Other experiments retain their existing optimized kernels.

## Native kernels across experiments, equal-time SCS and first-third burn-in

- Compile all canonical DCS, HnR, ordinary random-walk and spherical comparison loops.
- Keep tuning separate, freeze sampling kernels, and discard the first third of actual generated sampling iterations in every current experiment and generated reference chain.
- Switch the default 100D skewed ν=3 comparison to equal time; keep super-heavy tails at equal iteration counts.
- Display two decimal places and scientific MSE notation while retaining exact integer counts and full-precision exports.
- Audit posterior comparison changes and measure native implementation speedups with paired benchmarks.

## Single 100D SCS comparison and compiled DCS

- Simplified the default launcher to one skewed ν=3 comparison with DCS-RWM, DCS-BW and author SCS; previous audit choices are retired and their results retained.
- Restored original DCS VI and continuous smoothed 0.234 adaptation, with correct physical initialization and unclipped production density.
- Vendored the MIT-licensed paper-date SCS source unchanged and applied its original VI/step recipe to the shared normal-CDF target.
- Compiled DCS sampling loops, cached radial quantities, and added numerical/RNG parity checks; pinned Numba and llvmlite.
- Added 50 full-length chains per method, isolated sampling clocks, exact all-iteration radial estimates, original VsSCS panels and chain-level bias/variance reports.
- Routed the launcher, CLI and All seven entry to the same current SCS experiment.

## Joint radial calibration

- Added an independent R-and-step audit for the symmetric ν=3 and skewed ν=2 targets, with equal pilot sampling budgets and 8×8 candidate grids for each method.
- Added jointly selected, R=1, VI-R and R=√(d+1) comparisons; every case receives proposal pretuning followed by fresh equal-time validation.
- Added separate boundary probes that preserve the original DCS and SCS transitions, with exact crossing, step-out, rejection and acceptance counts.
- Added calibration profiles, heatmaps and full-precision tables alongside the existing estimator, error and sample-count figures.
- Added a no-command launcher option and Windows/Linux execution checks. Existing experiments and saved results remain available.

## Parallel CPU execution

- Added a CPU-worker selector and bounded independent chain processes, including logistic references.
- Added native Linux/Windows CPU affinity, explicit thread limits, worker provenance and compilation accounting; macOS scheduling remains OS-managed.
- Preserved deterministic chain seeds, matched NUTS/DCS budget ordering, atomic checkpoints and saved-worker-count resume.
- Added cross-platform worker checks and made Git optional for running downloaded ZIP copies.
- Earlier sequential-only checkpoints remain available for plotting; start a new run after updating the scientific source.

## 0.2.0 — corrected experimental implementation

- Centralized targets, ball/cap transforms, tuning and kernels; retained thin compatibility entry points.
- Corrected GARCH prior, shared constrained-coordinate handling and posterior reference provenance.
- Matched paper settings: d=200 RWM comparison, ν=3 fixed-step comparison, symmetric ν=0.5 target, 30-second logistic budget.
- Replaced the skew-t coordinate reference generator, used analytic radial references, and added bounded directional checks.
- Replaced pooled logistic references with independently generated NUTS references and diagnostic gates for paper use.
- Froze adaptation after warmup, removed artificial density clipping/cutoffs, synchronized timing and reported complete setup costs.
- Saved per-chain draws, seeds, tuning, work, failures and diagnostics; corrected RMSE aggregation; added plot-only and resume workflows.
- Pinned the full Python 3.12 CPU dependency closure, added mathematical tests and end-to-end CI, finalized MIT licensing and citation metadata.
- Added Windows/macOS launchers and a local browser interface for setup, running, resuming and viewing results without typing commands.

Previous manuscript figures remain archival. Version 0.2.0 requires new production simulations; it does not preserve old performance numbers.
