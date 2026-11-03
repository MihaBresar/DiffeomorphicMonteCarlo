# Changes

## 2026-11-03 — selectable targets, shared VI switch and optimized Numba comparison

- Replace the old visible SCS/boundary choices with a four-method JAX comparison (DCS-RWM, DCS-BW, SCS, DCS-BW-reflect) and a separate Numba DCS-BW/reflect versus author JAX SCS comparison. Keep old CLI and replot compatibility.
- Offer the original 100D skew-t(3) and matching 500D skew-t(3), both with DCS β=2. Use prespecified analytic radial references and thresholds.
- Add VI on/off for both map families. Off uses unit scale, zero centre/deformation and no fitting; all methods still receive separate step calibration. On uses equal fitting resources for both families.
- Reuse the same JAX calibrated state and step in optimized Numba DCS production. Fuse density/radius calculations, cache rejected-state values and accumulate exact radial summaries without full-state output; retain rejection repeats. Disclose the separate PCG64 production stream and unequal framework/output costs.
- Preserve equal sampling clocks, one-third burn-in, method-order rotation, CPU allocations, exact counts, two-decimal displays and full-precision exports. Add target/backend labels and tests for numerical kernels, calibration transfer, reports and public CLI routes.

## 2026-11-03 — reflected-BW ball-movement calibration, boundary v3

- Remove the reflected-only step ceiling of 2. Tune reflected BW by observed squared displacement of actual post-Metropolis states in ball coordinates, using rotating h/2, h, 2h trial windows within exactly the existing warmup-transition budget. Rejected moves contribute zero, and steps above 2 remain available.
- Freeze the selected step before production. Preserve the reflected transition, target, fitted maps, burn-in, sampling clock and ordinary BW/SCS calibration. Only the common finite-precision scale guards remain.
- Record candidate steps, iteration counts, acceptance, scores and selections in chain metadata and `reflected_tuning.csv`. Report final-round edge winners without claiming optimality. High acceptance is valid for this criterion; it is no longer forced toward 0.234 for reflected BW.
- Version the separate boundary experiment as `timed-boundary-dcs-scs-v3`. Keep historical v1/v2 results and replot labels unchanged; rerun all boundary methods together for new comparisons. Equal tuning resources do not imply identical tuning criteria or improved radial MSE.

## 2026-11-03 — current-comparison fairness audit

- Fix CPU allocation in partial final waves of both 100D comparisons. Use a smaller pool with the same ordered CPU subset for every method, established before JAX initialization. Record and check expected and actual allocations under `matched-wave-cpu-v2`; macOS remains explicitly OS-managed. Rerun all methods together for corrected timings, especially 50 chains on 32 workers.
- Preserve targets, kernels, map fitting, step calibration and seed rules. No new performance ranking is claimed.
- Correct stale descriptions of historical fixed-step/original-VI SCS execution and clarify which reports include ESS, R-hat and directional diagnostics.
- State the scope of radial-only results, the reflected-only step bound, conditional map fits and the limits of equal-budget tuning. Absolute VI losses omit different base constants and cannot rank map families as normalized KL values.
- Exercise a partial final wave in Linux/Windows CI. Give every pilot of the historical joint-radius CI smoke check one second via an explicit override, avoiding too few draws in its old 0.1-second Windows check; all scientific preset defaults remain unchanged.

## 2026-11-03 — boundary v2, seed batches and MSE variability

- Restore ordinary DCS-BW and author SCS's original step calibration in the boundary comparison, including the numerical log-scale ceiling of 15. Apply the extra step ceiling of 2 only to reflected BW, whose acceptance target may be unattainable. Save this as `timed-boundary-dcs-scs-v2`; the main comparison's protocol is unchanged.
- Preserve the historical boundary v1 runs and validation archive, which used a common ceiling of 2. A final scale below 2 does not prove that the old cap left warmup unchanged; rerun all three boundary methods for v2.
- Add seed batches for both 100D comparisons: default batch 0 retains sampling seed 901001, and batch b adds 1,000,000 × b while leaving map-fitting seed 42 fixed. The launcher offers batches 0–5; `--seed-batch` accepts 0–999 and is mutually exclusive with `--seed`.
- Export inexpensive empirical MSE uncertainty summaries from per-chain squared errors. These are descriptive, not formal 95% confidence intervals; the t(3) norm's infinite fourth moment limits variance-based MSE uncertainty.
- Document full Windows reruns and why timed iteration counts and estimates can differ even when the sampling seed is repeated. No new performance ranking is claimed.

## 2026-11-03 — separate reflected-boundary comparison

- Add a launcher experiment comparing direct-rejection DCS-BW, reversible reflected-chord DCS-BW and unchanged author SCS on the same 100D skewed t(3) target.
- Preserve the main comparison; both BW variants share their fitted map, with matched fitting, separate step calibration, recording, timing and first-third burn-in.
- Document proposal reversibility and distinguish chord reflection from SCS's spherical step-out. Test stationarity, transition symmetry, multiple reflections and numerical safeguards.
- Record and report step-ceiling saturation when the common acceptance target is unattainable.

## 2026-11-03 — exact DCS JAX efficiency improvements

- Generate uniform-ball proposals from one normalized vector of d+2 Gaussian draws, preserving the proposal law while removing the separate radial random draw.
- Simplify the β=2 radial map with the exact Möbius boundary-gap identity; retain the general formula for other β values.
- Reject outside-ball proposals before either target-density calculation, while recomputing both densities for admissible proposals.
- Preserve the original author SCS code, common fitting and warmup budgets, full-state recording, timing and first-third burn-in.
- Record `jax-dcs-exact-ball-v1` separately from the unchanged v3 experimental protocol. BW seeded trajectories change, so regenerate the full 100D comparison; other experiments are unchanged.
- Add a reproducible fixed-map, fixed-scale JAX implementation benchmark and preserve the earlier JAX results as historical evidence.

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
