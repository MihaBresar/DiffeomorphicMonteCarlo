# Changes

## 2026-11-06 — original SCS baseline only in active comparisons

- Keep only the original author VI/transition with fixed SCS h=0.1 in new five-method JAX and Numba comparisons. Reject acceptance-tuned SCS overrides at the current configuration and command-line entry points; retain the hidden fixed alias for compatible scripts.
- Preserve low-level historical policies, saved results and archived sensitivity evidence. Other manuscript experiments retain their explicitly documented SCS implementation and calibration.
- Clarify that current public DCS settings retain their fitted physical centres and default shared h=0.07. Separate zero-centre and post-VI calibration studies do not change those defaults or establish matched end-to-end tuning against original SCS.

## 2026-11-05 — identity-map optimization and explicit fixed SCS control

- Remove identity Möbius calculations from the JAX ball transform when the supplied deformation is exactly zero. Keep the complete nonzero-deformation path, radial map and Jacobian, proposal draws, domain checks and all-state recording. This also applies to billiards in their existing post-Möbius coordinates; it does not alter fitted maps or coordinate policies.
- Add the low-level `fixed_scs_move_v1` policy with a positive finite `scs_fixed_move`. It runs the unchanged author SCS transition at the supplied scale throughout discarded warmup and production, without adaptation. Keep the public author `h=0.1` default and its original execution path unchanged.
- Validate analytic densities/Jacobians, near-boundary behavior, original-author parity and fixed-step provenance. A serial paired throughput check found no universal speed gain; fresh complete comparison runs are required to assess timed performance on a given computer.

## 2026-11-05 — shared fixed DCS moves, version 11

- Share h=0.07 across all four DCS variants: BW, reflected BW and billiard use that uniform-ball increment radius; Gaussian RWM uses coordinate standard deviation h/√(d+2). Every variant runs its own kernel during discarded warmup without separate step optimization.
- Keep author SCS h=0.1. Preserve the former reflected-BW and billiard pilot policies as explicit historical controls. Keep the same five methods, targets, sampling budgets, timing and first-third production burn-in.
- Preserve version-10 matched fixed-cloud VI: Full 1,000 × 2,000, Smaller 400 × 500, Quick 20 × 128. Fresh-cloud VI studies remain separate experiments, not public defaults.
- Add `--dcs-fixed-move 0.15` as an explicit shared-move sensitivity setting; retain the historical method-specific calibration with `--dcs-step-rule acceptance`. Neither fixed value is claimed optimal or guaranteed to improve performance. Saved v10 results retain their original method-specific calibration; new v11 comparisons require fresh sampling of all methods together.
- Simplify launcher seed choices to Run 1 (default) through Run 9 without changing their underlying seed mapping.

## 2026-11-05 — fixed-cloud, matched-budget comparisons, version 10

- Require VI for both map families in every new 100D/500D five-method comparison. Remove the launcher VI-off choice and reject new off requests at public comparison entry points. Remove the generic untuned-map selection from the launcher and public command line; posterior identity maps remain unchanged.
- Match optimizer update counts and fixed reference-cloud sizes for DCS and SCS: Full 1,000 × 2,000, Smaller 400 × 500, Quick 20 × 128. Preserve author SCS VI; DCS now reuses its own fixed cloud. Parameterizations, seeds and clipping remain method-specific. Apply the same fixed-cloud budget policy to generic paired DCS/SCS experiments; other DCS experiments retain their fitting policy.
- Keep sampler kernels, proposal calibration and saved-result interpretation. Historical untuned-map reports and the version-9 development study remain labelled by their recorded settings. Rerun every method at all preset sizes because the DCS draw-reuse policy changes, including Smaller runs.
- Define the current scope as a comparison of fitted sampling procedures; equal fitting resources do not isolate proposal geometry from fitting or establish general superiority.

## 2026-11-05 — original SCS fitting throughout the repository

- Route every VI-enabled SCS fit through the pinned author's unchanged routine, including the generic experiments and audit entry points. Remove the in-house sphere optimizer and its fallback paths. Explicit VI-off maps remain untuned.
- Record SCS's separate fitting budget: 1,000 updates on 2,000 fixed draws for full runs, with reduced Quick/Smaller budgets. Preserve author seed 0, unit initialization, Adam 0.01 and fitting-only coordinate clipping. Include the executed author files in generic run fingerprints.
- The main version-9 100D/500D comparisons already used this routine. Their fitted maps and loss histories reproduce exactly; DCS fitting, production proposals and step rules are unchanged.
- Rerun the logistic and 50D super-heavy-tail comparisons for results using the revised fitting policy. Saved results retain their original provenance.

## 2026-11-05 — author SCS fitting and step-size sensitivity, version 9

- Use the pinned authors' unchanged VI routine for SCS with a fixed reference cloud, seed 0, coordinate clipping in the fitting objective, and their full 1,000-update/2,000-draw budget. Preserve DCS fitting and all production kernels.
- Default SCS to the authors' fixed h=0.1 after the prespecified development rule rejected acceptance tuning as the global default. Keep acceptance-target 0.234 adaptation as an explicit alternative. Retain billiards' own-kernel radial calibration, RWM/BW acceptance calibration and reflected-BW movement calibration, with equal discarded pilot transition counts.
- Add a matched-map SCS sensitivity runner that reuses a completed version-9 run's maps, seeds, pilot and production budgets and tests the opposite step policy, acceptance or fixed h=0.1. Both policies are explicitly selectable. Report both settings; preserve historical version-8 results separately.
- Disclose different VI recipes, the shared normal-CDF target, excluded preparation costs and radial-only scope. This is not the exact SCS paper experiment, equally optimal tuning, a marginal-superiority claim, or a reproduction of the original DCS paper's BW figures.
- Keep fresh launcher requests separate from saved-run requests and reserve output folders atomically, including when consecutive Windows timestamps collide. Update checks for seed batch 6.

## 2026-11-04 — own-kernel radial calibration, version 8

- Calibrate billiard production moves directly instead of inheriting a reflected-BW step. Give author SCS the same separate transition budget and radial pilot policy using its own transition.
- Select from discarded pilots by bounded squared changes in physical log(1 + norm), including repeated states after rejection. Freeze the selected scale before production; no final MSE or known truth selects it.
- Preserve RWM/BW acceptance tuning, reflected-BW movement tuning, every production kernel, VI fits and the fully untuned R=1 map option. Numba reuses the corresponding JAX pilot state and scale.
- Record separate `radial_tuning.csv` and report tables; keep historical reflected-BW billiard pilots and replot behavior labelled by their original protocol.
- Use fresh runs for version-8 results. Equal pilot transition counts do not imply equal pilot seconds; timing is reported separately.

## 2026-11-04 — radial comparisons, ESS and native billiards

- Use the same five active methods in JAX and optimized Numba comparisons:
  DCS-RWM, DCS-BW, SCS, DCS-BW-reflect and DCS-Billiard. Keep reflected BW
  with its existing ball-movement pilot and preserve historical reports.
- Add optimized native RWM and invariant-chord billiards with the shared JAX
  pilot, effective map, frozen step, and independent native production stream.
- Remove coordinate statistics from new comparisons. Retain analytic radial
  references, original MSE/count/time figures, and add functional ESS and ESS/s.
- Compute comparison ESS from unthinned post-burn-in scalar histories,
  including rejections. Bound concurrent FFT memory, separate diagnostic time,
  and preserve unavailable reasons instead of inventing counts for constant traces.
- Record both comparison protocols as version 7. New results require fresh
  runs; historical thinned files cannot reconstruct unthinned ESS.

## 2026-11-04 — optimized JAX billiards

- Add **DCS-Billiard** as the seventh JAX comparison method, alongside SCS,
  reflected BW and every existing control. Both 100D/500D targets and VI
  settings are available from the launcher. Numba remains unchanged.
- Use a closed-form specular endpoint with invariant extra full chords;
  there is no collision loop or extra Gaussian vector. Remove redundant
  outgoing-velocity work while retaining endpoint validity checks.
- Propose after Möbius, retaining the jointly fitted centre, scale and β;
  the active deformation is zero. Save fitted and effective method maps.
  Pretune with reflected BW in the active coordinates using the same total
  pilot budget, then freeze the step before production.
- Record JAX protocol `timed-sampler-comparison-v6`; Numba remains version 5.
  Keep all eight measurements, all-state recording, one-third burn-in, common
  CPU limits, counts, timings and previous figure families. Rerun the complete
  JAX comparison for new results; archived reports keep their original scope.
- Document the reversible proposal and the restricted radial-kernel identity.
  The added sampler is experimental; no universal MSE or speed ordering is
  asserted, and full-paper results require new measurements.

## 2026-11-04 — common JAX CPU runtime limit

- Set `PJRT_NPROC=1` before backend creation for every method and the
  coordinator. The earlier Eigen-only limits did not constrain all compiled
  kernels; extra per-worker CPU consumption distorted dimension-dependent
  RWM/BW wall-time rates on the laptop.
- Record `pjrt-single-compute-thread-v1` and `PJRT_NPROC` consistently in
  execution plans, manifests and generic chain records. Runtime helper threads
  may remain; the setting controls compute pools and operator parallelism.
- Preserve targets, proposals, tuning and version-5 scientific protocols.
  Rerun the complete comparison for corrected time, sample-count and MSE
  figures; historical timing cohorts must not be combined with new runs.

## 2026-11-04 — reflected Gaussian RWM with rotations

- Add DCS-RWM-rotate to both current comparisons, preserving every existing
  method: six methods in JAX and five in the optimized comparison.
- Use reflected Gaussian increments and the existing corrected angular move,
  selecting exactly one component with equal probability per iteration.
  Pretune both scales within the same total warmup-transition budget and
  retain separate first-third production burn-in.
- Match initial expected squared increment length with Gaussian coordinate
  scale `0.1/sqrt(d+2)`. Record Gaussian scale units and equivalent BW radius.
- Record version-5 protocols, preserve historical reports, and include the
  new method in every existing radial and X₁/X₂/X₃ MSE/count/time figure.
  Fresh comparisons rerun all methods; old results do not acquire this method.

## 2026-11-04 — reflected BW with fixed-radius rotations

- Add DCS-BW-rotate to both current comparisons: a fixed 50/50 random mixture
  of reflected BW and angular moves, implemented in JAX and optimized Numba.
  Each iteration performs and counts exactly one selected move.
- Rotate after the Möbius map so the angular proposal preserves physical
  radius about the fitted centre and uses the physical-density Metropolis ratio.
- Split the existing total warmup budget between reflected-step and
  unit-direction-movement pilots; freeze both scales before sampling.
  Retain separate first-third burn-in, eight functionals, targets and VI choices.
- Record protocols `timed-sampler-comparison-v4` and
  `timed-numba-comparison-v4`. Export angular candidates, frozen scales and
  angles alongside existing BW tuning records. Preserve historical report
  semantics and all figure families. Fresh timed comparisons rerun all methods.
- No existing result is relabeled as evidence for the new sampler; speed and
  MSE comparisons require measured runs.

## 2026-11-04 — mirrored and symmetric coordinate checks

- Add E[X₂], P(X₂ < −3), E[X₃] and P(X₃ > 3) to both comparison modes,
  both existing targets and both VI settings. Retain the radial and X₁ plots.
- Use the exact X₂/−X₁ symmetry for mirrored references and an analytic t(3)
  marginal reference for X₃. Preserve reference MCSE and full-precision exports.
- Accumulate all eight quantities on every post-burn-in state inside each
  sampling clock. The target, proposals, fitting and tuning are unchanged.
- Record protocols `timed-sampler-comparison-v3` and
  `timed-numba-comparison-v3`. Version-1 and version-2 reports keep their
  recorded scope; rerun all methods together to obtain X₂/X₃ results.
- Read UTF-8 reports explicitly in tests, fixing the previous Windows-only
  text-decoding failures.

## 2026-11-04 — first-coordinate estimates

- Add the mean of the first physical coordinate and its one-sided tail
  probability to the existing 100D and 500D skewed t(3) comparisons. Target
  densities, radial thresholds, DCS β=2 and physical initialization are unchanged.
- Accumulate all four functionals inside every sampling clock, including repeats
  after rejection and exact first-third burn-in. Both comparison modes retain
  VI on/off for both map families.
- Retain all original radial figure families and add separate first-coordinate
  estimate/MSE figures. Export independent coordinate reference values and MCSE
  with estimated references clearly labeled; analytic radial references remain.
- Record protocols `timed-sampler-comparison-v2` and
  `timed-numba-comparison-v2`; earlier saved reports retain their original
  radial-only definitions. Rerun all methods together for the new measurements
  and timing. SCS's pinned author source remains unchanged.

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
