# Validation of the corrected implementation

## Optimized JAX billiard addition (4 November 2026)

Protocol v6 adds `DCS-Billiard` to the JAX comparison, preserving the previous
six controls and the separate Numba comparison. Its endpoint expressions match
the reviewed optimized prototype. The candidate uses post-Möbius coordinates,
retains the jointly fitted centre/R/β, and reuses a reflected-BW pilot in those
coordinates before freezing the step. See [the method and limits](BILLIARD.md).

All **297 automated tests** and **four public-CLI Quick configurations
(56 chains)** passed locally: 100D/500D, VI on/off, two concurrent workers.
Checks cover independent billiard geometry/reversal, volume and transition
symmetry, exact pilot reuse, map provenance, all eight estimators, one-third
burn-in, figures and read-only report regeneration. These are execution checks,
not new MSE rankings. Full-paper settings remain 50 chains per method and
20 sampling seconds per chain, with 1500 separate pilot transitions.

[Validation record](../validation/billiard_checks.json). The author SCS source,
targets, references, existing proposal helpers and CPU policy are unchanged.
GitHub Actions checks Linux and Windows separately; its current status is
available in the repository's Actions tab.

## Common JAX CPU-pool correction (4 November 2026)

The earlier numerical-library flags did not constrain all JAX CPU runtime
pools. Current workers and the coordinator additionally set `PJRT_NPROC=1`
before JAX initialization, under execution policy
`pjrt-single-compute-thread-v1`. This applies equally to every method. It
limits compute pools and compiled parallel-task assignment; it does not
promise that the process has no auxiliary threads.

The saved four-worker Smaller runs showed a dimension-dependent reversal:
plain RWM/BW throughput was about 0.60× in 100D and 1.76× in 500D. The slower
method consumed about 1.90–2.21 CPU seconds per wall second. Isolated production
kernel replays with the common pool limit removed all recorded parallel-task
annotations and reduced CPU/wall ratios to about one. In that separate
single-process experiment, RWM/BW was 1.09× in 100D and 1.20× in 500D.
Boundary/acceptance diagnostics and frozen scales were unchanged. Alternative
exact BW generators did not improve on the existing generator in this probe.

All **288 tests**, **eight Quick configurations (88 chains)** and a separate
**four-worker, six-method timing check in 100D and 500D (48 chains)** passed.
The timing check used VI on, four chains per method, two sampling seconds per
chain and 1000 separate calibration transitions. The full validation record
contains the actual rates, CPU/wall ratios, settings and source hashes.

See [CPU-pool validation](../validation/jax_thread_pool_checks.json).
Sampling transitions, map/tuning algorithms, references, burn-in and estimators
are unchanged. **Rerun complete fixed-time comparisons** under the corrected
execution policy: changed throughput changes the number of generated samples
and thus the fixed-time MSE. Earlier results below describe their original
execution settings. The short timing check is not a new accuracy ranking.
Windows/Linux verification runs separately in GitHub Actions.

## Reflected Gaussian RWM with rotations (4 November 2026)

Version 5 adds **DCS-RWM-rotate** to the existing comparisons: six methods
in JAX and five in the optimized Numba/author-SCS comparison. One reflected
Gaussian move or one fixed-radius angular move is selected with probability
1/2 per iteration. Both scales are pretuned within the same total budget and
frozen before sampling. Gaussian coordinate standard deviation starts at
`0.1/sqrt(d+2)`; it is distinct from the BW proposal radius. Reports preserve
all eight estimates, the ten figure families and every earlier control.

All **286 automated tests** passed locally in **85.779 seconds**.
The complete eight-setting matrix passed for both Quick check (**88 chains**)
and Smaller reproduction (**880 chains**), covering 100D/500D skewed t(3),
VI on/off and both implementations. Tests cover reflected Gaussian symmetry,
physical-density corrections, stationarity checks, component selection,
pilot accounting, exact first-third discard and all-state MSE recomputation.
Matching JAX and Numba settings used identical maps, references, paired chain
seeds, final proposal scales and angular scales/angles for all common methods.
The prior transition controls and pinned author SCS sources remain unchanged.

| Local preset | Chains/method | Requested seconds/chain | Separate tuning transitions | Simultaneous workers | VI updates × draws/update | Independent reference draws |
|---|---:|---:|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 2 | 20 × 128 | 50,000 |
| Smaller reproduction | 20 | 2.00 | 1000 | 4 | 400 × 500 | 1,000,000 |

These local runs used sampling seed batch **1**, fit seed **42**, independent
reference seed **740317**, float64 CPU computation and the Apple M4 Pro laptop.
VI off used zero fitting draws for both maps. Each mixture split its tuning
transitions equally between its reflected and angular components. Sampling
time includes the first-third production discard and every repeated state;
fitting, reference preparation, compilation, tuning and reports are excluded.
The 300-transition clock checks can exceed the requested time, particularly
in Quick check; actual measured times and counts remain in each report.

See the [version-5 validation record](../validation/rwm_rotation_checks.json)
for source hashes, actual scales/angles, settings and backend matching checks.
Quick checks establish execution only. The Smaller results are exploratory,
conditional on one fitted map and one seed batch: they do not establish
universal dominance or publication-scale rankings. Heavy-tailed mean/norm
errors make descriptive MSE uncertainty fragile, and independent reference
error remains separate. The optimized comparison includes framework and
output-processing differences. Fresh full-paper runs and independent seed
batches are needed for stronger performance conclusions. Windows/Linux
checks for this revision run separately in GitHub Actions; no remote pass is
claimed here.

## Reflected-BW/rotation mixture (4 November 2026)

Version 4 adds **DCS-BW-rotate** in both JAX and optimized Numba. Each
iteration chooses one reflected BW or one fixed-radius angular update with
equal probability. Both scales are pretuned within the existing shared
warmup-transition budget and then frozen. With VI enabled, rotation takes
place after the Möbius transformation and preserves distance from the fitted
centre, with the corresponding proposal correction.

All **271 automated tests** and **eight quick comparison configurations**
passed locally on the Apple M4 Pro laptop, covering **72 chains**. Tests cover
reversibility, the independent physical-density ratio, target invariance,
50/50 component selection, complete pilot accounting, all-state statistics,
exact first-third burn-in and preservation of the existing plot families.
Matching JAX and Numba cases selected identical pairs of calibrated scales
and used identical fitted maps. The existing transition implementations,
map fitting and author SCS code are unchanged.

See the [rotation check record](../validation/rotation_checks.json), including
the actual quick-check scales and angles. These checks validate execution and
implementation consistency; they do not establish performance rankings.
Fresh full-paper runs are needed to assess the new sampler. Windows/Linux
checks for this revision run separately in GitHub Actions.

## Three-coordinate extension (4 November 2026)

Version 3 retains the version-2 radial and X₁ measurements and adds the X₂
mean/left tail and X₃ mean/right tail. The X₂ reference relations follow from
the target's reflection symmetry; the X₃ references use its analytic t(3)
marginal. Both comparison implementations accumulate all eight measurements
within their sampling clocks, counting repeated states and discarding exactly
the first third of timed iterations. The existing targets, fitting, tuning
and transition laws are unchanged.

All **252 automated tests** and **eight quick comparison configurations**
passed locally, covering **56 chains**. Checks include mirrored/analytic
reference values, all-state coordinate accumulation, frozen calibration,
unchanged prior trajectories and statistics, and all ten figure families.
The previous Windows-only HTML-decoding test failures were corrected by
reading reports explicitly as UTF-8. See the
[three-coordinate check record](../validation/three_coordinate_checks.json).
These quick runs check execution, not sampler rankings; full-paper runs and
the new remote Windows/Linux workflow are separate validation steps.

The version-2 smaller-run results below do not contain X₂/X₃ measurements
and must not be presented as validation of their performance rankings.
Fresh runs of all methods are required to obtain those measurements.

## Historical first-coordinate extension (4 November 2026)

Version 2 adds first-coordinate mean/tail measurements for the existing 100D
and 500D skewed t(3) targets. There are eight combinations of
target, implementation mode and VI setting. References distinguish analytic
radial constants from independently estimated values and retain reference MCSE.
New checks cover independent coordinate references, coordinate accumulation
including repeated states, exact first-third discard and additional reports.
The original radial plot families and historical replot support are preserved.
Fresh runs of every method are needed because all four measurements now enter
the timed production work.

The following 3 November record describes the earlier radial-only version; its
test counts and performance results must not be presented as validation of the
new coordinate measurements.

## Historical selectable-target comparisons (3 November 2026)

That suite covered **eight comparison combinations**: 100D and 500D skewed Student-t targets, two implementation modes, and VI on/off for both map families. Both targets use ν=3 and DCS β=2. Checks covered target-specific analytic references, exact one-third burn-in, frozen step calibration, native/JAX numerical agreement and partial-worker scheduling. SCS's pinned source files and licence remained unchanged. The machine-readable [check record](../validation/comparison_suite_checks.json) records those automated-test and execution-check results.

All **eight Smaller reproduction combinations** completed on the Apple M4 Pro laptop with four workers: **560 chains**, twenty per method/case, two seconds of production each, no retries or discarded attempts. The numerical sources stayed fixed during these runs. The shared JAX and native comparisons used identical fitted maps and step calibration for each target/VI setting. Guard flags are reported, including SCS saturation in 12 of 20 chains in each 100D no-VI comparison. These are conditional, exploratory estimates, not full-paper rankings.

See [the full-precision method results](../validation/comparison_smaller_results.csv), [map fitting results](../validation/comparison_smaller_results_fitting.csv), [observed comparisons](COMPARISON_RESULTS.md) and [the protocol](COMPARISONS.md). Remote Windows/Linux validation requires a successful run of the updated GitHub Actions workflow; no such run is claimed before publication.

The following sections retain historical validation records for earlier revisions.


The Windows queue follow-up passed **39 tests**, including eight competing
processes claiming 219 jobs exactly once across three synchronized rounds.
A 200D DCS-versus-NUTS smaller reproduction was deliberately stopped after
the previous release saved eight DCS chains. The updated code reused all
eight files byte-for-byte, completed eight NUTS chains, and resumed again with
all 16 checkpoints unchanged. The integration ran on macOS; native Windows
and Linux checks run in GitHub Actions. See
[`validation/windows_queue_checks.json`](../validation/windows_queue_checks.json)
and [the recovery instructions](WINDOWS_QUEUE_FIX.md).

The local verification record is [validation/local_checks.json](../validation/local_checks.json), including platform, source hashes, experiment fingerprints and full-length tuning checks.

- **15 numerical/API tests passed** under Python 3.12.14: NumPy/JAX density agreement and finite gradients; GARCH agreement with an independent physical-coordinate likelihood and Jacobian; centered/non-centered Eight Schools equivalence; inverse maps and finite-difference volume/surface Jacobians; near-boundary stability; exact radial expectations; independent skew-t generation versus quadrature; invariant-transition checks for all three ball kernels and SCS; correct independent-chain RMSE; honest unavailable diagnostics; reference order; checkpoint validation; and the standalone API.
- **All seven quick experiments passed**, including both logistic prior settings: 40 production smoke chains plus eight separate logistic reference smoke chains. Every figure and the combined posterior table was generated.
- **All nine full-length map fits passed**, using the paper optimization settings and sample sizes. This includes the small-β symmetric target and both logistic priors.
- **All seven plotting-only paths passed.** Resume reused all 40 production checkpoints without changing their SHA-256 hashes. Compilation checks and `pip check` passed.
- Logistic and fixed-step PNG outputs were visually inspected for readable labels, correct panel arrangement and explicit quick-result labelling.

These checks establish implementation consistency and executable workflows. They do **not** establish paper-scale performance, posterior convergence, CLTs, or accuracy of short smoke references. Full paper production chains and the high-accuracy independent logistic references must still be run. Results from the quick profile must not be used as publication evidence. The Linux CI workflow is provided; no completed remote CI run is claimed.

Constant functional traces receive null MCSE/ESS with an explanation; they are not reported as successful zero-error diagnostics. High-accuracy logistic reference preparation rejects failed diagnostic screens for paper use. PosteriorDB reference MCSE is reported rather than treating the finite draws as exact values.

Fixed sufficient theorem assumptions are not enforced on empirical examples. The logistic benchmark intentionally remains a test beyond those assumptions. Larger sensitivity, ablation, calibration and non-centered-baseline studies remain additional research designs, as distinguished in [MANUSCRIPT_CHANGES.md](MANUSCRIPT_CHANGES.md).
# Restored figures, CPU scheduling and VI update

The September 11 follow-up passed **30 tests**, all seven end-to-end quick
experiments with two workers, and full replot/resume checks with **58 NPZ files
unchanged**. Browser checks covered all-thread selection, restoring a saved run's
figures, supplementary grouping and full-precision numerical tables. A synthetic
16-core/32-thread topology test checks Ryzen-style allocation; the local machine
is macOS, so actual Ryzen performance remains to be measured on that machine.
See [`validation/plots_cpu_checks.json`](../validation/plots_cpu_checks.json) and
[the implementation audit](IMPLEMENTATION_AUDIT.md). GitHub Actions validates
native Linux/Windows execution separately.

The earlier validation records below describe their original code revisions.
