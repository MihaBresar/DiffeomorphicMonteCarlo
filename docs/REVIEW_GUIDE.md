# Reviewer guide

The launcher now defaults to the [reported zero-centre study](REPORTED_STUDY.md),
whose frozen maps, original results and calibration provenance are bundled in
`studies/zero_centre`. It is distinct from the fitted-centre version-11 protocol
discussed below. Current runs include both original SCS and SCS calibrated.
DCS and the calibrated control used equal VI evaluation and R/h pilot transition
budgets; original SCS retains its author defaults. This does not imply identical
map choices or equal elapsed preparation time.

This repository supports inspection and independent reruns of the corrected experiments and subsequent sampler extensions. Start with a small execution check, then inspect the recorded protocol before interpreting comparisons. The source commit and run manifest, rather than a figure filename, identify a result.

## Scope and evidence

The default reported-study comparison loads bundled full-budget maps without refitting. All four DCS variants use the recorded zero-centre, zero-deformation maps and calibrated radius/step settings; original SCS retains its author-fitted map and h=0.1. The added SCS calibrated
control retains the same author centre/observer with pilot-selected radius and
step under the same pilot budget as DCS. Its original JAX results and development record are bundled, while the Numba option is an implementation extension using those settings.

The separately retained **earlier fitted-map** JAX and optimized-Numba comparisons use version 11. They compare DCS-RWM, DCS-BW, SCS, reflected BW and DCS-Billiard on 100D/500D normal-CDF-skewed t(3) targets, with VI fitted for both map families. Billiards and reflected BW are subsequent extensions. Both recipes measure radial-tail probability and mean norm for the recorded sampling procedures. They cannot separate proposal geometry from map preparation or establish coordinate, no-VI or general sampler superiority.

Current runs incorporate implementation and calibration corrections and do not reproduce or validate the original manuscript figures retrospectively. The [manuscript relationship](MANUSCRIPT_CHANGES.md) identifies their scope. The [version-9 SCS study](AUTHOR_SCS_CALIBRATION.md) documents the SCS step-policy choice and reports both fixed and acceptance-tuned outcomes; it does not validate the v11 shared DCS step setting.

## Minimal independent check

Install Python 3.12 and the pinned CPU dependencies using the [README](../README.md). The desktop launcher provides the same experiment choices. From an activated environment:

```sh
python run_experiment.py reported-zero-centre --preset quick --workers 1 --dry-run
python run_experiment.py reported-zero-centre --preset quick --workers 1 --output-dir results/reviewer-check
python run_experiment.py --plot-only results/reviewer-check
```

Confirm a complete manifest and open `paper_values.html`. A quick result checks execution only. The reported-study smaller preset uses 20 chains × 3 seconds; full settings use 50 chains × 20 seconds. Every preset loads the same bundled maps without refitting. The default fresh sampling batch is 22; batches 20 and 21 repeat the reported streams, with hardware-dependent timed counts. Use a fresh output directory each time. To examine the dimensional extension, select `--target skew-t-500d`.

For this default study, verify the bundled map hashes, effective zero-centre DCS maps, recorded radius/step choices and calibration provenance against [the reported-study protocol](REPORTED_STUDY.md). The VI and R/h pilot counts match DCS against SCS calibrated. Distinguish these
counts from elapsed preparation time and the distinct retained map policies;
original SCS remains a separate author-default baseline.

## Review checklist for the earlier fitted-map version-11 recipe

The fitting budgets, fitted-centre descriptions and h=0.07 below apply to `sampler-comparison` and `sampler-comparison-numba`, not the default reported-study entry.

| Check | Evidence and interpretation |
|---|---|
| Exact target and observables | `configuration.json`, [target definitions](TARGETS.md), and `dcs_bench/comparison_targets.py`. All methods share the same physical target, all-ones pilot start, fixed radial thresholds and analytic references. The target is normal-CDF-skewed t(3), distinct from the SCS paper's Student-t-CDF example. |
| Source and execution provenance | `manifest.json`: protocol, source hashes, dependencies, CPU policy, worker allocation and complete/failed status. Do not pool results across incompatible source/runtime versions. |
| Fitted and effective maps | `maps.json`, `method_maps.json`, `effective_maps.csv`. RWM/BW/reflected BW share the ball map; billiards retain its centre/R/β with zero effective deformation after Möbius. There is no constrained refit. Both map families are fitted in new comparisons. |
| Author SCS implementation | Pinned files in `third_party/scs_paper`, their checksums, and `dcs_bench/scs_paper.py`. The author projection, VI routine and production transition are unchanged; adaptation/recording policies are disclosed separately. |
| Fitting resources | SCS: seed 0, fixed cloud, Adam 0.01, ±1,000 physical-coordinate clipping inside VI, no gradient clipping. DCS: seed 42, fixed cloud, Adam 0.01, gradient-norm cap 10, no coordinate clipping. Full evaluation budgets match: 1,000×2,000 per family. Each reuses its own 2,000-point cloud from its reference distribution. Equal evaluation counts do not imply equal fitting time or identical optimizers. Production target densities are unclipped. |
| Warmup budgets and step criteria | Same number of discarded transitions per chain, not necessarily equal seconds or optimal settings. BW, reflected BW and billiard share fixed increment radius h=0.07; RWM uses h/√(d+2) per coordinate. All four use their own kernel during discarded warmup without separate step optimization. The single SCS baseline uses fixed h=0.1. Inspect final steps, selected controls, guard hits, candidate scores and warmup time. |
| Frozen production and burn-in | Every scale freezes before production. The last pilot state starts production, then the first third of actual generated production iterations is discarded. Repeated rejection states remain in estimates and ESS. |
| Timing boundary | Production includes sampling, projection/statistic work and scalar ESS capture/flush. It excludes fits, compilation, pilots, synchronization waits, diagnostic FFTs and report generation. Inspect actual seconds and final-batch overshoot. |
| Implementation comparison | JAX uses common full-state scans and recording. Numba uses fused DCS loops against author JAX SCS; its rate includes framework/output differences and is not a pure geometry comparison. |
| Estimators and uncertainty | Per-chain means are compared with the same analytic truth; MSE gives each chain equal weight. Report all cases and functions. Fixed-map uncertainty excludes refitting; t(3) norm errors can make empirical MSE uncertainty unstable. |
| ESS | [Functional ESS](ESS.md) uses unthinned post-burn-in norm and tail-indicator histories. ESS/s sums per-chain ESS over full production seconds. Unavailable diagnostics and valid counts remain visible. It is not a convergence guarantee or coordinate ESS. |
| Seed independence | Sampling batch b starts at 901001+1,000,000b, with chain offsets; map-fitting seeds remain fixed. Timed stopping prevents bitwise count reproducibility. New batches test sampling variability conditional on the same maps. |

## Development selection and held-out assessment

SCS's default h=0.1 is its authors' skew-t setting. The version-9 development study also tested acceptance adaptation toward 0.234 under the same target, maps, seeds and discarded transition budget. The disclosed rule retained adaptation only if its observed MSE improved both radial functions in the full-author-VI 100D pair and the no-VI pair; otherwise it selected fixed h=0.1 globally. Adaptation failed that rule. All reduced-fit and full-fit outcomes are retained in the [study report](AUTHOR_SCS_CALIBRATION.md).

This is a development choice based on 100D evidence, not a proof of optimal tuning. The unchanged author VI routine at full settings uses its original 1,000-update/2,000-draw budget; the earlier fitted-map recipe's quick and smaller presets deliberately abbreviate that budget. The reported study instead loads its full-budget maps at every run size. Using the same author routine and h=0.1 in 500D is an extension, not a reproduction of an author 500D experiment or a separately optimized 500D setting.

Version 10 changed the full DCS fitting budget from 1,500 to 1,000 updates and introduced fixed clouds for both families. Version 11 preserves those maps and equal fitting resources, but changes all four DCS variants to the shared fixed covariance-matched move above. The earlier fitted-map entry points use the preset h=0.07 automatically, without manual step choices. Historical studies retain their recorded settings. No optimality or guaranteed ranking is asserted for that preset. Fresh-cloud VI for both families remains a separate experimental study, not a public default.

Freeze the version-11 protocol before evaluating a fresh, previously uninspected full-paper sampling batch. Rerun all five methods together and report the new batch separately from the version-9 and later development studies. A changed protocol after inspecting results needs a new independent assessment and disclosure. This guide does not assert that a new full batch has completed.

The earlier five-method compatibility recipes contain only author SCS VI and fixed h=0.1. Historical
step-policy sensitivity results remain in the archived study; they are not
active baseline choices. Retaining the original baseline does not establish
equally optimal tuning or isolate the transition geometry.

The reported zero-centre study records a later development stage. It changes the
production centre after VI and uses pilot-selected radius/step values
and post-Möbius coordinates for every DCS variant. It is now the default launcher
entry; the earlier fitted-map recipe retains fitted centres, uses h=0.07,
and uses post-Möbius coordinates only for billiards. A comparison of a developed DCS variant against original SCS is a
valid declared-baseline experiment, but its additional target-specific
development resources are not matched preparation for the two methods.
Report that scope, retain the development record and assess frozen choices
on fresh sampling seeds; do not describe it as differing only in the sampler.

## Other six experiments

All VI-enabled SCS fits call the pinned author optimizer. Production
implementations and proposal calibration still differ across experiment
families:

- Logistic and the ν=0.5 comparison use author SCS VI, the independently
  checked Numba SCS production implementation, and acceptance calibration
  toward 0.234. Their SCS VI budget is recorded separately from DCS: full
  1,000 updates on 2,000 fixed draws, reduced to 400 × 500 or 20 × 128 in
  Smaller or Quick presets. DCS matches those counts and also reuses a fixed
  cloud in these paired experiments. These production kernels are not represented as
  the unchanged author JAX transition. Author VI retains its unit-scale
  initialization and unconstrained centre/observer fit even on symmetric
  targets; DCS-specific radial-median and symmetry choices do not alter it.
- Generic DCS/random-walk/SCS kernels use Numba; NUTS uses BlackJAX or NumPyro.
  Different production block sizes and their actual time overshoot are recorded.
  These are implementation comparisons, not equal-framework measurements.
- Eight Schools uses centered coordinates for both methods. This isolates that
  parameterization; it does not compare against an optimized noncentered NUTS
  implementation. NUTS acceptance/depth settings are explicit, and divergences
  and depth-limit hits should accompany performance conclusions.
- Eight Schools/GARCH run NUTS for a fixed iteration budget, then give DCS the
  measured mean NUTS production time. The Quick preset adds a minimum DCS time
  for execution checking; Smaller and Full settings do not apply that floor.
  Quick runs are not performance comparisons.
- Generic method order is fixed. Independent repeats and a controlled system
  load are needed for timing conclusions; the synchronized comparison instead
  rotates method order across waves. With few waves, rotation does not fully
  balance every method's position.

Within each generic experiment, the physical target, estimands, reference,
production burn-in fraction and thinning policy are shared. ESS is computed
from the actual retained series and is not multiplied by thinning. Posterior
reference estimates are independent but approximate, with Monte Carlo error
reported. See [reproducibility](REPRODUCIBILITY.md), [HnR/NUTS](HNR_NUTS.md)
and the resolved configurations for details.

## Reproducing a claim or reporting a discrepancy

Record the commit, command or launcher settings, preset, target, fitting budgets, shared DCS step rule and move size, SCS step rule, seed batch and worker count. Include the complete manifest, resolved configuration and relevant summary/CSV rows. State whether a comparison concerns generated samples, MSE, functional ESS, ESS/s or end-to-end cost; these need not rank methods identically.

For a proposed discrepancy, preserve both results and use fresh folders. Avoid selecting seeds, functions or tuning settings after observing which favors a method. Longer chains and independent batches can test repeatability; increasing replicate count alone does not fix biased or insufficiently mixed chains. See [contributing](../CONTRIBUTING.md) for validation and reporting guidance.
