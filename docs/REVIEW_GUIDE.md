# Reviewer guide

This repository supports inspection and independent reruns of the corrected experiments and subsequent sampler extensions. Start with a small execution check, then inspect the recorded protocol before interpreting comparisons. The source commit and run manifest, rather than a figure filename, identify a result.

## Scope and evidence

The current JAX and optimized-Numba comparisons use version 9. They compare DCS-RWM, DCS-BW, SCS, reflected BW and DCS-Billiard on 100D/500D normal-CDF-skewed t(3) targets, with VI on/off. Billiards and reflected BW are subsequent extensions. The current suite measures radial-tail probability and mean norm; it cannot establish coordinate or general sampler superiority.

The original manuscript figures are [archival assets](../figures/archival/README.md) without complete run provenance. Current runs incorporate implementation and calibration corrections and do not validate those figures retrospectively. [Earlier coordinate measurements](COMPARISON_RESULTS.md), including cases favoring SCS, and [version-8 calibration results](CALIBRATION_RESULTS.md) remain available. The [version-9 SCS study](AUTHOR_SCS_CALIBRATION.md) reports both fixed and acceptance-tuned policies rather than only a favorable setting.

## Minimal independent check

Install Python 3.12 and the pinned CPU dependencies using the [README](../README.md). The desktop launcher provides the same experiment choices. From an activated environment:

```sh
python run_experiment.py sampler-comparison --preset quick --workers 1 --dry-run
python run_experiment.py sampler-comparison --preset quick --workers 1 --output-dir results/reviewer-check
python run_experiment.py --plot-only results/reviewer-check
```

Confirm a complete manifest and open `paper_values.html`. A quick result checks execution only. The smaller preset uses 20 chains × 2 seconds; full settings use 50 chains × 20 seconds. Use a fresh output directory each time. To evaluate a map ablation, change `--vi on` to `--vi off`; to examine the extension, select `--target skew-t-500d`.

## Review checklist

| Check | Evidence and interpretation |
|---|---|
| Exact target and observables | `configuration.json`, [target definitions](TARGETS.md), and `dcs_bench/comparison_targets.py`. All methods share the same physical target, all-ones pilot start, fixed radial thresholds and analytic references. The target is normal-CDF-skewed t(3), distinct from the SCS paper's Student-t-CDF example. |
| Source and execution provenance | `manifest.json`: protocol, source hashes, dependencies, CPU policy, worker allocation and complete/failed status. Do not pool results across incompatible source/runtime versions. |
| Fitted and effective maps | `maps.json`, `method_maps.json`, `effective_maps.csv`. RWM/BW/reflected BW share the ball map; billiards retain its centre/R/β with zero effective deformation after Möbius. There is no constrained refit. VI-off maps are untuned for both families. |
| Author SCS implementation | Pinned files in `third_party/scs_paper`, their checksums, and `dcs_bench/scs_paper.py`. The author projection, VI routine and production transition are unchanged; adaptation/recording policies are disclosed separately. |
| Fitting resources | SCS: seed 0, fixed cloud, Adam 0.01, ±1,000 physical-coordinate clipping inside VI, no gradient clipping. DCS: seed 42, fresh batches, Adam 0.01, gradient-norm cap 10, no coordinate clipping. Full budgets differ: 1,000×2,000 fixed versus 1,500×2,000 fresh. Production target densities are unclipped. |
| Pilot budgets and criteria | Same number of discarded transitions per chain, not necessarily equal pilot seconds or optimal tuning. DCS-RWM/BW target acceptance 0.234; reflected BW uses ball movement; billiards use their own radial-movement pilot. SCS defaults to fixed h=0.1. Inspect final steps, guard hits, candidate scores and pilot time. |
| Frozen production and burn-in | Every scale freezes before production. The last pilot state starts production, then the first third of actual generated production iterations is discarded. Repeated rejection states remain in estimates and ESS. |
| Timing boundary | Production includes sampling, projection/statistic work and scalar ESS capture/flush. It excludes fits, compilation, pilots, synchronization waits, diagnostic FFTs and report generation. Inspect actual seconds and final-batch overshoot. |
| Implementation comparison | JAX uses common full-state scans and recording. Numba uses fused DCS loops against author JAX SCS; its rate includes framework/output differences and is not a pure geometry comparison. |
| Estimators and uncertainty | Per-chain means are compared with the same analytic truth; MSE gives each chain equal weight. Report all cases and functions. Fixed-map uncertainty excludes refitting; t(3) norm errors can make empirical MSE uncertainty unstable. |
| ESS | [Functional ESS](ESS.md) uses unthinned post-burn-in norm and tail-indicator histories. ESS/s sums per-chain ESS over full production seconds. Unavailable diagnostics and valid counts remain visible. It is not a convergence guarantee or coordinate ESS. |
| Seed independence | Sampling batch b starts at 901001+1,000,000b, with chain offsets; map-fitting seeds remain fixed. Timed stopping prevents bitwise count reproducibility. New batches test sampling variability conditional on the same maps. |

## Development selection and held-out assessment

SCS's default h=0.1 is its authors' skew-t setting. The version-9 development study also tested acceptance adaptation toward 0.234 under the same target, maps, seeds and discarded transition budget. The disclosed rule retained adaptation only if its observed MSE improved both radial functions in the full-author-VI 100D pair and the no-VI pair; otherwise it selected fixed h=0.1 globally. Adaptation failed that rule. All reduced-fit and full-fit outcomes are retained in the [study report](AUTHOR_SCS_CALIBRATION.md).

This is a development choice based on 100D evidence, not a proof of optimal tuning. The unchanged author VI routine at full settings uses its original 1,000-update/2,000-draw budget; quick and smaller presets deliberately abbreviate that budget. Using the same author routine and h=0.1 in 500D is an extension, not a reproduction of an author 500D experiment or a separately optimized 500D setting.

The selected protocol is frozen before the planned fresh full-paper **sampling batch 8**. That batch is reserved for assessing the selected procedure, not for further choosing its default from the observed ranking. Development batch 7 and the full batch must be reported separately. A changed protocol after inspecting the full batch needs a new independent assessment and disclosure. This guide does not assert that the planned full batch has already completed.

To inspect SCS sensitivity conditional on a primary's saved maps, use:

```sh
python scripts/reproduce_scs_step_control.py results/reviewer-check --output-dir results/reviewer-step-control
```

The helper runs the opposite SCS step policy and validates saved source/configuration/map/allocation provenance. It performs no map refit. Original fit time and zero new fitting time are recorded separately; the latter is not an independent end-to-end fitting cost. Pilot RNG consumption differs between fixed and adaptive policies, so identical seed labels do not imply identical warmup proposals.

## Other six experiments

The version-9 author-SCS claims above apply to the two selectable 100D/500D
comparison modes. The other experiment families have their own recorded
implementation and calibration policies:

- Logistic and the ν=0.5 comparison use the independently checked Numba SCS
  implementation, generic map fitting and acceptance calibration toward 0.234.
  They do not call the pinned author VI/transition used in version 9.
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
  load are needed for timing conclusions; the five-method comparison instead
  rotates method order across waves. With few waves, rotation does not fully
  balance every method's position.

Within each generic experiment, the physical target, estimands, reference,
production burn-in fraction and thinning policy are shared. ESS is computed
from the actual retained series and is not multiplied by thinning. Posterior
reference estimates are independent but approximate, with Monte Carlo error
reported. See [reproducibility](REPRODUCIBILITY.md), [HnR/NUTS](HNR_NUTS.md)
and the resolved configurations for details.

## Reproducing a claim or reporting a discrepancy

Record the commit, command or launcher settings, preset, target, VI choice, SCS step rule, seed batch and worker count. Include the complete manifest, resolved configuration and relevant summary/CSV rows. State whether a comparison concerns generated samples, MSE, functional ESS, ESS/s or end-to-end cost; these need not rank methods identically.

For a proposed discrepancy, preserve both results and use fresh folders. Avoid selecting seeds, functions or tuning settings after observing which favors a method. Longer chains and independent batches can test repeatability; increasing replicate count alone does not fix biased or insufficiently mixed chains. See [contributing](../CONTRIBUTING.md) for validation and reporting guidance.
