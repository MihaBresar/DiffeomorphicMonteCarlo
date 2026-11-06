# Current sampler comparisons

Both launcher comparisons use the same five methods: **DCS-RWM, DCS-BW,
author SCS, DCS-BW-reflect and DCS-Billiard**. The two rotation mixtures are
removed from new comparison runs. New comparisons always fit both map families by VI; the VI-off option is removed. Previously saved reports remain readable.

| Identifier | Production implementations |
|---|---|
| `sampler-comparison` | All five methods in JAX |
| `sampler-comparison-numba` | DCS-RWM, DCS-BW, DCS-BW-reflect and DCS-Billiard in optimized Numba; unchanged author SCS in JAX |

Both modes use protocol version 11. RWM means DCS-RWM in transformed ball
coordinates. Its isotropic Gaussian proposal rejects exterior endpoints;
ordinary BW uses a uniform-ball increment and also rejects exterior endpoints.
Reflected BW folds exterior endpoints along the proposal chord before its
Metropolis test, using the shared ball map.
Billiards use the [invariant specular construction](BILLIARD.md).

## Targets, measurements and references

| Selector | Target and map | Radial threshold | Tail reference | Mean-norm reference |
|---|---|---:|---:|---:|
| `skew-t-100d` | Normal-CDF-skewed t, d=100, ν=3, α=(100,−100,0,…), DCS β=2 | 50.00 | 0.0107346153 | 13.7852602890 |
| `skew-t-500d` | Same skew and ν, d=500, DCS β=2 | 50√5 ≈ 111.80 | 0.0106799328 | 30.8864890662 |

New runs report only **radial-tail probability and mean norm**, their MSEs,
iteration counts and production time, plus functional ESS and ESS per second.
The X₁/X₂/X₃ means and coordinate tails are no longer computed or plotted.
The target and common physical all-ones starting point are unchanged.

Both reference values are analytic: skewing this t target by sign leaves its
radial law unchanged, and ‖X‖²/d follows F(d,ν). No coordinate-reference
simulation is performed. Thresholds are fixed before sampling, independent
of method, VI and pilot outcomes. CSV/JSON retain full precision.

## What is matched

- Same physical target, two radial functionals, analytic references and start.
- Same per-chain production-time budget, number of chains, separate discarded
  warmup-transition budget, 300-transition clock batches and first-third burn-in.
- Repeated states after rejection contribute to estimates and ESS.
- Same ordered CPU allocation and simultaneous chains per wave, including
  partial waves. Method order rotates across waves. Runtime helper threads
  may exist; all JAX compute pools use the common `PJRT_NPROC=1` policy.
- Each DCS method has the same JAX pilot, final pilot state and frozen step in
  either execution mode. Numba production has an independent PCG64 stream;
  cross-backend trajectories are not bitwise copies.

Every new comparison fits ball and sphere families once each. DCS β stays 2;
SCS latitude stays 1.1. Billiards retain the joint
ball fit's centre/R/β but run in post-Möbius coordinates, with effective δ=0.
This is not a new VI fit with δ constrained to zero. RWM/BW/reflected BW retain the original
fitted ball map. `method_maps.json`, `effective_maps.csv`, summary and manifest
record the distinction. Base-normalizing constants differ, so reported VI
losses are not comparable normalized KL values across families.

Versions 10 and 11 use the **pinned authors' VI routine for SCS** and match the
number of fitting objective/gradient evaluations across both families. DCS
uses a fixed reference cloud and a full budget of 1,000 updates. SCS retains
its original fixed-cloud routine, seed 0, Adam 0.01, and
physical-coordinate clipping at ±1,000 inside its VI objective, including
the Jacobian argument. It does not clip optimizer gradients. DCS reuses its
own fixed cloud, with seed 42, Adam 0.01, a gradient-norm cap of 10 and
no coordinate or CDF clipping. Both start at zero centre/deformation and
unit scale. Production target densities are unclipped for every method.

Full settings give each family 1,000 updates with 2,000 objective/gradient
evaluation points per update: 2,000,000 point evaluations per fit. SCS reuses
one fixed cloud of 2,000 draws; DCS independently draws and reuses its own
2,000-point cloud. Their reference distributions and actual draws differ.
Matching cloud sizes and evaluation counts does not make the fitting
parameterizations, clipping rules or preparation times identical. Fitting costs and parameters are recorded separately.
The comparison therefore measures fitted sampling procedures and cannot
isolate a proposal-geometry effect from the fitted transformations. Restricting
the public suite to fitted maps does not establish performance without VI.
The unchanged author VI and transition are applied to this repository's
normal-CDF-skewed t target, not the authors' Student-t-CDF-skewed example.
See [the author baseline and sensitivity study](AUTHOR_SCS_CALIBRATION.md).

Version 11 does not change these fitting settings or maps. Fresh-cloud VI for
both families is a separate experimental study, not the public default.
Zero-centre and post-VI radius/step studies are also separate: current public
runs retain the fitted physical centre and the stated default radius/step
policy. Their outcomes must not be labelled as results of these defaults.
Such extra, target-informed DCS development is not matched end-to-end tuning
against an unchanged author SCS baseline.

## Shared DCS move and separate warmup

All four DCS variants share one fixed move **h=0.07**. BW, reflected BW and
billiard draw the same uniform-ball increment distribution of radius h.
RWM uses Gaussian coordinate standard deviation **σ=h/√(d+2)**, matching the
increment covariance h²I/(d+2). This does not make the proposal distributions,
physical endpoints or acceptance probabilities identical. Billiards retain
their effective post-Möbius coordinates; this is part of their transition,
not a separate map fit or step optimization.

Each variant runs its actual production kernel during separate discarded
warmup, keeping the shared step fixed. Reflected BW and billiard receive no
additional step-selection pilot under this default. Numba production receives
the identical JAX warmup state and frozen step. Warmup time is recorded
separately from sampling time.

The launcher and public command-line entry points use this preset automatically.
There is no manual DCS h or calibration-policy choice. Internal configuration
helpers retain explicit overrides for historical studies; those are not the
current launcher protocol. The preset is not a demonstrated optimum.

The only SCS baseline in new five-method comparisons uses the authors' skew-t
**fixed h=0.1** throughout discarded warmup and production, with unchanged
author VI and transition code. It is not an adaptively selected step.
New acceptance-tuned SCS requests are rejected. The hidden
`--scs-step-rule fixed` alias remains for compatible scripts; it selects the
same single baseline.

Historical SCS sensitivity results and their original source revision remain
in the [development study](AUTHOR_SCS_CALIBRATION.md). They are archival
evidence, not additional active baselines. The earlier study's rule retained
acceptance adaptation only if it improved both radial functions in the
full-author-VI and no-VI checks; it failed that rule. Keeping this author
baseline does not establish that h=0.1 is optimal for the present target or
for 500D.

Under the historical adaptive control only, reflected BW tests rotating
h/2, h and 2h trials by squared ball movement, and billiard tests its own
kernel using bounded squared changes of physical log(1+norm). Rejections
score zero. Their saved histories remain readable; these controls are not offered by
current public entry points or represented as preset fixed-step warmup.

All methods receive the preset's same total discarded warmup-transition budget.
Under the default, all five methods use fixed-step equilibration. Adaptive
controls retain broad numerical guards and freeze before production. Equal
counts do not establish equal warmup time or optimal settings.
Gaussian RWM's recorded scale is a coordinate standard deviation;
BW/reflected-BW/billiard scale is the initial proposal-ball radius.
SCS uses the authors' fitting routine and fixed h=0.1; the active comparison
does not offer a separately calibrated or recentered SCS variant.

Equal pilot transition counts do not establish equally optimal tuning, nor
equal preparation time. Numerical step sizes have different geometric
meanings across kernels, and optional historical controls use different selection
criteria. Preserve historical sensitivity outcomes separately from the
active original-baseline comparison. Freeze the main protocol before evaluating a fresh full-paper
seed batch; the smaller study is development evidence, not an independent
validation of a choice informed by its results.

## ESS and timing

[Functional ESS](ESS.md) is computed separately for the radial-tail indicator
and norm using the unthinned post-burn-in scalar history. It describes the
actual functions used in the mean estimators, not rank-normalized coordinate
bulk/tail ESS. Thinning saved traces does not alter comparison ESS.

Scalar diagnostic capture occurs inside every production clock. Histories
are temporary and are discarded after diagnostics. ESS computation begins
after all chains in a wave stop sampling; only one chain's FFT workspace is
active at a time to bound memory. Computation and waiting are outside the
production clock and diagnostic time is recorded separately. Constant,
nonfinite or insufficient traces produce unavailable ESS with a reason.
A method total, mean and ESS/s are unavailable if any chain is unavailable;
valid-chain counts are shown, and incomplete subsets are not advertised as
complete totals. ESS/s divides total ESS by total production seconds,
including discarded sampling burn-in.

JAX uses full-state scans, physical projection and the same scalar recording
for every method. Native DCS caches current density/radius and fuses scalar
accumulation; SCS retains author JAX scans. Thus the Numba mode compares these
implementations, including their different framework/output costs, rather
than isolating geometric transition cost. The author SCS source is unchanged.

## Budgets and outputs

| Preset | Chains/method | Seconds/chain | Separate pilot | DCS VI updates × fixed cloud size | SCS VI updates × fixed cloud size |
|---|---:|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 20 × 128 | 20 × 128 |
| Smaller reproduction | 20 | 2 | 1,000 | 400 × 500 | 400 × 500 |
| Full paper settings | 50 | 20 | 1,500 | 1,000 × 2,000 | 1,000 × 2,000 |

The SCS cloud is reused at every update. Thus full settings evaluate 2,000
reference states over 1,000 updates, not just 2,000 target evaluations in
total. Smaller presets reduce the authors' fitting budget and do not claim
to reproduce the paper's full VI fit. Version 9 gave DCS 1,500 full fitting
updates and fresh draws at every update. Version 10 introduced 1,000 full
updates and a fixed reference cloud at every preset; version 11 retains this
fitting policy. Version 10's method-specific DCS calibration—acceptance-target
0.234 for ordinary RWM/BW, ball movement for reflected BW and radial movement
for billiards—is replaced by the v11 shared fixed-move default. Rerun all five methods together in
fresh folders for new v11 sampling comparisons at every preset. Old reports
retain their original fitted maps and step policies.

Compilation, fitting, pilots, synchronization,
ESS computation and final file/report generation are outside production time.
The first third of the actual generated iterations is discarded exactly.
All states in each completed 300-step block count; elapsed time and overshoot
are recorded. Saved trace thinning changes neither radial estimates nor ESS.

Reports retain `VsSCS`, `scs_100d_estimates`, `scs_100d_runtime` and
`scs_100d_bias_variance`, and add `scs_100d_ess`. Numbers use two displayed
decimals, exact integer counts, and full-precision CSV/JSON exports.
`functional_ess.csv` records ESS totals/means/medians, ESS/s, valid counts,
strides and unavailable reasons. The per-chain records retain diagnostics and
costs, without saving the large unthinned temporary scalar histories.

Historical protocols retain their original method lists and measurements;
replotting cannot reconstruct an unsaved sampler or unthinned ESS. Run all
methods together to compare new counts, MSE and ESS. ESS alone is not evidence
of convergence or absence of bias. MSE variability is conditional on fixed
fitted maps; the t(3) norm has no fourth moment, so rare errors can make
empirical MSE uncertainty unstable. Use longer runs and fresh seed batches
before drawing publication conclusions.

The active measurements support claims about radial-tail probability and mean
norm. They do not establish general sampler or marginal superiority.

See [click-to-run instructions](CLICK_TO_RUN.md), [rerun commands](RERUN.md)
and [the billiard proposal](BILLIARD.md).
