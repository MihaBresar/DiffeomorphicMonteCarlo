# Current sampler comparisons

Both launcher comparisons use the same five methods: **DCS-RWM, DCS-BW,
author SCS, DCS-BW-reflect and DCS-Billiard**. The two rotation mixtures are
removed from new comparison runs. Their saved reports remain readable.

| Identifier | Production implementations |
|---|---|
| `sampler-comparison` | All five methods in JAX |
| `sampler-comparison-numba` | DCS-RWM, DCS-BW, DCS-BW-reflect and DCS-Billiard in optimized Numba; unchanged author SCS in JAX |

Both modes use protocol version 9. RWM means DCS-RWM in transformed ball
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
- Same per-chain production-time budget, number of chains, total pilot
  transition budget, 300-transition clock batches and first-third burn-in.
- Repeated states after rejection contribute to estimates and ESS.
- Same ordered CPU allocation and simultaneous chains per wave, including
  partial waves. Method order rotates across waves. Runtime helper threads
  may exist; all JAX compute pools use the common `PJRT_NPROC=1` policy.
- Each DCS method has the same JAX pilot, final pilot state and frozen step in
  either execution mode. Numba production has an independent PCG64 stream;
  cross-backend trajectories are not bitwise copies.

VI **off** uses R=1, centre zero and deformation zero for both map families.
DCS β stays 2; SCS latitude stays 1.1. DCS steps are still pretuned; SCS
defaults to fixed h=0.1, with acceptance adaptation available separately.
VI **on** fits ball and sphere families once each. Billiards retain the joint
ball fit's centre/R/β but run in post-Möbius coordinates, with effective δ=0.
This is not a new VI fit with δ constrained to zero. RWM/BW/reflected BW retain the original
fitted ball map. `method_maps.json`, `effective_maps.csv`, summary and manifest
record the distinction. Base-normalizing constants differ, so reported VI
losses are not comparable normalized KL values across families.

Version 9 uses the **pinned authors' VI routine for SCS**, while leaving DCS
VI unchanged. SCS uses one fixed reference cloud, seed 0, Adam 0.01, and
physical-coordinate clipping at ±1,000 inside its VI objective, including
the Jacobian argument. It does not clip optimizer gradients. DCS uses fresh
base draws at each update, seed 42, Adam 0.01, a gradient-norm cap of 10 and
no coordinate or CDF clipping. Both start at zero centre/deformation and
unit scale. Production target densities are unclipped for every method.

These are different fitting recipes, not identical optimizers or equal
unique-draw budgets. Full settings use the authors' 1,000 updates on 2,000
fixed reference draws for SCS; DCS retains 1,500 updates with 2,000 fresh
draws per update. Fitting costs and parameters are recorded separately.
The unchanged author VI and transition are applied to this repository's
normal-CDF-skewed t target, not the authors' Student-t-CDF-skewed example.
See [the author baseline and sensitivity study](AUTHOR_SCS_CALIBRATION.md).

## Separate proposal calibration

DCS-Billiard runs its **actual production kernel** during its separate pilot
and selects by a radial-movement score. For consecutive accepted-or-repeated
physical states, set Δ = log(1 + ‖X_next‖) − log(1 + ‖X_current‖); the score is
mean Δ²/(1 + Δ²). Rejections score zero. This bounded proxy does not use target
truth, production MSE or ESS and is not a guarantee of optimal mixing.

The prespecified search tests five coarse scales h × (0.25, 1, 4, 16, 64),
then three refinement scales at (0.5, 1, 2) times the coarse winner. The existing
initial h remains 0.1. The complete warmup budget is allocated across these
eight windows: 125 transitions each for Smaller reproduction, and 187 or 188
for Full paper settings. Candidate execution order is randomized by the pilot
seed within each stage; chronological order and counts are saved. Numerical
scale guards still apply. This finite search is not a new geometric step cap.

All pilot observations are discarded. The last pilot state and selected scale
initialize production; no production adaptation is performed. Billiards use
their effective post-Möbius map. The
same JAX pilot is used for either JAX or Numba production. Equal transition
budgets need not take equal time; actual pilot seconds are reported separately
and excluded from production time. `radial_tuning.csv` records candidate
steps, scores, acceptance and selections, separately from reflected-BW pilots.

Ordinary DCS-RWM and DCS-BW use observed-acceptance dual averaging toward
0.234. SCS defaults to the authors' skew-t **fixed h=0.1**, including during
its discarded pilot. It is not an adaptively selected step. Acceptance
adaptation toward 0.234 remains an explicit alternative; it uses the
unchanged author transition and recorded sphere map. That adaptation is a
benchmark modification, rather than the authors' fixed-step skew-t recipe.

`--scs-step-rule acceptance` or `--scs-step-rule fixed` explicitly chooses
the SCS policy for a new comparison. The [sensitivity runner](AUTHOR_SCS_CALIBRATION.md#reproduction)
reuses a completed primary's exact maps and budgets and automatically runs
the opposite policy. The smaller study's pre-result rule retains acceptance
as the global default only if its MSE improves both radial functions in a
full-author-VI on check and the no-VI check; otherwise it falls back globally
to fixed h=0.1. Acceptance failed that rule, so the default is fixed h=0.1.
Both checks use the small production budget. All results are reported, and
a fresh full-paper batch evaluates that development choice.

Reflected BW retains its ball-movement pilot: up to eight rounds of
rotating h/2, h and 2h trials, selecting by observed squared movement in the
shared ball map with rejected moves scoring zero. It has no geometric step
ceiling of 2. Its trials remain in `reflected_tuning.csv`.

All methods receive the preset's same total pilot transition budget and freeze
their scales before sampling. All rules retain broad numerical guards; a guard
hit or equal budget does not establish optimal tuning. Gaussian RWM's scale is
a coordinate standard deviation; BW/billiard scale is a proposal-ball radius.
SCS uses the authors' fitting routine and defaults to their fixed h=0.1,
with acceptance calibration available for sensitivity checks.

Equal pilot transition counts do not establish equally optimal tuning, nor
equal preparation time. Numerical step sizes have different geometric
meanings across kernels, and the active methods use different selection
criteria. Report the SCS acceptance and fixed-step sensitivity outcomes
together. Freeze the main protocol before evaluating a fresh full-paper
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

| Preset | Chains/method | Seconds/chain | Separate pilot | DCS VI updates × fresh draws/update | SCS VI updates × fixed cloud size |
|---|---:|---:|---:|---:|---:|
| Quick check | 2 | 0.05 | 100 | 20 × 128 | 20 × 128 |
| Smaller reproduction | 20 | 2 | 1,000 | 400 × 500 | 400 × 500 |
| Full paper settings | 50 | 20 | 1,500 | 1,500 × 2,000 | 1,000 × 2,000 |

The SCS cloud is reused at every update. Thus full settings evaluate 2,000
reference states over 1,000 updates, not just 2,000 target evaluations in
total. Smaller presets reduce the authors' fitting budget and do not claim
to reproduce the paper's full VI fit.

VI off uses no fitting updates. Compilation, fitting, pilots, synchronization,
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
