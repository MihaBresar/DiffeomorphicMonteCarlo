# Relationship to the manuscript experiments

This repository contains corrected implementations and later sampler
extensions. It is not a claim that running the current default configuration
reproduces every numerical value in an earlier manuscript. The current
five-method comparison includes reflected BW and billiards, which are distinct
from the original ordinary BW proposal.

The active version-11 settings, including shared BW/reflected-BW/billiard increment radius h=0.07,
Gaussian RWM coordinate standard deviation h/√(d+2), SCS author VI, fixed
SCS h=0.1 and DCS sensitivity controls, are specified in [COMPARISONS.md](COMPARISONS.md).
The `paper` preset means the recorded full benchmark budget. Figures and
numerical claims for a publication must identify the exact source revision,
configuration and saved results used to produce them.

## Corrections represented in the repository

- Synthetic target densities and analytic references are centralized. For
  the 200D ν=3 radial example, the tail probability at norm 50 is
  0.0291914455 and the mean norm is 19.5196857147. The benchmark skew uses
  a normal CDF; it differs from the Student-t-CDF skew in the SCS paper.
- The 200D random-walk example uses its stated dimension. The super-heavy-tail
  example is symmetric with ν=0.5. Its mean norm does not exist, so
  comparisons use functions with defined reference values. DCS retains its
  symmetry-constrained fit and SCS uses the author fitting routine.
- Eight Schools uses log τ with the corresponding Jacobian for both samplers.
  GARCH includes μ, a fixed initial variance, the stated physical prior and
  the pinned PosteriorDB model/data definition. Posterior DCS uses the
  documented identity affine map and β=1.
- Adaptation and fitting are separate from production. Every new experiment
  discards the first third of its production iterations. The active SCS
  baseline and all four DCS comparison variants have fixed steps during
  discarded warmup and production. The DCS variants share one move scale,
  with the stated Gaussian normalization. Other proposal-calibration policies are recorded
  per method. Fitting recipes and initialization differ across experiment
  families and must be read from their resolved configuration.
- All VI-enabled SCS fits call the pinned authors' optimizer, including
  logistic and the ν=0.5 experiment. SCS uses its own fixed-cloud budget,
  seed 0 and unit initialization. Version 10 uses fixed clouds for both
  families in paired DCS/SCS experiments and matches their fitting budgets;
  DCS parameterization and clipping remain distinct. Generic SCS
  production remains the checked Numba implementation with acceptance-target
  0.234 step tuning, while the 100D/500D comparison retains author JAX
  production and the fixed-h=0.1 default.
- Version 11 retains version 10's fitted maps and changes all four DCS variants
  from method-specific calibration to a shared fixed move h=0.07,
  selected automatically by public entry points. Gaussian RWM scales by
  1/√(d+2) to match BW increment covariance. This requires new sampling and
  does not establish the preset as optimal. Fresh-cloud VI studies are
  separate experiments and are not public fitting defaults.
- NUTS comparison settings include target acceptance 0.8 and maximum depth 10.
  Reference chains have separate settings. Timing distinguishes preparation,
  production, diagnostics and total elapsed execution. Parallel timing is
  conditional on the recorded worker allocation and hardware load.
- Posterior reference estimates have Monte Carlo uncertainty; they are not
  exact truth. The posterior table's RMSE is the square root of the mean
  squared chain-level errors over chains and coordinates, rather than the
  error of one pooled estimate.
- Logistic Q–Q plots compare against independent reference quantiles. Their
  empirical 5–95% ranges describe variability across chains; they are not
  confidence bands proving a central limit theorem.

Norm-based empirical MSE under ν=3 can be unstable because the norm has no
fourth moment. Current radial results do not establish coordinate-marginal
superiority or universal sampler rankings. Empirical examples outside sufficient
theorem assumptions are valid experiments, but those theorems do not establish
convergence for them.

[TARGETS.md](TARGETS.md), [RERUN.md](RERUN.md), [ESS.md](ESS.md) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) define the target, output and timing
conventions used to interpret a new run. Existing paper figures require
separate reconciliation with their generating source and configuration.
