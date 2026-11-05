# Target densities and coordinate maps

`dcs_bench/targets.py` defines the same mathematical log density for NumPy and JAX. Arbitrary numerical epsilons, clipped CDF arguments, and fixed interior-ball cutoffs are absent from production target evaluations. Out-of-domain or nonrepresentable proposals are rejected; floating-point arithmetic still has a finite range.

## Normal-CDF-skewed multivariate Student t

The normalized density is `2 t_d,ν(x) Φ(αᵀx sqrt((ν+d)/(ν+||x||²)))`. This is the manuscript's normal-CDF formula. It should not be silently substituted with a Student-t-CDF skew family.

An exact iid construction first draws Y from the symmetric multivariate t, then returns Y with probability Φ(a(Y)) and -Y otherwise. Since t(-y)=t(y) and a(-y)=-a(y), the output density is `t(y)Φ(a(y)) + t(-y)[1-Φ(a(-y))] = 2t(y)Φ(a(y))`. Sign selection precedes neither the t scale mixture nor evaluation of a(Y).

Even functionals, including radial functions, retain the symmetric t law. Thus `||X||²/d ~ F(d,ν)`. For ν>1, `E||X|| = sqrt(ν) Γ((d+1)/2) Γ((ν-1)/2) / [Γ(d/2) Γ(ν/2)]`. At d=200, ν=3 the radial references are 0.029191445495738842 for radius ≥50 and 19.519685714651676 for the mean norm. For ν=0.5 the mean norm is not defined; the code instead uses `E log10(1+||X||)` evaluated by one-dimensional quadrature, with numerical error reported.

Directional checks use the unit skewness direction (e₁ if α=0): `1{uᵀX>0}` and `tanh(uᵀX/sqrt(ν))`. Their references are independent iid estimates with MCSE, or exact symmetry values 1/2 and 0. Coordinate quantiles use the corrected iid construction. One million iid reference draws are used for paper quantiles; exact radial constants need no Monte Carlo sample.

The [current comparison suite](COMPARISONS.md) offers the original 100D skewed
t(3) and a 500D version with the same skew coefficients. Both use DCS β=2. Their
radial thresholds remain 50 and 50√5. In addition to mean norm and radial tail
probability, reports include MSE, counts, time and functional ESS. The current
suite does not estimate coordinate means or tails. The target densities,
thresholds for the radial quantities and all-ones physical initialization
remain unchanged.

The analytic F-law radial references remain exact. Optional coordinate
reference utilities support diagnostic and saved-report checks. X₁ references
are independent estimates with recorded MCSE, using conditional sign
expectations and fixed known-mean controls. With α=(100,−100,0,…), X₂ has
the same law as −X₁, so its mean reference is negated and its left-tail
reference is copied from X₁, with shared MCSE. X₃ has an ordinary symmetric
t(3) marginal: its reference mean is zero and its tail reference is
`scipy.stats.t.sf(3, 3)`, both analytic. This marginal statement does not
imply independence of coordinates. The coordinate threshold is
fixed before benchmarking. Ordinary third-moment skewness is not defined at
ν=3 and is not an estimand.

The current comparison fits both map families by VI, initializing their maps
at unit scale. Other DCS experiment configurations may use the radial-median
initialization described below; author SCS fitting always retains its own
initialization.

## Logistic regression

The design uses legacy NumPy RandomState seed 42, 50 rows, 20 standard-normal predictors, centered/scaled to sample standard deviation 0.5 (ddof=0). Responses are `1{X[:,0]>0}`. The prior is an independent product of univariate t distributions with scale 2.5 and ν=4 or 6. The likelihood uses stable log-sigmoid evaluation.

This is an empirical example beyond the paper's sufficient radial assumptions. The repository imposes no theorem-assumption gate. Quantile agreement is an empirical accuracy diagnostic, not evidence of a CLT for normalized estimators. Independent reference NUTS uses exactly this density and disjoint seeds.

## Eight Schools

Physical parameter order: θ₁,…,θ₈, μ, τ. The centered hierarchy has μ~N(0,5²), τ~half-Cauchy(5), θⱼ~N(μ,τ²), and yⱼ~N(θⱼ,σⱼ²). Both production samplers operate on `(θ, μ, log τ)`, including the +log τ Jacobian. Reported draws and moments are in physical coordinates. This replaces the previous signed-τ extension with epsilon regularization.

The target implementation also supports `parameterization: noncentered`, using `(z,μ,log τ)`, θ=μ+τz, and the corresponding standard-normal z density. The main configuration remains centered. The non-centered change of variables is numerically tested; adding it to a production comparison requires its own explicit experiment design and rerun.

## GARCH(1,1)

Physical order: `(μ, α₀, α₁, β₁)`. There are **four parameters**, including the mean μ. The pinned Stan model sets `h[0]=sigma1²` and `h[t]=α₀+α₁(y[t-1]-μ)²+β₁ h[t-1]`; all observations follow N(μ,h[t]). Its prior is flat in physical coordinates on α₀>0, α₁>0, β₁>0, α₁+β₁<1.

Canonical unconstrained coordinates map as `α₀=exp(x₁)`, `α₁=sigmoid(x₂)`, `β₁=sigmoid(-x₂) sigmoid(x₃)`. The log Jacobian is `x₁ + log σ(x₂) + 2 log σ(-x₂) + log σ(x₃) + log σ(-x₃)`. Both NUTS and DCS use this same potential. In particular there is no extra `1/(1-α₁)` conditional-uniform prior normalization.

With the supplied fixed initial variance and finite dataset, approaching α₁+β₁=1 does not by itself make all conditional variances diverge. Treat this as a constrained posterior benchmark; the repository makes no unsupported radial-tail claim about it. The PosteriorDB GARCH dataset is simulated, not observed financial data.

## Transformations and kernels

DCS maps z to `μ + R Tδ(z)/(1-||Tδ(z)||^β)^(1/β)`. The inverse and full Jacobian are in `transforms.py`. For t targets an algebraically equivalent log density cancels large radial factors before evaluation. DCS-BW proposes a uniform displacement in a ball; DCS-RWM uses Gaussian displacements; DCS-HnR performs a slice transition on a random feasible chord. The last is a slice kernel, not an exact independent conditional draw.

SCS uses a spherical-cap projection with latitude 1.1 and the authors' great-circle step-out proposal. Its map and surface Jacobian are independently tested. Every VI-enabled SCS fit calls the pinned author implementation, including logistic and ν=0.5 targets. SCS VI uses seed 0, one fixed reference cloud, Adam 0.01 and coordinate clipping at ±1,000 inside the fitting objective, including its Jacobian argument; it does not clip gradients. Full SCS fitting uses 1,000 updates and 2,000 reference draws, with separate reduced budgets of 400 × 500 and 20 × 128 for Smaller and Quick presets. Production target evaluation is unclipped. In the version-10 100D/500D comparisons, production also calls the pinned author transition and defaults to fixed h=0.1, with acceptance-target 0.234 adaptation available as a sensitivity option. Generic logistic and ν=0.5 experiments retain their checked Numba SCS production kernel and acceptance-target 0.234 adaptation. See [the current protocol](COMPARISONS.md) and [author-source and calibration details](AUTHOR_SCS_CALIBRATION.md).

In experiments containing SCS, DCS now reuses one fixed reference cloud, matching SCS's update count and cloud size. It retains smooth bounded shift parameters, Adam 0.01 and optimization-only gradient clipping, without clipping target coordinates or CDF arguments. The 100D/500D full budget is 1,000 updates on 2,000 fixed points and the default DCS fit seed is 42. Each method draws its own cloud from its own reference distribution. DCS experiments without SCS retain their recorded fitting budgets, draw policies and seeds. For symmetric t targets DCS fixes location and Möbius shift at zero and fits only scale. SCS uses the author's unconstrained centre/observer fit, even for a symmetric target. These are distinct fitting choices, recorded separately.

The default DCS `ball_map_policy=mobius` fits R, centre and Möbius shift, subject to those symmetry constraints. Internal `vanilla` compatibility skips all map training and initialization, with R=1, centre=0 and Möbius shift=0, while keeping the configured β. This untuned option is no longer exposed for new launcher or public command-line runs; saved reports keep their original parameters. The current comparison suite requires VI for both map families; there is no VI-off selection for new runs. Billiards use post-Möbius coordinates with the joint fit's centre/R/β and effective δ=0, as detailed in [the construction](BILLIARD.md). The identity maps for Eight Schools and GARCH are unaffected. Fixed R=√d is retained only as a historical plotting label. Sharing the physical target and production budget does not make these fitting and proposal rules identical.

Synthetic t configurations with `map_scale_initialization=radial_median` initialize the fitted DCS scale by matching the median norm of its independent base sample to the known F-law radial median. This helps the small-β DCS fit traverse a large change in log scale within its optimization budget. The current 100D/500D comparison instead initializes at unit scale; vanilla DCS skips this initialization. SCS always uses the author's unit-scale initialization and receives no radial-median adjustment. Logistic fits also start at unit scale.
