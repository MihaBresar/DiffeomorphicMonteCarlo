# Target densities and coordinate maps

`dcs_bench/targets.py` defines the same mathematical log density for NumPy and JAX. Arbitrary numerical epsilons, clipped CDF arguments, and fixed interior-ball cutoffs are absent from production target evaluations. Out-of-domain or nonrepresentable proposals are rejected; floating-point arithmetic still has a finite range.

## Normal-CDF-skewed multivariate Student t

The normalized density is `2 t_d,ν(x) Φ(αᵀx sqrt((ν+d)/(ν+||x||²)))`. This is the manuscript's normal-CDF formula. It should not be silently substituted with a Student-t-CDF skew family.

An exact iid construction first draws Y from the symmetric multivariate t, then returns Y with probability Φ(a(Y)) and -Y otherwise. Since t(-y)=t(y) and a(-y)=-a(y), the output density is `t(y)Φ(a(y)) + t(-y)[1-Φ(a(-y))] = 2t(y)Φ(a(y))`. Sign selection precedes neither the t scale mixture nor evaluation of a(Y).

Even functionals, including radial functions, retain the symmetric t law. Thus `||X||²/d ~ F(d,ν)`. For ν>1, `E||X|| = sqrt(ν) Γ((d+1)/2) Γ((ν-1)/2) / [Γ(d/2) Γ(ν/2)]`. At d=200, ν=3 the radial references are 0.029191445495738842 for radius ≥50 and 19.519685714651676 for the mean norm. For ν=0.5 the mean norm is not defined; the code instead uses `E log10(1+||X||)` evaluated by one-dimensional quadrature, with numerical error reported.

Directional checks use the unit skewness direction (e₁ if α=0): `1{uᵀX>0}` and `tanh(uᵀX/sqrt(ν))`. Their references are independent iid estimates with MCSE, or exact symmetry values 1/2 and 0. Coordinate quantiles use the corrected iid construction. One million iid reference draws are used for paper quantiles; exact radial constants need no Monte Carlo sample.

The [current comparison suite](COMPARISONS.md) also offers the original 100D skewed t(3) and a 500D version with the same skew coefficients; their radial thresholds are 50 and 50√5 respectively. Its VI switch applies to both map families, and its maps initialize at unit scale. Other experiment configurations may use the radial-median initialization described below.

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

SCS uses a spherical-cap projection with latitude 1.1 and the authors' great-circle step-out proposal. Its map and surface Jacobian are independently tested. For symmetric t targets, location and observer/Möbius shifts remain zero; only scale is fit. Map training uses fresh independent base draws at each iteration, restoring the original stochastic VI procedure, and reverse KL with the exact target, smooth bounded shift parameterization, Adam and optimization-only gradient clipping. The target is not clipped during training. The default DCS `ball_map_policy=mobius` fits R, centre and Möbius shift (subject to the symmetry constraints above). The optional `vanilla` policy skips all map training and initialization, with R=1, centre=0 and Möbius shift=0, while keeping the configured β. In the other six experiments this DCS-only selector leaves SCS fitting unchanged. In the current comparison suite, the separate VI switch controls both families and all methods still pretune their proposal steps. The identity maps for Eight Schools and GARCH are unaffected. Fixed R=√d is retained only as a historical plotting label. These implementations do not claim bitwise replication of an upstream optimizer.

Synthetic t configurations with `map_scale_initialization=radial_median` initialize fitted DCS and SCS scales by matching the median norm of their independent base sample to the known F-law radial median. The current comparison suite instead initializes at unit scale; vanilla DCS skips this initialization. It prevents the small-β fit from exhausting its iterations while moving from R=1 across many log-scale units. It uses the same analytic target-scale information for both projection families; logistic fits start at unit scale.
