# Target densities and coordinate maps

`dcs_bench/targets.py` defines the common mathematical log density for the numerical backends. Production evaluations do not clip target coordinates or replace the skew CDF. Out-of-domain or nonrepresentable proposals are rejected; floating-point arithmetic has finite range.

## Normal-CDF-skewed multivariate Student t

The normalized density is

$$2t_{d,\nu}(x)\Phi\left(\alpha^\mathsf{T}x\sqrt{\frac{\nu+d}{\nu+\|x\|^2}}\right).$$

This normal-CDF family differs from a Student-t-CDF skew family. Exact independent draws first sample a symmetric t vector Y, then choose Y or −Y using the indicated normal-CDF probability. Symmetry gives the required normalized density.

Even functionals retain the symmetric radial law: `||X||²/d ~ F(d,ν)`. For ν>1,

$$E\|X\|=\sqrt{\nu}\,
\frac{\Gamma((d+1)/2)\Gamma((\nu-1)/2)}
{\Gamma(d/2)\Gamma(\nu/2)}.$$

The main comparison uses d=100 or 500, ν=3, α=(100,−100,0,…), DCS β=2 and radial thresholds 50 or 50√5. Both radial references are analytic. No coordinate-marginal statistics enter its report.

The 200D configurations use ν=3 and α=(20,−30,0,…). The 50D super-heavy-tail target is symmetric with ν=0.5 and DCS β=0.2. Its mean norm does not exist; `E log10(1+||X||)` is obtained by one-dimensional quadrature with numerical error reported. Tail probabilities follow the F law. Quantile references use independent exact draws where required.

## Logistic regression

The design uses NumPy RandomState seed 42, 50 observations and 20 standard-normal predictors, centred and scaled to standard deviation 0.5 (ddof=0). Responses are `1{X[:,0]>0}`. Independent Student-t priors have scale 2.5 and ν=4 or 6. The likelihood uses stable log-sigmoid evaluation.

This empirical target need not satisfy the manuscript's sufficient radial assumptions. Independent reference NUTS uses the same density and separate seeds. Quantile agreement is an empirical diagnostic, not proof of a CLT.

## Eight Schools

Physical order is θ₁,…,θ₈, μ, τ. The centred hierarchy uses μ~N(0,5²), τ~half-Cauchy(5), θⱼ~N(μ,τ²) and yⱼ~N(θⱼ,σⱼ²). Both samplers operate on (θ,μ,log τ), including the +log τ Jacobian, and report physical coordinates.

The target also supports a noncentred representation, but the configured experiment is centred. A different parameterization changes the experiment and should be recorded separately.

## GARCH(1,1)

Physical order is (μ,α₀,α₁,β₁). The pinned model sets `h[0]=sigma1²` and `h[t]=α₀+α₁(y[t−1]−μ)²+β₁h[t−1]`, with observations N(μ,h[t]). Its physical-coordinate prior is flat on α₀>0, α₁>0, β₁>0 and α₁+β₁<1.

Unconstrained coordinates map as α₀=exp(x₁), α₁=σ(x₂), β₁=σ(−x₂)σ(x₃). The log Jacobian is `x₁+log σ(x₂)+2log σ(−x₂)+log σ(x₃)+log σ(−x₃)`. DCS and NUTS use this same potential, without a conditional-uniform normalization factor.

This is a constrained posterior benchmark. Approaching the stationary boundary does not alone imply divergent conditional variances for the finite supplied dataset. PosteriorDB's GARCH data are simulated.

## Maps and fitting

DCS maps z to `μ+R Tδ(z)/(1−||Tδ(z)||^β)^(1/β)`, with inverse and full Jacobian in `transforms.py`. Equivalent radial cancellations improve numerical evaluation for t targets. BW uses uniform-ball increments, RWM Gaussian increments and HnR a slice transition on a random feasible chord. HnR is not an exact independent conditional draw.

The main comparison loads its fixed maps and parameters described in [the protocol](REPORTED_STUDY.md). All DCS centres/deformations are zero. Original SCS retains author VI; the extra SCS control has centre/observer zero with R/h chosen before that override. Neither family is refitted when a main-comparison run starts.

Every VI-enabled SCS fit uses the pinned original optimizer: seed 0, fixed cloud, Adam 0.01 and ±1,000 coordinate clipping within the fitting objective, including its Jacobian argument. Production remains unclipped. Full fitting is 1,000 updates on 2,000 fixed points; general Smaller and Quick settings use 400 × 500 and 20 × 128.

In paired DCS/SCS fitting, update counts and cloud sizes match. Each family uses its own reference distribution, seed and parameterization. DCS retains optimization-only gradient clipping and smooth bounded shifts. Symmetric t DCS fixes centre/deformation to zero while fitting scale; author SCS keeps its unconstrained centre/observer fit.

The generic synthetic DCS configurations can initialize scale from the known radial median; author SCS retains unit-scale initialization. Logistic fits start at unit scale. Eight Schools and GARCH use their specified identity maps.

SCS uses latitude 1.1 and the author great-circle step-out proposal. Main comparisons call the unmodified author transition; generic logistic and super-heavy-tail production use checked Numba transitions with acceptance-target 0.234 tuning. See [attribution](../THIRD_PARTY.md) and [billiard geometry](BILLIARD.md).
