# Validation of the corrected implementation

The local verification record is [validation/local_checks.json](../validation/local_checks.json), including platform, source hashes, experiment fingerprints and full-length tuning checks.

- **15 numerical/API tests passed** under Python 3.12.14: NumPy/JAX density agreement and finite gradients; GARCH agreement with an independent physical-coordinate likelihood and Jacobian; centered/non-centered Eight Schools equivalence; inverse maps and finite-difference volume/surface Jacobians; near-boundary stability; exact radial expectations; independent skew-t generation versus quadrature; invariant-transition checks for all three ball kernels and SCS; correct independent-chain RMSE; honest unavailable diagnostics; reference order; checkpoint validation; and the standalone API.
- **All seven quick experiments passed**, including both logistic prior settings: 40 production smoke chains plus eight separate logistic reference smoke chains. Every figure and the combined posterior table was generated.
- **All nine full-length map fits passed**, using the paper optimization settings and sample sizes. This includes the small-β symmetric target and both logistic priors.
- **All seven plotting-only paths passed.** Resume reused all 40 production checkpoints without changing their SHA-256 hashes. Compilation checks and `pip check` passed.
- Logistic and fixed-step PNG outputs were visually inspected for readable labels, correct panel arrangement and explicit quick-result labelling.

These checks establish implementation consistency and executable workflows. They do **not** establish paper-scale performance, posterior convergence, CLTs, or accuracy of short smoke references. Full paper production chains and the high-accuracy independent logistic references must still be run. Results from the quick profile must not be used as publication evidence. The Linux CI workflow is provided; no completed remote CI run is claimed.

Constant functional traces receive null MCSE/ESS with an explanation; they are not reported as successful zero-error diagnostics. High-accuracy logistic reference preparation rejects failed diagnostic screens for paper use. PosteriorDB reference MCSE is reported rather than treating the finite draws as exact values.

Fixed sufficient theorem assumptions are not enforced on empirical examples. The logistic benchmark intentionally remains a test beyond those assumptions. Larger sensitivity, ablation, calibration and non-centered-baseline studies remain additional research designs, as distinguished in [MANUSCRIPT_CHANGES.md](MANUSCRIPT_CHANGES.md).
