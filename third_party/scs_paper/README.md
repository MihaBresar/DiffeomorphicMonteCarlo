# Original SCS paper code

`src/scp_core.py`, `src/train.py` and `LICENSE` are byte-for-byte copies from
[liusf15/sub-cauchy-sampler](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565),
commit `ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565` (15 January 2026). This was
the latest upstream revision before the paper's first arXiv release.
`manifest.json` records SHA-256 checksums and upstream Git blob identities.
The original MIT copyright notice and licence are preserved in `LICENSE`.

The adapter in `dcs_bench/scs_paper.py` privately resolves `src.train` and
substitutes an identity decorator for the optional `jax_tqdm` progress display.
It does not modify the vendored files or install a global replacement module.
JAX and Optax are already repository dependencies; `jax_tqdm` is unnecessary
because no progress callback runs inside the benchmark.

The current [JAX and optimized-Numba comparisons](../../docs/REPORTED_STUDY.md)
invoke the authors' original projection and step-out transition. Every
VI-enabled SCS fit throughout the repository calls the original variational
optimizer, including logistic and the symmetric ν=0.5 experiment: fixed reference draws,
seed 0, Adam learning rate 0.01, and physical-coordinate clipping at ±1,000
inside the fitting objective for both target and Jacobian evaluation. Full
settings use 2,000 reference draws and 1,000 updates. The current frozen-map
comparison loads that full-budget fit at every preset without refitting.
Earlier fitted-map and generic Quick/Smaller presets use 20 × 128 and 400 × 500
update/draw budgets, recorded independently of DCS in `scs_vi`. DCS uses its own
fitting parameterization and gradient clipping. Version 10 matches the full
evaluation budget at 1,000 updates on each family's own fixed 2,000-point
cloud. SCS's original cloud reuse is preserved; DCS now reuses a cloud too.
The reference distributions, fitting seeds and clipping rules remain distinct.
Current frozen-map comparisons reuse both original fits. Earlier VI-off reports
retain their recorded unit scale and zero centre/deformation.
SCS fitting keeps the author's initialization and parameterization; DCS-only
radial-median initialization and exact symmetry constraints do not alter it.

The current six-method comparison includes original SCS at the authors' skew-t
step h=0.1 and **SCS calibrated (zero)**. The latter uses zero physical centre
and zero observer displacement, retaining its previous R/h selections without
recalibration after zeroing. This changes the supplied map parameters, not the
author projection or transition code. It is not the original author fitted map
or a constrained VI refit. Historical v1/v2 reports and calibration evidence
retain their original parameters; zero-centre-only controls with a fitted
observer are distinct from the current zero-centre/zero-observer control.
Their SCS latitude is 1.1 and the physical pilot start is all ones. Generic
logistic and ν=0.5 experiments retain the checked Numba SCS production kernel
and acceptance-target 0.234 calibration; using author VI does not make those
production kernels unmodified author code.

Production target evaluations remain unclipped. The clipping above belongs
only to the unchanged author fitting routine. Existing saved results retain
their recorded fitting and proposal-calibration policies; fresh runs use
author VI whenever SCS map fitting is enabled.

Short blocks of the original scan limit memory. Independent keys between
blocks preserve the author's transition distribution; they do not reproduce
the exact random sequence of one uninterrupted upstream scan. Every production
state, including repeats after rejection, contributes to the radial estimates.
Thinning affects saved traces only. Compilation and discarded warmup have
separate clocks; sampling is synchronized before its clock stops.

This uses **the SCS paper transition applied to our selected target**: the original DCS normal-CDF skew-t at ν=3 in 100D or 500D, with the recorded fitting and step policy. The author's published example
uses the Azzalini–Capitanio Student-t-CDF skew target at ν=2 and ν=1. It is not
presented as a reproduction of the author's target or published figures.
