# Original SCS source

`src/scp_core.py`, `src/train.py` and `LICENSE` are byte-for-byte copies from [liusf15/sub-cauchy-sampler](https://github.com/liusf15/sub-cauchy-sampler/tree/ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565), revision `ea6f27a6cb1d0ca90d8c1cd92c32fc3dcf8ea565`. `manifest.json` records SHA-256 checksums and upstream Git blob identities. Original MIT terms are preserved.

The adapter in `dcs_bench/scs_paper.py` privately resolves the source imports and replaces the optional progress decorator with an identity decorator. It does not modify vendored files or install a global replacement module.

Every VI-enabled SCS fit uses the original optimizer: seed 0, fixed reference draws, Adam 0.01, and physical-coordinate clipping at ±1,000 inside both fitting target and Jacobian evaluation. Production evaluations are unclipped. Full fitting uses 1,000 updates and 2,000 reference draws. Main comparisons load that full-budget fit at every preset. Other Quick/Smaller fits use 20 × 128 and 400 × 500 updates/draws.

The current six-method comparison includes original SCS at its author-fitted map and h=0.1, plus SCS calibrated (zero). The latter sets centre and observer to zero while retaining R/h selected on the fitted map before zeroing. It is not a constrained VI refit or a new calibration on the zero map. Both variants retain latitude 1.1 and the unchanged author projection and transition.

Short scan blocks bound memory. Independent block keys preserve the transition law, without reproducing the exact stream of an uninterrupted author scan. Every state, including repeats after rejection, enters radial estimates. Compilation and warmup have separate clocks; sampling is synchronized before its clock stops.

The Numba extension changes only DCS production; both comparison SCS variants stay in JAX. Logistic and t(0.5) experiments use separately checked Numba SCS production with acceptance-target 0.234 tuning, and are labelled accordingly. See [the full protocol](../../docs/REPORTED_STUDY.md) and [attribution](../../THIRD_PARTY.md).
