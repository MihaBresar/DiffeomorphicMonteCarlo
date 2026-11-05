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

The current [JAX and optimized-Numba comparisons](../../docs/COMPARISONS.md)
invoke the authors' original projection and step-out transition. With VI on,
SCS also calls the original variational optimizer: fixed reference draws,
seed 0, Adam learning rate 0.01, and physical-coordinate clipping at ±1,000
inside the fitting objective for both target and Jacobian evaluation. Full
settings use 2,000 reference draws and 1,000 updates. Quick and Smaller presets
use reduced budgets recorded in each configuration. DCS uses its own
fresh-draw fitting routine; the two fitting recipes are intentionally distinct.
With VI off, both map families use unit scale and zero centre/deformation.

SCS defaults to the authors' skew-t step h=0.1, with a discarded equilibration
phase and no step adaptation. Acceptance adaptation toward 0.234 is an explicit
sensitivity option. Both receive the same discarded transition budget and
freeze their production step. This choice and all measured alternatives are
reported in [the calibration study](../../docs/AUTHOR_SCS_CALIBRATION.md).
The SCS latitude is 1.1 and the physical pilot start is all ones.

Production target evaluations remain unclipped. The clipping above belongs
only to the unchanged author fitting routine. Historical configurations retain
their original fitting and proposal-calibration policies.

Short blocks of the original scan limit memory. Independent keys between
blocks preserve the author's transition distribution; they do not reproduce
the exact random sequence of one uninterrupted upstream scan. Every production
state, including repeats after rejection, contributes to the radial estimates.
Thinning affects saved traces only. Compilation and discarded warmup have
separate clocks; sampling is synchronized before its clock stops.

This uses **the SCS paper transition applied to our selected target**: the original DCS normal-CDF skew-t at ν=3 in 100D or 500D, with the recorded fitting and step policy. The author's published example
uses the Azzalini–Capitanio Student-t-CDF skew target at ν=2 and ν=1. It is not
presented as a reproduction of the author's target or published figures.
