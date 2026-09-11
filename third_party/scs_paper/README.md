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

The experiment invokes the author's VI optimizer and original step-out
transition. It uses fixed proposal step 0.1, latitude 1.1, an all-ones physical
initial state, and no production adaptation. Full VI uses the original fixed
2,000-draw reference batch, seed 0, 1,000 Adam steps and learning rate 0.01.
The original optimizer clips physical coordinates to ±1,000 for both its
target and Jacobian calculations. The sampling target is not clipped.

Short blocks of the original scan limit memory. Independent keys between
blocks preserve the author's transition distribution; they do not reproduce
the exact random sequence of one uninterrupted upstream scan. Every production
state, including repeats after rejection, contributes to the radial estimates.
Thinning affects saved traces only. Compilation and discarded warmup have
separate clocks; sampling is synchronized before its clock stops.

This is **the SCS paper algorithm and optimizer applied to the original DCS
experiment's normal-CDF skew target**, at ν=3. The author's published example
uses the Azzalini–Capitanio Student-t-CDF skew target at ν=2 and ν=1. It is not
presented as a reproduction of the author's target or published figures.
