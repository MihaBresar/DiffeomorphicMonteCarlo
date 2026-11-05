# Optimized billiard sampler

`DCS-Billiard` is available in both current five-method comparisons alongside
DCS-RWM, DCS-BW, DCS-BW-reflect and author SCS. Choose JAX for a common-framework comparison,
or optimized Numba DCS versus the author JAX SCS implementation. Both offer
100D and 500D skewed Student-t(3), with and without VI. This is an experimental
geometric proposal, not a claim of universal dominance.

## Running the full comparison

Extract the latest repository ZIP and open **Start.bat** on Windows. Choose:

1. **Five samplers: DCS-RWM, DCS-BW, SCS, reflected BW and DCS-Billiard (JAX)** or **Five samplers: optimized Numba DCS versus SCS**.
2. **100D** or **500D** under Target distribution.
3. **With VI** or **Without VI**.
4. **Full paper settings**, a sampling seed batch, and CPU workers.
5. **Start run**.

Full settings use 50 chains per method, 20.00 production seconds per chain,
and 1,500 separate pilot transitions. The first third of production is discarded.
Reports show mean norm, radial-tail probability, MSE, sample counts, timing,
functional ESS and ESS per second. They omit coordinate statistics. See
[the comparison protocol](COMPARISONS.md) and [ESS definitions](ESS.md).

Both implementations use the same own-kernel JAX pilot state and frozen step. Numba
production uses an independent PCG64 stream and fused density/radius summaries;
it is not a bitwise replay of JAX. Its endpoint follows the same invariant-chord
rule below, without fast-math approximations. The author SCS code stays in JAX.

## Active coordinates and VI

Write the fitted ball transformation as

$$w=M_\delta(z),\qquad
F(w)=\mu+\frac{R w}{(1-\|w\|^\beta)^{1/\beta}}.$$

The billiard move runs directly in **w**, after Möbius. The effective production
map retains the jointly fitted μ, R and β and has δ=0. The fitted δ influenced
the joint optimization of μ and R, but it has no direct production effect.
This is not a new VI fit constrained to δ=0. Reports retain both the original
fit and the effective method map so this difference is visible. With VI off,
μ=0, R=1 and δ=0 for all ball methods; β remains 2.

The density in these active coordinates is

$$g_w(w)=\pi(F(w))\,R^d(1-\|w\|^\beta)^{-1-d/\beta}.$$

A symmetric w proposal therefore uses the ordinary ratio
`min(1, g_w(w') / g_w(w))`. Equivalently, expressing it in z coordinates
introduces a reverse/forward proposal ratio
`J_M(z) / J_M(z')`, which cancels the Möbius determinants in the transformed
target ratio. No additional Möbius correction is applied when running in w.

The pilot starts afresh at the w coordinate corresponding to the common
physical all-ones point. Versions 8 and 9 calibrate **billiard moves themselves**,
using exactly the preset's warmup budget. The pilot scores bounded changes in
physical log(1 + norm), with rejections contributing zero, then freezes the
selected scale before production from its last pilot state. No production
MSE or known reference value enters this per-chain calibration.

Version 8 gave SCS the same radial-pilot policy and transition budget. Current
version 9 instead uses the authors' SCS VI and defaults to fixed h=0.1 during
its discarded pilot and production. Acceptance-target 0.234 adaptation remains
an explicit SCS sensitivity option. Both SCS policies keep the same number of
discarded pilot transitions as DCS. Their global default was selected in the
separately reported [development study](AUTHOR_SCS_CALIBRATION.md); it was not
selected within any retained production chain. See [the full protocol](COMPARISONS.md).

The version-6/7 runs instead calibrated reflected BW in the active coordinates
and transferred its step to billiards. Their saved reports retain that rule.
Changing calibration does not change the proposal or invariance argument below.

## Proposal and invariants

Draw the same uniform-ball increment as BW, with original length L and unit
direction v. For a finite interior position w, define

$$p=w\cdot v,\quad b=\|w-pv\|,\quad
a=\sqrt{1-\|w\|^2+p^2},\quad
\psi=2\operatorname{atan2}(a,b).$$

Here a is the half-chord length and b is the impact parameter. The extra
number of complete chord traversals is fixed by

$$\theta=\operatorname{atan2}(L,2a),\qquad
k=\left\lfloor\max\left(0,\frac{\theta}{\psi}-\frac{L}{2a}\right)\right\rfloor.$$

Follow specular billiard flow for length `L + 2ak`. The implementation computes
the endpoint directly; it does not loop over collisions. Its vector work is
O(d), it needs no extra Gaussian vector, and it performs one Metropolis test.
Zero increments give the identity. The rule is applied to **every proposal**,
including increments whose unextended endpoint lies inside the ball.

Specular billiard flow preserves position-volume times uniform directional
measure. Both b and the original L are constant along an orbit and unchanged
by momentum reversal. The selected travel time is consequently constant on
that orbit and even in momentum. Applying this invariant travel time, then
reversing momentum, is a measure-preserving involution. Refreshing the
isotropic direction and discarding momentum gives a position proposal
reversible for ball volume, so the target-only Metropolis ratio above applies.

The integer k is allowed to be discontinuous: its level sets are invariant
under the flow. A time chosen from the current target density, the initial
axial phase p, or only the event that the original proposal exits the ball
would not have this argument. The conceptual terminal momentum keeps original
length L, not the extended travel distance. The optimized implementation does
not construct that unused terminal momentum.

## Radial identity and limits

For the ordinary flight, let

$$m=\left\lfloor\frac{p+L+a}{2a}\right\rfloor,\qquad
s=(p+L+a)\bmod(2a)-a.$$

Adding k full chords replaces m by m+k but leaves s unchanged. The endpoint
squared radius is **b²+s²**, the same as chord-reflected BW with the same
increment. The added travel length satisfies `2ak ≤ θ < π/2`; formal collision
counts can be large near grazing without increasing loop work, because there
is no collision loop.

For a radial g_w and the same frozen scale/length law, the accepted radial
transition kernels therefore agree. The corresponding physical distance is
`||x−μ||`. This does **not** prove preservation of the original z-coordinate
BW kernel, the reported `||x||` when μ differs from zero, or equal-time radial
MSE on a skew target. Direction-dependent acceptance and changed proposal
coordinates must be assessed empirically.

The implementation retains finite-length, representable-count, phase and
interior-endpoint checks. Stable diameter and grazing formulas avoid avoidable
cancellation. Invalid results self-loop; endpoints are not projected into an
artificial interior shell. These numerical safeguards are distinct from the
exact-arithmetic reversibility argument. Compiler roundoff can make long
same-seed trajectories diverge even when common-input transition checks agree.

The reference construction builds on [Gryazina and Polyak's Billiard Walk](https://arxiv.org/abs/1211.3932).
Its invariant extra-chord rule and retained BW flight-length law differ from
their original flight-length choice. For the unchanged comparison algorithm,
see [the SCS proposal](https://arxiv.org/html/2601.11066v1#S2.SS2) and
[the pinned source provenance](../THIRD_PARTY.md).
