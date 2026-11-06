# Optimized billiard sampler

DCS-Billiard is one of the six methods in the current [comparison](REPORTED_STUDY.md). Both 100D and 500D targets are available in JAX and the Numba extension. No universal performance ranking is assumed.

All DCS methods share a zero-centre, zero-deformation ball map, β=2, and the dimension-specific R/h in the protocol. The source fit and production overrides remain separately recorded. Billiard warmup uses its own kernel for 1,500 discarded transitions; its preset step stays fixed.

Numba production starts from the corresponding JAX warmup state and uses an independent PCG64 stream. It follows the same invariant-chord rule, without fast-math or a collision loop. Author SCS remains in JAX.

## Active coordinates

The move operates after the Möbius transformation, in ball coordinates w, with

$$F(w)=\mu+\frac{Rw}{(1-\|w\|^\beta)^{1/\beta}},\qquad
g_w(w)=\pi(F(w))R^d(1-\|w\|^\beta)^{-1-d/\beta}.$$

Current comparison maps have μ=0 and δ=0. A symmetric w proposal uses the target-only Metropolis ratio. No extra Möbius correction is applied in these coordinates.

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
