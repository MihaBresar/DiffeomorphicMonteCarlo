"""Endpoint-only implementation of the unchanged invariant billiard proposal.

This implementation removes outgoing-momentum construction, which is
needed for the phase-flow oracle but not for a position MH transition. It also
uses squared-norm < 1 as the combined finite/interior endpoint check: any NaN
or infinite component makes that predicate false. Proposal geometry, invariant
extra chords, random draws and density work are unchanged.

The intended caller is the existing DCS scan, with a finite interior state and
uniform-ball increments at a finite calibrated scale in exp([-30,15]). Under
that contract, discarded outgoing-momentum finiteness checks are redundant:
represented count < 2**53, half-chord >= sqrt(machine ball gap), finite bounded
direction/transverse vectors and scale <= exp(15) keep that auxiliary velocity
far below float64 overflow. The proposal is evaluated in post-Mobius coordinates by timed_radial.
"""


def reflect_ball_billiard(state,increment):
    """Return (endpoint, valid), without constructing final velocity.

    If a is the half-chord and b the impact parameter, add k complete chords:
    k = floor(max(0, atan2(L, 2a)/(2 atan2(a, b)) - L/(2a))).
    The added distance depends only on billiard invariants b and original L,
    which are even under momentum reversal. Hence the invariant-time flow
    preserves phase volume and reversibility. Complete chords preserve the
    proposed radius. Apply this rule even when the raw endpoint is interior;
    restricting it to boundary crossings would invalidate that argument.
    """
    import jax
    import jax.numpy as jnp

    state=jnp.asarray(state);increment=jnp.asarray(increment,dtype=state.dtype)
    radius2=jnp.dot(state,state)

    def reject(_):return state,jnp.asarray(False)

    def raw_endpoint(_):
        raw=state+increment
        valid=jnp.dot(raw,raw)<1.
        return jax.lax.cond(valid,
            lambda _:(raw,jnp.asarray(True)),reject,None)

    def advance(_):
        # A finite norm implies every increment component is finite. No
        # separate vector-wide isfinite reduction is necessary.
        length=jnp.linalg.norm(increment)

        def positive_length(_):
            direction=increment/length;position=jnp.dot(state,direction)
            transverse=state-position*direction;impact=jnp.linalg.norm(transverse)
            half=jnp.sqrt((1.-radius2)+position*position);chord=2.*half
            psi=2.*jnp.arctan2(half,impact)
            theta=jnp.arctan2(length,chord)
            extra=jnp.floor(jnp.maximum(0.,theta/psi-length/chord))
            unfolded=position+length+half
            base_count=jnp.floor(unfolded/chord)
            count=base_count+extra
            along=jnp.mod(unfolded,chord)-half
            max_exact_integer=float(2**(jnp.finfo(state.dtype).nmant+1))
            represented=(jnp.isfinite(half)&(half>0.) & jnp.isfinite(psi)&(psi>0.) &
                jnp.isfinite(theta)&jnp.isfinite(extra)&(extra>=0.) &
                jnp.isfinite(unfolded)&jnp.isfinite(count)&(count>=0.) &
                (count<max_exact_integer))

            def represented_endpoint(_):
                def collisions(_):
                    def near_diameter(_):
                        parity=1.-2.*jnp.mod(count,2.)
                        angle=count*(2.*jnp.arctan2(impact,half))
                        cosine=parity*jnp.cos(angle);sine=-parity*jnp.sin(angle)
                        coefficient=jnp.where(impact>0.,
                            sine/jnp.where(impact>0.,impact,1.),-parity*2.*count/half)
                        return cosine,sine,coefficient,jnp.isfinite(angle)

                    def near_grazing(_):
                        angle=count*psi
                        cosine,sine=jnp.cos(angle),jnp.sin(angle)
                        return cosine,sine,sine/impact,jnp.isfinite(angle)

                    cosine,sine,coefficient,phase_valid=jax.lax.cond(
                        impact<=half,near_diameter,near_grazing,None)
                    point=((along*cosine+impact*sine)*direction+
                           (cosine-along*coefficient)*transverse)
                    # As in the original DCS acceptance guard, the squared
                    # norm simultaneously rejects nonfinite and exterior points.
                    valid=phase_valid & (jnp.dot(point,point)<1.)
                    return jax.lax.cond(valid,
                        lambda _:(point,jnp.asarray(True)),reject,None)

                return jax.lax.cond(count==0.,raw_endpoint,collisions,None)

            return jax.lax.cond(represented,represented_endpoint,reject,None)

        return jax.lax.cond(length==0.,raw_endpoint,
            lambda _:jax.lax.cond(jnp.isfinite(length)&(length>0.),positive_length,reject,None),None)

    # This scalar already-computed norm check also rejects invalid starts;
    # it avoids a separate all-isfinite reduction over the state vector.
    return jax.lax.cond(radius2<1.,advance,reject,None)
