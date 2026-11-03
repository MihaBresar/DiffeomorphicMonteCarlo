"""Reversible chord reflection for isotropic proposals in the open unit ball.

This is a reflected-chord ball walk, not specular billiards or the SCS spherical
step-out. Conditional on an unoriented line, the signed proposal displacement
is symmetric. Reflecting that displacement on the line's chord gives a symmetric
interval proposal kernel, hence a proposal reversible for ball volume. A target
density ratio is therefore the appropriate Metropolis correction.
"""


def reflect_ball_proposal(x, increment):
    """Return ``(endpoint, valid)`` after reflecting a displacement on its chord.

    ``x`` is a finite point strictly inside the unit ball. An admissible raw
    proposal is returned unchanged. Otherwise, a triangular wave folds the
    entire displacement onto the chord, handling any number of reflections
    without a loop or reflection cap. The original isotropic increment law is
    retained by the caller.

    Exact boundary endpoints have probability zero in real arithmetic. Invalid
    or boundary-rounded results return ``(x, False)`` for an MH rejection; we
    never project or clip them into a thin interior shell. This is a finite-
    precision safeguard, not an additional mathematical boundary region or a
    claim of exact reversibility on a floating-point grid.
    """
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(x)
    increment = jnp.asarray(increment, dtype=x.dtype)
    raw = x + increment

    def inside(point):
        return jnp.all(jnp.isfinite(point)) & (jnp.sum(point * point) < 1.)

    def reflect(_):
        length = jnp.linalg.norm(increment)

        def finite_displacement(_):
            direction = increment / length
            position = jnp.dot(x, direction)
            transverse = x - position * direction
            # Reuse the original ball gap instead of subtracting a separately
            # rounded transverse norm from one near grazing chords.
            half_length_squared = (1. - jnp.dot(x, x)) + position * position

            def fold(_):
                half_length = jnp.sqrt(half_length_squared)
                chord_length = 2. * half_length
                phase = jnp.mod(position + half_length + length, 2. * chord_length)
                reflected_position = chord_length - jnp.abs(phase - chord_length) - half_length
                endpoint = transverse + reflected_position * direction
                return jax.lax.cond(inside(endpoint),
                    lambda _: (endpoint, jnp.asarray(True)),
                    lambda _: (x, jnp.asarray(False)), None)

            return jax.lax.cond(jnp.isfinite(half_length_squared) & (half_length_squared > 0.),
                                fold, lambda _: (x, jnp.asarray(False)), None)

        return jax.lax.cond(jnp.isfinite(length) & (length > 0.),
                            finite_displacement, lambda _: (x, jnp.asarray(False)), None)

    return jax.lax.cond(inside(raw), lambda _: (raw, jnp.asarray(True)), reflect, None)


def reflect_ball_step(x, increment):
    """Return the reflected point, with a self-loop for numerical invalidity.

    Metropolis callers should use :func:`reflect_ball_proposal` and count a
    false validity flag as rejection rather than as an accepted zero move.
    """
    return reflect_ball_proposal(x, increment)[0]
