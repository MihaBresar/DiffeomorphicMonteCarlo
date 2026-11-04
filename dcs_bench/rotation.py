"""Reflected ball walk mixed with a reversible fixed-radius angular update.

The angular update rotates w=Mobius_delta(z), so it preserves ||x-mu|| even
with a fitted map. Its proposal is symmetric for w-volume, not z-volume.
The Mobius proposal correction below is therefore essential when VI is on.
Production chooses one component per transition with fixed probability 1/2.
"""
import math
from numbers import Integral, Real

import numpy as np

from .ball_tuning import (calibrate_ball_esjd, MAX_ROUNDS, CANDIDATES_PER_ROUND,
                          NUMERICAL_LOG_MIN, NUMERICAL_LOG_MAX)
from .transforms import BallMap, mobius

POLICY = 'split_ball_angular_esjd_v1'
ANGULAR_PROBABILITY = .5
INITIAL_ANGULAR_SCALE = 1.


def angular_proposal(state, normal, angular_scale, delta):
    """Return (proposal, log proposal correction, valid), without clipping.

    The angle is 2*atan(angular_scale), covering (0, pi) for positive scales.
    ``normal`` has d independent standard-normal components. Projection onto
    the tangent plane gives a uniform tangent direction. Reverse rotation uses
    an equally likely tangent direction, making the w-space kernel symmetric.

    For a z-space log density, add log|DM_delta(z)|-log|DM_delta(z')| to its
    difference. At fixed w-radius the resulting ratio equals pi(x')/pi(x).
    Exact degenerate directions and floating-point boundary hits self-loop.
    """
    import jax.numpy as jnp

    state, normal, delta = jnp.asarray(state), jnp.asarray(normal), jnp.asarray(delta)
    w, log_old_mobius = mobius(state, delta, jnp)
    radius = jnp.linalg.norm(w)
    direction = w / jnp.where(radius > 0., radius, 1.)
    tangent = normal - direction * jnp.dot(direction, normal)
    tangent_norm = jnp.linalg.norm(tangent)
    angle = 2. * jnp.arctan(angular_scale)
    rotated = (jnp.cos(angle) * w + jnp.sin(angle) * radius * tangent /
               jnp.where(tangent_norm > 0., tangent_norm, 1.))
    proposed, _ = mobius(rotated, -delta, jnp)
    _, log_new_mobius = mobius(proposed, delta, jnp)
    correction = log_old_mobius - log_new_mobius
    valid = ((radius > 0.) & (radius < 1.) & (tangent_norm > 0.) &
             (angular_scale > 0.) & jnp.isfinite(angular_scale) &
             jnp.all(jnp.isfinite(proposed)) & jnp.isfinite(correction) &
             (jnp.sum(rotated * rotated) < 1.) & (jnp.sum(proposed * proposed) < 1.))
    return jnp.where(valid, proposed, state), jnp.where(valid, correction, 0.), valid


def build_components(mapping_dict, target, reflected_components):
    """Build full-state scans using the supplied unchanged reflected-BW map.

    ``reflected_components`` is (block, projection, initial, density), as
    returned by timed_radial._dcs_components. The production scan fuses the
    component choice and proposal rather than nesting single-step scans.
    It returns exactly one post-MH state for each requested transition.
    """
    import jax
    import jax.numpy as jnp
    from .ball_boundary import reflect_ball_proposal

    mapping = BallMap(**mapping_dict)
    if mapping.d < 2 or mapping.d != target.d:
        raise ValueError('Angular mixtures require matching dimensions of at least two')
    reflected_block, projection, initial, density = reflected_components
    delta = jnp.asarray(mapping.delta)
    dimension = mapping.d

    def angular_step(state, normal, scale, logu):
        proposed, correction, valid = angular_proposal(state, normal, scale, delta)
        accept = jax.lax.cond(valid,
            lambda _: (logu < density(proposed) - density(state) + correction) * 1.,
            lambda _: jnp.asarray(0.), operand=None)
        return jnp.where(accept > 0., proposed, state), accept

    def angular_block(state, key, count, scale):
        def step(carry, index):
            current, rng = carry
            rng, normal_key, accept_key = jax.random.split(rng, 3)
            normal = jax.random.normal(normal_key, shape=(dimension,))
            current, accept = angular_step(current, normal, scale,
                                           jnp.log(jax.random.uniform(accept_key)))
            return (current, rng), (current, accept)
        return jax.lax.scan(step, (state, key), jnp.arange(count))[1]

    def block(state, key, count, scales):
        def step(carry, index):
            current, rng = carry
            # Independent selection/proposal/MH randomness. The d+2 Gaussian
            # gives the existing exact uniform-ball increment in the BW branch;
            # its first d coordinates give the angular branch's tangent normal.
            rng, selection_key, normal_key, accept_key = jax.random.split(rng, 4)
            normal = jax.random.normal(normal_key, shape=(dimension + 2,))
            logu = jnp.log(jax.random.uniform(accept_key))

            def reflected(_):
                increment = scales[0] * normal[:dimension] / jnp.linalg.norm(normal)
                proposed, valid = reflect_ball_proposal(current, increment)
                accept = jax.lax.cond(valid,
                    lambda _: (logu < density(proposed) - density(current)) * 1.,
                    lambda _: jnp.asarray(0.), operand=None)
                return jnp.where(accept > 0., proposed, current), accept

            current, accept = jax.lax.cond(
                jax.random.bernoulli(selection_key, ANGULAR_PROBABILITY),
                lambda _: angular_step(current, normal[:dimension], scales[1], logu),
                reflected, operand=None)
            return (current, rng), (current, accept)
        return jax.lax.scan(step, (state, key), jnp.arange(count))[1]

    def direction(states):
        w, _ = mobius(states, delta, jnp)
        radius = jnp.linalg.norm(w, axis=-1, keepdims=True)
        return w / jnp.where(radius > 0., radius, 1.)

    return dict(block=block, angular_block=angular_block, reflect_block=reflected_block,
                projection=projection, initial=initial, density=density, direction=direction)


def _calibrate_angular(block, direction, initial, key, warmup, initial_scale):
    """The same rotating local pilot search, scored in unit directions."""
    import jax
    import jax.numpy as jnp

    rounds = min(MAX_ROUNDS, warmup // CANDIDATES_PER_ROUND)
    windows = rounds * CANDIDATES_PER_ROUND
    window_size, remainder = divmod(warmup, windows)
    offsets = jnp.asarray([-math.log(2.), 0., math.log(2.)], dtype=initial.dtype)

    def round_step(carry, round_index):
        state, rng, log_center, accepts = carry
        candidates = jnp.exp(jnp.clip(log_center + offsets, NUMERICAL_LOG_MIN, NUMERICAL_LOG_MAX))
        history = jnp.zeros((3, 5), dtype=initial.dtype)

        def trial_step(trial_carry, trial_index):
            current, trial_rng, accepts_so_far, rows = trial_carry
            candidate_index = (trial_index + round_index) % 3
            scale = candidates[candidate_index]
            trial_rng, subkey = jax.random.split(trial_rng)
            extra = round_index * 3 + trial_index < remainder

            def evaluate(count):
                states, accepted = block(current, subkey, count, scale)
                directions = direction(states)
                previous = jnp.concatenate((direction(current)[None, :], directions[:-1]), axis=0)
                squared = jnp.sum((directions - previous) ** 2, axis=1)
                return states[-1], squared.sum(), accepted.sum()

            endpoint, squared, accepted = jax.lax.cond(extra,
                lambda _: evaluate(window_size + 1), lambda _: evaluate(window_size), operand=None)
            count = window_size + extra.astype(jnp.int32)
            rows = rows.at[candidate_index].set(jnp.asarray(
                [scale, count, squared / count, accepted / count, 0.], dtype=initial.dtype))
            return (endpoint, trial_rng, accepts_so_far + accepted, rows), None

        (state, rng, accepts, history), _ = jax.lax.scan(trial_step,
            (state, rng, accepts, history), jnp.arange(3))
        winner = jnp.argmax(history[:, 2])
        history = history.at[winner, 4].set(1.)
        return (state, rng, jnp.log(candidates[winner]), accepts), history

    final, history = jax.lax.scan(round_step,
        (initial, key, jnp.asarray(math.log(initial_scale), dtype=initial.dtype),
         jnp.asarray(0., dtype=initial.dtype)), jnp.arange(rounds))
    return final[0], jnp.exp(final[2]), final[3], history


def calibrate(components, initial, key, warmup, initial_bw_scale,
              initial_angular_scale=INITIAL_ANGULAR_SCALE, scale_ceiling=None):
    """Split the existing warmup budget between the two separately tuned scales.

    Floor(warmup/2) transitions tune reflected BW with ball ESJD. The rest
    continue from that endpoint and tune angular ESJD. Neither phase uses
    production samples or estimation errors. The angular phase holds centered
    physical radius fixed, so its pilot score is conditional on that radius;
    this finite local search is not a global-efficiency guarantee.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 6:
        raise ValueError('Split ball/angular calibration requires at least 6 warmup transitions')
    for name, value in [('initial_bw_scale', initial_bw_scale),
                        ('initial_angular_scale', initial_angular_scale)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0.:
            raise ValueError(f'{name} must be positive and finite')
    import jax
    import jax.numpy as jnp

    bw_count = int(warmup) // 2
    angular_count = int(warmup) - bw_count
    bw_key, angular_key = jax.random.split(key)
    state, bw_scale, bw_accepts, bw_history = calibrate_ball_esjd(
        components['reflect_block'], initial, bw_key, bw_count, initial_bw_scale, scale_ceiling)
    state, angular_scale, angular_accepts, angular_history = _calibrate_angular(
        components['angular_block'], components['direction'], state, angular_key,
        angular_count, initial_angular_scale)
    return (state, jnp.stack((bw_scale, angular_scale)), bw_accepts + angular_accepts,
            (bw_history, angular_history))
