"""Equal-budget calibration of each production kernel's radial movement.

The bounded score uses the actual accepted physical states, including repeated
states on rejection. It is a pilot movement proxy, not ESS or an MSE estimate.
The complete search is specified before production and every pilot is discarded.
"""
import math
from numbers import Integral, Real

POLICY = 'own_kernel_radial_pilot_v1'
COARSE_FACTORS = (.25, 1., 4., 16., 64.)
REFINE_FACTORS = (.5, 1., 2.)
WINDOWS = len(COARSE_FACTORS) + len(REFINE_FACTORS)
NUMERICAL_LOG_MIN = -30.
NUMERICAL_LOG_MAX = 15.
HISTORY_COLUMNS = ('scale', 'iterations', 'radial_score', 'acceptance',
                   'selected', 'trial_order', 'stage')
SCORE_DEFINITION = ('Mean bounded squared change of log(1+physical norm): '
    'delta^2/(1+delta^2), including zero changes on rejection; not ESS or radial MSE')
STEP_POLICY = ('Own production-kernel pilots: five fixed coarse scales then three '
    'local refinements, equal transition windows and seeded randomized trial order; '
    'maximize bounded log(1+physical norm) movement, then freeze')


def log1p_norm(physical):
    """Evaluate log(1+norm) without overflow for finite physical coordinates."""
    import jax.numpy as jnp

    magnitude = jnp.max(jnp.abs(physical), axis=-1)
    safe = jnp.where(magnitude > 0., magnitude, 1.)
    scaled = jnp.sqrt(jnp.sum((physical / safe[..., None]) ** 2, axis=-1))
    log_norm = jnp.log(safe) + jnp.log(jnp.where(scaled > 0., scaled, 1.))
    return jnp.where(magnitude > 0., jnp.logaddexp(0., log_norm), 0.)


def radial_jump_scores(physical):
    """One bounded score for each adjacent pair, with no accepted-only filtering."""
    import jax.numpy as jnp

    delta = jnp.diff(log1p_norm(physical))
    squared = delta * delta
    return squared / (1. + squared)


def calibrate_radial(block, projection, initial, key, warmup, initial_scale,
                     scale_ceiling=None):
    """Run exactly ``warmup`` transitions of the supplied production kernel.

    Five coarse candidates are ``initial_scale * (.25, 1, 4, 16, 64)``.
    Three refinement candidates are the coarse winner times ``(.5, 1, 2)``.
    The eight windows split the budget equally, with remainder transitions
    assigned to the earliest chronological windows. Trial order is randomized
    independently within each stage. All trials continue from the previous
    endpoint; the last endpoint, not a selected past state, starts production.

    History is stored in ascending candidate order within each stage. Its
    zero-based ``trial_order`` records actual global chronology; ``stage`` is
    one for coarse and two for refinement. Exact score ties select the lowest
    candidate. Numerical log limits are safeguards, not a geometric ceiling.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < WINDOWS:
        raise ValueError('Own-kernel radial calibration requires at least eight warmup transitions')
    if isinstance(initial_scale, bool) or not isinstance(initial_scale, Real) or \
            not math.isfinite(initial_scale) or initial_scale <= 0.:
        raise ValueError('initial_scale must be positive and finite')
    if scale_ceiling is not None and (isinstance(scale_ceiling, bool) or
            not isinstance(scale_ceiling, Real) or not math.isfinite(scale_ceiling) or
            scale_ceiling < initial_scale or math.log(scale_ceiling) < NUMERICAL_LOG_MIN):
        raise ValueError('scale_ceiling must be finite and at least initial_scale and exp(-30)')

    import jax
    import jax.numpy as jnp

    initial = jnp.asarray(initial, dtype=jnp.result_type(initial, float(initial_scale)))
    window_size, remainder = divmod(int(warmup), WINDOWS)
    log_ceiling = NUMERICAL_LOG_MAX if scale_ceiling is None else math.log(scale_ceiling)

    def stage(state, rng, center, accepted_total, factors, offset, stage_index):
        candidates = jnp.exp(jnp.clip(jnp.log(center) + jnp.log(jnp.asarray(factors)),
                                      NUMERICAL_LOG_MIN, log_ceiling))
        rng, order_key = jax.random.split(rng)
        order = jax.random.permutation(order_key, len(factors))
        history = jnp.zeros((len(factors), len(HISTORY_COLUMNS)), dtype=initial.dtype)

        def trial(carry, trial_index):
            current, trial_rng, accepts_so_far, rows = carry
            candidate_index = order[trial_index]
            scale = candidates[candidate_index]
            trial_rng, subkey = jax.random.split(trial_rng)
            chronological_index = offset + trial_index
            extra = chronological_index < remainder

            def evaluate(count):
                states, accepted = block(current, subkey, count, scale)
                physical = projection(jnp.concatenate((current[None, ...], states), axis=0))
                return states[-1], jnp.sum(radial_jump_scores(physical)), jnp.sum(accepted)

            endpoint, score_sum, accepts = jax.lax.cond(extra,
                lambda _: evaluate(window_size + 1), lambda _: evaluate(window_size), operand=None)
            count = window_size + extra.astype(jnp.int32)
            row = jnp.asarray([scale, count, score_sum / count, accepts / count,
                               0., chronological_index, stage_index], dtype=initial.dtype)
            rows = rows.at[candidate_index].set(row)
            return (endpoint, trial_rng, accepts_so_far + accepts, rows), None

        (state, rng, accepted_total, history), _ = jax.lax.scan(trial,
            (state, rng, accepted_total, history), jnp.arange(len(factors)))
        winner = jnp.argmax(history[:, 2])
        history = history.at[winner, 4].set(1.)
        return state, rng, candidates[winner], accepted_total, history

    state, rng, center, accepts, coarse = stage(initial, key,
        jnp.asarray(initial_scale, dtype=initial.dtype), jnp.asarray(0., dtype=initial.dtype),
        COARSE_FACTORS, 0, 1)
    state, _, scale, accepts, refined = stage(state, rng, center, accepts,
        REFINE_FACTORS, len(COARSE_FACTORS), 2)
    return state, scale, accepts, jnp.concatenate((coarse, refined), axis=0)
