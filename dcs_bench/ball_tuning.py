"""Discarded-pilot movement calibration for the reflected-chord ball walk.

The score is mean squared displacement of actual post-Metropolis ball states,
including zero displacement on rejection. It is not an acceptance-rate target,
a radial-error score, or a claim of globally optimal tuning. The proposal is
unchanged, and the selected scale is frozen before production sampling.
"""
import math
from numbers import Integral, Real

POLICY = 'ball_esjd_pilot_v1'
MAX_ROUNDS = 8
CANDIDATES_PER_ROUND = 3
STEP_FACTOR = 2.
NUMERICAL_LOG_MIN = -30.
NUMERICAL_LOG_MAX = 15.
HISTORY_COLUMNS = ('scale', 'iterations', 'mean_sjd', 'mean_acceptance', 'selected')


def calibrate_ball_esjd(block, initial, key, warmup, initial_scale, scale_ceiling=None):
    """Return final pilot state, frozen scale, total accepts, and trial history.

    ``block(state, key, count, scale)`` must return the same complete post-MH
    state and acceptance arrays as the production kernel. ``warmup`` is a
    static integer. All requested transitions are allocated across at most
    eight rounds of the local candidates ``h/2, h, 2h``. Remainder transitions
    go to the first trial windows in chronological execution order.

    Trial order rotates by round, but history is stored in ascending candidate
    order with columns ``HISTORY_COLUMNS``. Every trial continues from the
    preceding trial's final state, including between rounds. Exact score ties
    select the smallest candidate. The returned state is the last evaluated
    state, not an earlier endpoint of the winning trial.

    The default scale safeguards are the existing numerical log limits -30
    and 15; there is no ball-diameter cutoff. ``scale_ceiling`` is supported
    only for explicit historical/custom settings. A finite noisy local search
    need not identify a global optimum or a stationary expected displacement.
    """
    if isinstance(warmup, bool) or not isinstance(warmup, Integral) or warmup < 3:
        raise ValueError('Ball movement calibration requires at least 3 warmup iterations')
    if isinstance(initial_scale, bool) or not isinstance(initial_scale, Real) or \
            not math.isfinite(initial_scale) or initial_scale <= 0.:
        raise ValueError('initial_scale must be positive and finite')
    if scale_ceiling is not None and (isinstance(scale_ceiling, bool) or
            not isinstance(scale_ceiling, Real) or not math.isfinite(scale_ceiling) or
            scale_ceiling < initial_scale or math.log(scale_ceiling) < NUMERICAL_LOG_MIN):
        raise ValueError('scale_ceiling must be finite and at least initial_scale and exp(-30)')

    import jax
    import jax.numpy as jnp

    rounds = min(MAX_ROUNDS, int(warmup) // CANDIDATES_PER_ROUND)
    windows = rounds * CANDIDATES_PER_ROUND
    window_size, remainder = divmod(int(warmup), windows)
    log_ceiling = NUMERICAL_LOG_MAX if scale_ceiling is None else math.log(scale_ceiling)
    initial = jnp.asarray(initial, dtype=jnp.result_type(initial, float(initial_scale)))
    offsets = jnp.asarray([-math.log(STEP_FACTOR), 0., math.log(STEP_FACTOR)], dtype=initial.dtype)

    def round_step(carry, round_index):
        state, rng, log_center, total_accepts = carry
        candidates = jnp.exp(jnp.clip(log_center + offsets, NUMERICAL_LOG_MIN, log_ceiling))
        history = jnp.zeros((CANDIDATES_PER_ROUND, len(HISTORY_COLUMNS)), dtype=initial.dtype)

        def trial_step(trial_carry, trial_index):
            current, trial_rng, accepts_so_far, rows = trial_carry
            candidate_index = (trial_index + round_index) % CANDIDATES_PER_ROUND
            scale = candidates[candidate_index]
            trial_rng, subkey = jax.random.split(trial_rng)
            chronological_index = round_index * CANDIDATES_PER_ROUND + trial_index
            extra = chronological_index < remainder

            def evaluate(count):
                states, accepted = block(current, subkey, count, scale)
                previous = jnp.concatenate((current[None, ...], states[:-1]), axis=0)
                squared = jnp.sum((states - previous) ** 2, axis=tuple(range(1, states.ndim)))
                return states[-1], jnp.sum(squared), jnp.sum(accepted)

            # Both branches have identical output shapes even though the
            # underlying scan lengths differ by one transition.
            endpoint, squared_sum, accepted_sum = jax.lax.cond(extra,
                lambda _: evaluate(window_size + 1), lambda _: evaluate(window_size), operand=None)
            count = window_size + extra.astype(jnp.int32)
            row = jnp.asarray([scale, count, squared_sum / count, accepted_sum / count, 0.],
                              dtype=initial.dtype)
            rows = rows.at[candidate_index].set(row)
            return (endpoint, trial_rng, accepts_so_far + accepted_sum, rows), None

        (state, rng, total_accepts, history), _ = jax.lax.scan(trial_step,
            (state, rng, total_accepts, history), jnp.arange(CANDIDATES_PER_ROUND))
        # Candidates are ascending. argmax chooses the first exact tie, so
        # high acceptance with no movement never forces increasing scales.
        winner = jnp.argmax(history[:, 2])
        history = history.at[winner, 4].set(1.)
        return (state, rng, jnp.log(candidates[winner]), total_accepts), history

    initial_carry = (initial, key, jnp.asarray(math.log(initial_scale), dtype=initial.dtype),
                     jnp.asarray(0., dtype=initial.dtype))
    (state, _, log_scale, total_accepts), history = jax.lax.scan(
        round_step, initial_carry, jnp.arange(rounds))
    return state, jnp.exp(log_scale), total_accepts, history
