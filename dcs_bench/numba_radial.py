"""Optimized native DCS production with the existing JAX calibration.

This is an implementation comparison, not a common-framework timing claim.
The proposal, target, frozen step and initial production state agree with the
JAX DCS procedure. Native production uses its own domain-separated PCG64 stream.
Current density and physical norm are cached across rejections; every resulting
state, including repeats, contributes to the same 300-transition summaries.
"""
from copy import deepcopy
import time

import numpy as np
from numba import njit
from scipy.special import gammaln

from .fast_dcs import _log_ndtr
from .transforms import BallMap

BATCH_SIZE = 300
PREFIX_STRIDE = 100
PRODUCTION_DOMAIN = 0x4E424457  # NB DW: separate from JAX calibration streams.


@njit(nogil=True, error_model='numpy')
def _evaluate(z, mu, delta, alpha, constants, skewed):
    """Stable transformed density and physical norm, sharing the map traversal."""
    d = len(z)
    beta, scale, tail, delta2, logscale, logdelta, normalizer = constants
    r2, dot = 0., 0.
    for j in range(d):
        r2 += z[j] * z[j]
        dot += z[j] * delta[j]
    if not np.isfinite(r2) or r2 >= 1.:
        return -np.inf, np.nan
    denominator = 1. + 2. * dot + delta2 * r2
    if denominator <= 0. or not np.isfinite(denominator):
        return -np.inf, np.nan
    logmo = logdelta - d * np.log(denominator)
    # Exact gap identity avoids subtracting the rounded Mobius radius for beta2.
    if beta == 2.:
        eps = (1. - delta2) * (1. - r2) / denominator
        logeps = np.log(eps)
        e = np.sqrt(eps)
    else:
        rho2 = 0.
        for j in range(d):
            w = ((1. - delta2) * z[j] + (1. + 2. * dot + r2) * delta[j]) / denominator
            rho2 += w * w
        if not np.isfinite(rho2) or rho2 >= 1.:
            return -np.inf, np.nan
        logeps = 0. if rho2 == 0. else np.log(-np.expm1(.5 * beta * np.log(rho2)))
        e = np.exp(logeps / beta)
    q, skew = 0., 0.
    for j in range(d):
        w = ((1. - delta2) * z[j] + (1. + 2. * dot + r2) * delta[j]) / denominator
        value = mu[j] * e + scale * w
        q += value * value
        if skewed:
            skew += alpha[j] * value
    rootq = np.sqrt(q)
    s = tail * e * e + q
    lp = (normalizer + (tail / beta - 1.) * logeps + logscale + logmo
          - (tail + d) / 2. * np.log(s))
    if skewed:
        lp += np.log(2.) + _log_ndtr(skew * np.sqrt((tail + d) / s), 0)
    return (lp, rootq / e) if np.isfinite(lp) else (-np.inf, np.nan)


@njit(nogil=True, error_model='numpy')
def _evaluate_extended(z, mu, delta, alpha, constants, skewed):
    """Density, physical radius and X[:3] in one stable map traversal."""
    d = len(z)
    beta, scale, tail, delta2, logscale, logdelta, normalizer = constants
    r2, dot = 0., 0.
    for j in range(d):
        r2 += z[j] * z[j]
        dot += z[j] * delta[j]
    if not np.isfinite(r2) or r2 >= 1.:
        return -np.inf, np.nan, np.nan, np.nan, np.nan
    denominator = 1. + 2. * dot + delta2 * r2
    if denominator <= 0. or not np.isfinite(denominator):
        return -np.inf, np.nan, np.nan, np.nan, np.nan
    logmo = logdelta - d * np.log(denominator)
    if beta == 2.:
        eps = (1. - delta2) * (1. - r2) / denominator
        logeps = np.log(eps)
        e = np.sqrt(eps)
    else:
        rho2 = 0.
        for j in range(d):
            w = ((1. - delta2) * z[j] + (1. + 2. * dot + r2) * delta[j]) / denominator
            rho2 += w * w
        if not np.isfinite(rho2) or rho2 >= 1.:
            return -np.inf, np.nan, np.nan, np.nan, np.nan
        logeps = 0. if rho2 == 0. else np.log(-np.expm1(.5 * beta * np.log(rho2)))
        e = np.exp(logeps / beta)
    q, skew, first, second, third = 0., 0., 0., 0., 0.
    for j in range(d):
        w = ((1. - delta2) * z[j] + (1. + 2. * dot + r2) * delta[j]) / denominator
        value = mu[j] * e + scale * w
        q += value * value
        if j == 0:
            first = value / e
        elif j == 1:
            second = value / e
        elif j == 2:
            third = value / e
        if skewed:
            skew += alpha[j] * value
    s = tail * e * e + q
    lp = (normalizer + (tail / beta - 1.) * logeps + logscale + logmo
          - (tail + d) / 2. * np.log(s))
    if skewed:
        lp += np.log(2.) + _log_ndtr(skew * np.sqrt((tail + d) / s), 0)
    return ((lp, np.sqrt(q) / e, first, second, third) if np.isfinite(lp)
            else (-np.inf, np.nan, np.nan, np.nan, np.nan))


@njit(nogil=True, error_model='numpy')
def _proposal_into(state, increment, reflected, proposed):
    """Fill a scratch proposal with the same direct/chord-reflected endpoint."""
    raw2 = 0.
    for j in range(len(state)):
        proposed[j] = state[j] + increment[j]
        raw2 += proposed[j] * proposed[j]
    if np.isfinite(raw2) and raw2 < 1.:
        return True
    if not reflected:
        return False
    length2, state2, dot = 0., 0., 0.
    for j in range(len(state)):
        length2 += increment[j] * increment[j]
        state2 += state[j] * state[j]
        dot += state[j] * increment[j]
    if not np.isfinite(length2) or length2 <= 0.:
        return False
    length = np.sqrt(length2)
    position = dot / length
    half2 = (1. - state2) + position * position
    if not np.isfinite(half2) or half2 <= 0.:
        return False
    half = np.sqrt(half2)
    chord = 2. * half
    phase = (position + half + length) % (2. * chord)
    new_position = chord - abs(phase - chord) - half
    norm2 = 0.
    for j in range(len(state)):
        direction = increment[j] / length
        transverse = state[j] - position * direction
        proposed[j] = transverse + new_position * direction
        norm2 += proposed[j] * proposed[j]
    return np.isfinite(norm2) and norm2 < 1.


@njit(nogil=True, error_model='numpy')
def _transition_from_increment(state, lp, radius, increment, logu, reflected,
                               mu, delta, alpha, constants, skewed, proposed):
    if not _proposal_into(state, increment, reflected, proposed):
        return lp, radius, False
    next_lp, next_radius = _evaluate(proposed, mu, delta, alpha, constants, skewed)
    if logu < next_lp - lp:
        for j in range(len(state)):
            state[j] = proposed[j]
        return next_lp, next_radius, True
    return lp, radius, False


@njit(nogil=True, error_model='numpy')
def _run_block(rng, state, lp, radius, scale, count, thin, threshold, reflected,
               mu, delta, alpha, constants, skewed):
    d = len(state)
    prefixes = np.zeros((count // PREFIX_STRIDE, 3))
    norms = np.empty(count // thin)
    increment, proposed = np.empty(d), np.empty(d)
    tails, norm_sum, accepted = 0, 0., 0
    for i in range(count):
        normals = rng.standard_normal(d + 2)
        full2 = 0.
        for j in range(d + 2):
            full2 += normals[j] * normals[j]
        factor = scale / np.sqrt(full2)
        for j in range(d):
            increment[j] = factor * normals[j]
        lp, radius, accept = _transition_from_increment(
            state, lp, radius, increment, np.log(rng.random()), reflected,
            mu, delta, alpha, constants, skewed, proposed)
        tails += int(radius >= threshold)
        norm_sum += radius
        accepted += int(accept)
        if (i + 1) % PREFIX_STRIDE == 0:
            row = (i + 1) // PREFIX_STRIDE - 1
            prefixes[row, 0], prefixes[row, 1], prefixes[row, 2] = tails, norm_sum, accepted
        if (i + 1) % thin == 0:
            norms[(i + 1) // thin - 1] = radius
    return lp, radius, prefixes, norms


@njit(nogil=True, error_model='numpy')
def _transition_extended(state, lp, radius, first, second, third, increment, logu, reflected,
                         mu, delta, alpha, constants, skewed, proposed):
    if not _proposal_into(state, increment, reflected, proposed):
        return lp, radius, first, second, third, False
    next_lp, next_radius, next_first, next_second, next_third = _evaluate_extended(
        proposed, mu, delta, alpha, constants, skewed)
    if logu < next_lp - lp:
        for j in range(len(state)):
            state[j] = proposed[j]
        return next_lp, next_radius, next_first, next_second, next_third, True
    return lp, radius, first, second, third, False


@njit(nogil=True, error_model='numpy')
def _run_block_extended(rng, state, lp, radius, first, second, third, scale, count, thin,
                        threshold, coordinate_threshold, reflected, mu, delta,
                        alpha, constants, skewed):
    """Accumulate every state; thinning affects only the returned traces."""
    d = len(state)
    prefixes = np.zeros((count // PREFIX_STRIDE, 9))
    saved = np.empty((count // thin, 4))
    increment, proposed = np.empty(d), np.empty(d)
    tails, norm_sum, accepted, first_sum, first_tails = 0, 0., 0, 0., 0
    second_sum, second_tails, third_sum, third_tails = 0., 0, 0., 0
    for i in range(count):
        normals = rng.standard_normal(d + 2)
        full2 = 0.
        for j in range(d + 2):
            full2 += normals[j] * normals[j]
        factor = scale / np.sqrt(full2)
        for j in range(d):
            increment[j] = factor * normals[j]
        lp, radius, first, second, third, accept = _transition_extended(
            state, lp, radius, first, second, third, increment, np.log(rng.random()), reflected,
            mu, delta, alpha, constants, skewed, proposed)
        tails += int(radius >= threshold)
        norm_sum += radius
        accepted += int(accept)
        first_sum += first
        first_tails += int(first > coordinate_threshold)
        second_sum += second
        second_tails += int(second < -coordinate_threshold)
        third_sum += third
        third_tails += int(third > coordinate_threshold)
        if (i + 1) % PREFIX_STRIDE == 0:
            row = (i + 1) // PREFIX_STRIDE - 1
            prefixes[row, 0], prefixes[row, 1], prefixes[row, 2] = tails, norm_sum, accepted
            prefixes[row, 3], prefixes[row, 4] = first_sum, first_tails
            prefixes[row, 5], prefixes[row, 6] = second_sum, second_tails
            prefixes[row, 7], prefixes[row, 8] = third_sum, third_tails
        if (i + 1) % thin == 0:
            row = (i + 1) // thin - 1
            saved[row, 0], saved[row, 1] = radius, first
            saved[row, 2], saved[row, 3] = second, third
    return lp, radius, first, second, third, prefixes, saved


def _inputs(mapping_dict, config):
    mapping = BallMap(**mapping_dict)
    spec = config.get('target', config)
    d = int(spec.get('dimension', mapping.d))
    if mapping.d != d:
        raise ValueError('Numba map and target dimensions differ')
    kind_name = spec.get('kind', 'skew_t')
    if kind_name != 'skew_t':
        raise ValueError(f'Numba radial production does not support {kind_name}')
    tail = float(spec.get('nu', 3.))
    if not np.isfinite(tail) or tail <= 0.:
        raise ValueError('Target degrees of freedom must be positive and finite')
    alpha = np.zeros(d)
    for index, value in spec.get('alpha', {}).items():
        alpha[int(index)] = value
    normalizer = (gammaln((tail + d) / 2.) - gammaln(tail / 2.)
                  + tail / 2. * np.log(tail) - d / 2. * np.log(np.pi))
    delta2 = float(mapping.delta @ mapping.delta)
    constants = np.array([mapping.beta, mapping.scale, tail, delta2,
                          d * np.log(mapping.scale), d * np.log1p(-delta2), normalizer])
    return (np.ascontiguousarray(mapping.mu), np.ascontiguousarray(mapping.delta),
            alpha, constants, bool(np.any(alpha)))


class NumbaRadial:
    """Native production initialized from the identical JAX pretuning result.

    The caller supplies the already-calibrated JAX backend. Its compile cost
    (including an unused JAX production program) remains in recorded setup cost.
    No calibration is rerun and no native production randomness selects a step.
    """
    def __init__(self, mapping, config, seed, calibration):
        if config['algorithm'] not in ('DCS-BW', 'DCS-BW-reflect'):
            raise ValueError('Optimized Numba comparison supports BW and reflected BW')
        self.coordinate_metrics = bool(config.get('coordinate_metrics', False))
        self.extended = self.coordinate_metrics
        self.inputs = _inputs(mapping, config)
        self.state = np.array(calibration.state, dtype=np.float64, order='C', copy=True)
        if self.state.shape != (len(self.inputs[0]),) or not np.all(np.isfinite(self.state)):
            raise ValueError('Invalid calibrated ball state')
        self.scale = self._scale = float(calibration.scale)
        if not np.isfinite(self.scale) or self.scale <= 0.:
            raise ValueError('Invalid calibrated proposal scale')
        self.thin = int(config['thin'])
        if self.thin < 1 or PREFIX_STRIDE % self.thin:
            raise ValueError('Native thin must divide100')
        self.threshold = float(config.get('threshold', 50.))
        self.coordinate_threshold = float(config.get('coordinate_threshold', 3.))
        if self.extended and not np.isfinite(self.coordinate_threshold):
            raise ValueError('Coordinate threshold must be finite')
        if self.extended and len(self.state) < 3:
            raise ValueError('Coordinate measurements require at least three dimensions')
        self.reflected = config['algorithm'] == 'DCS-BW-reflect'
        self.warmup_seconds = float(calibration.warmup_seconds)
        self.warmup_acceptance = calibration.warmup_acceptance
        self.initial_scale = float(getattr(calibration, 'initial_scale', config.get('initial_scale', .1)))
        self.tuning_policy = getattr(calibration, 'tuning_policy', None)
        self.tuning_metadata = deepcopy(getattr(calibration, 'tuning_metadata', {}))
        started = time.perf_counter()
        if self.extended:
            self.lp, self.radius, self.first, self.second, self.third = _evaluate_extended(self.state, *self.inputs)
        else:
            self.lp, self.radius = _evaluate(self.state, *self.inputs)
        if (not np.isfinite(self.lp) or not np.isfinite(self.radius)
                or (self.extended and not np.all(np.isfinite([self.first, self.second, self.third])))):
            raise ValueError('Nonfinite native initial density or physical radius')
        # Compile with zero transitions and a disposable RNG before timing.
        if self.extended:
            _run_block_extended(np.random.default_rng(0), self.state.copy(), self.lp, self.radius,
                self.first, self.second, self.third, self.scale, 0, self.thin, self.threshold, self.coordinate_threshold,
                self.reflected, *self.inputs)
        else:
            _run_block(np.random.default_rng(0), self.state.copy(), self.lp, self.radius,
                       self.scale, 0, self.thin, self.threshold, self.reflected, *self.inputs)
        self.numba_compile_seconds = time.perf_counter() - started
        self.compile_seconds = float(calibration.compile_seconds) + self.numba_compile_seconds
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), PRODUCTION_DOMAIN]))
        self.implementation = ('Optimized Numba DCS production; unchanged JAX pretuning; cached current '
                               'density and norm; fused300-transition all-state radial accumulation; float64 without fast-math')
        self.tuning_metadata.update(
            native_compile_seconds=self.numba_compile_seconds,
            calibration_backend='Identical JAX calibration and frozen starting state',
            production_rng='NumPy PCG64; SeedSequence([chain_seed,0x4E424457]); separate from calibration',
            native_statistic_policy='Every post-Metropolis state contributes; current norm reused on rejection')
        if self.coordinate_metrics:
            self.implementation = self.implementation.replace('density and norm', 'density, norm and first three coordinates')
            self.implementation = self.implementation.replace('radial accumulation', 'radial and coordinate accumulation')
            self.tuning_metadata['native_statistic_policy'] = (
                'Every post-Metropolis state contributes; current norm and first three coordinates reused on rejection')

    def batch(self):
        if self.extended:
            self.lp, self.radius, self.first, self.second, self.third, prefixes, saved = _run_block_extended(
                self.rng, self.state, self.lp, self.radius, self.first, self.second, self.third, self._scale, BATCH_SIZE,
                self.thin, self.threshold, self.coordinate_threshold, self.reflected, *self.inputs)
            return prefixes, saved
        self.lp, self.radius, prefixes, norms = _run_block(
            self.rng, self.state, self.lp, self.radius, self._scale, BATCH_SIZE,
            self.thin, self.threshold, self.reflected, *self.inputs)
        return prefixes, norms
