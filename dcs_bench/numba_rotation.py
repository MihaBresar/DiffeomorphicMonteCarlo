"""Native reflected-BW/rotation mixture with shared, frozen JAX pretuning.

Each iteration selects exactly one component with probability one half. The
rotation acts after the Mobius transformation, preserving physical distance
from the fitted centre. Its proposal is symmetric there, not in raw ball
coordinates, so its Metropolis ratio removes the Mobius Jacobian change.
"""
from copy import deepcopy
import time

import numpy as np
from numba import njit

from . import numba_radial as radial

PRODUCTION_DOMAIN = 0x4E42524F  # NBRO; independent of calibration and older kernels.


@njit(nogil=True, error_model='numpy')
def _mobius_into(state, delta, delta2, logdelta, output):
    r2, dot = 0., 0.
    for j in range(len(state)):
        r2 += state[j] * state[j]
        dot += state[j] * delta[j]
    denominator = 1. + 2. * dot + delta2 * r2
    if not np.isfinite(r2) or r2 >= 1. or not np.isfinite(denominator) or denominator <= 0.:
        return np.nan
    for j in range(len(state)):
        output[j] = ((1. - delta2) * state[j]
                     + (1. + 2. * dot + r2) * delta[j]) / denominator
    return logdelta - len(state) * np.log(denominator)


@njit(nogil=True, error_model='numpy')
def _angular_proposal_from_w(w, normal, angular_scale, delta, delta2, logdelta,
                             next_w, proposed):
    """Fill a rotated proposal; return validity and its forward Mobius log-J."""
    r2, dot = 0., 0.
    for j in range(len(w)):
        r2 += w[j] * w[j]
        dot += w[j] * normal[j]
    if (not np.isfinite(r2) or r2 <= 0. or r2 >= 1.
            or not np.isfinite(angular_scale) or angular_scale <= 0.):
        return False, np.nan
    tangent2 = 0.
    for j in range(len(w)):
        next_w[j] = normal[j] - w[j] * dot / r2
        tangent2 += next_w[j] * next_w[j]
    if not np.isfinite(tangent2) or tangent2 <= 0.:
        return False, np.nan
    theta = 2. * np.arctan(angular_scale)
    tangent_scale = np.sin(theta) * np.sqrt(r2 / tangent2)
    cosine = np.cos(theta)
    next_r2, next_dot = 0., 0.
    for j in range(len(w)):
        next_w[j] = cosine * w[j] + tangent_scale * next_w[j]
        next_r2 += next_w[j] * next_w[j]
        next_dot += next_w[j] * delta[j]
    if not np.isfinite(next_r2) or next_r2 >= 1.:
        return False, np.nan
    # M_delta^{-1}=M_{-delta}; no clipping or radial projection is applied.
    denominator = 1. - 2. * next_dot + delta2 * next_r2
    if not np.isfinite(denominator) or denominator <= 0.:
        return False, np.nan
    proposed2, proposed_dot = 0., 0.
    for j in range(len(w)):
        proposed[j] = ((1. - delta2) * next_w[j]
                       - (1. - 2. * next_dot + next_r2) * delta[j]) / denominator
        proposed2 += proposed[j] * proposed[j]
        proposed_dot += proposed[j] * delta[j]
    new_denominator = 1. + 2. * proposed_dot + delta2 * proposed2
    valid = (np.isfinite(proposed2) and proposed2 < 1.
             and np.isfinite(new_denominator) and new_denominator > 0.)
    return valid, logdelta - len(w) * np.log(new_denominator) if valid else np.nan


@njit(nogil=True, error_model='numpy')
def angular_proposal(state, normal, angular_scale, delta):
    """Deterministic proposal oracle shared with the JAX parity tests."""
    delta2 = np.dot(delta, delta)
    logdelta = len(state) * np.log1p(-delta2)
    w, next_w, proposed = np.empty_like(state), np.empty_like(state), state.copy()
    old_logmo = _mobius_into(state, delta, delta2, logdelta, w)
    if not np.isfinite(old_logmo):
        return state.copy(), 0., False
    valid, new_logmo = _angular_proposal_from_w(
        w, normal, angular_scale, delta, delta2, logdelta, next_w, proposed)
    if not valid:
        return state.copy(), 0., False
    return proposed, old_logmo - new_logmo, True


@njit(nogil=True, error_model='numpy')
def _run_block(rng, state, w, lp, radius, first, second, third, logmo,
               bw_scale, angular_scale, count, thin, threshold, coordinate_threshold,
               mu, delta, alpha, constants, skewed):
    """Fused all-state statistics, with cached values after rejection."""
    d = len(state)
    prefixes = np.zeros((count // radial.PREFIX_STRIDE, 9))
    saved = np.empty((count // thin, 4))
    increment, proposed, next_w = np.empty(d), np.empty(d), np.empty(d)
    totals = np.zeros(9)
    delta2, logdelta = constants[3], constants[5]
    for i in range(count):
        angular = rng.random() < .5
        normals = rng.standard_normal(d + 2)
        logu = np.log(rng.random())
        correction = 0.
        if angular:
            valid, next_logmo = _angular_proposal_from_w(
                w, normals, angular_scale, delta, delta2, logdelta, next_w, proposed)
            if valid:
                correction = logmo - next_logmo
        else:
            full2 = 0.
            for j in range(d + 2):
                full2 += normals[j] * normals[j]
            factor = bw_scale / np.sqrt(full2)
            for j in range(d):
                increment[j] = factor * normals[j]
            valid = radial._proposal_into(state, increment, True, proposed)
        accepted = False
        if valid:
            next_lp, next_radius, next_first, next_second, next_third = radial._evaluate_extended(
                proposed, mu, delta, alpha, constants, skewed)
            if logu < next_lp - lp + correction:
                # Refresh the cache from the represented new state, rather
                # than accumulating forward/inverse roundoff in a second state.
                next_logmo = _mobius_into(proposed, delta, delta2, logdelta, next_w)
                if np.isfinite(next_logmo):
                    for j in range(d):
                        state[j], w[j] = proposed[j], next_w[j]
                    lp, radius = next_lp, next_radius
                    first, second, third, logmo = next_first, next_second, next_third, next_logmo
                    accepted = True
        totals[0] += int(radius >= threshold)
        totals[1] += radius
        totals[2] += int(accepted)
        totals[3] += first
        totals[4] += int(first > coordinate_threshold)
        totals[5] += second
        totals[6] += int(second < -coordinate_threshold)
        totals[7] += third
        totals[8] += int(third > coordinate_threshold)
        if (i + 1) % radial.PREFIX_STRIDE == 0:
            prefixes[(i + 1) // radial.PREFIX_STRIDE - 1] = totals
        if (i + 1) % thin == 0:
            row = (i + 1) // thin - 1
            saved[row, 0], saved[row, 1] = radius, first
            saved[row, 2], saved[row, 3] = second, third
    return lp, radius, first, second, third, logmo, prefixes, saved


class NumbaRotation:
    """Native production using the exact calibrated JAX state and two scales."""
    def __init__(self, mapping, config, seed, calibration):
        if config['algorithm'] != 'DCS-BW-rotate':
            raise ValueError('NumbaRotation requires DCS-BW-rotate')
        self.inputs = radial._inputs(mapping, config)
        self.coordinate_metrics = bool(config.get('coordinate_metrics', False))
        self.state = np.array(calibration.state, dtype=np.float64, order='C', copy=True)
        minimum_dimension = 3 if self.coordinate_metrics else 2
        if (self.state.shape != (len(self.inputs[0]),) or len(self.state) < minimum_dimension
                or not np.all(np.isfinite(self.state))):
            raise ValueError('Rotation requires a finite calibrated ball state of compatible dimension')
        self.scale = self._scale = float(calibration.scale)
        self.angular_scale = float(calibration.angular_scale)
        if not np.all(np.isfinite([self.scale, self.angular_scale])) or min(self.scale, self.angular_scale) <= 0.:
            raise ValueError('Both calibrated proposal scales must be positive and finite')
        self.thin = int(config['thin'])
        if self.thin < 1 or radial.PREFIX_STRIDE % self.thin:
            raise ValueError('Native thin must divide 100')
        self.threshold = float(config.get('threshold', 50.))
        self.coordinate_threshold = float(config.get('coordinate_threshold', 3.))
        if not np.all(np.isfinite([self.threshold, self.coordinate_threshold])):
            raise ValueError('Measurement thresholds must be finite')
        self.warmup_seconds = float(calibration.warmup_seconds)
        self.warmup_acceptance = calibration.warmup_acceptance
        self.initial_scale = float(getattr(calibration, 'initial_scale', config.get('initial_scale', .1)))
        self.tuning_policy = getattr(calibration, 'tuning_policy', None)
        self.tuning_metadata = deepcopy(getattr(calibration, 'tuning_metadata', {}))
        started = time.perf_counter()
        self.lp, self.radius, self.first, self.second, self.third = radial._evaluate_extended(self.state, *self.inputs)
        self.w = np.empty_like(self.state)
        self.logmo = _mobius_into(self.state, self.inputs[1], self.inputs[3][3], self.inputs[3][5], self.w)
        if not np.all(np.isfinite([self.lp, self.radius, self.first, self.second, self.third, self.logmo])):
            raise ValueError('Nonfinite native initial density or coordinates')
        _run_block(np.random.default_rng(0), self.state.copy(), self.w.copy(), self.lp,
            self.radius, self.first, self.second, self.third, self.logmo, self.scale, self.angular_scale,
            0, self.thin, self.threshold, self.coordinate_threshold, *self.inputs)
        self.numba_compile_seconds = time.perf_counter() - started
        self.compile_seconds = float(calibration.compile_seconds) + self.numba_compile_seconds
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), PRODUCTION_DOMAIN]))
        self.implementation = ('Optimized Numba 50/50 reflected-BW/rotation mixture; cached density, '
            'Mobius state, physical norm and first three coordinates; fused all-state accumulation; '
            'shared JAX pretuning; float64 without fast-math')
        self.tuning_metadata.update(native_compile_seconds=self.numba_compile_seconds,
            calibration_backend='Identical JAX calibration, two frozen scales and starting state',
            production_rng='NumPy PCG64; SeedSequence([chain_seed,0x4E42524F]); separate from calibration',
            mixture_angular_probability=.5, updates_per_iteration=1,
            angular_scale=self.angular_scale, angular_angle_radians=float(2. * np.arctan(self.angular_scale)),
            angular_radius_policy='Preserve distance from the fitted physical centre, rotating after Mobius',
            native_statistic_policy='Every post-Metropolis state contributes, including cached rejection repeats')

    def batch(self):
        self.lp, self.radius, self.first, self.second, self.third, self.logmo, prefixes, saved = _run_block(
            self.rng, self.state, self.w, self.lp, self.radius, self.first, self.second, self.third, self.logmo,
            self.scale, self.angular_scale, radial.BATCH_SIZE, self.thin, self.threshold,
            self.coordinate_threshold, *self.inputs)
        return (prefixes, saved) if self.coordinate_metrics else (prefixes[:, :3], saved[:, 0])
