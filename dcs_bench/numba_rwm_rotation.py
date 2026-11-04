"""Optimized reflected Gaussian RWM / angular-rotation mixture.

The existing JAX pilot supplies the initial state and both frozen scales.
Production chooses exactly one update per iteration. Gaussian increments are
reflected on their ball chord; angular proposals retain the fitted-centre
radius and the required Mobius proposal correction. All post-MH states enter
statistics, including rejection repeats. This module does not alter BW paths.
"""
from copy import deepcopy
import time

import numpy as np
from numba import njit

from . import numba_radial as radial
from .numba_rotation import _mobius_into, _angular_proposal_from_w

PRODUCTION_DOMAIN = 0x4E52524F  # NRRO; distinct from BW rotation and calibration.


@njit(nogil=True, error_model='numpy')
def _run_block(rng, state, w, lp, radius, first, second, third, logmo,
               gaussian_scale, angular_scale, count, thin, threshold, coordinate_threshold,
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
        normals = rng.standard_normal(d)
        logu = np.log(rng.random())
        correction = 0.
        if angular:
            valid, next_logmo = _angular_proposal_from_w(
                w, normals, angular_scale, delta, delta2, logdelta, next_w, proposed)
            if valid:
                correction = logmo - next_logmo
        else:
            for j in range(d):
                increment[j] = gaussian_scale * normals[j]
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


class NumbaRWMRotation:
    """Native production using the exact calibrated JAX state and two scales."""
    def __init__(self, mapping, config, seed, calibration):
        if config['algorithm'] != 'DCS-RWM-rotate':
            raise ValueError('NumbaRWMRotation requires DCS-RWM-rotate')
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
        self.initial_scale = float(getattr(calibration, 'initial_scale',
            float(config.get('initial_scale', .1)) / np.sqrt(len(self.state)+2.)))
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
        self.implementation = ('Optimized Numba 50/50 reflected-Gaussian-RWM/rotation mixture; cached density, '
            'Mobius state, physical norm and first three coordinates; fused all-state accumulation; '
            'shared JAX pretuning; float64 without fast-math')
        self.tuning_metadata.update(native_compile_seconds=self.numba_compile_seconds,
            calibration_backend='Identical JAX calibration, two frozen scales and starting state',
            production_rng='NumPy PCG64; SeedSequence([chain_seed,0x4E52524F]); separate from calibration',
            mixture_angular_probability=.5, updates_per_iteration=1,
            gaussian_step_sd=self.scale, gaussian_step_rms=float(self.scale*np.sqrt(len(self.state))),
            proposal_length_policy='Isotropic Gaussian displacement with per-coordinate SD sigma; chord reflection',
            angular_scale=self.angular_scale, angular_angle_radians=float(2. * np.arctan(self.angular_scale)),
            angular_radius_policy='Preserve distance from the fitted physical centre, rotating after Mobius',
            native_statistic_policy='Every post-Metropolis state contributes, including cached rejection repeats')

    def batch(self):
        self.lp, self.radius, self.first, self.second, self.third, self.logmo, prefixes, saved = _run_block(
            self.rng, self.state, self.w, self.lp, self.radius, self.first, self.second, self.third, self.logmo,
            self.scale, self.angular_scale, radial.BATCH_SIZE, self.thin, self.threshold,
            self.coordinate_threshold, *self.inputs)
        return (prefixes, saved) if self.coordinate_metrics else (prefixes[:, :3], saved[:, 0])
