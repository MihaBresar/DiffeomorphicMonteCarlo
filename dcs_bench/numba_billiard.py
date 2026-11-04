"""Fused native production for the reviewed post-Mobius billiard proposal.

The JAX pilot supplies the state and frozen scale. Production uses a separate
PCG64 stream, caches the current density/radius after rejections, and records
every post-Metropolis state. No fast-math or target-specific clipping is used.
"""
from copy import deepcopy
import time

import numpy as np
from numba import njit

from . import numba_radial as radial

BILLIARD_DOMAIN = 0x4E42494C  # NBIL, independent of calibration and older methods.
RWM_DOMAIN = 0x4E52574D       # NRWM.


@njit(nogil=True, error_model='numpy')
def _proposal_into(state, increment, proposed):
    """Same invariant complete-chord extension as billiard.reflect_ball_billiard."""
    radius2, length2, dot = 0., 0., 0.
    for j in range(len(state)):
        radius2 += state[j] * state[j]
        length2 += increment[j] * increment[j]
        dot += state[j] * increment[j]
    if not radius2 < 1. or not np.isfinite(length2):
        return False
    length = np.sqrt(length2)
    if length == 0.:
        raw2 = 0.
        for j in range(len(state)):
            proposed[j] = state[j] + increment[j]
            raw2 += proposed[j] * proposed[j]
        return raw2 < 1.
    position = dot / length
    impact2 = 0.
    for j in range(len(state)):
        transverse = state[j] - position * increment[j] / length
        impact2 += transverse * transverse
    impact = np.sqrt(impact2)
    half = np.sqrt((1. - radius2) + position * position)
    chord = 2. * half
    psi = 2. * np.arctan2(half, impact)
    theta = np.arctan2(length, chord)
    extra = np.floor(max(0., theta / psi - length / chord))
    unfolded = position + length + half
    count = np.floor(unfolded / chord) + extra
    along = unfolded % chord - half
    if (not np.isfinite(half) or half <= 0. or not np.isfinite(psi) or psi <= 0.
            or not np.isfinite(theta) or not np.isfinite(extra) or extra < 0.
            or not np.isfinite(unfolded) or not np.isfinite(count)
            or count < 0. or count >= float(2**53)):
        return False
    if count == 0.:
        norm2 = 0.
        for j in range(len(state)):
            proposed[j] = state[j] + increment[j]
            norm2 += proposed[j] * proposed[j]
        return norm2 < 1.
    if impact <= half:
        parity = 1. - 2. * (count % 2.)
        angle = count * (2. * np.arctan2(impact, half))
        cosine, sine = parity * np.cos(angle), -parity * np.sin(angle)
        coefficient = sine / impact if impact > 0. else -parity * 2. * count / half
    else:
        angle = count * psi
        cosine, sine = np.cos(angle), np.sin(angle)
        coefficient = sine / impact
    if not np.isfinite(angle):
        return False
    forward = along * cosine + impact * sine
    sideways = cosine - along * coefficient
    norm2 = 0.
    for j in range(len(state)):
        direction = increment[j] / length
        transverse = state[j] - position * direction
        proposed[j] = forward * direction + sideways * transverse
        norm2 += proposed[j] * proposed[j]
    return norm2 < 1.


@njit(nogil=True, error_model='numpy')
def _transition_from_increment(state, lp, radius, increment, logu, billiard,
                               mu, delta, alpha, constants, skewed, proposed):
    valid = (_proposal_into(state, increment, proposed) if billiard else
             radial._proposal_into(state, increment, False, proposed))
    if not valid:
        return lp, radius, False
    next_lp, next_radius = radial._evaluate(proposed, mu, delta, alpha, constants, skewed)
    if logu < next_lp - lp:
        for j in range(len(state)):
            state[j] = proposed[j]
        return next_lp, next_radius, True
    return lp, radius, False


@njit(nogil=True, error_model='numpy')
def _run_block(rng, state, lp, radius, scale, count, thin, threshold, billiard,
               mu, delta, alpha, constants, skewed):
    d = len(state)
    prefixes = np.zeros((count // radial.PREFIX_STRIDE, 3))
    norms = np.empty(count // thin)
    increment, proposed = np.empty(d), np.empty(d)
    tails, norm_sum, accepts = 0, 0., 0
    for i in range(count):
        if billiard:
            normals = rng.standard_normal(d + 2)
            full2 = 0.
            for j in range(d + 2):
                full2 += normals[j] * normals[j]
            factor = scale / np.sqrt(full2)
        else:
            normals = rng.standard_normal(d)
            factor = scale
        for j in range(d):
            increment[j] = factor * normals[j]
        lp, radius, accept = _transition_from_increment(state, lp, radius,
            increment, np.log(rng.random()), billiard, mu, delta, alpha, constants, skewed, proposed)
        tails += int(radius >= threshold)
        norm_sum += radius
        accepts += int(accept)
        if (i + 1) % radial.PREFIX_STRIDE == 0:
            row = (i + 1) // radial.PREFIX_STRIDE - 1
            prefixes[row, 0], prefixes[row, 1], prefixes[row, 2] = tails, norm_sum, accepts
        if (i + 1) % thin == 0:
            norms[(i + 1) // thin - 1] = radius
    return lp, radius, prefixes, norms


@njit(nogil=True, error_model='numpy')
def _run_block_extended(rng, state, lp, radius, first, second, third, scale, count, thin,
                        threshold, coordinate_threshold, billiard, mu, delta, alpha, constants, skewed):
    """Optional legacy-coordinate API; current comparisons use radial output."""
    d = len(state)
    prefixes = np.zeros((count // radial.PREFIX_STRIDE, 9))
    saved = np.empty((count // thin, 4))
    increment, proposed = np.empty(d), np.empty(d)
    totals = np.zeros(9)
    for i in range(count):
        if billiard:
            normals = rng.standard_normal(d + 2)
            full2 = 0.
            for j in range(d + 2):
                full2 += normals[j] * normals[j]
            factor = scale / np.sqrt(full2)
        else:
            normals = rng.standard_normal(d)
            factor = scale
        for j in range(d):
            increment[j] = factor * normals[j]
        logu = np.log(rng.random())
        valid = (_proposal_into(state, increment, proposed) if billiard else
                 radial._proposal_into(state, increment, False, proposed))
        accept = False
        if valid:
            next_lp, next_radius, next_first, next_second, next_third = radial._evaluate_extended(
                proposed, mu, delta, alpha, constants, skewed)
            if logu < next_lp - lp:
                for j in range(d):
                    state[j] = proposed[j]
                lp, radius = next_lp, next_radius
                first, second, third = next_first, next_second, next_third
                accept = True
        totals[0] += int(radius >= threshold)
        totals[1] += radius
        totals[2] += int(accept)
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
    return lp, radius, first, second, third, prefixes, saved


class _NumbaBallProduction:
    """Shared native driver; the supplied JAX calibration is never repeated."""
    def __init__(self, mapping, config, seed, calibration, *, billiard):
        expected = 'DCS-Billiard' if billiard else 'DCS-RWM'
        if config['algorithm'] != expected:
            raise ValueError(f'Native backend requires {expected}')
        self.billiard = billiard
        mapping = deepcopy(mapping)
        if billiard:
            mapping['delta'] = [0.] * int(mapping['dimension'])
        self.inputs = radial._inputs(mapping, config)
        self.coordinate_metrics = bool(config.get('coordinate_metrics', False))
        self.state = np.array(calibration.state, dtype=np.float64, order='C', copy=True)
        if self.state.shape != (len(self.inputs[0]),) or not np.all(np.isfinite(self.state)):
            raise ValueError('Invalid calibrated ball state')
        self.scale = self._scale = float(calibration.scale)
        if not np.isfinite(self.scale) or self.scale <= 0.:
            raise ValueError('Invalid calibrated proposal scale')
        self.thin = int(config['thin'])
        if self.thin < 1 or radial.PREFIX_STRIDE % self.thin:
            raise ValueError('Native thin must divide 100')
        self.threshold = float(config.get('threshold', 50.))
        self.coordinate_threshold = float(config.get('coordinate_threshold', 3.))
        if not np.all(np.isfinite([self.threshold, self.coordinate_threshold])):
            raise ValueError('Measurement thresholds must be finite')
        if self.coordinate_metrics and len(self.state) < 3:
            raise ValueError('Coordinate measurements require at least three dimensions')
        self.warmup_seconds = float(calibration.warmup_seconds)
        self.warmup_acceptance = calibration.warmup_acceptance
        self.initial_scale = float(getattr(calibration, 'initial_scale', config.get('initial_scale', .1)))
        self.tuning_policy = getattr(calibration, 'tuning_policy', None)
        self.tuning_metadata = deepcopy(getattr(calibration, 'tuning_metadata', {}))
        started = time.perf_counter()
        if self.coordinate_metrics:
            self.lp, self.radius, self.first, self.second, self.third = radial._evaluate_extended(self.state, *self.inputs)
            valid = np.all(np.isfinite([self.lp, self.radius, self.first, self.second, self.third]))
        else:
            self.lp, self.radius = radial._evaluate(self.state, *self.inputs)
            valid = np.all(np.isfinite([self.lp, self.radius]))
        if not valid:
            raise ValueError('Nonfinite native initial density or radius')
        if self.coordinate_metrics:
            _run_block_extended(np.random.default_rng(0), self.state.copy(), self.lp, self.radius,
                self.first, self.second, self.third, self.scale, 0, self.thin, self.threshold,
                self.coordinate_threshold, billiard, *self.inputs)
        else:
            _run_block(np.random.default_rng(0), self.state.copy(), self.lp, self.radius,
                self.scale, 0, self.thin, self.threshold, billiard, *self.inputs)
        self.numba_compile_seconds = time.perf_counter() - started
        self.compile_seconds = float(calibration.compile_seconds) + self.numba_compile_seconds
        domain = BILLIARD_DOMAIN if billiard else RWM_DOMAIN
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), domain]))
        self.implementation = ('Optimized Numba ' + expected + '; shared JAX calibration; cached density '
            'and radius; fused all-state accumulation; float64 without fast-math')
        self.tuning_metadata.update(native_compile_seconds=self.numba_compile_seconds,
            calibration_backend='Identical JAX calibration, frozen scale and starting state',
            production_rng=f'NumPy PCG64; SeedSequence([chain_seed,{domain:#x}]); separate from calibration',
            native_statistic_policy='Every post-Metropolis state contributes, including cached rejection repeats')
        if billiard:
            self.tuning_metadata.update(effective_map=mapping,
                native_boundary_policy='Same invariant-extra-chord specular billiard endpoint as JAX')
        else:
            self.tuning_metadata.update(scale_parameter='Gaussian coordinate standard deviation',
                native_boundary_policy='Direct rejection outside the ball, same as JAX DCS-RWM')

    def batch(self):
        if self.coordinate_metrics:
            self.lp, self.radius, self.first, self.second, self.third, prefixes, saved = _run_block_extended(
                self.rng, self.state, self.lp, self.radius, self.first, self.second, self.third,
                self._scale, radial.BATCH_SIZE, self.thin, self.threshold,
                self.coordinate_threshold, self.billiard, *self.inputs)
            return prefixes, saved
        self.lp, self.radius, prefixes, norms = _run_block(self.rng, self.state, self.lp, self.radius,
            self._scale, radial.BATCH_SIZE, self.thin, self.threshold, self.billiard, *self.inputs)
        return prefixes, norms


class NumbaBilliard(_NumbaBallProduction):
    """Native billiard in post-Mobius coordinates, retaining fitted centre/R/beta."""
    def __init__(self, mapping, config, seed, calibration):
        super().__init__(mapping, config, seed, calibration, billiard=True)
