"""Minimal sampler interface independent of the benchmark catalogue."""
import numpy as np
from .transforms import BallMap
from .samplers import run_metropolis


def sample(log_density, initial, draws=10000, warmup=1500, seed=0, beta=2.,
           algorithm='DCS-BW', mapping=None, thin=1):
    """Return physical-space samples and diagnostics for a NumPy log density.

    The callable accepts one finite vector and returns an unnormalized log
    density. It must include any constraint-transform Jacobian the caller uses.
    Gradients are unnecessary. Warmup adapts only the proposal scale; map
    parameters stay fixed throughout this call.
    """
    initial_point = np.asarray(initial, dtype=float)
    if initial_point.ndim != 1 or not np.all(np.isfinite(initial_point)):
        raise ValueError('initial must be a finite vector')
    if draws < 4 or warmup < 1 or thin < 1:
        raise ValueError('Require draws >= 4, warmup >= 1, thin >= 1')
    if algorithm not in ('DCS-BW', 'DCS-RWM', 'DCS-HnR'):
        raise ValueError('Choose DCS-BW, DCS-RWM, or DCS-HnR')
    class CallableTarget:
        kind = 'custom'
        d = len(initial_point)
        def initial(self): return initial_point.copy()
        def log_prob(self, x): return log_density(x)
        def physical(self, x): return x
    target = CallableTarget()
    mapping = mapping or BallMap(target.d, beta=beta)
    config = dict(warmup=warmup, iterations=draws*thin, thin=thin)
    return run_metropolis(target, algorithm, mapping, config, seed)
