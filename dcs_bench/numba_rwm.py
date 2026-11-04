"""Optimized ordinary DCS-RWM with the identical frozen JAX calibration.

The Gaussian coordinate standard deviation is unchanged. Proposals outside
the ball are rejected; this is neither a reflected nor a rotation mixture.
"""
from .numba_billiard import _NumbaBallProduction


class NumbaRWM(_NumbaBallProduction):
    def __init__(self, mapping, config, seed, calibration):
        super().__init__(mapping, config, seed, calibration, billiard=False)
