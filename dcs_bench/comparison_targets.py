"""Prespecified Student-t targets for the JAX and optimized-Numba suites.

Target definitions and thresholds do not depend on pilot outcomes or on
whether map fitting is enabled.
"""
from copy import deepcopy
import math

TARGET_IDS = ('skew-t-100d', 'skew-t-500d')

_TARGETS = {
    'skew-t-100d': {
        'id': 'skew-t-100d',
        'title': '100D skewed Student-t (nu=3)',
        'target': {'kind': 'skew_t', 'dimension': 100, 'nu': 3.,
                   'alpha': {'0': 100., '1': -100.}},
        'beta': 2., 'threshold': 50., 'coordinate_threshold': 3.,
        'reference_note': 'Analytic multivariate-t radial law; the normal-CDF sign skew leaves this law unchanged.',
        'threshold_policy': 'Original fixed radial threshold 50.',
    },
    'skew-t-500d': {
        'id': 'skew-t-500d',
        'title': '500D skewed Student-t (nu=3)',
        'target': {'kind': 'skew_t', 'dimension': 500, 'nu': 3.,
                   'alpha': {'0': 100., '1': -100.}},
        'beta': 2., 'threshold': 50.*math.sqrt(5.), 'coordinate_threshold': 3.,
        'reference_note': 'Analytic multivariate-t radial law; the normal-CDF sign skew leaves this law unchanged.',
        'threshold_policy': 'Original threshold 50 multiplied by sqrt(d/100); fixed before sampling.',
    },

}


def comparison_target(target_id):
    """Return independent config data; callers cannot mutate the target registry."""
    if target_id not in _TARGETS:
        raise ValueError(f'Unknown comparison target: {target_id}; choose one of {TARGET_IDS}')
    return deepcopy(_TARGETS[target_id])
