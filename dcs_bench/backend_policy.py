"""Lightweight backend declarations shared by configuration and native kernels."""

NUMBA_TARGETS = {'skew_t': 0, 'logistic': 1, 'eight_schools': 2, 'garch': 3}
NUMBA_ALGORITHMS = {'DCS-RWM': 0, 'DCS-BW': 1, 'DCS-HnR': 2,
                    'RWM-Gaussian': 0, 'RWM-stable': 3, 'SCS': 4}


def validate_backend_configuration(config):
    """Require the declared native backend instead of silently falling back."""
    requested = config.get('metropolis_backend', 'auto')
    if requested not in ('auto', 'numba'):
        raise ValueError('metropolis_backend must be auto or numba')
    if requested == 'numba':
        unsupported_targets = [target['kind'] for target in config['targets']
                               if target['kind'] not in NUMBA_TARGETS]
        unsupported_methods = [method for method in config['algorithms']
                               if method != 'NUTS' and method not in NUMBA_ALGORITHMS]
        if unsupported_targets or unsupported_methods:
            raise ValueError('Required Numba backend does not support targets '
                             f'{unsupported_targets} or algorithms {unsupported_methods}')
    targets_supported = all(target['kind'] in NUMBA_TARGETS for target in config['targets'])
    return {method: ('NumPyro NUTS (JAX)' if config['mode'] == 'nuts_matched_time'
                      else 'BlackJAX NUTS (JAX)') if method == 'NUTS' else
                    'Numba CPU' if targets_supported and method in NUMBA_ALGORITHMS else 'NumPy fallback'
            for method in config['algorithms']}
