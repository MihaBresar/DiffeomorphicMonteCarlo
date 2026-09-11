"""Independent diagnostics of the published stored map fits; no MCMC is run.

Run with the repository's pinned Python environment. Both projection families
are assessed on 100,000 fresh common base draws per family. Parameter ablations
are evaluations of already saved maps, not fits or choices based on chain MSE.

Example: python scripts/check_map_geometry.py --output-file results/map_geometry.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betainc, gammaln
from scipy.stats import f as f_distribution

from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import uniform_ball, uniform_cap


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sphere_forward_batch(z, parameters):
    """Vectorized exact formula used by SphereMap.forward and VI objective."""
    d, latitude, scale = parameters['dimension'], parameters['latitude'], parameters['scale']
    mu, observer = np.asarray(parameters['mu']), np.asarray(parameters['observer'])
    ell = z[:, -1] + 1
    m = (latitude - ell) / latitude
    yhat = (z[:, :-1] - (ell / latitude)[:, None] * observer) / m[:, None]
    v = yhat - observer
    numerator = m * np.sum(v * v, axis=1) + v @ observer + latitude - latitude ** 2 * (1 - m)
    assert np.all(m > 0) and np.all(numerator > 0)
    lj = d * np.log(scale) + np.log(numerator) - d * np.log(m) - np.log(latitude)
    return mu + scale * yhat, lj


def map_variants(parameters, shift_name, axis):
    mu, shift = np.asarray(parameters['mu']), np.asarray(parameters[shift_name])
    zero = np.zeros_like(mu).tolist()
    variants = {
        'full': parameters,
        'alpha_aligned': dict(parameters, mu=(axis * (mu @ axis)).tolist(),
                              **{shift_name: (axis * (shift @ axis)).tolist()}),
        'zero_mu': dict(parameters, mu=zero),
        'zero_shift': dict(parameters, **{shift_name: zero}),
        'scale_only': dict(parameters, mu=zero, **{shift_name: zero}),
        'vanilla': dict(parameters, mu=zero, scale=1., **{shift_name: zero}),
    }
    geometry = {}
    for name, vector in [('mu', mu), (shift_name, shift)]:
        geometry[name] = {'norm': float(np.linalg.norm(vector)),
                          'parallel_coefficient': float(vector @ axis),
                          'orthogonal_norm': float(np.linalg.norm(vector - axis * (vector @ axis)))}
    return variants, geometry


def diagnose_family(family, parameters, target, axis, seed, batches=10, batch_size=10000):
    shift_name = 'delta' if family == 'ball' else 'observer'
    variants, geometry = map_variants(parameters, shift_name, axis)
    losses = {key: [] for key in variants}
    rng = np.random.default_rng(seed)
    agreement = []
    for batch in range(batches):
        z = uniform_ball(rng, batch_size, target.d) if family == 'ball' else uniform_cap(rng, batch_size, target.d, parameters['latitude'])
        for key, value in variants.items():
            if family == 'ball':
                x, jacobian = BallMap(**value).forward(z)
            else:
                x, jacobian = sphere_forward_batch(z, value)
                if batch == 0:
                    checks = [SphereMap(**value).forward(point) for point in z[:100]]
                    reference_x = np.stack([p[0] for p in checks])
                    reference_j = np.asarray([p[1] for p in checks])
                    np.testing.assert_allclose(x[:100], reference_x, rtol=5e-13, atol=5e-13)
                    np.testing.assert_allclose(jacobian[:100], reference_j, rtol=5e-13, atol=5e-13)
                    agreement.append({'variant': key, 'points': 100,
                                      'max_forward_absolute_error': float(np.max(np.abs(x[:100] - reference_x))),
                                      'max_log_jacobian_absolute_error': float(np.max(np.abs(jacobian[:100] - reference_j)))})
            losses[key].append(-target.log_prob(x) - jacobian)
    losses = {key: np.concatenate(values) for key, values in losses.items()}
    d = target.d
    log_volume = d / 2 * np.log(np.pi) - gammaln(d / 2 + 1) if family == 'ball' else (
        np.log(2) + (d + 1) / 2 * np.log(np.pi) - gammaln((d + 1) / 2)
        + np.log(betainc(d / 2, d / 2, parameters['latitude'] / 2)))
    rows = {}
    for key, values in losses.items():
        difference = values - losses['full']
        se = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
        mean = float(np.mean(values))
        diff = float(np.mean(difference))
        rows[key] = {'loss_mean': mean, 'loss_mc_se': float(np.std(values, ddof=1) / np.sqrt(len(values))),
                     'estimated_reverse_kl': float(mean - log_volume),
                     'paired_loss_difference_vs_full': diff, 'paired_difference_mc_se': se,
                     'paired_normal_95_mc_interval': [diff - 1.959963984540054 * se, diff + 1.959963984540054 * se]}
    return {'seed': seed, 'batches': batches, 'batch_size': batch_size, 'draws_per_variant': batches * batch_size,
            'base_log_volume': float(log_volume), 'fitted_scale': parameters['scale'],
            'fitted_parameter_geometry': geometry,
            'variants': variants, 'map_hashes': {key: digest(value) for key, value in variants.items()},
            'rows': rows, 'vectorized_sphere_formula_checks': agreement}


def symmetric_radial_geometry():
    d, nu = 100, 3.

    def score(scale):
        expectation = quad(lambda q: d / 2 * q ** (d / 2 - 1) * scale ** 2 * q /
                           (nu * (1 - q) + scale ** 2 * q), 0., 1., epsabs=1e-12)[0]
        return d - (d + nu) * expectation

    optimum = brentq(score, .1, 10.)
    probabilities = [.05, .5, .95, .99]
    physical = np.sqrt(d * f_distribution.ppf(probabilities, d, nu))
    rows = []
    for scale in [1., optimum, float(np.sqrt(nu))]:
        radius = physical / np.sqrt(scale ** 2 + physical ** 2)
        rows.append({'scale': scale, 'transformed_radius_quantiles': radius.tolist(),
                     'boundary_distance_at_physical_quantiles': (1 - radius).tolist()})
    return {'d': d, 'nu': nu, 'beta': 2., 'mu': 0., 'delta': 0.,
            'transformed_density_up_to_constant': '(1-r^2)^((nu-2)/2) * [nu+(R^2-nu)*r^2]^(-(nu+d)/2) * R^d',
            'uniform_ball_radius_squared_distribution': 'Beta(d/2, 1)',
            'reverse_kl_stationary_equation': 'd - (nu+d) E[R^2 q / (nu(1-q)+R^2 q)] = 0, q~Beta(d/2,1)',
            'reverse_kl_optimal_scalar_R': optimum,
            'stationary_equation_residual': score(optimum), 'quantile_probabilities': probabilities,
            'physical_radius_quantiles': physical.tolist(), 'transformed_quantiles': rows,
            'exact_uniform_ball_control': {'nu': 2., 'beta': 2., 'R': float(np.sqrt(2)), 'mu': 0., 'delta': 0.,
                'note': 'The transformed density is constant on the unit ball; mean-norm MSE is not finite for this nu=2 control.'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-file', type=Path, required=True,
                        help='New JSON file; existing files are never overwritten.')
    args = parser.parse_args(argv)
    destination = args.output_file.resolve()
    if destination.exists():
        parser.error(f'Use a new output file: {destination}')
    inputs = ROOT / 'validation/fixed_step_maps.json'
    saved = json.loads(inputs.read_text())
    spec = {'kind': 'skew_t', 'dimension': 100, 'nu': 3, 'alpha': {'0': 100, '1': -100}}
    if saved['config']['targets'] != [spec]:
        raise ValueError('The published maps must describe the declared 100D nu=3 skew-t target.')
    input_sha256 = sha256_file(inputs)
    paths = [Path(__file__), ROOT / 'dcs_bench/targets.py', ROOT / 'dcs_bench/transforms.py',
             ROOT / 'dcs_bench/tuning.py']
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    target = Target(spec)
    axis = target.alpha / np.linalg.norm(target.alpha)
    output = {'purpose': 'Independent map-objective and radial-geometry diagnostics; no MCMC comparison or production-MSE tuning.',
              'target': spec, 'input_file': inputs.relative_to(ROOT).as_posix(), 'input_sha256': input_sha256,
              'source_sha256': hashes,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'ball': diagnose_family('ball', saved['maps']['current']['ball'], target, axis, 569311),
              'sphere': diagnose_family('sphere', saved['maps']['current']['sphere'], target, axis, 569312),
              'symmetric_radial_geometry': symmetric_radial_geometry(),
              'limitations': [
                  'One saved map fit per family; results are conditional on those fits.',
                  'The same fresh base draws are shared by map variants within each family, but ball and cap bases differ.',
                  'Reported Monte Carlo intervals describe independent-base integration uncertainty, not uncertainty from refitting or from MCMC.',
                  'Lower reverse KL does not imply lower asymptotic variance or faster mixing for a specific transition kernel.',
                  'Projection onto alpha is justified as an explicit synthetic-target symmetry ablation; it is not a generic posterior-map fix.',
                  'Existing symmetric-target VI constrains centres and shifts to zero, so it tests fitted scalar R rather than a nonzero Mobius deformation.',
              ]}
    if input_sha256 != sha256_file(inputs) or hashes != {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}:
        raise RuntimeError('A source or input file changed during the calculation.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for family in ['ball', 'sphere']:
        rows = output[family]['rows']
        print(f'{family}: full KL={rows["full"]["estimated_reverse_kl"]!r}; '
              f'projected-minus-full loss={rows["alpha_aligned"]["paired_loss_difference_vs_full"]!r}; '
              f'paired MC SE={rows["alpha_aligned"]["paired_difference_mc_se"]!r}')
    print('Report:', destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
