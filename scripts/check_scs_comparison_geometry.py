"""Read-only numerical checks for the 100D SCS comparison geometry.

Run with the repository Python environment:
    python scripts/check_scs_comparison_geometry.py

Uses the compact, fixed fitted-map fixture tests/fixtures/scs_geometry_fits.json.
Checks the production-map Jacobians and common physical initialization, then
compares SCS cap step-out with independently iterated first-return and reverse
geodesic paths. These calculations do not measure performance or rank samplers.

The third JSON line explicitly labels constructed extreme-roundoff examples in
the unchanged upstream SCS function. Nonfinite outputs in those examples are
intentional demonstrations, not evidence that ordinary benchmark runs fail.
No production source or experiment results are modified. The numerical
operations and RNG draw order match the archived independent audit snippets.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from dcs_bench import scs_paper, timed_radial
from dcs_bench.targets import Target

scs_paper._author()
import jax
import jax.numpy as jnp

core = scs_paper._author()
rng = np.random.default_rng(147829)
s = json.loads((ROOT / 'tests/fixtures/scs_geometry_fits.json').read_text())
target = Target(s['config']['target'])
errors = {}
for method, family in [('DCS-BW', 'ball'), ('SCS', 'sphere')]:
    mapping = s['maps'][family]
    _, projection, initial, density = timed_radial._components(mapping, target.spec, method)
    jacobian = jax.jit(jax.jacfwd(projection))
    points = []
    if family == 'ball':
        for radius in (.5, .99, .99999):
            direction = rng.normal(size=100)
            direction /= np.linalg.norm(direction)
            points.append(radius * direction)
    else:
        for height in (-.5, .0, .09999):
            direction = rng.normal(size=100)
            direction /= np.linalg.norm(direction)
            points.append(np.r_[np.sqrt(1-height**2)*direction, height+1.])
    residuals = []
    for point in points:
        x = jnp.array(point)
        physical = projection(x)
        matrix = np.asarray(jacobian(x))
        if family == 'sphere':
            normal = point.copy()
            normal[-1] -= 1
            basis = np.linalg.svd(normal[None, :], full_matrices=True)[2][1:].T
            matrix = matrix @ basis
        observed = np.linalg.slogdet(matrix)[1]
        recorded = float(density(x) - target.log_prob(physical, 'jax'))
        residuals.append(float(abs(observed-recorded)))
    errors[method] = {
        'surface_or_volume_logjac_errors': residuals,
        'initial_physical_max_error': float(np.max(np.abs(np.asarray(projection(initial))-1.))),
    }
print(json.dumps(errors))

# For a cap-crossing step the reverse geodesic, using the same angular step,
# must skip the same dark points and return to the initial bright point.
checks = []
for d in (2, 5, 100):
    n = 3000
    x = rng.normal(size=(n, d+1))
    x /= np.linalg.norm(x, axis=1)[:, None]
    x = x[x[:, -1] < .1]
    v = rng.normal(size=x.shape)
    v -= x * np.sum(x*v, axis=1)[:, None]
    v /= np.linalg.norm(v, axis=1)[:, None]
    alpha = rng.uniform(.001, np.pi/2-1e-5, len(x))
    y = x*np.cos(alpha)[:, None] + v*np.sin(alpha)[:, None]
    crossing = y[:, -1] >= .1
    x, v, alpha, y = x[crossing], v[crossing], alpha[crossing], y[crossing]
    f = jax.jit(jax.vmap(lambda a, b: core.stepout_dark_side(a, b, .1)))
    endpoint = np.asarray(f(jnp.array(x), jnp.array(y)))
    # Independently count repeated rotations until first bright visit.
    k = np.ones(len(x), dtype=int)
    expected = y.copy()
    for _ in range(10000):
        dark = expected[:, -1] >= .1
        if not np.any(dark):
            break
        k[dark] += 1
        expected[dark] = (x[dark]*np.cos(k[dark]*alpha[dark])[:, None]
                          + v[dark]*np.sin(k[dark]*alpha[dark])[:, None])
    else:
        raise RuntimeError('Unexpected long first-return path')
    tangent_end = -x*np.sin(k*alpha)[:, None] + v*np.cos(k*alpha)[:, None]
    reverse = endpoint*np.cos(alpha)[:, None] - tangent_end*np.sin(alpha)[:, None]
    recross = reverse[:, -1] >= .1
    returned = reverse.copy()
    returned[recross] = np.asarray(f(jnp.array(endpoint[recross]), jnp.array(reverse[recross])))
    checks.append({
        'dimension': d, 'crossings': len(x),
        'max_first_return_error': float(np.max(np.abs(endpoint-expected))),
        'max_reverse_error': float(np.max(np.abs(returned-x))),
        'nonfinite': int(np.sum(~np.all(np.isfinite(endpoint), axis=1))),
        'outside': int(np.sum(endpoint[:, -1] >= .1)),
    })
print(json.dumps({'scs_reverse_path': checks}))

# The second original snippet: constructed tiny-angle upstream edge cases.
# These are not observations of failures in the ordinary benchmark runs.
rows = []
for gap in (1e-8, 1e-10, 1e-12, 1e-14):
    h = .1-gap
    x = np.array([np.sqrt(1-h*h), 0., h])
    v = np.array([-h, 0., np.sqrt(1-h*h)])
    theta = 2*gap/np.sqrt(1-h*h)
    y = x*np.cos(theta) + v*np.sin(theta)
    out = np.asarray(core.stepout_dark_side(jnp.asarray(x), jnp.asarray(y), .1))
    rows.append({
        'cap_gap': gap, 'actual_angle': theta,
        'dot': float(jnp.dot(jnp.asarray(x), jnp.asarray(y))),
        'crosses': bool(y[-1] > .1),
        'finite_output': bool(np.all(np.isfinite(out))),
    })
print(json.dumps({'upstream_extreme_roundoff_examples': rows,
                  'scope': 'Constructed numerical edge cases in unchanged author SCS; current benchmark runs are not shown affected.'}))
