"""Canonical targets used by every sampler and reference generator.

NumPy and JAX evaluate the same formulas, without clipping the skewing CDF
argument or adding epsilon constants to posterior densities.
"""
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]


def read_archive(path):
    path = Path(path)
    bundle = ROOT/'data/posteriordb'
    if path.is_relative_to(bundle):
        manifest = json.loads((bundle/'manifest.json').read_text())
        expected = manifest['files'][str(path.relative_to(bundle))]['sha256']
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'PosteriorDB checksum mismatch: {path}')
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.json')]
        if len(names) != 1:
            raise ValueError(f'Expected one JSON payload in {path}')
        return json.loads(archive.read(names[0]))


class Target:
    def __init__(self, spec):
        self.spec = dict(spec)
        self.kind = spec['kind']
        self.d = spec['dimension']
        if self.kind == 'skew_t':
            self.nu = float(spec['nu'])
            self.alpha = np.zeros(self.d)
            for index, value in spec.get('alpha', {}).items():
                self.alpha[int(index)] = value
            self.names = [f'x[{i+1}]' for i in range(self.d)]
        elif self.kind == 'logistic':
            self.nu = float(spec['nu'])
            rng = np.random.RandomState(spec.get('data_seed', 42))
            raw = rng.randn(spec['n_observations'], self.d)
            self.design = (raw-raw.mean(axis=0))/(2*raw.std(axis=0))
            self.y = (self.design[:, 0] > 0).astype(float)
            self.names = [f'beta[{i+1}]' for i in range(self.d)]
        elif self.kind == 'eight_schools':
            data = read_archive(ROOT/'data/posteriordb/data/data/eight_schools.json.zip')
            self.y, self.sigma = np.array(data['y']), np.array(data['sigma'])
            self.names = [f'theta[{i+1}]' for i in range(8)] + ['mu', 'tau']
        elif self.kind == 'garch':
            data = read_archive(ROOT/'data/posteriordb/data/data/garch.json.zip')
            self.y, self.sigma1 = np.array(data['y']), data['sigma1']
            self.names = ['mu', 'alpha0', 'alpha1', 'beta1']
        else:
            raise ValueError(f'Unknown target: {self.kind}')

    def physical(self, x, backend='numpy'):
        xp, sp = modules(backend)
        x = xp.asarray(x)
        if self.kind == 'eight_schools':
            tau = xp.exp(x[..., 9])
            theta = x[..., :8]
            if self.spec.get('parameterization') == 'noncentered':
                theta = x[..., 8, None] + tau[..., None]*theta
            return xp.concatenate([theta, x[..., 8, None], tau[..., None]], axis=-1)
        if self.kind == 'garch':
            a = sp.expit(x[..., 2])
            b = sp.expit(-x[..., 2])*sp.expit(x[..., 3])
            return xp.stack([x[..., 0], xp.exp(x[..., 1]), a, b], axis=-1)
        return x

    def initial(self):
        x = np.zeros(self.d)
        if self.kind == 'eight_schools':
            x[:8] = self.y if self.spec.get('parameterization') != 'noncentered' else 0
            x[8], x[9] = float(self.y.mean()), np.log(5.)
        elif self.kind == 'garch':
            x[0], x[1] = float(self.y.mean()), 0.
        elif self.kind == 'skew_t':
            x[:] = 1.
        return x

    def log_prob(self, x, backend='numpy'):
        xp, sp = modules(backend)
        x = xp.asarray(x)
        if self.kind == 'skew_t':
            r2 = xp.sum(x*x, axis=-1)
            nu, d = self.nu, self.d
            log_t = (sp.gammaln((nu+d)/2)-sp.gammaln(nu/2)
                     -d/2*xp.log(nu*xp.pi)-(nu+d)/2*xp.log1p(r2/nu))
            arg = (x @ xp.asarray(self.alpha))*xp.sqrt((nu+d)/(nu+r2))
            return xp.log(2.) + log_t + sp.log_ndtr(arg)
        if self.kind == 'logistic':
            nu, scale = self.nu, self.spec['prior_scale']
            constant = sp.gammaln((nu+1)/2)-sp.gammaln(nu/2)-xp.log(scale*xp.sqrt(nu*xp.pi))
            prior = xp.sum(constant-(nu+1)/2*xp.log1p((x/scale)**2/nu), axis=-1)
            logits = x @ xp.asarray(self.design).T
            signed_logits = (2*xp.asarray(self.y)-1)*logits
            return prior - xp.sum(xp.logaddexp(0., -signed_logits), axis=-1)
        if self.kind == 'eight_schools':
            p = self.physical(x, backend)
            theta, mu, tau = p[..., :8], p[..., 8], p[..., 9]
            obs = -0.5*xp.sum(((xp.asarray(self.y)-theta)/xp.asarray(self.sigma))**2, axis=-1)
            prior = -0.5*(mu/5)**2 - xp.logaddexp(0., 2*(x[..., 9]-xp.log(5.)))
            if self.spec.get('parameterization') == 'noncentered':
                hierarchy = -0.5*xp.sum(x[..., :8]**2, axis=-1)
            else:
                hierarchy = -8*x[..., 9]-0.5*xp.sum(((theta-mu[..., None])/tau[..., None])**2, axis=-1)
            return obs + prior + hierarchy + x[..., 9]
        # Flat density in physical (mu,alpha0,alpha1,beta1), with exact Jacobian.
        p = self.physical(x, backend)
        mu, a0, a1, b1 = (p[i] for i in range(4))
        logsigmoid = lambda q: -xp.logaddexp(0., -q)
        logjac = x[1]+logsigmoid(x[2])+2*logsigmoid(-x[2])+logsigmoid(x[3])+logsigmoid(-x[3])
        if backend == 'jax':
            import jax
            def step(h, previous_y):
                h = a0+a1*(previous_y-mu)**2+b1*h
                return h, h
            _, tail = jax.lax.scan(step, xp.asarray(self.sigma1**2), xp.asarray(self.y[:-1]))
            h = xp.concatenate([xp.asarray([self.sigma1**2]), tail])
        else:
            h = np.empty(len(self.y))
            h[0] = self.sigma1**2
            for i in range(1, len(h)):
                h[i] = a0+a1*(self.y[i-1]-mu)**2+b1*h[i-1]
        lp = logjac-0.5*xp.sum(xp.log(h)+(xp.asarray(self.y)-mu)**2/h)
        return xp.where(xp.isfinite(lp), lp, -xp.inf)


def modules(backend):
    if backend == 'numpy':
        return np, special
    if backend == 'jax':
        import jax.numpy as jnp
        import jax.scipy.special as jsp
        return jnp, jsp
    raise ValueError(backend)
