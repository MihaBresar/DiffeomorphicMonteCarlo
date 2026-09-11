"""Exact change-of-variable formulas; no artificial interior-ball cutoff."""
import numpy as np
from .targets import modules


def mobius(z, delta, xp=np):
    r2 = xp.sum(z*z, axis=-1)
    delta2 = xp.sum(delta*delta)
    dot = z @ delta
    denominator = 1+2*dot+delta2*r2
    w = ((1-delta2)*z+(1+2*dot+r2)[..., None]*delta)/denominator[..., None]
    logjac = z.shape[-1]*(xp.log1p(-delta2)-xp.log(denominator))
    return w, logjac


class BallMap:
    def __init__(self, dimension, beta=2., mu=None, scale=1., delta=None):
        self.d, self.beta = dimension, float(beta)
        self.mu = np.zeros(dimension) if mu is None else np.asarray(mu, dtype=float)
        self.delta = np.zeros(dimension) if delta is None else np.asarray(delta, dtype=float)
        self.scale = float(scale)
        if self.beta <= 0 or self.scale <= 0 or np.linalg.norm(self.delta) >= 1:
            raise ValueError('beta and scale must be positive; ||delta|| must be < 1')
        if not np.all(np.isfinite(self.mu)) or not np.all(np.isfinite(self.delta)) or not np.isfinite(self.scale):
            raise ValueError('Nonfinite map parameters')

    def parameters(self):
        return {'dimension': self.d, 'beta': self.beta, 'mu': self.mu.tolist(),
                'scale': self.scale, 'delta': self.delta.tolist()}

    def forward(self, z):
        z = np.asarray(z)
        if np.any(np.linalg.norm(z, axis=-1) >= 1):
            raise ValueError('Point outside the open ball')
        w, log_mobius = mobius(z, self.delta)
        rho = np.linalg.norm(w, axis=-1)
        if np.any(rho >= 1):
            raise FloatingPointError('Map boundary cannot be represented at this precision')
        with np.errstate(divide='ignore'):
            eps = -np.expm1(self.beta*np.log(rho))
        x = self.mu + self.scale*w/np.exp(np.log(eps)/self.beta)[..., None]
        logjac = self.d*np.log(self.scale)-(1+self.d/self.beta)*np.log(eps)+log_mobius
        return x, logjac

    def inverse(self, x):
        v = (np.asarray(x)-self.mu)/self.scale
        r = np.linalg.norm(v, axis=-1)
        with np.errstate(divide='ignore'):
            factor = np.exp(np.logaddexp(0., self.beta*np.log(r))/self.beta)
        w = v/factor[..., None]
        z, _ = mobius(w, -self.delta)
        if np.any(np.linalg.norm(z, axis=-1) >= 1):
            raise FloatingPointError('Initial point maps onto the floating-point boundary')
        return z

    def bind_log_density(self, target):
        """Freeze map invariants once; retain the validated algebraic density."""
        from scipy.special import gammaln, log_ndtr
        d, beta, scale = self.d, self.beta, self.scale
        mu, delta = self.mu.copy(), self.delta.copy()
        delta2 = float(delta@delta)
        logscale, logdelta = d*np.log(scale), d*np.log1p(-delta2)
        target_lp = target.bind_log_prob() if hasattr(target, 'bind_log_prob') else target.log_prob
        skew_t = target.kind == 'skew_t'
        if skew_t:
            nu, alpha = target.nu, target.alpha.copy()
            skewed = bool(np.any(alpha))
            constant = gammaln((nu+d)/2)-gammaln(nu/2)+nu/2*np.log(nu)-d/2*np.log(np.pi)
        def evaluate(z):
            r2 = z@z
            if r2 >= 1:
                return -np.inf
            if delta2:
                dot = z@delta
                denominator = 1+2*dot+delta2*r2
                w = ((1-delta2)*z+(1+2*dot+r2)*delta)/denominator
                logmo = logdelta-d*np.log(denominator)
            else:
                w, logmo = z, 0.
            rho = np.linalg.norm(w)
            if rho >= 1:
                return -np.inf
            logeps = 0. if rho == 0 else np.log(-np.expm1(beta*np.log(rho)))
            e = np.exp(logeps/beta)
            if skew_t:
                scaled_x = mu*e+scale*w
                s = nu*e*e+scaled_x@scaled_x
                lp = constant+(nu/beta-1)*logeps+logscale-(nu+d)/2*np.log(s)+logmo
                if skewed:
                    lp += np.log(2.)+log_ndtr((alpha@scaled_x)*np.sqrt((nu+d)/s))
                return lp
            return target_lp(mu+scale*w/e)+logscale-(1+d/beta)*logeps+logmo
        return evaluate

    def log_density(self, target, z):
        # Evaluate t target after algebraic cancellation, keeping the same
        # normalization as direct log_prob + log|J| for validation.
        if np.linalg.norm(z) >= 1:
            return -np.inf
        try:
            if target.kind != 'skew_t':
                x, logjac = self.forward(z)
                lp = target.log_prob(x)+logjac
            else:
                from scipy.special import gammaln, log_ndtr
                w, logmo = mobius(z, self.delta)
                rho = np.linalg.norm(w)
                if rho >= 1:
                    return -np.inf
                eps = 1. if rho == 0 else -np.expm1(self.beta*np.log(rho))
                e = np.exp(np.log(eps)/self.beta)
                scaled_x = self.mu*e+self.scale*w
                s = target.nu*e*e+np.sum(scaled_x**2)
                d, nu = target.d, target.nu
                constant = np.log(2.)+gammaln((nu+d)/2)-gammaln(nu/2)+nu/2*np.log(nu)-d/2*np.log(np.pi)
                lp = (constant+(nu/self.beta-1)*np.log(eps)+d*np.log(self.scale)
                      -(nu+d)/2*np.log(s)+logmo
                      +log_ndtr((target.alpha @ scaled_x)*np.sqrt((nu+d)/s)))
            return float(lp) if np.isfinite(lp) else -np.inf
        except (FloatingPointError, ValueError, OverflowError):
            return -np.inf


class SphereMap:
    def __init__(self, dimension, mu=None, scale=1., observer=None, latitude=1.1):
        self.d, self.scale, self.latitude = dimension, float(scale), float(latitude)
        self.mu = np.zeros(dimension) if mu is None else np.asarray(mu)
        self.observer = np.zeros(dimension) if observer is None else np.asarray(observer)
        if self.scale <= 0 or not (1 <= self.latitude < 2) or np.sum(self.observer**2)+(self.latitude-1)**2 >= 1:
            raise ValueError('Invalid sphere map parameters')

    def parameters(self):
        return {'dimension': self.d, 'mu': self.mu.tolist(), 'scale': self.scale,
                'observer': self.observer.tolist(), 'latitude': self.latitude}

    def forward(self, z):
        ell = z[-1]+1
        lo = self.latitude
        if ell >= lo:
            raise ValueError('Point outside bright cap')
        m = (lo-ell)/lo
        yhat = (z[:-1]-(ell/lo)*self.observer)/m
        v = yhat-self.observer
        numerator = m*np.dot(v,v)+np.dot(v,self.observer)+lo-lo*lo*(1-m)
        if numerator <= 0:
            raise FloatingPointError('Nonpositive sphere Jacobian')
        logjac = self.d*np.log(self.scale)+np.log(numerator)-self.d*np.log(m)-np.log(lo)
        return self.mu+self.scale*yhat, logjac

    def inverse(self, y):
        v = (np.asarray(y)-self.mu)/self.scale
        h, lo = self.observer, self.latitude
        a = np.sum((v-h)**2)+lo*lo
        b = np.dot(v-h,h)-lo*(lo-1)
        c = np.sum(h*h)+lo*lo-2*lo
        root = np.sqrt(b*b-a*c)
        m = -c/(root+b) if b > 0 else (-b+root)/a
        return np.r_[m*v+(1-m)*h, (1-m)*lo-1]

    def log_density(self, target, z):
        try:
            x, lj = self.forward(z)
            lp = target.log_prob(x)+lj
            return float(lp) if np.isfinite(lp) else -np.inf
        except (FloatingPointError, ValueError, OverflowError):
            return -np.inf

    def bind_log_density(self, target):
        """Apply the same constant-caching optimization to the SCS comparator."""
        d, scale, lo = self.d, self.scale, self.latitude
        mu, observer = self.mu.copy(), self.observer.copy()
        constant = d*np.log(scale)-np.log(lo)
        target_lp = target.bind_log_prob() if hasattr(target, 'bind_log_prob') else target.log_prob
        def evaluate(z):
            ell = z[-1]+1
            if ell >= lo:
                return -np.inf
            m = (lo-ell)/lo
            yhat = (z[:-1]-(ell/lo)*observer)/m
            v = yhat-observer
            numerator = m*np.dot(v, v)+np.dot(v, observer)+lo-lo*lo*(1-m)
            if numerator <= 0:
                return -np.inf
            return target_lp(mu+scale*yhat)+constant+np.log(numerator)-d*np.log(m)
        return evaluate
