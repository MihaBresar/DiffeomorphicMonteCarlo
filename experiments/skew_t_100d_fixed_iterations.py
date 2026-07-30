"""Fixed-iteration 100-dimensional skew-t comparison.

This benchmark evaluates DCS-RWM, DCS-BW/PBW, and SCS at ν=2.  The original
paper-scale settings remain defaults; run
``python run_experiment.py skew-t-100d-fixed`` for a saved profile.
"""

import os
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import minimize

from reproducibility import (
    configure_multiprocessing,
    env_int,
    finalize_experiment,
    worker_count,
)

# --- JAX CONFIGURATION ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
jax.config.update("jax_enable_x64", True)  
import jax.numpy as jnp
from jax.scipy.special import gammaln as jgammaln
import jax.scipy.stats as jstats

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS (Matched to Paper Sec 4.1)
# ==============================================================================
D = 100
NU = 2.0  # Paper tests nu=2 (heavy-tailed) and nu=1 (skew Cauchy)
ALPHA = np.zeros(D)
ALPHA[0] = 100.0  # High skewness on first coordinate
ALPHA[1] = -100.0 # High negative skewness on second coordinate
THRESHOLD = 50.0

EXPERIMENT_ID = "skew-t-100d-fixed"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 20)

# --- Paper-scale fixed-length MCMC setup ---
NUM_ITERS = env_int("HEAVY_TAILS_NUM_ITERS", 500_000)
BURN_IN = 100
THINNING = 50

BETA_VAL = 2.0  # Locked to preserve polynomial tail decay for PRWM/PBW/PHR
REFERENCE_SAMPLES = env_int("HEAVY_TAILS_REFERENCE_SAMPLES", 10_000_000)
REFERENCE_BATCH_SIZE = env_int("HEAVY_TAILS_REFERENCE_BATCH_SIZE", 500_000)
OPTIMIZATION_STEPS = env_int("HEAVY_TAILS_OPTIMIZATION_STEPS", 1_500, minimum=0)
OPTIMIZATION_SAMPLES = env_int("HEAVY_TAILS_OPTIMIZATION_SAMPLES", 2_000)

Q_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

# ==============================================================================
# 2. VARIATIONAL OPTIMIZERS (KL DIVERGENCE)
# ==============================================================================
def optimize_prwm_hyperparameters():
    print("\n" + "="*90)
    print(f"OPTIMIZING DCS-RWM/DCS-BW (Location μ, Scale R, Möbius Shift δ)".center(90))
    print("="*90)
    
    @jax.jit
    def prwm_kl_loss(params, key):
        mu = params[:D]
        raw_delta = params[D:2*D]
        log_R = params[-1]
        R = jnp.exp(log_R)
        
        delta_norm = jnp.sqrt(jnp.sum(raw_delta**2) + 1e-12)
        delta = jnp.tanh(delta_norm) * (raw_delta / delta_norm) * 0.999
        delta_sq = jnp.sum(delta**2)
        
        k1, k2 = jax.random.split(key)
        v = jax.random.normal(k1, (OPTIMIZATION_SAMPLES, D))
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        u = jax.random.uniform(k2, (OPTIMIZATION_SAMPLES, 1)) ** (1.0 / D)
        z = v * u
        rho2 = jnp.sum(z**2, axis=1)
        
        dot_z_delta = jnp.dot(z, delta)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * rho2
        
        term1 = (1.0 - delta_sq) * z
        term2 = (1.0 + 2.0 * dot_z_delta + rho2)[:, None] * delta[None, :]
        w = (term1 + term2) / denom[:, None]
        
        log_J_moebius = D * jnp.log(1.0 - delta_sq) - D * jnp.log(denom)
        
        w_norm = jnp.clip(jnp.linalg.norm(w, axis=1), 0.0, 1.0 - 1e-7)
        eps = jnp.clip(1.0 - w_norm**BETA_VAL, 1e-12, 1.0)
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        x_mod = mu * eps_pow1[:, None] + R * w
        S = NU * eps_pow2 + jnp.sum(x_mod**2, axis=1)
        
        core_log = (NU / BETA_VAL - 1.0) * jnp.log(eps) + D * log_R - ((NU + D) / 2.0) * jnp.log(S)
        
        x_scaled = x_mod / jnp.sqrt(S)[:, None]
        arg = jnp.dot(x_scaled, jnp.array(ALPHA)) * jnp.sqrt(NU + D)
        log_cdf = jstats.norm.logcdf(jnp.clip(arg, -35.0, 35.0))
        
        return jnp.mean(-(core_log + log_cdf + log_J_moebius))

    @jax.jit
    def step(params, state, key):
        loss, grads = jax.value_and_grad(prwm_kl_loss)(params, key)
        t, m, v = state
        t += 1
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * (grads ** 2)
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        params = params - 0.01 * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        return params, (t, m, v), loss

    params = jnp.concatenate([jnp.zeros(D), jnp.zeros(D), jnp.array([jnp.log(1.0)])])
    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % max(1, OPTIMIZATION_STEPS // 6) == 0:
            print(f"Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[-1]):.4f}")

    mu_opt = np.array(params[:D])
    raw_delta = np.array(params[D:2*D])
    delta_norm = np.sqrt(np.sum(raw_delta**2) + 1e-12)
    delta_opt = np.tanh(delta_norm) * (raw_delta / delta_norm) * 0.999
    return mu_opt, delta_opt, float(jnp.exp(params[-1]))

def optimize_scs_hyperparameters():
    print("\n" + "="*90)
    print("OPTIMIZING SCS (Location μ, Scale R, Observer ho, Fixed lo=1.1)".center(90))
    print("="*90)
    
    def log_prob_skew_t_jax(x):
        r2 = jnp.sum(x**2, axis=-1)
        log_td = (jgammaln((NU + D) / 2.0) - jgammaln(NU / 2.0) - (D / 2.0) * jnp.log(NU * jnp.pi) - ((NU + D) / 2.0) * jnp.log1p(r2 / NU))
        arg = jnp.dot(x, jnp.array(ALPHA)) * jnp.sqrt((NU + D) / (NU + r2))
        return jnp.log(2.0) + log_td + jstats.norm.logcdf(jnp.clip(arg, -35.0, 35.0))
        
    @jax.jit
    def scs_kl_loss(params, key):
        mu = params[:D]
        raw_ho = params[D:2*D]
        log_R = params[2*D]
        lo = 1.1 # Fixed latitude
        
        R = jnp.exp(log_R)
        r_max = jnp.sqrt(1.0 - (lo - 1.0)**2) - 1e-4
        ho_norm = jnp.sqrt(jnp.sum(raw_ho**2) + 1e-12)
        ho = r_max * jnp.tanh(ho_norm) * (raw_ho / ho_norm)

        v = jax.random.normal(key, (OPTIMIZATION_SAMPLES, D+1))
        u = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        l_x, h_x = u[:, D] + 1.0, u[:, :D]
        
        valid_mask = l_x < (lo - 1e-5)
        safe_l_x = jnp.where(valid_mask, l_x, lo - 1e-5)
        
        denom = lo - safe_l_x
        y_hat = (lo / denom[:, None]) * h_x - (safe_l_x / denom)[:, None] * ho
        y = R * y_hat + mu

        M = 1.0 - safe_l_x / lo
        y_hat_minus_ho = y_hat - ho
        num = M * jnp.sum(y_hat_minus_ho**2, axis=1) + jnp.sum(y_hat_minus_ho * ho, axis=1) + lo - (lo**2)*(1.0 - M)
        log_den = D * jnp.log(jnp.maximum(M, 1e-10)) + jnp.log(lo)
        log_J = D * log_R + jnp.log(jnp.maximum(num, 1e-10)) - log_den

        lp = log_prob_skew_t_jax(y)
        safe_loss = jnp.where(valid_mask, -log_J - lp, 0.0)
        return jnp.sum(safe_loss) / jnp.maximum(jnp.sum(valid_mask), 1.0)

    @jax.jit
    def step(params, state, key):
        loss, grads = jax.value_and_grad(scs_kl_loss)(params, key)
        t, m, v = state
        t += 1
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * (grads ** 2)
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        params = params - 0.01 * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        return params, (t, m, v), loss

    # Initialize: mu=0, ho=0, R=1.0
    params = jnp.concatenate([jnp.zeros(D), jnp.zeros(D), jnp.array([jnp.log(1.0)])]) 
    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % max(1, OPTIMIZATION_STEPS // 6) == 0:
            print(f"Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[2*D]):.4f} | l_o: 1.1000")

    mu_opt = np.array(params[:D])
    raw_ho = params[D:2*D]
    lo_opt = 1.1
    
    r_max = np.sqrt(1.0 - (lo_opt - 1.0)**2) - 1e-4
    ho_norm = np.sqrt(np.sum(raw_ho**2) + 1e-12)
    ho_opt = np.array(r_max * np.tanh(ho_norm) * (raw_ho / ho_norm))
    
    return mu_opt, ho_opt, float(jnp.exp(params[2*D])), lo_opt


# ==============================================================================
# 3. MONTE CARLO REFERENCE, HELPERS & INVERSE MAPPINGS
# ==============================================================================
def get_monte_carlo_reference(
    n_samples: int = REFERENCE_SAMPLES,
    batch_size: int = REFERENCE_BATCH_SIZE,
):
    """Generate the fixed-seed Monte Carlo reference used for diagnostics."""
    if n_samples % batch_size:
        raise ValueError("REFERENCE_SAMPLES must be divisible by REFERENCE_BATCH_SIZE.")
    print(f"\nGenerating {n_samples} Monte Carlo reference draws in batches of {batch_size}...")
    
    @jax.jit
    def generate_batch(key):
        k1, k2, k3 = jax.random.split(key, 3)
        X = jax.random.normal(k1, (batch_size, D))
        Y = jax.random.normal(k2, (batch_size,))
        dot_prod = X[:, 0] * ALPHA[0] + X[:, 1] * ALPHA[1]
        Z = jnp.where((Y < dot_prod)[:, None], X, -X)
        V = jax.random.gamma(k3, NU / 2.0, (batch_size,)) * 2.0
        skew_t = Z / jnp.sqrt(V / NU)[:, None]
        norms = jnp.linalg.norm(skew_t, axis=1)
        return jnp.sum(norms >= THRESHOLD), jnp.sum(norms), skew_t[:, :4]

    key = jax.random.PRNGKey(999)
    sum_indicator, sum_norm = 0.0, 0.0
    exact_samples_4d = np.empty((n_samples, 4))
    
    for i in range(n_samples // batch_size):
        key, subkey = jax.random.split(key)
        ind_sum, norm_sum, skew_t_4 = generate_batch(subkey)
        sum_indicator += float(ind_sum)
        sum_norm += float(norm_sum)
        exact_samples_4d[i * batch_size:(i + 1) * batch_size, :] = np.array(skew_t_4)
        
    return np.quantile(exact_samples_4d, Q_LEVELS, axis=0), sum_indicator / n_samples, sum_norm / n_samples

def scs_inverse_map(y_target, mu, ho, R, lo):
    """Analytical inverse of Sub-Cauchy Projection (Eq. 1 in paper)"""
    y_hat = (y_target - mu) / R
    y_minus_ho = y_hat - ho
    norm_y_minus_ho_sq = np.sum(y_minus_ho**2)
    dot_term = np.dot(y_minus_ho, ho) - lo * (lo - 1.0)
    c_term = np.sum(ho**2) + lo**2 - 2.0 * lo
    denom = norm_y_minus_ho_sq + lo**2
    
    # Quadratic solution for M
    M = (-dot_term + np.sqrt(max(0.0, dot_term**2 - denom * c_term))) / denom
    
    h_x = M * y_hat + (1.0 - M) * ho
    l_x = (1.0 - M) * lo
    
    x_sphere = np.concatenate([h_x, [l_x - 1.0]])
    x_sphere /= np.linalg.norm(x_sphere)  # Renormalize to ensure it's precisely on unit sphere
    return x_sphere

def get_z_inverse_custom(target_x, opt_mu, opt_delta, opt_R):
    """Numerical inversion to find starting 'z' for custom projected samplers"""
    delta_sq = np.sum(opt_delta**2)
    def loss(z):
        z_norm2 = np.sum(z**2)
        if z_norm2 >= 0.9999: return 1e9
        dot_z_delta = np.dot(z, opt_delta)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
        term1 = (1.0 - delta_sq) * z
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
        w = (term1 + term2) / denom
        w_norm = np.linalg.norm(w)
        if w_norm >= 0.9999: return 1e9
        eps = 1.0 - w_norm**BETA_VAL
        eps_pow1 = eps**(1.0 / BETA_VAL)
        x_mod = opt_mu * eps_pow1 + opt_R * w
        return np.sum((x_mod - target_x)**2)
    
    res = minimize(loss, np.zeros(D), method='L-BFGS-B', bounds=[(-0.98, 0.98)]*D)
    return res.x

def process_samples(samples):
    if len(samples) > 0:
        norms = np.linalg.norm(samples, axis=1)
        return np.quantile(samples[:, :4], Q_LEVELS, axis=0), np.mean(norms >= THRESHOLD), np.mean(norms)
    return np.zeros((len(Q_LEVELS), 4)), 0.0, 0.0


# ==============================================================================
# 4. WORKERS (Fixed Run Lengths & Initialization)
# ==============================================================================
def worker_phr(args):
    seed, opt_mu, opt_delta, opt_R = args
    rng = np.random.default_rng(seed)
    delta_sq = np.sum(opt_delta**2)
    
    # Initialize exactly at target point (1, 1, ..., 1)
    target_x = np.ones(D)
    z = get_z_inverse_custom(target_x, opt_mu, opt_delta, opt_R)
    
    def get_log_pi_z_stable(z_pos):
        from scipy.stats import norm
        z_norm2 = np.sum(z_pos**2)
        if z_norm2 >= 0.9999999: return -1e20 
        
        dot_z_delta = np.dot(z_pos, opt_delta)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
        term1 = (1.0 - delta_sq) * z_pos
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
        w = (term1 + term2) / denom
        
        log_J_moebius = D * np.log(1.0 - delta_sq) - D * np.log(denom)
        
        w_norm = np.linalg.norm(w)
        if w_norm >= 0.9999999: return -1e20
        
        eps = 1.0 - w_norm**BETA_VAL
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        x_mod = opt_mu * eps_pow1 + opt_R * w
        S = NU * eps_pow2 + np.sum(x_mod**2)
        
        core_log = (NU / BETA_VAL - 1.0) * np.log(eps) + D * np.log(opt_R) - ((NU + D) / 2.0) * np.log(S)
        x_scaled = x_mod / np.sqrt(S)
        arg = np.dot(ALPHA, x_scaled) * np.sqrt(NU + D)
        
        return core_log + norm.logcdf(arg) + log_J_moebius
        
    lp_curr = get_log_pi_z_stable(z)
    
    trace = []
    
    start_time = time.time()
    for iters in range(1, NUM_ITERS + 1):
        theta_dir = rng.normal(0, 1, D)
        theta_dir /= np.linalg.norm(theta_dir)
        dot = np.dot(z, theta_dir)
        disc = np.sqrt(max(0.0, dot**2 + 1.0 - np.sum(z**2)))
        L, R_bound = -dot - disc, -dot + disc
        
        log_u = lp_curr + np.log(rng.random() + 1e-30)
        for _ in range(50):
            eta = rng.uniform(L, R_bound)
            z_prop = z + eta * theta_dir
            lp_prop = get_log_pi_z_stable(z_prop)
            if lp_prop > log_u:
                z, lp_curr = z_prop, lp_prop
                break
            else:
                if eta < 0: L = eta
                else: R_bound = eta
                
        # Thinning Logic
        if iters > BURN_IN and (iters - BURN_IN) % THINNING == 0:
            # Transform to target space before storing
            dot_z_delta = np.dot(z, opt_delta)
            z_norm2 = np.sum(z**2)
            denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
            term1 = (1.0 - delta_sq) * z
            term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
            w = (term1 + term2) / denom
            w_norm = np.linalg.norm(w)
            eps_pow1 = (1.0 - w_norm**BETA_VAL)**(1.0 / BETA_VAL)
            final_x = opt_mu + opt_R * w / eps_pow1
            trace.append(final_x)
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    trace = np.array(trace)
    q, ind, norm = process_samples(trace)
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': NUM_ITERS, 'acc_rate': 1.0, 'time': elapsed_time}

def worker_prwm(args):
    seed, opt_mu, opt_delta, opt_R = args
    rng = np.random.default_rng(seed)
    delta_sq = np.sum(opt_delta**2)
    
    target_x = np.ones(D)
    z = get_z_inverse_custom(target_x, opt_mu, opt_delta, opt_R)
    
    def get_log_pi_z_stable(z_pos):
        from scipy.stats import norm
        z_norm2 = np.sum(z_pos**2)
        if z_norm2 >= 0.9999999: return -1e20 
        
        dot_z_delta = np.dot(z_pos, opt_delta)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
        term1 = (1.0 - delta_sq) * z_pos
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
        w = (term1 + term2) / denom
        
        log_J_moebius = D * np.log(1.0 - delta_sq) - D * np.log(denom)
        
        w_norm = np.linalg.norm(w)
        if w_norm >= 0.9999999: return -1e20
        
        eps = 1.0 - w_norm**BETA_VAL
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        x_mod = opt_mu * eps_pow1 + opt_R * w
        S = NU * eps_pow2 + np.sum(x_mod**2)
        
        core_log = (NU / BETA_VAL - 1.0) * np.log(eps) + D * np.log(opt_R) - ((NU + D) / 2.0) * np.log(S)
        x_scaled = x_mod / np.sqrt(S)
        arg = np.dot(ALPHA, x_scaled) * np.sqrt(NU + D)
        
        return core_log + norm.logcdf(arg) + log_J_moebius
        
    lp_curr = get_log_pi_z_stable(z)
    
    h = 0.1
    mu_da, target_acc, gamma_da, t0, kappa = np.log(10 * h), 0.234, 0.05, 10.0, 0.75
    h_bar_log, H_bar = 0.0, 0.0

    trace, accepts = [], 0
    start_time = time.time()
    
    for iters in range(1, NUM_ITERS + 1):
        z_prop = z + rng.normal(0, h, D)
        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)
            alpha = min(1.0, np.exp(lp_prop - lp_curr))
            if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
                z, lp_curr = z_prop, lp_prop
                accepts += 1
        else: alpha = 0.0
            
        H_bar = (1.0 - 1.0/(iters + t0)) * H_bar + (1.0/(iters + t0)) * (target_acc - alpha)
        log_h = mu_da - (np.sqrt(iters) / gamma_da) * H_bar
        h = np.exp(log_h)
        weight = iters**(-kappa)
        h_bar_log = weight * log_h + (1.0 - weight) * h_bar_log
        h = np.exp(h_bar_log) 

        if iters > BURN_IN and (iters - BURN_IN) % THINNING == 0:
            dot_z_delta = np.dot(z, opt_delta)
            z_norm2 = np.sum(z**2)
            denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
            term1 = (1.0 - delta_sq) * z
            term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
            w = (term1 + term2) / denom
            w_norm = np.linalg.norm(w)
            eps_pow1 = (1.0 - w_norm**BETA_VAL)**(1.0 / BETA_VAL)
            final_x = opt_mu + opt_R * w / eps_pow1
            trace.append(final_x)
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    trace = np.array(trace)
    q, ind, norm = process_samples(trace)
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': NUM_ITERS, 'acc_rate': accepts / NUM_ITERS, 'step': h, 'time': elapsed_time}

def worker_pbw(args):
    seed, opt_mu, opt_delta, opt_R = args
    rng = np.random.default_rng(seed)
    delta_sq = np.sum(opt_delta**2)
    
    target_x = np.ones(D)
    z = get_z_inverse_custom(target_x, opt_mu, opt_delta, opt_R)
    
    def get_log_pi_z_stable(z_pos):
        from scipy.stats import norm
        z_norm2 = np.sum(z_pos**2)
        if z_norm2 >= 0.9999999: return -1e20 
        
        dot_z_delta = np.dot(z_pos, opt_delta)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
        term1 = (1.0 - delta_sq) * z_pos
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
        w = (term1 + term2) / denom
        
        log_J_moebius = D * np.log(1.0 - delta_sq) - D * np.log(denom)
        
        w_norm = np.linalg.norm(w)
        if w_norm >= 0.9999999: return -1e20
        
        eps = 1.0 - w_norm**BETA_VAL
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        x_mod = opt_mu * eps_pow1 + opt_R * w
        S = NU * eps_pow2 + np.sum(x_mod**2)
        
        core_log = (NU / BETA_VAL - 1.0) * np.log(eps) + D * np.log(opt_R) - ((NU + D) / 2.0) * np.log(S)
        x_scaled = x_mod / np.sqrt(S)
        arg = np.dot(ALPHA, x_scaled) * np.sqrt(NU + D)
        
        return core_log + norm.logcdf(arg) + log_J_moebius
        
    lp_curr = get_log_pi_z_stable(z)
    
    r_step = 0.1
    mu_da, target_acc, gamma_da, t0, kappa = np.log(10 * r_step), 0.234, 0.05, 10.0, 0.75
    h_bar_log, H_bar = 0.0, 0.0

    trace, accepts = [], 0
    start_time = time.time()
    
    for iters in range(1, NUM_ITERS + 1):
        direction = rng.normal(0, 1, D)
        direction /= np.linalg.norm(direction)
        rho = r_step * (rng.random() ** (1.0 / D))
        z_prop = z + rho * direction
        
        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)
            alpha = min(1.0, np.exp(lp_prop - lp_curr))
            if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
                z, lp_curr = z_prop, lp_prop
                accepts += 1
        else: alpha = 0.0
            
        H_bar = (1.0 - 1.0/(iters + t0)) * H_bar + (1.0/(iters + t0)) * (target_acc - alpha)
        log_r = mu_da - (np.sqrt(iters) / gamma_da) * H_bar
        r_step = np.exp(log_r)
        weight = iters**(-kappa)
        h_bar_log = weight * log_r + (1.0 - weight) * h_bar_log
        r_step = np.exp(h_bar_log)

        if iters > BURN_IN and (iters - BURN_IN) % THINNING == 0:
            dot_z_delta = np.dot(z, opt_delta)
            z_norm2 = np.sum(z**2)
            denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
            term1 = (1.0 - delta_sq) * z
            term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta
            w = (term1 + term2) / denom
            w_norm = np.linalg.norm(w)
            eps_pow1 = (1.0 - w_norm**BETA_VAL)**(1.0 / BETA_VAL)
            final_x = opt_mu + opt_R * w / eps_pow1
            trace.append(final_x)
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    trace = np.array(trace)
    q, ind, norm = process_samples(trace)
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': NUM_ITERS, 'acc_rate': accepts / NUM_ITERS, 'step': r_step, 'time': elapsed_time}

def worker_scs(args):
    """Optimized SCS Worker (Matches Section 4.1 exactly)"""
    seed, opt_mu, opt_ho, opt_R, opt_lo = args
    rng = np.random.default_rng(seed)
    lo = opt_lo
    
    # Initialize exactly at target point (1, 1, ..., 1)
    target_y = np.ones(D)
    x = scs_inverse_map(target_y, opt_mu, opt_ho, opt_R, lo)
    
    def log_prob_skew_t_np_scs(y, nu, alpha, d):
        r2 = np.sum(y**2)
        log_td = (gammaln((nu + d) / 2) - gammaln(nu / 2) - (d / 2) * np.log(nu * np.pi) - ((nu + d) / 2) * np.log1p(r2 / nu))
        arg = np.dot(alpha, y) * np.sqrt((nu + d) / (nu + r2))
        from scipy.stats import norm
        return np.log(2.0) + log_td + norm.logcdf(np.clip(arg, -35.0, 35.0))

    def log_target_sphere(x_sphere):
        h_x, l_x_unshifted = x_sphere[:-1], x_sphere[-1]
        l_x = l_x_unshifted + 1.0
        denom = lo - l_x
        y_hat = (lo / denom) * h_x - (l_x / denom) * opt_ho
        y = opt_R * y_hat + opt_mu
        M = 1.0 - l_x / lo
        y_hat_minus_ho = y_hat - opt_ho
        num = M * np.sum(y_hat_minus_ho**2) + np.dot(y_hat_minus_ho, opt_ho) + lo - (lo**2)*(1.0 - M)
        log_den = D * np.log(max(M, 1e-10)) + np.log(lo)
        log_J = D * np.log(opt_R) + np.log(max(num, 1e-10)) - log_den
        return log_prob_skew_t_np_scs(y, NU, ALPHA, D) + log_J, y
        
    lp_curr, y_curr = log_target_sphere(x)
    
    h = 0.1 # Fixed step size for SCS as outlined in paper
    trace, accepts = [], 0

    start_time = time.time()
    for iters in range(1, NUM_ITERS + 1):
        delta = rng.normal(0, h, D+1)
        delta -= np.dot(x, delta) * x
        x_prop = (x + delta) / np.linalg.norm(x + delta)
        
        if x_prop[-1] > lo - 1:
            dot_xx = np.clip(np.dot(x, x_prop), -1.0, 1.0)
            alpha_angle = np.arccos(dot_xx)
            u = x_prop - dot_xx * x
            u_norm = np.linalg.norm(u)
            if u_norm > 1e-12 and alpha_angle > 1e-12:
                u /= u_norm
                R_lat = np.sqrt(x[-1]**2 + u[-1]**2)
                if R_lat >= np.abs(lo - 1):
                    phi = np.arccos(np.clip(x[-1] / R_lat, -1.0, 1.0))
                    gamma = np.arccos(np.clip((lo - 1) / R_lat, -1.0, 1.0))
                    K = int(np.floor((phi + gamma) / alpha_angle)) + 1
                    x_prop = np.cos(K * alpha_angle) * x + np.sin(K * alpha_angle) * u
        
        lp_prop, y_prop = log_target_sphere(x_prop)
        if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
            x, lp_curr, y_curr = x_prop, lp_prop, y_prop
            accepts += 1
            
        if iters > BURN_IN and (iters - BURN_IN) % THINNING == 0:
            trace.append(y_curr.copy())
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    trace = np.array(trace)
    q, ind, norm = process_samples(trace)
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': NUM_ITERS, 'acc_rate': accepts / NUM_ITERS, 'step': h, 'time': elapsed_time}

# ==============================================================================
# 5. ORCHESTRATION & PLOTTING
# ==============================================================================
def run_comparison():
    pr_mu, pr_delta, pr_R = optimize_prwm_hyperparameters()
    sc_mu, sc_ho, sc_R, sc_lo = optimize_scs_hyperparameters()
    
    true_q, true_indicator, true_norm = get_monte_carlo_reference()
    
    cores = worker_count(NUM_RUNS)
    print(f"\n--- Launching {NUM_RUNS} chains for 3 algorithms ({NUM_ITERS} iterations each) ---")

    with multiprocessing.Pool(processes=cores) as pool:
        prwm_results = pool.map(
            worker_prwm,
            [(142 + i, pr_mu, pr_delta, pr_R) for i in range(NUM_RUNS)]
        )

        pbw_results = pool.map(
            worker_pbw,
            [(242 + i, pr_mu, pr_delta, pr_R) for i in range(NUM_RUNS)]
        )

        scs_results = pool.map(
            worker_scs,
            [(342 + i, sc_mu, sc_ho, sc_R, sc_lo) for i in range(NUM_RUNS)]
        )

    all_results = [
        prwm_results,
        pbw_results,
        scs_results
    ]

    labels = [
        f'DCS-RWM (β={BETA_VAL})', 
        f'DCS-BW (β={BETA_VAL})', 
        f'SCS (lo={sc_lo:.2f})'
    ]

    colors = [
        '#FF8C00',
        '#9370DB',
        '#FF1493'
    ]

    avg_times = [np.mean([r['time'] for r in res]) for res in all_results]

    print("\n" + "="*115)
    print(f"SAMPLING EFFICIENCY (FIXED RUN: {NUM_ITERS} ITERS)".center(115))
    print("="*115)

    for i in range(len(labels)):
        acc_str = f" | Acc: {np.mean([r['acc_rate'] for r in all_results[i]]):.1%}"
        time_str = f" | Avg Time: {avg_times[i]:.2f}s"

        step_str = (
            f" | Final Step: {np.mean([r['step'] for r in all_results[i]]):.4f}"
            if 'step' in all_results[i][0]
            else " | Final Step: N/A (Auto)"
        )

        print(
            f"{labels[i]:<32} "
            f"Iterations: {NUM_ITERS:<7.0f}"
            f"{acc_str}{step_str}{time_str}"
        )

    print("="*115)

    extract = lambda results, key: [r[key] for r in results]

    q_all = [
        np.array(extract(res, 'quantiles'))
        for res in all_results
    ]

    data_indicator = [
        extract(res, 'mean_indicator')
        for res in all_results
    ]

    data_norm = [
        extract(res, 'mean_norm')
        for res in all_results
    ]

    mse_indicator = [
        np.mean((np.array(d) - true_indicator)**2)
        for d in data_indicator
    ]

    mse_norm = [
        np.mean((np.array(d) - true_norm)**2)
        for d in data_norm
    ]

    print("\n" + "="*105)
    print(f"ERGODIC AVERAGES & MEAN SQUARED ERROR (ACROSS {NUM_RUNS} CHAINS)".center(105))
    print("="*105)
    print(
        f"{'Sampler':<32} | "
        f"{'E[I{||x||_2>=50}]':<16} | "
        f"{'MSE':<16} | "
        f"{'E[||x||_2]':<17} | "
        f"{'MSE':<16}"
    )
    print("-" * 105)
    print(
        f"{'Monte Carlo reference':<32} | "
        f"{true_indicator:<16.6f} | "
        f"{'-':<16} | "
        f"{true_norm:<17.6f} | "
        f"{'-':<16}"
    )
    print("-" * 105)

    for i, name in enumerate(labels):
        print(
            f"{name:<32} | "
            f"{np.mean(data_indicator[i]):<16.6f} | "
            f"{mse_indicator[i]:<16.2e} | "
            f"{np.mean(data_norm[i]):<17.6f} | "
            f"{mse_norm[i]:<16.2e}"
        )

    print("="*105)

    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
    fig1.suptitle(
        f"Marginal Q–Q Diagnostics: Skew-t (d={D}, ν={NU})",
        fontsize=16
    )

    for dim in range(4):
        ax = axes1[dim]
        exact_dim_q = true_q[:, dim]

        ax.plot(
            [np.min(exact_dim_q), np.max(exact_dim_q)],
            [np.min(exact_dim_q), np.max(exact_dim_q)],
            'k--',
            alpha=0.5,
            label='y = x'
        )

        for i, name in enumerate(labels):
            dim_data = q_all[i][:, :, dim]

            ax.plot(
                exact_dim_q,
                np.median(dim_data, axis=0),
                marker='o',
                markersize=4,
                color=colors[i],
                label=name,
                alpha=0.8
            )

            ax.fill_between(
                exact_dim_q,
                np.percentile(dim_data, 5, axis=0),
                np.percentile(dim_data, 95, axis=0),
                color=colors[i],
                alpha=0.15
            )

        ax.set_title(f"$x_{dim+1}$")
        ax.set_xlabel("Monte Carlo Reference Quantiles")
        ax.set_ylabel("Estimated Quantiles")
        ax.grid(True, linestyle=':', alpha=0.6)

        if dim == 0:
            ax.legend()

    fig1.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        f"Chainwise Tail and Norm Estimates (d={D}, ν={NU}; {NUM_RUNS} chains)",
        fontsize=16,
        fontweight='bold'
    )

    def make_boxplot(ax, data, exact_val, title):
        bplot = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=labels,
            showmeans=True,
            meanline=True
        )

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(
            exact_val,
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Reference: {exact_val:.4f}'
        )

        ax.set_title(title, fontsize=14)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.tick_params(axis='x', rotation=15)
        ax.legend()

    make_boxplot(
        axes2[0],
        data_indicator,
        true_indicator,
        rf"$\mathbb{{E}}[\mathbb{{I}}\{{\lVert x\rVert_2 \geq {THRESHOLD}\}}]$"
    )

    make_boxplot(
        axes2[1],
        data_norm,
        true_norm,
        r"$\mathbb{E}[\lVert x\rVert_2]$"
    )

    fig2.tight_layout()

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle(
        "Estimator Error and Sampling-Output Diagnostics",
        fontsize=16,
        fontweight='bold'
    )

    def make_barplot(ax, heights, title, ylabel, is_scientific=True):
        bars = ax.bar(
            labels,
            heights,
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")

        for bar in bars:
            h = bar.get_height()
            label_text = f'{h:.2e}' if is_scientific else f'{h:.2f}'

            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    make_barplot(
        axes3[0],
        mse_indicator,
        rf"MSE of $\hat{{\mathbb{{E}}}}[\mathbb{{I}}\{{\lVert x\rVert_2 \geq {THRESHOLD}\}}]$",
        "Mean Squared Error"
    )

    make_barplot(
        axes3[1],
        mse_norm,
        r"MSE of $\hat{\mathbb{E}}[\lVert x\rVert_2]$",
        "Mean Squared Error"
    )

    make_barplot(
        axes3[2],
        avg_times,
        "Average Runtime (s)",
        "Seconds",
        is_scientific=False
    )

    fig3.tight_layout()

    finalize_experiment(
        EXPERIMENT_ID,
        {"marginal_qq": fig1, "ergodic_averages": fig2, "mse_and_runtime": fig3},
        {
            "reference": {
                "tail_probability": true_indicator,
                "euclidean_norm_mean": true_norm,
                "samples": REFERENCE_SAMPLES,
                "seed": 999,
            },
            "mse": {"tail_probability": mse_indicator, "euclidean_norm_mean": mse_norm},
            "average_runtime_seconds": dict(zip(labels, avg_times)),
        },
        {
            "dimension": D,
            "nu": NU,
            "alpha_nonzero": {"x1": float(ALPHA[0]), "x2": float(ALPHA[1])},
            "num_runs": NUM_RUNS,
            "iterations_per_chain": NUM_ITERS,
            "burn_in_iterations": BURN_IN,
            "thinning": THINNING,
            "reference_samples": REFERENCE_SAMPLES,
            "reference_batch_size": REFERENCE_BATCH_SIZE,
            "optimization_steps": OPTIMIZATION_STEPS,
            "optimization_samples": OPTIMIZATION_SAMPLES,
            "seed_schedule": {
                "dcs_rwm_optimizer": 42,
                "scs_optimizer": 42,
                "monte_carlo_reference": 999,
                "dcs_rwm_chains": "142 + chain_index",
                "dcs_bw_pbw_chains": "242 + chain_index",
                "scs_chains": "342 + chain_index",
            },
            "worker_processes": cores,
            "timing": "Optimizer fitting and reference generation are outside the fixed-chain runtime measurements.",
        },
    )


def main() -> None:
    multiprocessing.freeze_support()
    configure_multiprocessing()
    run_comparison()


if __name__ == "__main__":
    main()
