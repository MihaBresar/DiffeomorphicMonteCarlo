"""50-dimensional skew-t benchmark at ν=0.5.

The implemented comparison contains DCS-RWM, DCS-BW/PBW, and SCS.  The SCS
optimizer below is retained as development code, but the paper run uses the
fixed SCS worker configuration in ``run_comparison``.  Use
``python run_experiment.py skew-t-50d-nu-0.5`` for reproducible artifacts.
"""

import os
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

from reproducibility import (
    configure_multiprocessing,
    env_float,
    env_int,
    finalize_experiment,
    worker_count,
)

# --- JAX CONFIGURATION ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
jax.config.update("jax_enable_x64", True)  # Prevent underflow in extreme tails
import jax.numpy as jnp
from jax.scipy.special import gammaln as jgammaln
import jax.scipy.stats as jstats

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
D = 50
NU = 0.5  # Strongly heavy-tailed setting
ALPHA = np.zeros(D)
ALPHA[0] = 1.0
ALPHA[1] = -1.0
THRESHOLD = 50.0

EXPERIMENT_ID = "skew-t-50d-nu-0.5"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 50)
TIME_BUDGET = env_float("HEAVY_TAILS_TIME_BUDGET", 10.0, minimum=0.001)

# Generalized Fixed Geometry
# Fixed proposal-tail parameter used by the original ν=0.5 benchmark.
BETA_VAL = 0.2
REFERENCE_SAMPLES = env_int("HEAVY_TAILS_REFERENCE_SAMPLES", 10_000_000)
REFERENCE_BATCH_SIZE = env_int("HEAVY_TAILS_REFERENCE_BATCH_SIZE", 500_000)
OPTIMIZATION_STEPS = env_int("HEAVY_TAILS_OPTIMIZATION_STEPS", 1_000, minimum=0)
OPTIMIZATION_SAMPLES = env_int("HEAVY_TAILS_OPTIMIZATION_SAMPLES", 2_000)
WARMUP_STEPS = env_int("HEAVY_TAILS_WARMUP_STEPS", 1_500, minimum=0)

Q_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

# ==============================================================================
# 2. VARIATIONAL OPTIMIZERS (KL DIVERGENCE)
# ==============================================================================
def optimize_prwm_hyperparameters():
    print("\n" + "="*80)
    print(f"OPTIMIZING PRWM/PBW (Location μ, Scale R) WITH FIXED BETA = {BETA_VAL}".center(80))
    print("="*80)
    
    @jax.jit
    def prwm_kl_loss(params, key):
        mu = params[:D]
        log_R = params[-1]
        R = jnp.exp(log_R)
        
        # Sample z ~ Uniform(B(0, 1))
        k1, k2 = jax.random.split(key)
        v = jax.random.normal(k1, (OPTIMIZATION_SAMPLES, D))
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        u = jax.random.uniform(k2, (OPTIMIZATION_SAMPLES, 1)) ** (1.0 / D)
        z = v * u
        rho = jnp.clip(jnp.linalg.norm(z, axis=1), 0.0, 1.0 - 1e-7)
        
        # THE GENERALIZED EXACT CANCELLATION MATH (Stable for ANY Beta!)
        eps = jnp.clip(1.0 - rho**BETA_VAL, 1e-12, 1.0)
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        # Stabilized Core Polynomial
        x_mod = mu * eps_pow1[:, None] + R * z
        S = NU * eps_pow2 + jnp.sum(x_mod**2, axis=1)
        
        # Canceled Log Density
        core_log = (NU / BETA_VAL - 1.0) * jnp.log(eps) + D * log_R - ((NU + D) / 2.0) * jnp.log(S)
        
        # Stabilized Skewness CDF
        x_scaled = x_mod / jnp.sqrt(S)[:, None]
        arg = jnp.dot(x_scaled, jnp.array(ALPHA)) * jnp.sqrt(NU + D)
        log_cdf = jstats.norm.logcdf(jnp.clip(arg, -35.0, 35.0))
        
        return jnp.mean(-(core_log + log_cdf))

    @jax.jit
    def step(params, state, key):
        loss, grads = jax.value_and_grad(prwm_kl_loss)(params, key)
        t, m, v = state
        t += 1
        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * (grads ** 2)
        m_hat = m / (1 - 0.9 ** t)
        v_hat = v / (1 - 0.999 ** t)
        params = params - 0.05 * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        return params, (t, m, v), loss

    params = jnp.concatenate([jnp.zeros(D), jnp.array([jnp.log(1.0)])])
    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % max(1, OPTIMIZATION_STEPS // 4) == 0:
            print(f"Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[-1]):.4f}")

    return np.array(params[:D]), float(jnp.exp(params[-1]))

def optimize_scs_hyperparameters(lo=1.1):
    """Development-only SCS geometry optimizer; not used by the paper preset."""
    print("\n" + "="*80)
    print("OPTIMIZING SCS (Location μ, Scale R, Observer ho)".center(80))
    print("="*80)
    
    def log_prob_skew_t_jax(x):
        r2 = jnp.sum(x**2, axis=-1)
        log_td = (jgammaln((NU + D) / 2.0) - jgammaln(NU / 2.0) - (D / 2.0) * jnp.log(NU * jnp.pi) - ((NU + D) / 2.0) * jnp.log1p(r2 / NU))
        arg = jnp.dot(x, jnp.array(ALPHA)) * jnp.sqrt((NU + D) / (NU + r2))
        return jnp.log(2.0) + log_td + jstats.norm.logcdf(jnp.clip(arg, -35.0, 35.0))
        
    @jax.jit
    def scs_kl_loss(params, key):
        mu, raw_ho, log_R = params[:D], params[D:2*D], params[-1]
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
        params = params - 0.05 * m_hat / (jnp.sqrt(v_hat) + 1e-8)
        return params, (t, m, v), loss

    params = jnp.concatenate([jnp.zeros(D), jnp.ones(D) * 1e-4, jnp.array([jnp.log(1.0)])]) 
    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % max(1, OPTIMIZATION_STEPS // 4) == 0:
            print(f"Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[-1]):.4f}")

    mu_opt = np.array(params[:D])
    raw_ho = params[D:2*D]
    r_max = np.sqrt(1.0 - (lo - 1.0)**2) - 1e-4
    ho_norm = np.sqrt(np.sum(raw_ho**2) + 1e-12)
    ho_opt = np.array(r_max * np.tanh(ho_norm) * (raw_ho / ho_norm))
    return mu_opt, ho_opt, float(jnp.exp(params[-1]))

# ==============================================================================
# 3. MONTE CARLO REFERENCE GENERATOR & WORKER UTILS
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
        return jnp.sum(norms >= THRESHOLD), jnp.sum(jnp.log10(1.0 + norms)), skew_t[:, :4]

    key = jax.random.PRNGKey(999)
    sum_indicator, sum_log_norm = 0.0, 0.0
    exact_samples_4d = np.empty((n_samples, 4))
    
    for i in range(n_samples // batch_size):
        key, subkey = jax.random.split(key)
        ind_sum, log_norm_sum, skew_t_4 = generate_batch(subkey)
        sum_indicator += float(ind_sum)
        sum_log_norm += float(log_norm_sum)
        exact_samples_4d[i * batch_size:(i + 1) * batch_size, :] = np.array(skew_t_4)
        
    return np.quantile(exact_samples_4d, Q_LEVELS, axis=0), sum_indicator / n_samples, sum_log_norm / n_samples

def log_prob_skew_t_np(y, nu, alpha, d):
    from scipy.stats import t as student_t
    r2 = np.sum(y**2)
    log_td = (gammaln((nu + d) / 2) - gammaln(nu / 2) - (d / 2) * np.log(nu * np.pi) - ((nu + d) / 2) * np.log1p(r2 / nu))
    arg = np.dot(alpha, y) * np.sqrt((nu + d) / (nu + r2))
    return np.log(2.0) + log_td + np.log(student_t.cdf(np.clip(arg, -35.0, 35.0), df=nu + d) + 1e-15)

def process_samples(samples):
    if len(samples) > 0:
        norms = np.linalg.norm(samples, axis=1)
        return np.quantile(samples[:, :4], Q_LEVELS, axis=0), np.mean(norms >= THRESHOLD), np.mean(np.log10(1.0 + norms))
    return np.zeros((len(Q_LEVELS), 4)), 0.0, 0.0

# ==============================================================================
# 4. WORKERS (WITH GENERALIZED ANALYTICAL STABILIZATION)
# ==============================================================================
def worker_prwm(args):
    time_budget, seed, opt_mu, opt_R = args
    rng = np.random.default_rng(seed)
    
    z = rng.normal(0, 0.1, D)
    z = z / np.linalg.norm(z) * 0.1
    
    # THE GENERALIZED EXACT CANCELLATION MATH (Stable for ANY Beta!)
    def get_log_pi_z_stable(z_pos):
        from scipy.stats import norm
        rho = np.linalg.norm(z_pos)
        if rho >= 0.9999999: return -1e20 
        
        eps = 1.0 - rho**BETA_VAL
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        x_mod = opt_mu * eps_pow1 + opt_R * z_pos
        S = NU * eps_pow2 + np.sum(x_mod**2)
        
        core_log = (NU / BETA_VAL - 1.0) * np.log(eps) + D * np.log(opt_R) - ((NU + D) / 2.0) * np.log(S)
        
        x_scaled = x_mod / np.sqrt(S)
        arg = np.dot(ALPHA, x_scaled) * np.sqrt(NU + D)
        return core_log + norm.logcdf(arg)
        
    lp_curr = get_log_pi_z_stable(z)
    
    h = 0.1
    mu_da, target_acc, gamma_da, t0, kappa = np.log(10 * h), 0.234, 0.05, 10.0, 0.75
    h_bar_log, H_bar = 0.0, 0.0

    for w in range(1, WARMUP_STEPS + 1):
        z_prop = z + rng.normal(0, h, D)
        rho_prop = np.linalg.norm(z_prop)
        if rho_prop < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)
            alpha = min(1.0, np.exp(lp_prop - lp_curr))
            if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
                z, lp_curr = z_prop, lp_prop
        else: alpha = 0.0
            
        H_bar = (1.0 - 1.0/(w + t0)) * H_bar + (1.0/(w + t0)) * (target_acc - alpha)
        log_h = mu_da - (np.sqrt(w) / gamma_da) * H_bar
        h = np.exp(log_h)
        weight = w**(-kappa)
        h_bar_log = weight * log_h + (1.0 - weight) * h_bar_log
        
    h = np.exp(h_bar_log) 

    trace, iters, accepts, start_time = [], 0, 0, time.time()
    while (time.time() - start_time) < time_budget:
        iters += 1
        z_prop = z + rng.normal(0, h, D)
        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)
            if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
                z, lp_curr = z_prop, lp_prop
                accepts += 1
        if iters % 10 == 0: trace.append(z.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn:]
    
    if len(final_z) > 0:
        nz_norm = np.linalg.norm(final_z, axis=1, keepdims=True)
        final_x = opt_mu + opt_R * final_z / (1.0 - nz_norm**BETA_VAL)**(1.0 / BETA_VAL)
    else: final_x = np.empty((0, D))
        
    q, ind, norm = process_samples(final_x)
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': iters, 'acc_rate': accepts / max(1, iters), 'step': h}

def worker_pbw(args):
    time_budget, seed, opt_mu, opt_R = args
    rng = np.random.default_rng(seed)
    
    z = rng.normal(0, 0.1, D)
    z = z / np.linalg.norm(z) * 0.1
    
    def get_log_pi_z_stable(z_pos):
        from scipy.stats import norm
        rho = np.linalg.norm(z_pos)
        if rho >= 0.9999999: return -1e20 
        
        eps = 1.0 - rho**BETA_VAL
        eps_pow1 = eps**(1.0 / BETA_VAL)
        eps_pow2 = eps**(2.0 / BETA_VAL)
        
        x_mod = opt_mu * eps_pow1 + opt_R * z_pos
        S = NU * eps_pow2 + np.sum(x_mod**2)
        
        core_log = (NU / BETA_VAL - 1.0) * np.log(eps) + D * np.log(opt_R) - ((NU + D) / 2.0) * np.log(S)
        
        x_scaled = x_mod / np.sqrt(S)
        arg = np.dot(ALPHA, x_scaled) * np.sqrt(NU + D)
        return core_log + norm.logcdf(arg)
        
    lp_curr = get_log_pi_z_stable(z)
    
    r_step = 0.1
    mu_da, target_acc, gamma_da, t0, kappa = np.log(10 * r_step), 0.234, 0.05, 10.0, 0.75
    h_bar_log, H_bar = 0.0, 0.0

    for w in range(1, WARMUP_STEPS + 1):
        direction = rng.normal(0, 1, D)
        direction /= np.linalg.norm(direction)
        rho = r_step * (rng.random() ** (1.0 / D))
        z_prop = z + rho * direction
        
        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)
            alpha = min(1.0, np.exp(lp_prop - lp_curr))
            if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
                z, lp_curr = z_prop, lp_prop
        else: alpha = 0.0
            
        H_bar = (1.0 - 1.0/(w + t0)) * H_bar + (1.0/(w + t0)) * (target_acc - alpha)
        log_r = mu_da - (np.sqrt(w) / gamma_da) * H_bar
        r_step = np.exp(log_r)
        weight = w**(-kappa)
        h_bar_log = weight * log_r + (1.0 - weight) * h_bar_log
        
    r_step = np.exp(h_bar_log)

    trace, iters, accepts, start_time = [], 0, 0, time.time()
    while (time.time() - start_time) < time_budget:
        iters += 1
        direction = rng.normal(0, 1, D)
        direction /= np.linalg.norm(direction)
        rho = r_step * (rng.random() ** (1.0 / D))
        z_prop = z + rho * direction
        
        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)
            if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
                z, lp_curr = z_prop, lp_prop
                accepts += 1
        if iters % 10 == 0: trace.append(z.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn:]
    
    if len(final_z) > 0:
        nz_norm = np.linalg.norm(final_z, axis=1, keepdims=True)
        final_x = opt_mu + opt_R * final_z / (1.0 - nz_norm**BETA_VAL)**(1.0 / BETA_VAL)
    else: final_x = np.empty((0, D))
        
    q, ind, norm = process_samples(final_x)
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': iters, 'acc_rate': accepts / max(1, iters), 'step': r_step}

def worker_scs(args):
    """Vanilla SCS Worker (Bounded Step Size)"""
    time_budget, seed = args
    rng = np.random.default_rng(seed)
    lo = 1.1
    x = np.concatenate([np.zeros(D), [-1.0]]) 
    
    def log_prob_skew_t_np_scs(y, nu, alpha, d):
        r2 = np.sum(y**2)
        log_td = (gammaln((nu + d) / 2) - gammaln(nu / 2) - (d / 2) * np.log(nu * np.pi) - ((nu + d) / 2) * np.log1p(r2 / nu))
        arg = np.dot(alpha, y) * np.sqrt((nu + d) / (nu + r2))
        from scipy.stats import norm
        return np.log(2.0) + log_td + norm.logcdf(np.clip(arg, -35.0, 35.0))

    def log_target_sphere(x_sphere):
        h_x, l_x = x_sphere[:-1], x_sphere[-1] + 1.0
        denom = lo - l_x
        y = (lo / denom) * h_x
        M = 1.0 - l_x / lo
        num = M * np.sum(y**2) + lo - (lo**2)*(1.0 - M)
        log_den = D * np.log(max(M, 1e-10)) + np.log(lo)
        log_J = np.log(max(num, 1e-10)) - log_den
        return log_prob_skew_t_np_scs(y, NU, ALPHA, D) + log_J, y
        
    lp_curr, y_curr = log_target_sphere(x)
    
    h = 0.1
    mu_da, target_acc, gamma_da, t0, kappa = np.log(10 * h), 0.234, 0.05, 10.0, 0.75
    h_bar_log, H_bar = 0.0, 0.0

    for w in range(1, WARMUP_STEPS + 1):
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
        alpha = min(1.0, np.exp(lp_prop - lp_curr))
        if np.log(rng.random() + 1e-30) < lp_prop - lp_curr:
            x, lp_curr, y_curr = x_prop, lp_prop, y_prop
            
        H_bar = (1.0 - 1.0/(w + t0)) * H_bar + (1.0/(w + t0)) * (target_acc - alpha)
        log_h = mu_da - (np.sqrt(w) / gamma_da) * H_bar
        
        h = min(np.exp(log_h), np.pi) 
        weight = w**(-kappa)
        h_bar_log = weight * np.log(h) + (1.0 - weight) * h_bar_log
        
    h = min(np.exp(h_bar_log), np.pi)

    trace, iters, accepts, start_time = [], 0, 0, time.time()
    while (time.time() - start_time) < time_budget:
        iters += 1
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
            
        if iters % 10 == 0: trace.append(y_curr.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    q, ind, norm = process_samples(trace[burn:])
    return {'quantiles': q, 'mean_indicator': ind, 'mean_norm': norm, 'iters': iters, 'acc_rate': accepts / max(1, iters), 'step': h}

# ==============================================================================
# 5. ORCHESTRATION & PLOTTING
# ==============================================================================
def run_comparison():
    pr_mu, pr_R = optimize_prwm_hyperparameters()
    true_q, true_indicator, true_norm = get_monte_carlo_reference()
    
    cores = worker_count(NUM_RUNS)
    print(f"\n--- Launching {NUM_RUNS} chains for PRWM, PBW, and SCS ({TIME_BUDGET}s each) ---")
    with multiprocessing.Pool(processes=cores) as pool:
        prwm_results = pool.map(worker_prwm, [(TIME_BUDGET, 142 + i, pr_mu, pr_R) for i in range(NUM_RUNS)])
        pbw_results = pool.map(worker_pbw, [(TIME_BUDGET, 242 + i, pr_mu, pr_R) for i in range(NUM_RUNS)])
        scs_results = pool.map(worker_scs, [(TIME_BUDGET, 342 + i) for i in range(NUM_RUNS)])

    all_results = [prwm_results, pbw_results, scs_results]
    labels = [f'DCS-RWM (β={BETA_VAL})', f'DCS-BW (β={BETA_VAL})', 'SCS']
    colors = ['#FF8C00', '#9370DB', '#FF1493']

    avg_iters = [np.mean([r['iters'] for r in res]) for res in all_results]

    print("\n" + "="*95)
    print(f"SAMPLING EFFICIENCY (TIME-EQUATED: {TIME_BUDGET}s)".center(95))
    print("="*95)
    for i in range(len(labels)):
        acc_str = f" | Acc: {np.mean([r['acc_rate'] for r in all_results[i]]):.1%}"
        step_str = f" | Final Step: {np.mean([r['step'] for r in all_results[i]]):.4f}"
        print(f"{labels[i]:<25} Avg Iterations: {avg_iters[i]:<6.0f}{acc_str}{step_str}")
    print("="*95)

    extract = lambda results, key: [r[key] for r in results]
    q_all = [np.array(extract(res, 'quantiles')) for res in all_results]
    data_indicator = [extract(res, 'mean_indicator') for res in all_results]
    data_norm = [extract(res, 'mean_norm') for res in all_results]

    mse_indicator = [np.mean((np.array(d) - true_indicator)**2) for d in data_indicator]
    mse_norm = [np.mean((np.array(d) - true_norm)**2) for d in data_norm]

    print("\n" + "="*95)
    print(f"ERGODIC AVERAGES & MEAN SQUARED ERROR (ACROSS {NUM_RUNS} CHAINS)".center(95))
    print("="*95)
    print(f"{'Sampler':<25} | {'E[I{||x||_2>=50}]':<16} | {'MSE':<16} | {'E[log10(1+||x||_2)]':<17} | {'MSE':<16}")
    print("-" * 95)
    print(f"{'Monte Carlo reference':<25} | {true_indicator:<16.6f} | {'-':<16} | {true_norm:<17.6f} | {'-':<16}")
    print("-" * 95)
    for i, name in enumerate(labels):
        print(f"{name:<25} | {np.mean(data_indicator[i]):<16.6f} | {mse_indicator[i]:<16.2e} | {np.mean(data_norm[i]):<17.6f} | {mse_norm[i]:<16.2e}")
    print("="*95)

    fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
    fig1.suptitle(f"Marginal Q–Q Diagnostics: Skew-t (d={D}, ν={NU})", fontsize=16)
    for dim in range(4):
        ax = axes1[dim]
        exact_dim_q = true_q[:, dim]
        ax.plot([np.min(exact_dim_q), np.max(exact_dim_q)], [np.min(exact_dim_q), np.max(exact_dim_q)], 'k--', alpha=0.5, label='y = x')
        for i, name in enumerate(labels):
            dim_data = q_all[i][:, :, dim]
            ax.plot(exact_dim_q, np.median(dim_data, axis=0), marker='o', markersize=4, color=colors[i], label=name, alpha=0.8)
            ax.fill_between(exact_dim_q, np.percentile(dim_data, 5, axis=0), np.percentile(dim_data, 95, axis=0), color=colors[i], alpha=0.15)
        ax.set_title(f"$x_{dim+1}$")
        ax.set_xlabel("Monte Carlo Reference Quantiles")
        ax.set_ylabel("Estimated Quantiles")
        ax.grid(True, linestyle=':', alpha=0.6)
        if dim == 0: ax.legend()
    fig1.tight_layout()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(f"Chainwise Tail and Norm Estimates (d={D}, ν={NU}; {NUM_RUNS} chains)", fontsize=16, fontweight='bold')
    def make_boxplot(ax, data, exact_val, title):
        bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True, meanline=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.axhline(exact_val, color='r', linestyle='--', linewidth=2, label=f'Reference: {exact_val:.4f}')
        ax.set_title(title, fontsize=14); ax.grid(axis='y', linestyle=':', alpha=0.7); ax.tick_params(axis='x', rotation=15); ax.legend()
    make_boxplot(axes2[0], data_indicator, true_indicator, rf"$\mathbb{{E}}[\mathbb{{I}}\{{\lVert x\rVert_2 \geq {THRESHOLD}\}}]$")
    make_boxplot(axes2[1], data_norm, true_norm, r"$\mathbb{E}[\log_{10}(1+\lVert x\rVert_2)]$")
    fig2.tight_layout()

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle(f"Estimator Error and Sampling-Output Diagnostics ({TIME_BUDGET}s budget)", fontsize=16, fontweight='bold')
    def make_barplot(ax, heights, title, ylabel, is_scientific=True):
        bars = ax.bar(labels, heights, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title(title, fontsize=14); ax.set_ylabel(ylabel); ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        for bar in bars:
            h = bar.get_height()
            label_text = f'{h:.2e}' if is_scientific else f'{int(h):,}'
            ax.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    make_barplot(axes3[0], mse_indicator, rf"MSE of $\hat{{\mathbb{{E}}}}[\mathbb{{I}}\{{\lVert x\rVert_2 \geq {THRESHOLD}\}}]$", "Mean Squared Error")
    make_barplot(axes3[1], mse_norm, r"MSE of $\hat{\mathbb{E}}[\log_{10}(1+\lVert x\rVert_2)]$", "Mean Squared Error")
    make_barplot(axes3[2], avg_iters, "Average Iterations Generated", "Iterations", is_scientific=False)
    fig3.tight_layout()

    finalize_experiment(
        EXPERIMENT_ID,
        {"marginal_qq": fig1, "ergodic_averages": fig2, "mse_and_iterations": fig3},
        {
            "reference": {
                "tail_probability": true_indicator,
                "mean_log10_one_plus_euclidean_norm": true_norm,
                "samples": REFERENCE_SAMPLES,
                "seed": 999,
            },
            "mse": {"tail_probability": mse_indicator, "log_norm": mse_norm},
            "average_iterations_per_chain": dict(zip(labels, avg_iters)),
        },
        {
            "dimension": D,
            "nu": NU,
            "alpha_nonzero": {"x1": float(ALPHA[0]), "x2": float(ALPHA[1])},
            "num_runs": NUM_RUNS,
            "time_budget_seconds_per_chain": TIME_BUDGET,
            "reference_samples": REFERENCE_SAMPLES,
            "reference_batch_size": REFERENCE_BATCH_SIZE,
            "optimization_steps": OPTIMIZATION_STEPS,
            "optimization_samples": OPTIMIZATION_SAMPLES,
            "warmup_steps": WARMUP_STEPS,
            "seed_schedule": {
                "optimizer": 42,
                "monte_carlo_reference": 999,
                "dcs_rwm_chains": "142 + chain_index",
                "dcs_bw_pbw_chains": "242 + chain_index",
                "scs_chains": "342 + chain_index",
            },
            "worker_processes": cores,
            "timing": "Reference generation, optimizer fitting, and adaptation are outside the timed sampling budget.",
        },
    )


def main() -> None:
    multiprocessing.freeze_support()
    configure_multiprocessing()
    run_comparison()


if __name__ == "__main__":
    main()
