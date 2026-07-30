"""Robust logistic-regression benchmark with Student-t priors.

The experiment compares DCS-BW/PBW and SCS for ν=4 and ν=6.  Its Q–Q panels
are a *pooled-sampler diagnostic*: the horizontal reference pools samples from
the two evaluated samplers and is not independent ground truth.  Use
``python run_experiment.py logistic-regression`` for saved outputs.
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
jax.config.update("jax_enable_x64", True)  
import jax.numpy as jnp
from jax.scipy.special import gammaln as jgammaln

# ==============================================================================
# 1. CONFIGURATION & DATA GENERATION
# ==============================================================================
D = 20
N = 50
PRIOR_SCALE = 2.5
NU_VALUES = [4, 6]

EXPERIMENT_ID = "logistic-regression"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 20)
TIME_BUDGET = env_float("HEAVY_TAILS_TIME_BUDGET", 20.0, minimum=0.001)
BETA_VAL = 2.0
OPTIMIZATION_STEPS = env_int("HEAVY_TAILS_OPTIMIZATION_STEPS", 4_000, minimum=0)
OPTIMIZATION_SAMPLES = env_int("HEAVY_TAILS_OPTIMIZATION_SAMPLES", 2_000)
WARMUP_STEPS = env_int("HEAVY_TAILS_WARMUP_STEPS", 1_500, minimum=0)

Q_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

np.random.seed(42)
X_raw = np.random.randn(N, D) 
X_np = (X_raw - np.mean(X_raw, axis=0)) / (np.std(X_raw, axis=0) * 2.0) 
Y_np = (X_np[:, 0] > 0).astype(float) 

X_jax = jnp.array(X_np)
Y_jax = jnp.array(Y_np)

# ==============================================================================
# 2. TARGET DENSITIES (Parametrized by NU)
# ==============================================================================
@jax.jit
def log_prob_target_jax(beta_batch, nu):
    beta_batch = jnp.atleast_2d(beta_batch)
    term1 = jgammaln((nu + 1.0) / 2.0) - jgammaln(nu / 2.0) - 0.5 * jnp.log(nu * jnp.pi * PRIOR_SCALE**2)
    log_prior = jnp.sum(term1 - ((nu + 1.0) / 2.0) * jnp.log1p((beta_batch / PRIOR_SCALE)**2 / nu), axis=-1)
    
    logits = jnp.dot(beta_batch, X_jax.T) 
    log_lik = jnp.sum(Y_jax * logits - jnp.logaddexp(0.0, logits), axis=-1)
    return jnp.squeeze(log_prior + log_lik)

def log_prob_target_np(beta, nu):
    from scipy.stats import t as student_t
    log_prior = np.sum(student_t.logpdf(beta, df=nu, loc=0.0, scale=PRIOR_SCALE))
    logits = np.dot(X_np, beta)
    log_lik = np.sum(Y_np * logits - np.logaddexp(0.0, logits))
    return log_prior + log_lik

# ==============================================================================
# 3. VARIATIONAL OPTIMIZERS
# ==============================================================================
def optimize_prwm_hyperparameters(nu):
    print(f"\n--- OPTIMIZING PRWM/PBW (ν={nu}) ---")
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
        x = mu + R * w / eps_pow1[:, None]
        log_J_radial = D * log_R - ((D + BETA_VAL) / BETA_VAL) * jnp.log(eps)
        
        lp = log_prob_target_jax(x, nu)
        return jnp.mean(-(log_J_radial + log_J_moebius + lp))

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

    params = jnp.concatenate([jnp.zeros(D), jnp.zeros(D), jnp.array([jnp.log(1.0)])])
    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % max(1, OPTIMIZATION_STEPS // 8) == 0:
            print(f"  Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[-1]):.4f}")

    mu_opt = np.array(params[:D])
    raw_delta = np.array(params[D:2*D])
    delta_norm = np.sqrt(np.sum(raw_delta**2) + 1e-12)
    delta_opt = np.tanh(delta_norm) * (raw_delta / delta_norm) * 0.999
    return mu_opt, delta_opt, float(jnp.exp(params[-1]))

def optimize_scs_hyperparameters(nu):
    print(f"--- OPTIMIZING SCS (ν={nu}, Fixed l_o=1.1) ---")
    fixed_lo = 1.1 
    
    @jax.jit
    def scs_kl_loss(params, key):
        mu = params[:D]
        raw_ho = params[D:2*D]
        log_R = params[2*D]
        R = jnp.exp(log_R)

        r_max = jnp.sqrt(1.0 - (fixed_lo - 1.0)**2) - 1e-4
        ho_norm = jnp.sqrt(jnp.sum(raw_ho**2) + 1e-12)
        ho = r_max * jnp.tanh(ho_norm) * (raw_ho / ho_norm)

        v = jax.random.normal(key, (OPTIMIZATION_SAMPLES, D+1))
        u = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        l_x, h_x = u[:, D] + 1.0, u[:, :D]
        
        valid_mask = l_x < (fixed_lo - 1e-5)
        safe_l_x = jnp.where(valid_mask, l_x, fixed_lo - 1e-5)
        
        denom = fixed_lo - safe_l_x
        y_hat = (fixed_lo / denom[:, None]) * h_x - (safe_l_x / denom)[:, None] * ho
        y = R * y_hat + mu

        M = 1.0 - safe_l_x / fixed_lo
        y_hat_minus_ho = y_hat - ho
        num = M * jnp.sum(y_hat_minus_ho**2, axis=1) + jnp.sum(y_hat_minus_ho * ho, axis=1) + fixed_lo - (fixed_lo**2)*(1.0 - M)
        log_den = D * jnp.log(jnp.maximum(M, 1e-10)) + jnp.log(fixed_lo)
        log_J = D * log_R + jnp.log(jnp.maximum(num, 1e-10)) - log_den

        lp = log_prob_target_jax(y, nu)
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
        
        # Learning rate used by the experiment configuration.
        params = params - 0.01 * m_hat / (jnp.sqrt(v_hat) + 1e-8) 
        return params, (t, m, v), loss

    params = jnp.concatenate([jnp.zeros(D), jnp.zeros(D), jnp.array([jnp.log(1.0)])]) 
    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)
        if i % max(1, OPTIMIZATION_STEPS // 8) == 0:
            print(f"  Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[2*D]):.4f}")

    mu_opt = np.array(params[:D])
    raw_ho = np.array(params[D:2*D])
    
    r_max = np.sqrt(1.0 - (fixed_lo - 1.0)**2) - 1e-4
    ho_norm = np.sqrt(np.sum(raw_ho**2) + 1e-12)
    ho_opt = np.array(r_max * np.tanh(ho_norm) * (raw_ho / ho_norm))
    
    return mu_opt, ho_opt, float(jnp.exp(params[2*D])), fixed_lo


# ==============================================================================
# 4. WORKER UTILS & SAMPLERS
# ==============================================================================
def worker_prwm(args):
    time_budget, seed, opt_mu, opt_delta, opt_R, nu = args
    rng = np.random.default_rng(seed)
    delta_sq = np.sum(opt_delta**2)
    z = rng.normal(0, 0.1, D)
    z = z / np.linalg.norm(z) * 0.1
    
    def get_log_pi_z_stable(z_pos):
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
        x = opt_mu + opt_R * w / eps_pow1
        log_J_radial = D * np.log(opt_R) - ((D + BETA_VAL) / BETA_VAL) * np.log(eps)
        return log_prob_target_np(x, nu) + log_J_radial + log_J_moebius
        
    lp_curr = get_log_pi_z_stable(z)
    h = 0.1
    mu_da, target_acc, gamma_da, t0, kappa = np.log(10 * h), 0.234, 0.05, 10.0, 0.75
    h_bar_log, H_bar = 0.0, 0.0

    for w in range(1, WARMUP_STEPS + 1):
        z_prop = z + rng.normal(0, h, D)
        if np.linalg.norm(z_prop) < 1.0:
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
        trace.append(z.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn::10] 
    
    if len(final_z) > 0:
        dot_z_delta = np.sum(final_z * opt_delta, axis=1)
        z_norm2 = np.sum(final_z**2, axis=1)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
        term1 = (1.0 - delta_sq) * final_z
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2)[:, None] * opt_delta
        w = (term1 + term2) / denom[:, None]
        w_norm = np.linalg.norm(w, axis=1, keepdims=True)
        eps_pow1 = (1.0 - w_norm**BETA_VAL)**(1.0 / BETA_VAL)
        final_x = opt_mu + opt_R * w / eps_pow1
    else: 
        final_x = np.empty((0, D))
        
    return {'trace': final_x, 'iters': iters, 'acc_rate': accepts / max(1, iters), 'step': h}

def worker_pbw(args):
    time_budget, seed, opt_mu, opt_delta, opt_R, nu = args
    rng = np.random.default_rng(seed)
    delta_sq = np.sum(opt_delta**2)
    z = rng.normal(0, 0.1, D)
    z = z / np.linalg.norm(z) * 0.1
    
    def get_log_pi_z_stable(z_pos):
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
        x = opt_mu + opt_R * w / eps_pow1
        log_J_radial = D * np.log(opt_R) - ((D + BETA_VAL) / BETA_VAL) * np.log(eps)
        return log_prob_target_np(x, nu) + log_J_radial + log_J_moebius
        
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
        trace.append(z.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn::10]
    
    if len(final_z) > 0:
        dot_z_delta = np.sum(final_z * opt_delta, axis=1)
        z_norm2 = np.sum(final_z**2, axis=1)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2
        term1 = (1.0 - delta_sq) * final_z
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2)[:, None] * opt_delta
        w = (term1 + term2) / denom[:, None]
        w_norm = np.linalg.norm(w, axis=1, keepdims=True)
        eps_pow1 = (1.0 - w_norm**BETA_VAL)**(1.0 / BETA_VAL)
        final_x = opt_mu + opt_R * w / eps_pow1
    else: 
        final_x = np.empty((0, D))
        
    return {'trace': final_x, 'iters': iters, 'acc_rate': accepts / max(1, iters), 'step': r_step}

def worker_scs(args):
    time_budget, seed, opt_mu, opt_ho, opt_R, opt_lo, nu = args
    rng = np.random.default_rng(seed)
    lo = opt_lo
    x = np.concatenate([np.zeros(D), [-1.0]]) 
    
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
        return log_prob_target_np(y, nu) + log_J, y
        
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
            
        trace.append(y_curr.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    return {'trace': trace[burn::10], 'iters': iters, 'acc_rate': accepts / max(1, iters), 'step': h}


# ==============================================================================
# 5. ORCHESTRATION & PLOTTING
# ==============================================================================
def run_comparison():
    # Only compare PBW and SCS
    samplers = ['DCS-BW', 'SCS']
    colors = {
        'DCS-BW': '#9370DB',
        'SCS': '#FF1493'
    }

    traces_dict = {nu: {'DCS-BW': [], 'SCS': []} for nu in NU_VALUES}
    metrics_dict = {nu: {} for nu in NU_VALUES}

    cores = worker_count(NUM_RUNS)

    summary_statistics = {}
    for nu in NU_VALUES:
        # PBW uses the PRWM/PBW variational map parameters
        pr_mu, pr_delta, pr_R = optimize_prwm_hyperparameters(nu)

        # SCS uses its own optimized parameters
        sc_mu, sc_ho, sc_R, sc_lo = optimize_scs_hyperparameters(nu)

        print(
            f"\n--- Launching {NUM_RUNS} chains for PBW and SCS "
            f"(ν={nu}) with {TIME_BUDGET}s budget ---"
        )

        with multiprocessing.Pool(processes=cores) as pool:
            pbw_results = pool.map(
                worker_pbw,
                [
                    (TIME_BUDGET, 242 + i, pr_mu, pr_delta, pr_R, nu)
                    for i in range(NUM_RUNS)
                ]
            )

            scs_results = pool.map(
                worker_scs,
                [
                    (TIME_BUDGET, 342 + i, sc_mu, sc_ho, sc_R, sc_lo, nu)
                    for i in range(NUM_RUNS)
                ]
            )

        metrics_dict[nu]['DCS-BW'] = pbw_results
        metrics_dict[nu]['SCS'] = scs_results

        for name, res_group in zip(['DCS-BW', 'SCS'], [pbw_results, scs_results]):
            min_len = min(len(r['trace']) for r in res_group)
            stacked = np.array([r['trace'][-min_len:] for r in res_group])
            traces_dict[nu][name] = stacked

    # ---------------------------------------------------------
    # Print Efficiency, Mean & Variance for β1
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("SAMPLER PERFORMANCE & β_1 ESTIMATES".center(85))
    print("=" * 85)
    print(
        f"{'ν':<3} | {'Sampler':<6} | {'Avg Iters':<10} | "
        f"{'Acc Rate':<8} | {'Step Size':<10} | "
        f"{'Mean(β_1)':<10} | {'Var(β_1)':<15}"
    )
    print("-" * 85)

    for nu in NU_VALUES:
        summary_statistics[str(nu)] = {}
        for sampler in samplers:
            res_group = metrics_dict[nu][sampler]

            avg_iters = np.mean([r['iters'] for r in res_group])
            avg_acc = np.mean([r['acc_rate'] for r in res_group])
            avg_step = np.mean([r['step'] for r in res_group])

            beta_1_samples = traces_dict[nu][sampler][:, :, 0].flatten()
            mean_est = np.mean(beta_1_samples)
            var_est = np.var(beta_1_samples)

            print(
                f"{int(nu):<3} | {sampler:<6} | {avg_iters:<10.0f} | "
                f"{avg_acc:<8.1%} | {avg_step:<10.4f} | "
                f"{mean_est:<10.4f} | {var_est:<15.4f}"
            )
            summary_statistics[str(nu)][sampler] = {
                "average_iterations_per_chain": avg_iters,
                "average_acceptance_rate": avg_acc,
                "average_step_size": avg_step,
                "beta_1_mean": mean_est,
                "beta_1_variance": var_est,
            }

        print("-" * 85)

    # ---------------------------------------------------------
    # Generate 2x2 pooled-sampler Q–Q diagnostic table.
    # Rows: ν values
    # Columns: PBW and SCS
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        len(NU_VALUES),
        len(samplers),
        figsize=(12, 10),
        squeeze=False
    )

    fig.suptitle(
        r"Pooled-Sampler Q–Q Diagnostic for $\beta_1$ in Robust Logistic Regression",
        fontsize=16
    )

    for row, nu in enumerate(NU_VALUES):
        # This is deliberately labeled as a pooled-sampler diagnostic, not truth.
        all_stacked_beta1 = np.concatenate([
            traces_dict[nu][s][:, :, 0].flatten()
            for s in samplers
        ])

        reference_q = np.quantile(all_stacked_beta1, Q_LEVELS)

        for col, sampler in enumerate(samplers):
            ax = axes[row, col]

            # y = x reference line
            ax.plot(
                [np.min(reference_q), np.max(reference_q)],
                [np.min(reference_q), np.max(reference_q)],
                'k-.',
                alpha=0.5,
                label='y=x' if row == 0 and col == 0 else None
            )

            # Quantiles across chains
            chain_quantiles = np.quantile(
                traces_dict[nu][sampler][:, :, 0],
                Q_LEVELS,
                axis=1
            )

            median_q = np.median(chain_quantiles, axis=1)
            lower_q = np.percentile(chain_quantiles, 5, axis=1)
            upper_q = np.percentile(chain_quantiles, 95, axis=1)

            ax.plot(
                reference_q,
                median_q,
                marker='o',
                markersize=4,
                color=colors[sampler],
                label=sampler
            )

            ax.fill_between(
                reference_q,
                lower_q,
                upper_q,
                color=colors[sampler],
                alpha=0.15
            )

            ax.set_title(f"{sampler}, $\\nu = {int(nu)}$")
            ax.grid(True, linestyle=':', alpha=0.6)

            if row == len(NU_VALUES) - 1:
                ax.set_xlabel("Pooled PBW+SCS Quantiles (diagnostic)")

            if col == 0:
                ax.set_ylabel("Sampler Quantiles")

            if row == 0 and col == 0:
                ax.legend(loc='upper left')

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    finalize_experiment(
        EXPERIMENT_ID,
        {"pooled_sampler_qq": fig},
        {
            "sampler_statistics": summary_statistics,
            "qq_reference": "Pooled DCS-BW/PBW and SCS samples; diagnostic only, not independent ground truth.",
        },
        {
            "observations": N,
            "dimension": D,
            "prior_scale": PRIOR_SCALE,
            "nu_values": NU_VALUES,
            "data_seed": 42,
            "num_runs": NUM_RUNS,
            "time_budget_seconds_per_chain": TIME_BUDGET,
            "optimization_steps": OPTIMIZATION_STEPS,
            "optimization_samples": OPTIMIZATION_SAMPLES,
            "warmup_steps": WARMUP_STEPS,
            "seed_schedule": {
                "synthetic_data": 42,
                "optimizers_per_nu": 42,
                "dcs_bw_pbw_chains_per_nu": "242 + chain_index",
                "scs_chains_per_nu": "342 + chain_index",
            },
            "worker_processes": cores,
            "timing": "Optimizer fitting and warmup are outside the timed sampling budget.",
        },
    )

    return fig, axes

def main() -> None:
    multiprocessing.freeze_support()
    configure_multiprocessing()
    run_comparison()


if __name__ == "__main__":
    main()
