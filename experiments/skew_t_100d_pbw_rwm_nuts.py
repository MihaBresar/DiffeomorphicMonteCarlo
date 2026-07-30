"""100-dimensional skew-t benchmark with projection sampling and baselines.

It compares DCS-BW/PBW, Gaussian random-walk Metropolis, symmetric
alpha-stable random-walk Metropolis, and BlackJAX NUTS.  Use
``python run_experiment.py skew-t-100d-baselines`` for reproducible artifacts.
"""

import os
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, log_ndtr
import arviz as az

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
import jax.scipy.stats as jstats
import blackjax

# ==============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
D = 100
NU = 3.0

ALPHA = np.zeros(D)
ALPHA[0] = 20.0
ALPHA[1] = -30.0

THRESHOLD = 50.0

EXPERIMENT_ID = "skew-t-100d-baselines"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 50)
TIME_BUDGET = env_float("HEAVY_TAILS_TIME_BUDGET", 10.0, minimum=0.001)
BETA_VAL = 2.0
REFERENCE_SAMPLES = env_int("HEAVY_TAILS_REFERENCE_SAMPLES", 10_000_000)
REFERENCE_BATCH_SIZE = env_int("HEAVY_TAILS_REFERENCE_BATCH_SIZE", 500_000)
OPTIMIZATION_STEPS = env_int("HEAVY_TAILS_OPTIMIZATION_STEPS", 1_500, minimum=0)
OPTIMIZATION_SAMPLES = env_int("HEAVY_TAILS_OPTIMIZATION_SAMPLES", 2_000)
NUTS_WARMUP = env_int("HEAVY_TAILS_NUTS_WARMUP", 1_000, minimum=0)
PBW_WARMUP = env_int("HEAVY_TAILS_WARMUP_STEPS", 1_500, minimum=0)

Q_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

# --- RWM settings ---
RWM_WARMUP = env_int("HEAVY_TAILS_WARMUP_STEPS", 1_500, minimum=0)

# RWM is now fast. Storing every full 100D sample can use too much RAM.
# We store only first 10 dimensions + full norm. RWM_THIN=2 is safer on 32 workers.
# If you have enough RAM, you can set this to 1.
RWM_THIN = 2

RWM_TARGET_ACC = 0.234
STABLE_ALPHA = 0.5

MAX_ABS_POSITION = 1e100
RWM_TIME_CHECK_INTERVAL = 128

# RWM diagnostic storage: first 4 dimensions for Q–Q, first 10 for ESS, norm separately.
RWM_STORE_DIMS = min(D, 10)
RWM_STORAGE_CHUNK = 65_536

# ==============================================================================
# Fast NumPy target constants
# ==============================================================================
LOG2 = np.log(2.0)
INV_NU = 1.0 / NU

LOG_T_CONST = (
    gammaln((NU + D) / 2.0)
    - gammaln(NU / 2.0)
    - (D / 2.0) * np.log(NU * np.pi)
)

LOG_T_POWER = 0.5 * (NU + D)
ARG_NUMERATOR = NU + D

ALPHA_NZ = np.flatnonzero(ALPHA)
ALPHA_NZ_VALS = ALPHA[ALPHA_NZ].astype(np.float64)

ALPHA_JAX = jnp.asarray(ALPHA)

# ==============================================================================
# 2. VARIATIONAL OPTIMIZER FOR PROJECTED / PBW METHOD
# ==============================================================================
def optimize_prwm_hyperparameters(target_log_prob, D, proposal_beta):
    print("\n" + "=" * 90)
    print("OPTIMIZING PBW MAPPING (Location μ, Scale R, Möbius Shift δ)".center(90))
    print(f"Fixed: β={proposal_beta}".center(90))
    print("=" * 90)

    @jax.jit
    def prwm_kl_loss(params, key):
        mu = params[:D]
        raw_delta = params[D:2 * D]
        log_R = params[-1]
        R = jnp.exp(log_R)

        delta_norm = jnp.sqrt(jnp.sum(raw_delta ** 2) + 1e-12)
        delta = jnp.tanh(delta_norm) * (raw_delta / delta_norm) * 0.999
        delta_sq = jnp.sum(delta ** 2)

        k1, k2 = jax.random.split(key)

        v = jax.random.normal(k1, (OPTIMIZATION_SAMPLES, D))
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)

        u = jax.random.uniform(k2, (OPTIMIZATION_SAMPLES, 1)) ** (1.0 / D)
        z = v * u

        rho2 = jnp.sum(z ** 2, axis=1)

        dot_z_delta = jnp.dot(z, delta)
        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * rho2

        term1 = (1.0 - delta_sq) * z
        term2 = (1.0 + 2.0 * dot_z_delta + rho2)[:, None] * delta[None, :]

        w = (term1 + term2) / denom[:, None]

        log_J_moebius = D * jnp.log(1.0 - delta_sq) - D * jnp.log(denom)

        w_norm = jnp.clip(jnp.linalg.norm(w, axis=1), 0.0, 1.0 - 1e-7)
        eps = jnp.clip(1.0 - w_norm ** proposal_beta, 1e-12, 1.0)
        eps_pow1 = eps ** (1.0 / proposal_beta)

        x_final = mu + R * w / eps_pow1[:, None]

        log_J_radial = D * log_R - (D / proposal_beta + 1.0) * jnp.log(eps)

        lp = target_log_prob(x_final)

        return jnp.mean(-(lp + log_J_radial + log_J_moebius))

    @jax.jit
    def step(params, state, key):
        loss, grads = jax.value_and_grad(prwm_kl_loss)(params, key)

        t, m, v = state
        t += 1

        m = 0.9 * m + 0.1 * grads
        v = 0.999 * v + 0.001 * (grads ** 2)

        m_hat = m / (1.0 - 0.9 ** t)
        v_hat = v / (1.0 - 0.999 ** t)

        params = params - 0.05 * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        return params, (t, m, v), loss

    params = jnp.concatenate(
        [
            jnp.zeros(D),
            jnp.zeros(D),
            jnp.array([jnp.log(1.0)]),
        ]
    )

    state = (0, jnp.zeros_like(params), jnp.zeros_like(params))
    key = jax.random.PRNGKey(42)

    for i in range(OPTIMIZATION_STEPS + 1):
        key, subkey = jax.random.split(key)
        params, state, loss = step(params, state, subkey)

        if i % max(1, OPTIMIZATION_STEPS // 6) == 0:
            print(f"Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[-1]):.4f}")

    mu_opt = np.array(params[:D])

    raw_delta = np.array(params[D:2 * D])
    delta_norm = np.sqrt(np.sum(raw_delta ** 2) + 1e-12)
    delta_opt = np.tanh(delta_norm) * (raw_delta / delta_norm) * 0.999

    return mu_opt, delta_opt, float(jnp.exp(params[-1]))

# ==============================================================================
# 3. MONTE CARLO REFERENCE GENERATOR & TARGET DEFINITIONS
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

        return (
            jnp.sum(norms >= THRESHOLD),
            jnp.sum(norms),
            skew_t[:, :4],
        )

    key = jax.random.PRNGKey(999)

    sum_indicator = 0.0
    sum_norm = 0.0

    exact_samples_4d = np.empty((n_samples, 4))

    for i in range(n_samples // batch_size):
        key, subkey = jax.random.split(key)

        ind_sum, norm_sum, skew_t_4 = generate_batch(subkey)

        sum_indicator += float(ind_sum)
        sum_norm += float(norm_sum)

        exact_samples_4d[i * batch_size:(i + 1) * batch_size, :] = np.array(skew_t_4)

    return (
        np.quantile(exact_samples_4d, Q_LEVELS, axis=0),
        sum_indicator / n_samples,
        sum_norm / n_samples,
    )


def numpy_target_skew_t(x):
    """
    Fast scalar NumPy log-density for the current skew-t target.
    Optimized to remove array-dispatch overhead for scalars.
    """
    r2 = float(np.dot(x, x))
    log_td = LOG_T_CONST - LOG_T_POWER * np.log1p(r2 * INV_NU)

    if ALPHA_NZ.size:
        dot_alpha = float(np.dot(ALPHA_NZ_VALS, x[ALPHA_NZ]))
    else:
        dot_alpha = 0.0

    arg = dot_alpha * np.sqrt(ARG_NUMERATOR / (NU + r2))
    
    # Fast scalar clip outperforms np.clip on single values
    if arg < -35.0: 
        arg = -35.0
    elif arg > 35.0: 
        arg = 35.0

    return float(LOG2 + log_td + log_ndtr(arg))


def log_prob_skew_t_np(y, nu, alpha, d):
    """
    Generic slower version retained for compatibility.
    Workers should use numpy_target_skew_t.
    """
    y = np.asarray(y, dtype=np.float64)

    r2 = float(np.dot(y, y))

    log_td = (
        gammaln((nu + d) / 2.0)
        - gammaln(nu / 2.0)
        - (d / 2.0) * np.log(nu * np.pi)
        - ((nu + d) / 2.0) * np.log1p(r2 / nu)
    )

    arg = float(np.dot(alpha, y)) * np.sqrt((nu + d) / (nu + r2))
    arg = np.clip(arg, -35.0, 35.0)

    return float(np.log(2.0) + log_td + log_ndtr(arg))


@jax.jit
def jax_target_skew_t(x):
    r2 = jnp.sum(x ** 2, axis=-1)

    log_td = (
        jgammaln((NU + D) / 2.0)
        - jgammaln(NU / 2.0)
        - (D / 2.0) * jnp.log(NU * jnp.pi)
        - ((NU + D) / 2.0) * jnp.log1p(r2 / NU)
    )

    arg = jnp.dot(x, ALPHA_JAX) * jnp.sqrt((NU + D) / (NU + r2))

    return jnp.log(2.0) + log_td + jstats.norm.logcdf(jnp.clip(arg, -35.0, 35.0))


def process_samples(samples):
    if len(samples) > 0:
        norms = np.linalg.norm(samples, axis=1)

        return (
            np.quantile(samples[:, :4], Q_LEVELS, axis=0),
            np.mean(norms >= THRESHOLD),
            np.mean(norms),
        )

    return np.zeros((len(Q_LEVELS), 4)), 0.0, 0.0


def process_rwm_reduced_samples(samples_small, norms):
    """
    samples_small contains only first RWM_STORE_DIMS coordinates.
    norms contains full-dimensional Euclidean norm.
    """
    if len(samples_small) > 0:
        return (
            np.quantile(samples_small[:, :4], Q_LEVELS, axis=0),
            np.mean(norms >= THRESHOLD),
            np.mean(norms),
        )

    return np.zeros((len(Q_LEVELS), 4)), 0.0, 0.0


def mean_ess_first_dims(samples, max_dims=10, max_points=150_000):
    """
    Robust ESS helper for one chain.
    Uses only first max_dims coordinates.
    If very long, uses an evenly spaced diagnostic subset.
    """
    try:
        samples = np.asarray(samples)

        if samples.ndim != 2 or samples.shape[0] < 4:
            return 0.0

        n = samples.shape[0]

        if n > max_points:
            idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
            samples = samples[idx]

        n_dims = min(max_dims, samples.shape[1])

        ess_vals = az.ess(np.expand_dims(samples[:, :n_dims], axis=0))

        if hasattr(ess_vals, "to_array"):
            vals = ess_vals.to_array().values
        elif hasattr(ess_vals, "values"):
            vals = ess_vals.values
        else:
            vals = np.asarray(ess_vals)

        return float(np.nanmean(vals))

    except Exception:
        return 0.0


def safe_target_log_prob_np(target_np, x):
    """
    Guarded target evaluation.
    """
    try:
        absmax = np.max(np.abs(x))

        if (not np.isfinite(absmax)) or absmax > MAX_ABS_POSITION:
            return -np.inf

        lp = target_np(x)

        if not np.isfinite(lp):
            return -np.inf

        return float(lp)

    except Exception:
        return -np.inf

# ==============================================================================
# 4. SAMPLERS
# ==============================================================================

# ------------------------------------------------------------------------------
# 4a. PROJECTED / PBW WORKER
# ------------------------------------------------------------------------------
def worker_pbw(args):
    time_budget, seed, opt_mu, opt_delta, opt_R, target_np = args

    rng = np.random.default_rng(seed)

    delta_sq = np.sum(opt_delta ** 2)

    z = rng.normal(0.0, 0.1, D)
    z = z / np.linalg.norm(z) * 0.1

    def get_log_pi_z_stable(z_pos):
        z_norm2 = np.sum(z_pos ** 2)

        if z_norm2 >= 0.9999999:
            return -1e20

        dot_z_delta = np.dot(z_pos, opt_delta)

        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2

        term1 = (1.0 - delta_sq) * z_pos
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2) * opt_delta

        w = (term1 + term2) / denom

        log_J_moebius = D * np.log(1.0 - delta_sq) - D * np.log(denom)

        w_norm = np.linalg.norm(w)

        if w_norm >= 0.9999999:
            return -1e20

        eps = max(1.0 - w_norm ** BETA_VAL, 1e-12)
        eps_pow1 = eps ** (1.0 / BETA_VAL)

        x_final = opt_mu + opt_R * w / eps_pow1

        log_J_radial = D * np.log(opt_R) - (D / BETA_VAL + 1.0) * np.log(eps)

        return target_np(x_final) + log_J_radial + log_J_moebius

    lp_curr = get_log_pi_z_stable(z)

    # Tuning phase outside budget
    r_step = 0.1

    mu_da = np.log(10.0 * r_step)
    target_acc = 0.234
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75

    h_bar_log = 0.0
    H_bar = 0.0

    for w_iter in range(1, PBW_WARMUP + 1):
        direction = rng.normal(0.0, 1.0, D)
        direction /= np.linalg.norm(direction)

        rho = r_step * (rng.random() ** (1.0 / D))

        z_prop = z + rho * direction

        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)

            log_alpha = lp_prop - lp_curr
            alpha_acc = min(1.0, np.exp(min(0.0, log_alpha)))

            if np.log(rng.random() + 1e-300) < log_alpha:
                z = z_prop
                lp_curr = lp_prop
        else:
            alpha_acc = 0.0

        H_bar = (
            (1.0 - 1.0 / (w_iter + t0)) * H_bar
            + (1.0 / (w_iter + t0)) * (target_acc - alpha_acc)
        )

        log_r = mu_da - (np.sqrt(w_iter) / gamma_da) * H_bar
        r_step = np.exp(log_r)

        weight = w_iter ** (-kappa)
        h_bar_log = weight * log_r + (1.0 - weight) * h_bar_log

    r_step = np.exp(h_bar_log)

    # Timed sampling phase
    trace = []
    iters = 0
    accepts = 0

    start_time = time.time()

    while (time.time() - start_time) < time_budget:
        iters += 1

        direction = rng.normal(0.0, 1.0, D)
        direction /= np.linalg.norm(direction)

        rho = r_step * (rng.random() ** (1.0 / D))

        z_prop = z + rho * direction

        if np.linalg.norm(z_prop) < 1.0:
            lp_prop = get_log_pi_z_stable(z_prop)

            log_alpha = lp_prop - lp_curr

            if np.log(rng.random() + 1e-300) < log_alpha:
                z = z_prop
                lp_curr = lp_prop
                accepts += 1

        if iters % 10 == 0:
            trace.append(z.copy())

    trace = np.array(trace)

    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn:]

    mean_ess = 0.0

    if len(final_z) > 0:
        dot_z_delta = np.sum(final_z * opt_delta, axis=1)
        z_norm2 = np.sum(final_z ** 2, axis=1)

        denom = 1.0 + 2.0 * dot_z_delta + delta_sq * z_norm2

        term1 = (1.0 - delta_sq) * final_z
        term2 = (1.0 + 2.0 * dot_z_delta + z_norm2)[:, None] * opt_delta

        w = (term1 + term2) / denom[:, None]

        w_norm = np.linalg.norm(w, axis=1, keepdims=True)
        eps = np.maximum(1.0 - w_norm ** BETA_VAL, 1e-12)
        eps_pow1 = eps ** (1.0 / BETA_VAL)

        final_x = opt_mu + opt_R * w / eps_pow1

        mean_ess = mean_ess_first_dims(final_x)

    else:
        final_x = np.empty((0, D))

    q, ind, norm_val = process_samples(final_x)

    return {
        "quantiles": q,
        "mean_indicator": ind,
        "mean_norm": norm_val,
        "iters": iters,
        "stored_samples": len(trace),
        "acc_rate": accepts / max(1, iters),
        "step": r_step,
        "ess": mean_ess,
    }

# ------------------------------------------------------------------------------
# 4b. RANDOM-WALK METROPOLIS HELPERS
# ------------------------------------------------------------------------------
def symmetric_alpha_stable_rvs(rng, alpha, size):
    """
    Chambers-Mallows-Stuck sampler for iid symmetric alpha-stable variables.
    For alpha=1, use NumPy's fast standard Cauchy sampler.
    """
    if not (0.0 < alpha < 2.0):
        raise ValueError("alpha must be in (0, 2).")

    if np.isclose(alpha, 1.0):
        return rng.standard_cauchy(size=size)

    U = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)
    W = rng.exponential(1.0, size=size)

    tiny = np.finfo(np.float64).tiny

    sin_alpha_U = np.sin(alpha * U)
    sign = np.sign(sin_alpha_U)

    cos_U = np.maximum(np.cos(U), tiny)
    cos_term = np.maximum(np.cos((1.0 - alpha) * U), tiny)
    W = np.maximum(W, tiny)

    log_abs = (
        np.log(np.abs(sin_alpha_U) + tiny)
        - (1.0 / alpha) * np.log(cos_U)
        + ((1.0 - alpha) / alpha) * (np.log(cos_term) - np.log(W))
    )

    log_abs = np.clip(log_abs, -745.0, 700.0)

    out = sign * np.exp(log_abs)
    out[sign == 0.0] = 0.0

    return out


def _run_rwm(time_budget, seed, target_np, proposal_kind):
    """
    Highly optimized generic random-walk Metropolis worker.
    Uses chunked RNG generation to minimize loop overhead.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, 1.0, D)
    
    try:
        lp_curr = target_np(x)
    except Exception:
        lp_curr = -np.inf

    if not np.isfinite(lp_curr):
        x = np.zeros(D)
        lp_curr = target_np(x)

    log_step_min = -250.0
    log_step_max = 5.0

    is_gaussian = (proposal_kind == "gaussian")
    is_cauchy = (proposal_kind == "stable" and np.isclose(STABLE_ALPHA, 1.0))

    if is_gaussian:
        step_size = 2.38 / np.sqrt(D)
        proposal_name = "Gaussian RWM"
    elif proposal_kind == "stable":
        step_size = 2.38 / (D ** (1.0 / STABLE_ALPHA))
        step_size = float(np.clip(step_size, np.exp(log_step_min), np.exp(log_step_max)))
        proposal_name = f"Stable RWM alpha={STABLE_ALPHA:g}"
    else:
        raise ValueError(f"Unknown proposal_kind={proposal_kind}")

    def draw_increment(scale, size=None):
        shape = D if size is None else size
        if is_gaussian:
            return scale * rng.standard_normal(shape)
        if is_cauchy:
            return scale * rng.standard_cauchy(size=shape)
        return scale * symmetric_alpha_stable_rvs(rng, STABLE_ALPHA, shape)

    # --- Tuning Phase ---
    mu_da = np.log(10.0 * step_size)
    target_acc = RWM_TARGET_ACC
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75
    h_bar_log = 0.0
    H_bar = 0.0

    for w_iter in range(1, RWM_WARMUP + 1):
        x_prop = x + draw_increment(step_size)
        
        try:
            lp_prop = target_np(x_prop)
        except Exception:
            lp_prop = -np.inf

        log_alpha = lp_prop - lp_curr
        if not np.isfinite(log_alpha):
            log_alpha = -np.inf

        accept_prob = 1.0 if log_alpha >= 0.0 else (float(np.exp(log_alpha)) if np.isfinite(log_alpha) else 0.0)

        if np.log(rng.random() + 1e-300) < log_alpha:
            x = x_prop
            lp_curr = lp_prop

        H_bar = (1.0 - 1.0 / (w_iter + t0)) * H_bar + (1.0 / (w_iter + t0)) * (target_acc - accept_prob)
        log_step = mu_da - (np.sqrt(w_iter) / gamma_da) * H_bar
        
        if log_step < log_step_min: log_step = log_step_min
        elif log_step > log_step_max: log_step = log_step_max
        
        step_size = float(np.exp(log_step))
        weight = w_iter ** (-kappa)
        h_bar_log = weight * log_step + (1.0 - weight) * h_bar_log

    step_size = float(np.exp(np.clip(h_bar_log, log_step_min, log_step_max)))

    # --- Timed Sampling Phase ---
    accepts = 0
    iters = 0

    storage_capacity = RWM_STORAGE_CHUNK
    samples_small = np.empty((storage_capacity, RWM_STORE_DIMS), dtype=np.float64)
    norms = np.empty(storage_capacity, dtype=np.float64)
    n_stored = 0

    # Chunk allocations to bypass RNG call overhead
    CHUNK_SIZE = 16384
    chunk_idx = CHUNK_SIZE
    rand_uniforms = np.empty(CHUNK_SIZE)
    increments = np.empty((CHUNK_SIZE, D))

    deadline = time.perf_counter() + time_budget

    while time.perf_counter() < deadline:
        for _ in range(RWM_TIME_CHECK_INTERVAL):
            
            # Refill the RNG chunks when empty
            if chunk_idx >= CHUNK_SIZE:
                rand_uniforms = np.log(rng.random(CHUNK_SIZE) + 1e-300)
                increments = draw_increment(step_size, size=(CHUNK_SIZE, D))
                chunk_idx = 0

            iters += 1
            x_prop = x + increments[chunk_idx]
            log_u = rand_uniforms[chunk_idx]
            chunk_idx += 1

            try:
                lp_prop = target_np(x_prop)
            except Exception:
                lp_prop = -np.inf

            log_alpha = lp_prop - lp_curr

            if log_u < log_alpha:
                x = x_prop
                lp_curr = lp_prop
                accepts += 1

            if iters % RWM_THIN == 0:
                # Inline capacity management for speed
                if n_stored >= storage_capacity:
                    new_capacity = storage_capacity + RWM_STORAGE_CHUNK
                    new_samples = np.empty((new_capacity, RWM_STORE_DIMS), dtype=np.float64)
                    new_norms = np.empty(new_capacity, dtype=np.float64)
                    
                    new_samples[:n_stored] = samples_small[:n_stored]
                    new_norms[:n_stored] = norms[:n_stored]
                    
                    samples_small = new_samples
                    norms = new_norms
                    storage_capacity = new_capacity

                samples_small[n_stored, :] = x[:RWM_STORE_DIMS]
                norms[n_stored] = np.sqrt(np.dot(x, x))
                n_stored += 1

    # Cleanup and processing
    samples_small = samples_small[:n_stored]
    norms = norms[:n_stored]

    burn = int(n_stored * 0.2) if n_stored > 10 else 0
    final_small = samples_small[burn:]
    final_norms = norms[burn:]

    if len(final_small) > 0:
        q, ind, norm_val = process_rwm_reduced_samples(final_small, final_norms)
        mean_ess = mean_ess_first_dims(final_small)
    else:
        q = np.zeros((len(Q_LEVELS), 4))
        ind, norm_val, mean_ess = 0.0, 0.0, 0.0

    return {
        "quantiles": q,
        "mean_indicator": ind,
        "mean_norm": norm_val,
        "iters": iters,
        "stored_samples": n_stored,
        "acc_rate": accepts / max(1, iters),
        "step": step_size,
        "ess": mean_ess,
        "proposal": proposal_name,
    }


def worker_rwm_gaussian(args):
    time_budget, seed, target_np = args

    return _run_rwm(
        time_budget=time_budget,
        seed=seed,
        target_np=target_np,
        proposal_kind="gaussian",
    )


def worker_rwm_stable(args):
    time_budget, seed, target_np = args

    return _run_rwm(
        time_budget=time_budget,
        seed=seed,
        target_np=target_np,
        proposal_kind="stable",
    )

# ------------------------------------------------------------------------------
# 4c. JAX NUTS WORKER
# ------------------------------------------------------------------------------
def worker_nuts(args):
    time_budget, seed = args

    rng_key = jax.random.PRNGKey(seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    initial_position = jax.random.normal(init_key, (D,))

    # Tuning phase outside budget
    adapt = blackjax.window_adaptation(blackjax.nuts, jax_target_skew_t)

    (base_state, parameters), _ = adapt.run(
        warmup_key,
        initial_position,
        num_steps=NUTS_WARMUP,
    )

    kernel = blackjax.nuts(
        jax_target_skew_t,
        **parameters,
        max_num_doublings=6,
    ).step

    state = base_state

    @jax.jit
    def step(state, key):
        state, info = kernel(key, state)
        return state, state.position, info.acceptance_rate

    # Compile before timing
    compile_key, sample_key = jax.random.split(sample_key)

    state, _, _ = step(state, compile_key)
    state.position.block_until_ready()

    # Timed sampling phase
    start_time = time.time()

    samples = []
    acceptance = []
    iters = 0

    while (time.time() - start_time) < time_budget:
        step_key = jax.random.fold_in(sample_key, iters)

        state, pos, acc = step(state, step_key)

        # Make timing more honest for JAX async execution.
        pos.block_until_ready()

        if iters % 10 == 0:
            samples.append(np.array(pos))

        acceptance.append(float(acc))
        iters += 1

    samples_np = np.array(samples)

    burn = int(len(samples_np) * 0.2) if len(samples_np) > 10 else 0
    final_chain = samples_np[burn:]

    if len(final_chain) > 0:
        q, ind, norm_val = process_samples(final_chain)
        mean_ess = mean_ess_first_dims(final_chain)
    else:
        q = np.zeros((len(Q_LEVELS), 4))
        ind = 0.0
        norm_val = 0.0
        mean_ess = 0.0

    return {
        "quantiles": q,
        "mean_indicator": ind,
        "mean_norm": norm_val,
        "iters": iters,
        "stored_samples": len(samples_np),
        "acc_rate": np.mean(acceptance) if acceptance else 0.0,
        "ess": mean_ess,
    }

# ==============================================================================
# 5. ORCHESTRATION & PLOTTING
# ==============================================================================
def run_comparison():
    pr_mu, pr_delta, pr_R = optimize_prwm_hyperparameters(
        jax_target_skew_t,
        D,
        proposal_beta=BETA_VAL,
    )

    true_q, true_indicator, true_norm = get_monte_carlo_reference()

    cores = worker_count(NUM_RUNS)

    # --------------------------------------------------------------------------
    # Projected / PBW
    # --------------------------------------------------------------------------
    print(
        f"\n--- Launching PBW {NUM_RUNS} parallel chains "
        f"({TIME_BUDGET}s each) on {cores} cores ---"
    )

    with multiprocessing.Pool(processes=cores) as pool:
        pbw_results = pool.map(
            worker_pbw,
            [
                (
                    TIME_BUDGET,
                    242 + i,
                    pr_mu,
                    pr_delta,
                    pr_R,
                    numpy_target_skew_t,
                )
                for i in range(NUM_RUNS)
            ],
        )

    # --------------------------------------------------------------------------
    # Gaussian RWM
    # --------------------------------------------------------------------------
    print(
        f"\n--- Launching Gaussian RWM {NUM_RUNS} parallel chains "
        f"({TIME_BUDGET}s each) on {cores} cores ---"
    )

    with multiprocessing.Pool(processes=cores) as pool:
        rwm_gaussian_results = pool.map(
            worker_rwm_gaussian,
            [
                (
                    TIME_BUDGET,
                    1242 + i,
                    numpy_target_skew_t,
                )
                for i in range(NUM_RUNS)
            ],
        )

    # --------------------------------------------------------------------------
    # Symmetric alpha-stable RWM
    # --------------------------------------------------------------------------
    print(
        f"\n--- Launching alpha={STABLE_ALPHA} symmetric-stable RWM "
        f"{NUM_RUNS} parallel chains ({TIME_BUDGET}s each) on {cores} cores ---"
    )

    with multiprocessing.Pool(processes=cores) as pool:
        rwm_stable_results = pool.map(
            worker_rwm_stable,
            [
                (
                    TIME_BUDGET,
                    2242 + i,
                    numpy_target_skew_t,
                )
                for i in range(NUM_RUNS)
            ],
        )

    # --------------------------------------------------------------------------
    # NUTS
    # --------------------------------------------------------------------------
    print(
        f"\n--- Launching NUTS {NUM_RUNS} parallel chains "
        f"({TIME_BUDGET}s each) on {cores} cores ---"
    )

    with multiprocessing.Pool(processes=cores) as pool:
        nuts_results = pool.map(
            worker_nuts,
            [
                (
                    TIME_BUDGET,
                    842 + i,
                )
                for i in range(NUM_RUNS)
            ],
        )

    all_results = [
        pbw_results,
        rwm_gaussian_results,
        rwm_stable_results,
        nuts_results,
    ]

    labels = [
        f"Optimized DCS-BW β={BETA_VAL}",
        "RWM Gaussian",
        rf"RWM Stable α={STABLE_ALPHA}",
        "JAX NUTS",
    ]

    colors = [
        "#9370DB",
        "#1E90FF",
        "#2E8B57",
        "#FF8C00",
    ]

    avg_iters = [
        np.mean([r["iters"] for r in res])
        for res in all_results
    ]

    avg_stored = [
        np.mean([r.get("stored_samples", np.nan) for r in res])
        for res in all_results
    ]

    total_ess = [
        np.sum([r["ess"] for r in res])
        for res in all_results
    ]

    avg_steps = [
        np.mean([r.get("step", np.nan) for r in res])
        for res in all_results
    ]

    print("\n" + "=" * 140)
    print(f"SAMPLING EFFICIENCY (TIME-EQUATED PURE SAMPLING: {TIME_BUDGET}s)".center(140))
    print("=" * 140)

    for i in range(len(labels)):
        acc = np.mean([r["acc_rate"] for r in all_results[i]])

        step_str = ""
        if not np.isnan(avg_steps[i]):
            step_str = f" | Step: {avg_steps[i]:.3e}"

        stored_str = ""
        if not np.isnan(avg_stored[i]):
            stored_str = f" | Stored: {avg_stored[i]:<9.0f}"

        print(
            f"{labels[i]:<32} "
            f"Avg Iters/Chain: {avg_iters[i]:<9.0f}"
            f"{stored_str}"
            f" | Acc: {acc:<7.1%}"
            f" | Total ESS: {total_ess[i]:<9.0f}"
            f" | ESS/sec: {total_ess[i] / TIME_BUDGET:<8.1f}"
            f"{step_str}"
        )

    print("=" * 140)

    extract = lambda results, key: [r[key] for r in results]

    q_all = [
        np.array(extract(res, "quantiles"))
        for res in all_results
    ]

    data_indicator = [
        extract(res, "mean_indicator")
        for res in all_results
    ]

    data_norm = [
        extract(res, "mean_norm")
        for res in all_results
    ]

    mse_indicator = [
        np.mean((np.array(d) - true_indicator) ** 2)
        for d in data_indicator
    ]

    mse_norm = [
        np.mean((np.array(d) - true_norm) ** 2)
        for d in data_norm
    ]

    print("\n" + "=" * 115)
    print(f"ERGODIC AVERAGES & MEAN SQUARED ERROR ACROSS {NUM_RUNS} CHAINS".center(115))
    print("=" * 115)

    print(
        f"{'Sampler':<34} | "
        f"{'E[I{||x||_2>=50}]':<16} | "
        f"{'MSE':<16} | "
        f"{'E[||x||_2]':<17} | "
        f"{'MSE':<16}"
    )

    print("-" * 115)

    print(
        f"{'Monte Carlo reference':<34} | "
        f"{true_indicator:<16.6f} | "
        f"{'-':<16} | "
        f"{true_norm:<17.6f} | "
        f"{'-':<16}"
    )

    print("-" * 115)

    for i, name in enumerate(labels):
        print(
            f"{name:<34} | "
            f"{np.mean(data_indicator[i]):<16.6f} | "
            f"{mse_indicator[i]:<16.2e} | "
            f"{np.mean(data_norm[i]):<17.6f} | "
            f"{mse_norm[i]:<16.2e}"
        )

    print("=" * 115)

    # --------------------------------------------------------------------------
    # Figure 1: marginal Q–Q diagnostics
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))

    fig1.suptitle(
        f"Marginal Q–Q Diagnostics: Skew-t d={D}, ν={NU}",
        fontsize=16,
    )

    for dim in range(4):
        ax = axes1[dim]

        exact_dim_q = true_q[:, dim]

        ax.plot(
            [np.min(exact_dim_q), np.max(exact_dim_q)],
            [np.min(exact_dim_q), np.max(exact_dim_q)],
            "k--",
            alpha=0.5,
            label="y = x",
        )

        for i, name in enumerate(labels):
            dim_data = q_all[i][:, :, dim]

            ax.plot(
                exact_dim_q,
                np.median(dim_data, axis=0),
                marker="o",
                markersize=4,
                color=colors[i],
                label=name,
                alpha=0.8,
            )

            ax.fill_between(
                exact_dim_q,
                np.percentile(dim_data, 5, axis=0),
                np.percentile(dim_data, 95, axis=0),
                color=colors[i],
                alpha=0.15,
            )

        ax.set_title(f"$x_{dim + 1}$")
        ax.set_xlabel("Monte Carlo Reference Quantiles")
        ax.set_ylabel("Estimated Quantiles")
        ax.grid(True, linestyle=":", alpha=0.6)

        if dim == 0:
            ax.legend(fontsize=8)

    fig1.tight_layout()

    # --------------------------------------------------------------------------
    # Figure 2: Ergodic-average boxplots
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    fig2.suptitle(
        f"Chainwise Tail and Norm Estimates (d={D}, ν={NU}; {NUM_RUNS} chains)",
        fontsize=16,
        fontweight="bold",
    )

    def make_boxplot(ax, data, exact_val, title):
        bplot = ax.boxplot(
            data,
            patch_artist=True,
            tick_labels=labels,
            showmeans=True,
            meanline=True,
        )

        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(
            exact_val,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Reference: {exact_val:.4f}",
        )

        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.7)
        ax.tick_params(axis="x", rotation=20)
        ax.legend()

    make_boxplot(
        axes2[0],
        data_indicator,
        true_indicator,
        rf"$\mathbb{{E}}[\mathbb{{I}}\{{\lVert x\rVert_2 \geq {THRESHOLD}\}}]$",
    )

    make_boxplot(
        axes2[1],
        data_norm,
        true_norm,
        r"$\mathbb{E}[\lVert x\rVert_2]$",
    )

    fig2.tight_layout()

 # --------------------------------------------------------------------------
    # Figure 3: MSE
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

    fig3.suptitle(
        f"Estimator Error and Sampling-Output Diagnostics ({TIME_BUDGET}s budget)",
        fontsize=16,
        fontweight="bold",
    )

    def make_barplot(ax, heights, title, ylabel, is_scientific=True):
        bars = ax.bar(
            labels,
            heights,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")

        for bar in bars:
            h = bar.get_height()

            if is_scientific:
                label_text = f"{h:.2e}"
            else:
                label_text = f"{int(h):,}"

            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    make_barplot(
        axes3[0],
        mse_indicator,
        rf"MSE of $\hat{{\mathbb{{E}}}}[\mathbb{{I}}\{{\lVert x\rVert_2 \geq {THRESHOLD}\}}]$",
        "Mean Squared Error",
    )

    make_barplot(
        axes3[1],
        mse_norm,
        r"MSE of $\hat{\mathbb{E}}[\lVert x\rVert_2]$",
        "Mean Squared Error",
    )

    fig3.tight_layout()

    finalize_experiment(
        EXPERIMENT_ID,
        {"marginal_qq": fig1, "ergodic_averages": fig2, "mse": fig3},
        {
            "reference": {
                "tail_probability": true_indicator,
                "euclidean_norm_mean": true_norm,
                "samples": REFERENCE_SAMPLES,
                "seed": 999,
            },
            "mse": {"tail_probability": mse_indicator, "euclidean_norm_mean": mse_norm},
            "average_iterations_per_chain": dict(zip(labels, avg_iters)),
            "aggregate_ess": dict(zip(labels, total_ess)),
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
            "rwm_warmup_steps": RWM_WARMUP,
            "pbw_warmup_steps": PBW_WARMUP,
            "nuts_warmup_steps": NUTS_WARMUP,
            "seed_schedule": {
                "optimizer": 42,
                "monte_carlo_reference": 999,
                "dcs_bw_pbw_chains": "242 + chain_index",
                "gaussian_rwm_chains": "1242 + chain_index",
                "stable_rwm_chains": "2242 + chain_index",
                "nuts_chains": "842 + chain_index",
            },
            "worker_processes": cores,
            "timing": "Reference generation, optimizer fitting, adaptation, and JIT compilation are outside the timed sampling budget.",
        },
    )


def main() -> None:
    multiprocessing.freeze_support()
    configure_multiprocessing()
    run_comparison()


if __name__ == "__main__":
    main()
