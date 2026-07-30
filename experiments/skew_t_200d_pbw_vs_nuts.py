"""200-dimensional skew-t benchmark: DCS-BW/PBW versus BlackJAX NUTS.

The paper-scale configuration remains the module default.  Use
``python run_experiment.py skew-t-200d-nuts`` for a saved, headless run or its
``--preset paper`` profile to retain those defaults.
"""

import os
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import norm
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
D = 200
NU = 3.0  
ALPHA = np.zeros(D)
ALPHA[0] = 20.0
ALPHA[1] = -30.0
THRESHOLD = 50.0

EXPERIMENT_ID = "skew-t-200d-nuts"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 50)
TIME_BUDGET = env_float("HEAVY_TAILS_TIME_BUDGET", 20.0, minimum=0.001)
BETA_VAL = 2.0
REFERENCE_SAMPLES = env_int("HEAVY_TAILS_REFERENCE_SAMPLES", 10_000_000)
REFERENCE_BATCH_SIZE = env_int("HEAVY_TAILS_REFERENCE_BATCH_SIZE", 500_000)
OPTIMIZATION_STEPS = env_int("HEAVY_TAILS_OPTIMIZATION_STEPS", 1_500, minimum=0)
OPTIMIZATION_SAMPLES = env_int("HEAVY_TAILS_OPTIMIZATION_SAMPLES", 2_000)
WARMUP_STEPS = env_int("HEAVY_TAILS_WARMUP_STEPS", 1_500, minimum=0)
NUTS_WARMUP = env_int("HEAVY_TAILS_NUTS_WARMUP", 1_000, minimum=0)

Q_LEVELS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

# ==============================================================================
# 2. VARIATIONAL OPTIMIZER (KL DIVERGENCE) FOR PBW
# ==============================================================================
def optimize_prwm_hyperparameters(target_log_prob, D, proposal_beta):
    print("\n" + "="*90)
    print(f"OPTIMIZING PBW MAPPING (Location μ, Scale R, Möbius Shift δ)".center(90))
    print(f"Fixed: β={proposal_beta}".center(90))
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
        eps = jnp.clip(1.0 - w_norm**proposal_beta, 1e-12, 1.0)
        eps_pow1 = eps**(1.0 / proposal_beta)
        
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
        if i % max(1, OPTIMIZATION_STEPS // 6) == 0:
            print(f"Iter {i:4d} | KL Loss: {loss:.4f} | R: {jnp.exp(params[-1]):.4f}")

    mu_opt = np.array(params[:D])
    raw_delta = np.array(params[D:2*D])
    delta_norm = np.sqrt(np.sum(raw_delta**2) + 1e-12)
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

def log_prob_skew_t_np(y, nu, alpha, d):
    r2 = np.sum(y**2)
    log_td = (gammaln((nu + d) / 2) - gammaln(nu / 2) - (d / 2) * np.log(nu * np.pi) - ((nu + d) / 2) * np.log1p(r2 / nu))
    arg = np.dot(alpha, y) * np.sqrt((nu + d) / (nu + r2))
    return np.log(2.0) + log_td + norm.logcdf(np.clip(arg, -35.0, 35.0))

def numpy_target_skew_t(x):
    return log_prob_skew_t_np(x, NU, ALPHA, D)

@jax.jit
def jax_target_skew_t(x):
    r2 = jnp.sum(x**2, axis=-1)
    log_td = (jgammaln((NU + D) / 2.0) - jgammaln(NU / 2.0) - (D / 2.0) * jnp.log(NU * jnp.pi) - ((NU + D) / 2.0) * jnp.log1p(r2 / NU))
    arg = jnp.dot(x, jnp.array(ALPHA)) * jnp.sqrt((NU + D) / (NU + r2))
    return jnp.log(2.0) + log_td + jstats.norm.logcdf(jnp.clip(arg, -35.0, 35.0))

def process_samples(samples):
    if len(samples) > 0:
        norms = np.linalg.norm(samples, axis=1)
        return np.quantile(samples[:, :4], Q_LEVELS, axis=0), np.mean(norms >= THRESHOLD), np.mean(norms)
    return np.zeros((len(Q_LEVELS), 4)), 0.0, 0.0

# ==============================================================================
# 4. SAMPLERS: PBW WORKER AND JAX NUTS
# ==============================================================================
def worker_pbw(args):
    time_budget, seed, opt_mu, opt_delta, opt_R, target_np = args
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
        
        eps = max(1.0 - w_norm**BETA_VAL, 1e-12)
        eps_pow1 = eps**(1.0 / BETA_VAL)
        
        x_final = opt_mu + opt_R * w / eps_pow1
        log_J_radial = D * np.log(opt_R) - (D / BETA_VAL + 1.0) * np.log(eps)
        
        return target_np(x_final) + log_J_radial + log_J_moebius
        
    lp_curr = get_log_pi_z_stable(z)
    
    # --- Tuning Phase (Outside Budget) ---
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

    # --- Timed Sampling Phase ---
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
        
        # Only save every 10th step to match NUTS and save memory
        if iters % 10 == 0: trace.append(z.copy())
            
    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn:]
    
    mean_ess = 0.0
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
        
        # Calculate Effective Sample Size (using first 10 dims for speed)
        try:
            ess_vals = az.ess(np.expand_dims(final_x[:, :10], 0))
            mean_ess = np.mean(list(ess_vals.values()))
        except Exception:
            mean_ess = 0.0
    else: 
        final_x = np.empty((0, D))
        
    q, ind, norm = process_samples(final_x)
    return {
        'quantiles': q, 
        'mean_indicator': ind, 
        'mean_norm': norm, 
        'iters': iters, 
        'acc_rate': accepts / max(1, iters), 
        'step': r_step,
        'ess': mean_ess
    }

def worker_nuts(args):
    time_budget, seed = args
    rng_key = jax.random.PRNGKey(seed)
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    
    initial_position = jax.random.normal(init_key, (D,))
    
    # --- Tuning Phase (Outside Budget) ---
    adapt = blackjax.window_adaptation(blackjax.nuts, jax_target_skew_t)
    (base_state, parameters), _ = adapt.run(
        warmup_key,
        initial_position,
        num_steps=NUTS_WARMUP,
    )
    
    kernel = blackjax.nuts(
        jax_target_skew_t, 
        **parameters, 
        max_num_doublings=6  
    ).step
    
    # Use the state produced by adaptation, rather than discarding warmup.
    state = base_state
    
    @jax.jit
    def step(state, key):
        state, info = kernel(key, state)
        return state, state.position, info.acceptance_rate

    # Dummy run to force compilation before starting the clock
    dummy_key = jax.random.split(sample_key, 1)[0]
    state, _, _ = step(state, dummy_key)
    state.position.block_until_ready()

    # --- Timed Sampling Phase ---
    start_time = time.time()
    samples, acceptance = [], []
    iters = 0
    
    while (time.time() - start_time) < time_budget:
        step_key = jax.random.fold_in(sample_key, iters)
        state, pos, acc = step(state, step_key)
        
        # Synchronize before timing/host transfer on accelerator backends.
        pos.block_until_ready()
        # Save every 10th step to match PBW thinning.
        if iters % 10 == 0:
            samples.append(np.array(pos))
        acceptance.append(float(acc))
        iters += 1
        
    samples_np = np.array(samples)
    burn = int(len(samples_np) * 0.2) if len(samples_np) > 10 else 0
    final_chain = samples_np[burn:]
    
    mean_ess = 0.0
    if len(final_chain) > 0:
        q, ind, norm = process_samples(final_chain)
        # Calculate Effective Sample Size (using first 10 dims for speed)
        try:
            ess_vals = az.ess(np.expand_dims(final_chain[:, :10], 0))
            mean_ess = np.mean(list(ess_vals.values()))
        except Exception:
            mean_ess = 0.0
    else:
        q, ind, norm = np.zeros((len(Q_LEVELS), 4)), 0.0, 0.0
        
    return {
        'quantiles': q, 
        'mean_indicator': ind, 
        'mean_norm': norm, 
        'iters': iters, 
        'acc_rate': np.mean(acceptance) if acceptance else 0.0,
        'ess': mean_ess
    }

# ==============================================================================
# 5. ORCHESTRATION & PLOTTING
# ==============================================================================
def run_comparison():
    pr_mu, pr_delta, pr_R = optimize_prwm_hyperparameters(jax_target_skew_t, D, proposal_beta=BETA_VAL)
    true_q, true_indicator, true_norm = get_monte_carlo_reference()
    
    cores = worker_count(NUM_RUNS)
    
    print(f"\n--- Launching PBW {NUM_RUNS} parallel chains ({TIME_BUDGET}s each) on {cores} cores ---")
    with multiprocessing.Pool(processes=cores) as pool:
        pbw_results = pool.map(worker_pbw, [(TIME_BUDGET, 242 + i, pr_mu, pr_delta, pr_R, numpy_target_skew_t) for i in range(NUM_RUNS)])
        
    print(f"--- Launching NUTS {NUM_RUNS} parallel chains ({TIME_BUDGET}s each) on {cores} cores ---")
    with multiprocessing.Pool(processes=cores) as pool:
        nuts_results = pool.map(worker_nuts, [(TIME_BUDGET, 842 + i) for i in range(NUM_RUNS)])

    all_results = [pbw_results, nuts_results]
    labels = [
        f'Optimized DCS-BW (β={BETA_VAL})', 
        'BlackJAX NUTS'
    ]
    colors = ['#9370DB', '#FF8C00']

    avg_iters = [np.mean([r['iters'] for r in res]) for res in all_results]
    avg_ess = [np.sum([r['ess'] for r in res]) for res in all_results]

    print("\n" + "="*115)
    print(f"SAMPLING EFFICIENCY (TIME-EQUATED PURE SAMPLING: {TIME_BUDGET}s)".center(115))
    print("="*115)
    for i in range(len(labels)):
        acc_str = f" | Acc: {np.mean([r['acc_rate'] for r in all_results[i]]):.1%}"
        ess_str = f" | Total ESS: {avg_ess[i]:<7.0f} | ESS/sec: {avg_ess[i]/TIME_BUDGET:<6.1f}"
        print(f"{labels[i]:<30} Avg Iters/Chain: {avg_iters[i]:<7.0f}{acc_str}{ess_str}")
    print("="*115)

    extract = lambda results, key: [r[key] for r in results]
    q_all = [np.array(extract(res, 'quantiles')) for res in all_results]
    data_indicator = [extract(res, 'mean_indicator') for res in all_results]
    data_norm = [extract(res, 'mean_norm') for res in all_results]

    mse_indicator = [np.mean((np.array(d) - true_indicator)**2) for d in data_indicator]
    mse_norm = [np.mean((np.array(d) - true_norm)**2) for d in data_norm]

    print("\n" + "="*105)
    print(f"ERGODIC AVERAGES & MEAN SQUARED ERROR (ACROSS {NUM_RUNS} CHAINS)".center(105))
    print("="*105)
    print(f"{'Sampler':<32} | {'E[I{||x||_2>=50}]':<16} | {'MSE':<16} | {'E[||x||_2]':<17} | {'MSE':<16}")
    print("-" * 105)
    print(f"{'Monte Carlo reference':<32} | {true_indicator:<16.6f} | {'-':<16} | {true_norm:<17.6f} | {'-':<16}")
    print("-" * 105)
    for i, name in enumerate(labels):
        print(f"{name:<32} | {np.mean(data_indicator[i]):<16.6f} | {mse_indicator[i]:<16.2e} | {np.mean(data_norm[i]):<17.6f} | {mse_norm[i]:<16.2e}")
    print("="*105)

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
    make_boxplot(axes2[1], data_norm, true_norm, r"$\mathbb{E}[\lVert x\rVert_2]$")
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
    make_barplot(axes3[1], mse_norm, r"MSE of $\hat{\mathbb{E}}[\lVert x\rVert_2]$", "Mean Squared Error")
    make_barplot(axes3[2], avg_ess, "Total Effective Sample Size (ESS)", "ESS", is_scientific=False) # Plot ESS instead of raw iterations!
    fig3.tight_layout()

    finalize_experiment(
        EXPERIMENT_ID,
        {"marginal_qq": fig1, "ergodic_averages": fig2, "mse_and_ess": fig3},
        {
            "reference": {
                "tail_probability": true_indicator,
                "euclidean_norm_mean": true_norm,
                "samples": REFERENCE_SAMPLES,
                "seed": 999,
            },
            "mse": {"tail_probability": mse_indicator, "euclidean_norm_mean": mse_norm},
            "average_iterations_per_chain": dict(zip(labels, avg_iters)),
            "aggregate_ess": dict(zip(labels, avg_ess)),
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
            "pbw_warmup_steps": WARMUP_STEPS,
            "nuts_warmup_steps": NUTS_WARMUP,
            "seed_schedule": {
                "optimizer": 42,
                "monte_carlo_reference": 999,
                "dcs_bw_pbw_chains": "242 + chain_index",
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
