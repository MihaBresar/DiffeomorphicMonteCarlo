"""PosteriorDB GARCH(1,1) benchmark.

This module compares NumPyro NUTS with the projected slice/HnR implementation
on the GARCH(1,1) target.  The bundled data are the small PosteriorDB dataset
documented in ``data/README.md``.  Run it through
``python run_experiment.py garch11`` to save all artifacts and metadata.
"""

import os
import time
import json
import multiprocessing
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.special import expit
import matplotlib.pyplot as plt

from reproducibility import (
    configure_multiprocessing,
    env_int,
    finalize_experiment,
    worker_count,
)

# --- JAX CONFIGURATION ---
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# ==============================================================================
# 1. LOAD DATA & FIXED REFERENCE SUMMARIES
# ==============================================================================
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "raw" / "garch11.json"
DATA_PATH = Path(os.environ.get("HEAVY_TAILS_GARCH_DATA", DEFAULT_DATA_PATH)).expanduser()


def get_garch_data(data_path: Path = DATA_PATH):
    """Load and validate the bundled PosteriorDB GARCH(1,1) data."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"GARCH data not found at {data_path}. See data/README.md or set "
            "HEAVY_TAILS_GARCH_DATA to a JSON file containing T, y, and sigma1."
        )

    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"T", "y", "sigma1"}
    missing_keys = required_keys.difference(data)
    if missing_keys:
        raise ValueError(f"GARCH data are missing required keys: {sorted(missing_keys)}")

    y = np.asarray(data["y"], dtype=float)
    t_length = int(data["T"])
    sigma1 = float(data["sigma1"])
    if len(y) != t_length:
        raise ValueError(f"GARCH data declare T={t_length}, but contain {len(y)} observations.")
    return y, t_length, sigma1

def get_posteriordb_reference():
    """Return the fixed reference summaries used in the original experiment."""
    print("Loading fixed PosteriorDB reference means and second moments...")
    
    names = ["mu", "alpha0", "alpha1", "beta1"]
    mean_values = [5.05001794660039, 1.47075973803898, 0.567284282813872, 0.293024546082117]
    mean_sq_values = [25.5180634, 2.490076, 0.3379668, 0.1014308]
    
    true_means = {k: v for k, v in zip(names, mean_values)}
    true_sqs = {k: v for k, v in zip(names, mean_sq_values)}
    
    return true_means, true_sqs

# Loaded at import time so spawned workers use the same portable data path.
Y, T_len, SIGMA1 = get_garch_data()
D = 4  # mu, alpha0, alpha1, beta1

EXPERIMENT_ID = "garch11"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 50)
NUM_NUTS_SAMPLES = env_int("HEAVY_TAILS_NUTS_SAMPLES", 150_000)
NUTS_WARMUP = env_int("HEAVY_TAILS_NUTS_WARMUP", 1_000, minimum=0)

# ==============================================================================
# 2. NUMPYRO NUTS WORKER
# ==============================================================================
def worker_nuts_garch(args):
    seed, num_samples = args

    def model(y, sigma1):
        # Stan default priors are improper uniform over the constraints
        mu = numpyro.sample('mu', dist.ImproperUniform(dist.constraints.real, (), event_shape=()))
        alpha0 = numpyro.sample('alpha0', dist.ImproperUniform(dist.constraints.positive, (), event_shape=()))
        alpha1 = numpyro.sample('alpha1', dist.Uniform(0.0, 1.0))
        beta1 = numpyro.sample('beta1', dist.Uniform(0.0, 1.0 - alpha1))
        
        def transition(h_prev, y_prev):
            h_t = alpha0 + alpha1 * (y_prev - mu)**2 + beta1 * h_prev
            return h_t, h_t
            
        _, h_tail = jax.lax.scan(transition, sigma1**2, y[:-1])
        h = jnp.concatenate([jnp.array([sigma1**2]), h_tail])
        
        numpyro.sample('obs', dist.Normal(mu, jnp.sqrt(h)), obs=y)

    numpyro.set_host_device_count(1)
    start_time = time.time()
    kernel = NUTS(model, target_accept_prob=0.8)
    mcmc = MCMC(
        kernel,
        num_warmup=NUTS_WARMUP,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), y=jnp.array(Y), sigma1=SIGMA1)
    
    samples = mcmc.get_samples()
    elapsed = time.time() - start_time
    
    # Pack parameters to match PosteriorDB order
    beta_trace = np.column_stack([samples['mu'], samples['alpha0'], samples['alpha1'], samples['beta1']])
    
    return {
        'mean_x': np.mean(beta_trace, axis=0),
        'mean_x2': np.mean(beta_trace**2, axis=0),
        'time': elapsed,
        'iters': num_samples
    }

# ==============================================================================
# 3. PROJECTED SLICE / HIT-AND-RUN WORKER
# ==============================================================================
def worker_phr_slice_garch(args):
    time_budget, seed = args
    rng = np.random.default_rng(seed)

    # Initial state
    x = np.zeros(D)
    x[0] = np.mean(Y) # mu
    curr_z = x / (1.0 + np.linalg.norm(x))

    def get_target_z(z_pos):
        nz = np.linalg.norm(z_pos)
        if nz >= 0.999999: return -1e20
        
        x_m = z_pos / (1.0 - nz)
        mu_unc = x_m[0]
        alpha0_unc = x_m[1]
        alpha1_unc = x_m[2]
        beta1_unc = x_m[3]
        
        # Suppress Numpy math warnings for extreme unconstrained proposals
        with np.errstate(all='ignore'):
            # Transform unconstrained space to constrained GARCH space
            alpha0 = np.exp(alpha0_unc)
            alpha1 = expit(alpha1_unc)
            beta1 = (1.0 - alpha1) * expit(beta1_unc)
            
            # Log Jacobian of the transformation
            log_J = alpha0_unc 
            log_J += np.log(alpha1 + 1e-12) + np.log(1.0 - alpha1 + 1e-12)
            log_J += np.log(beta1 + 1e-12) + np.log(1.0 - expit(beta1_unc) + 1e-12)
            
            # Likelihood (Sequential volatility accumulation)
            lp = log_J
            curr_h = SIGMA1**2
            lp += -0.5 * np.log(curr_h) - 0.5 * (Y[0] - mu_unc)**2 / curr_h
            
            for t in range(1, T_len):
                curr_h = alpha0 + alpha1 * (Y[t-1] - mu_unc)**2 + beta1 * curr_h
                lp += -0.5 * np.log(curr_h) - 0.5 * (Y[t] - mu_unc)**2 / curr_h
                
            lp -= (D + 1) * np.log(1.0 - nz)
            
            # Safely reject any overflow/NaN conditions
            if np.isnan(lp) or np.isinf(lp):
                return -1e20
                
        return lp

    curr_log = get_target_z(curr_z)
    trace = []
    start_time = time.time()
    iters = 0
    
    while (time.time() - start_time) < time_budget:
        iters += 1
        theta_dir = rng.normal(0, 1, D)
        theta_dir /= np.linalg.norm(theta_dir)
        
        dot = np.sum(curr_z * theta_dir)
        nz_sq = np.sum(curr_z**2)
        disc = np.sqrt(max(0.0, dot**2 + (1.0 - nz_sq)))
        L, R = -dot - disc, -dot + disc
        
        log_y = curr_log + np.log(rng.random() + 1e-30)
        
        # Dynamic slice bracket
        for _ in range(50):
            eta = rng.uniform(L, R)
            z_prop = curr_z + eta * theta_dir
            lp_prop = get_target_z(z_prop)
            
            if lp_prop > log_y:
                curr_z = z_prop
                curr_log = lp_prop
                break
            else:
                if eta < 0: L = eta
                else: R = eta
                
        if iters % 10 == 0:
            trace.append(curr_z.copy())

    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn:]
    
    if len(final_z) > 0:
        nz = np.linalg.norm(final_z, axis=1, keepdims=True)
        final_x = final_z / (1.0 - nz)
        
        # Recover constrained parameters for evaluation
        final_mu = final_x[:, 0]
        final_alpha0 = np.exp(final_x[:, 1])
        final_alpha1 = expit(final_x[:, 2])
        final_beta1 = (1.0 - final_alpha1) * expit(final_x[:, 3])
        final_trace = np.column_stack([final_mu, final_alpha0, final_alpha1, final_beta1])
    else:
        final_trace = np.empty((0, D))
    
    return {
        'mean_x': np.mean(final_trace, axis=0) if len(final_trace) > 0 else np.zeros(D),
        'mean_x2': np.mean(final_trace**2, axis=0) if len(final_trace) > 0 else np.zeros(D),
        'iters': iters
    }

# ==============================================================================
# 4. ORCHESTRATION & PLOTTING
# ==============================================================================
def run_comparison():
    true_means, true_sqs = get_posteriordb_reference()
    
    num_runs = NUM_RUNS
    num_nuts_samples = NUM_NUTS_SAMPLES
    cores = worker_count(num_runs)
    
    print(f"\n--- 1. Launching {num_runs} NUTS chains ({num_nuts_samples} samples) ---")
    nuts_args = [(42 + i, num_nuts_samples) for i in range(num_runs)]
    start_nuts = time.time()
    with multiprocessing.Pool(processes=cores) as pool:
        nuts_results = pool.map(worker_nuts_garch, nuts_args)
    
    nuts_times = [r['time'] for r in nuts_results]
    avg_time = np.mean(nuts_times)
    print(f"NUTS finished. Average time per chain: {avg_time:.2f}s")
    
    print(
        f"\n--- 2. Launching {num_runs} projected slice/HnR chains "
        f"(time-matched to {avg_time:.2f}s) ---"
    )
    phr_args = [(avg_time, 42 + i) for i in range(num_runs)]
    with multiprocessing.Pool(processes=cores) as pool:
        phr_results = pool.map(worker_phr_slice_garch, phr_args)

    # --- AGGREGATION ---
    nuts_mean_x = np.mean([r['mean_x'] for r in nuts_results], axis=0)
    nuts_mean_x2 = np.mean([r['mean_x2'] for r in nuts_results], axis=0)
    
    phr_mean_x = np.mean([r['mean_x'] for r in phr_results], axis=0)
    phr_mean_x2 = np.mean([r['mean_x2'] for r in phr_results], axis=0)

    # --- TABLE ---
    print("\n" + "="*115)
    print("FIRST- AND SECOND-MOMENT COMPARISON (Fixed PosteriorDB References)".center(115))
    print("="*115)
    
    cw = 12
    print(f"{'Parameter':<10} | {'Reference E[X]':<{cw}} | {'NUTS E[X]':<{cw}} | {'Projected E[X]':<{cw}} | {'Reference E[X^2]':<{cw}} | {'NUTS E[X^2]':<{cw}} | {'Projected E[X^2]':<{cw}}")
    print("-" * 115)
    
    sq_err_nuts_1st = []; sq_err_phr_1st = []
    sq_err_nuts_2nd = []; sq_err_phr_2nd = []
    
    # We strictly map the 4 dimensions output by our MCMC to the expected names
    mcmc_params = ["mu", "alpha0", "alpha1", "beta1"]
    
    for i, name in enumerate(mcmc_params):
        n_m = nuts_mean_x[i]; n_sq = nuts_mean_x2[i]
        p_m = phr_mean_x[i]; p_sq = phr_mean_x2[i]
        
        t_m = true_means.get(name, None)
        t_sq = true_sqs.get(name, None)
        
        if t_m is not None and t_sq is not None:
            sq_err_nuts_1st.append((n_m - t_m)**2); sq_err_phr_1st.append((p_m - t_m)**2)
            sq_err_nuts_2nd.append((n_sq - t_sq)**2); sq_err_phr_2nd.append((p_sq - t_sq)**2)
            print(f"{name:<10} | {t_m:<{cw}.3f} | {n_m:<{cw}.3f} | {p_m:<{cw}.3f} | {t_sq:<{cw}.3f} | {n_sq:<{cw}.3f} | {p_sq:<{cw}.3f}")
        else:
            print(f"{name:<10} | {'N/A':<{cw}} | {n_m:<{cw}.3f} | {p_m:<{cw}.3f} | {'N/A':<{cw}} | {n_sq:<{cw}.3f} | {p_sq:<{cw}.3f}")
            
    print("-" * 115)
    
    if len(sq_err_nuts_1st) > 0:
        rmse_n_1 = np.sqrt(np.mean(sq_err_nuts_1st)); rmse_p_1 = np.sqrt(np.mean(sq_err_phr_1st))
        rmse_n_2 = np.sqrt(np.mean(sq_err_nuts_2nd)); rmse_p_2 = np.sqrt(np.mean(sq_err_phr_2nd))
        print(f"{'RMSE':<10} | {'-':<{cw}} | {rmse_n_1:<{cw}.4f} | {rmse_p_1:<{cw}.4f} | {'-':<{cw}} | {rmse_n_2:<{cw}.4f} | {rmse_p_2:<{cw}.4f}")
    else:
        print(f"{'RMSE':<10} | {'-':<{cw}} | {'N/A':<{cw}} | {'N/A':<{cw}} | {'-':<{cw}} | {'N/A':<{cw}} | {'N/A':<{cw}}")
    print("="*115)

    # --- Q-Q PLOTS (SECOND MOMENTS) ---
    nuts_mom2 = np.array([r['mean_x2'] for r in nuts_results])  
    phr_mom2 = np.array([r['mean_x2'] for r in phr_results])    

    # Index 2 is alpha1, Index 3 is beta1 (The constraint boundaries)
    idx_alpha = 2
    idx_beta = 3

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Chain-wise Q–Q Plots of GARCH(1,1) Second-Moment Estimates across {num_runs} Chains",
        fontsize=16,
    )

    # ROW 1: ALPHA1
    stats.probplot(nuts_mom2[:, idx_alpha], dist="norm", plot=axes[0, 0])
    axes[0, 0].get_lines()[1].set_color('red')
    axes[0, 0].set_title(r"NUTS: Second Moment of $\alpha_1$ ($E[\alpha_1^2]$)")
    axes[0, 0].set_ylabel(r"Sample Quantiles ($X^2$)")

    stats.probplot(phr_mom2[:, idx_alpha], dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[1].set_color('red')
    axes[0, 1].set_title(r"Projected slice/HnR: Second Moment of $\alpha_1$ ($E[\alpha_1^2]$)")
    axes[0, 1].set_ylabel("")

    # ROW 2: BETA1
    stats.probplot(nuts_mom2[:, idx_beta], dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[1].set_color('red')
    axes[1, 0].set_title(r"NUTS: Second Moment of $\beta_1$ ($E[\beta_1^2]$)")
    axes[1, 0].set_ylabel(r"Sample Quantiles ($X^2$)")

    stats.probplot(phr_mom2[:, idx_beta], dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[1].set_color('red')
    axes[1, 1].set_title(r"Projected slice/HnR: Second Moment of $\beta_1$ ($E[\beta_1^2]$)")
    axes[1, 1].set_ylabel("")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    avg_iters_nuts = float(np.mean([r["iters"] for r in nuts_results]))
    avg_iters_projected = float(np.mean([r["iters"] for r in phr_results]))
    finalize_experiment(
        EXPERIMENT_ID,
        {"second_moment_qq": fig},
        {
            "average_nuts_chain_seconds": avg_time,
            "first_moment_rmse": {"nuts": rmse_n_1, "projected_slice_hnr": rmse_p_1},
            "second_moment_rmse": {"nuts": rmse_n_2, "projected_slice_hnr": rmse_p_2},
            "average_iterations": {"nuts": avg_iters_nuts, "projected_slice_hnr": avg_iters_projected},
        },
        {
            "data_path": DATA_PATH,
            "data_length": T_len,
            "num_runs": num_runs,
            "nuts_samples": num_nuts_samples,
            "nuts_warmup": NUTS_WARMUP,
            "worker_processes": cores,
            "seed_schedule": "42 + chain_index for both methods",
            "reference": "fixed PosteriorDB summary statistics embedded in this module",
            "timing": (
                "NUTS elapsed time includes MCMC construction, warmup, and draws; "
                "its mean is used as the projected slice/HnR per-chain budget."
            ),
        },
    )


def main() -> None:
    multiprocessing.freeze_support()
    configure_multiprocessing()
    run_comparison()


if __name__ == "__main__":
    main()
