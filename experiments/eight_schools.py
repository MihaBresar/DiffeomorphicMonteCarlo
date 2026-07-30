"""Eight Schools centered-funnel benchmark.

This experiment compares NumPyro NUTS with the projected slice/HnR sampler
implemented below.  Use ``python run_experiment.py eight-schools`` rather than
editing module-level constants; the public runner records the configuration,
figures, summaries, and runtime metadata for every invocation.
"""

import os
import time
import multiprocessing
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from reproducibility import (
    configure_multiprocessing,
    env_int,
    finalize_experiment,
    worker_count,
)

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# ==============================================================================
# 1. POSTERIORDB REFERENCE SUMMARIES
# ==============================================================================
def get_posteriordb_reference():
    """Return the fixed PosteriorDB reference summaries used in this study."""
    print("Loading fixed PosteriorDB reference means and second moments...")
    
    true_means = {
        "theta[1]": 6.150502, "theta[2]": 4.939581, "theta[3]": 3.905906, "theta[4]": 4.796016, 
        "theta[5]": 3.614436, "theta[6]": 4.051147, "theta[7]": 6.317169, "theta[8]": 4.883996, 
        "mu": 4.410518, "tau": 3.602059
    }

    # Fixed reference E[X^2] values supplied with the original experiment.
    provided_sqs_list = [
        69.36345, 45.9787, 43.13923, 45.76135, 
        34.35767, 39.4135, 64.93269, 52.12845, 
        30.40302, 23.20407
    ]
    
    true_sqs = {k: v for k, v in zip(true_means.keys(), provided_sqs_list)}
    
    return true_means, true_sqs

# ==============================================================================
# 2. DATA
# ==============================================================================
Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
J = 8
D = J + 2

EXPERIMENT_ID = "eight-schools"
NUM_RUNS = env_int("HEAVY_TAILS_NUM_RUNS", 50)
NUM_NUTS_SAMPLES = env_int("HEAVY_TAILS_NUTS_SAMPLES", 150_000)
NUTS_WARMUP = env_int("HEAVY_TAILS_NUTS_WARMUP", 1_000, minimum=0)

# ==============================================================================
# 3. NUTS WORKER
# ==============================================================================
def worker_nuts_8schools(args):
    seed, num_samples = args

    def model(J, sigma, y):
        mu = numpyro.sample('mu', dist.Normal(0, 5))
        tau = numpyro.sample('tau', dist.HalfCauchy(5))
        with numpyro.plate('J', J):
            theta = numpyro.sample('theta', dist.Normal(mu, tau))
            numpyro.sample('obs', dist.Normal(theta, sigma), obs=y)

    numpyro.set_host_device_count(1)
    start_time = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=NUTS_WARMUP,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), J=J, sigma=SIGMA, y=Y)
    
    samples = mcmc.get_samples()
    elapsed = time.time() - start_time
    
    # Pack parameters: theta[1..8], mu, tau
    beta_trace = np.column_stack([samples['theta'], samples['mu'], samples['tau']])
    
    return {
        'mean_x': np.mean(beta_trace, axis=0),
        'mean_x2': np.mean(beta_trace**2, axis=0),
        'time': elapsed,
        'iters': num_samples
    }

# ==============================================================================
# 4. PROJECTED SLICE / HIT-AND-RUN WORKER
# ==============================================================================
def worker_phr_slice(args):
    time_budget, seed = args
    rng = np.random.default_rng(seed)

    x = np.zeros(D)
    x[0:8] = Y.copy()
    x[8] = np.mean(Y)
    x[9] = np.std(Y)
    
    curr_z = x / (1.0 + np.linalg.norm(x))

    def get_target_z(z_pos):
        nz = np.linalg.norm(z_pos)
        if nz >= 0.999999: return -1e20
        
        x_m = z_pos / (1.0 - nz)
        thetas = x_m[0:8]
        mu, tau = x_m[8], x_m[9]
        tau_abs = np.abs(tau)
        
        lp = -np.log(1.0 + (tau_abs/5.0)**2) - 0.5*(mu/5.0)**2
        z_h = (thetas - mu) / (tau_abs + 1e-9)
        lp += -J*np.log(tau_abs + 1e-9) - 0.5*np.sum(z_h**2)
        z_obs = (Y - thetas) / SIGMA
        lp += -0.5*np.sum(z_obs**2)
        lp -= (D+1)*np.log(1.0 - nz)
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

        
        log_y = curr_log + np.log(rng.random() + 1e-300)

        L_shrink, R_shrink = L, R

        while True:
            eta = rng.uniform(L_shrink, R_shrink)
            z_prop = curr_z + eta * theta_dir
            lp_prop = get_target_z(z_prop)

            if lp_prop >= log_y:
             curr_z = z_prop
             curr_log = lp_prop
             break

            if eta < 0.0:
               L_shrink = eta
            else:
             R_shrink = eta

        if iters % 10 == 0:
         trace.append(curr_z.copy())

    trace = np.array(trace)
    burn = int(len(trace) * 0.2) if len(trace) > 10 else 0
    final_z = trace[burn:]
    
    if len(final_z) > 0:
        nz = np.linalg.norm(final_z, axis=1, keepdims=True)
        final_x = final_z / (1.0 - nz)
        final_x[:, 9] = np.abs(final_x[:, 9])
    else:
        final_x = np.empty((0, D))
    
    return {
        'mean_x': np.mean(final_x, axis=0) if len(final_x) > 0 else np.zeros(D),
        'mean_x2': np.mean(final_x**2, axis=0) if len(final_x) > 0 else np.zeros(D),
        'iters': iters
    }

# ==============================================================================
# 5. ORCHESTRATION, REPORTING & PLOTTING
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
        nuts_results = pool.map(worker_nuts_8schools, nuts_args)
    
    nuts_times = [r['time'] for r in nuts_results]
    avg_time = np.mean(nuts_times)
    print(f"NUTS finished. Average time per chain: {avg_time:.2f}s")
    
    print(
        f"\n--- 2. Launching {num_runs} projected slice/HnR chains "
        f"(time-matched to {avg_time:.2f}s) ---"
    )
    phr_args = [(avg_time, 42 + i) for i in range(num_runs)]
    with multiprocessing.Pool(processes=cores) as pool:
        phr_results = pool.map(worker_phr_slice, phr_args)

    # --- AGGREGATION ---
    nuts_mean_x = np.mean([r['mean_x'] for r in nuts_results], axis=0)
    nuts_mean_x2 = np.mean([r['mean_x2'] for r in nuts_results], axis=0)
    
    phr_mean_x = np.mean([r['mean_x'] for r in phr_results], axis=0)
    phr_mean_x2 = np.mean([r['mean_x2'] for r in phr_results], axis=0)

    param_names = ["theta[1]", "theta[2]", "theta[3]", "theta[4]", "theta[5]", "theta[6]", "theta[7]", "theta[8]", "mu", "tau"]

    # --- TABLE ---
    print("\n" + "="*115)
    print("1ST AND 2ND MOMENT COMPARISON (Official PosteriorDB Reference)".center(115))
    print("="*115)
    
    cw = 12
    print(f"{'Parameter':<10} | {'Reference E[X]':<{cw}} | {'NUTS E[X]':<{cw}} | {'Projected E[X]':<{cw}} | {'Reference E[X^2]':<{cw}} | {'NUTS E[X^2]':<{cw}} | {'Projected E[X^2]':<{cw}}")
    print("-" * 115)
    
    sq_err_nuts_1st = []; sq_err_phr_1st = []
    sq_err_nuts_2nd = []; sq_err_phr_2nd = []
    
    for i, name in enumerate(param_names):
        t_m = true_means[name]; t_sq = true_sqs[name]
        n_m = nuts_mean_x[i]; n_sq = nuts_mean_x2[i]
        p_m = phr_mean_x[i]; p_sq = phr_mean_x2[i]
        
        sq_err_nuts_1st.append((n_m - t_m)**2); sq_err_phr_1st.append((p_m - t_m)**2)
        sq_err_nuts_2nd.append((n_sq - t_sq)**2); sq_err_phr_2nd.append((p_sq - t_sq)**2)
        
        print(f"{name:<10} | {t_m:<{cw}.3f} | {n_m:<{cw}.3f} | {p_m:<{cw}.3f} | {t_sq:<{cw}.3f} | {n_sq:<{cw}.3f} | {p_sq:<{cw}.3f}")
        
    print("-" * 115)
    print(f"{'RMSE':<10} | {'-':<{cw}} | {np.sqrt(np.mean(sq_err_nuts_1st)):<{cw}.3f} | {np.sqrt(np.mean(sq_err_phr_1st)):<{cw}.3f} | {'-':<{cw}} | {np.sqrt(np.mean(sq_err_nuts_2nd)):<{cw}.3f} | {np.sqrt(np.mean(sq_err_phr_2nd)):<{cw}.3f}")
    print("="*115)

    # ==========================================================================
    # --- NICE Q-Q PLOTS (SECOND MOMENTS) ---
    # ==========================================================================
    # Extract the second moments from the dictionaries for plotting
    nuts_mom2 = np.array([r['mean_x2'] for r in nuts_results])  # Shape: (N_CHAINS, D)
    phr_mom2 = np.array([r['mean_x2'] for r in phr_results])    # Shape: (N_CHAINS, D)

    # Indices: theta[1] is 0, tau is 9
    idx_theta1 = 0
    idx_tau = 9

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Chain-wise Q–Q Plots of Second-Moment Estimates across {num_runs} Chains", fontsize=16)

    # --- ROW 1: TAU (The Funnel/Heavy Tail driver) ---
    stats.probplot(nuts_mom2[:, idx_tau], dist="norm", plot=axes[0, 0])
    axes[0, 0].get_lines()[1].set_color('red')
    axes[0, 0].set_title(r"NUTS: Second Moment of $\tau$ ($E[\tau^2]$)")
    axes[0, 0].set_ylabel(r"Sample Quantiles ($X^2$)")

    stats.probplot(phr_mom2[:, idx_tau], dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[1].set_color('red')
    axes[0, 1].set_title(r"Projected slice/HnR: Second Moment of $\tau$ ($E[\tau^2]$)")
    axes[0, 1].set_ylabel("")

    # --- ROW 2: THETA_1 (School A) ---
    stats.probplot(nuts_mom2[:, idx_theta1], dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[1].set_color('red')
    axes[1, 0].set_title(r"NUTS: Second Moment of $\theta_1$ ($E[\theta_1^2]$)")
    axes[1, 0].set_ylabel(r"Sample Quantiles ($X^2$)")

    stats.probplot(phr_mom2[:, idx_theta1], dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[1].set_color('red')
    axes[1, 1].set_title(r"Projected slice/HnR: Second Moment of $\theta_1$ ($E[\theta_1^2]$)")
    axes[1, 1].set_ylabel("")

    # Visual Styles
    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ==========================================================================
    # --- BAR PLOTS: MSE AND SAMPLING EFFICIENCY ---
    # ==========================================================================
    # 1. Calculate overall MSE across all parameters
    mse_nuts_1st = np.mean(sq_err_nuts_1st)
    mse_phr_1st = np.mean(sq_err_phr_1st)
    
    mse_nuts_2nd = np.mean(sq_err_nuts_2nd)
    mse_phr_2nd = np.mean(sq_err_phr_2nd)
    
    # 2. Extract average iterations
    avg_iters_nuts = np.mean([r['iters'] for r in nuts_results])
    avg_iters_phr = np.mean([r['iters'] for r in phr_results])

    labels = ['NUTS', 'Projected slice/HnR']
    colors = ['#1f77b4', '#2ca02c']

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(
        f"Moment MSE & Sampling Efficiency ({avg_time:.2f}s Budget)",
        fontsize=16,
        fontweight='bold'
    )

    def make_barplot(ax, heights, title, ylabel, is_scientific=True):
        bars = ax.bar(labels, heights, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")

        for bar in bars:
            h = bar.get_height()
            label_text = f'{h:.2e}' if is_scientific else f'{int(h):,}'
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    make_barplot(
        axes2[0],
        [mse_nuts_1st, mse_phr_1st],
        r"MSE of First Moment $\hat{\mathbb{E}}[X]$",
        "Mean Squared Error",
        is_scientific=True
    )

    make_barplot(
        axes2[1],
        [mse_nuts_2nd, mse_phr_2nd],
        r"MSE of Second Moment $\hat{\mathbb{E}}[X^2]$",
        "Mean Squared Error",
        is_scientific=True
    )

    make_barplot(
        axes2[2],
        [avg_iters_nuts, avg_iters_phr],
        "Average Iterations Generated",
        "Iterations",
        is_scientific=False
    )

    fig2.tight_layout()


    # ==========================================================================
    # --- BOXPLOTS: CHAIN-WISE RMSE DISTRIBUTION ---
    # ==========================================================================
    true_mean_vec = np.array([true_means[name] for name in param_names])
    true_sq_vec = np.array([true_sqs[name] for name in param_names])

    nuts_chain_means = np.array([r['mean_x'] for r in nuts_results])
    phr_chain_means = np.array([r['mean_x'] for r in phr_results])

    nuts_chain_sqs = np.array([r['mean_x2'] for r in nuts_results])
    phr_chain_sqs = np.array([r['mean_x2'] for r in phr_results])

    nuts_rmse_1st = np.sqrt(np.mean((nuts_chain_means - true_mean_vec) ** 2, axis=1))
    phr_rmse_1st = np.sqrt(np.mean((phr_chain_means - true_mean_vec) ** 2, axis=1))

    nuts_rmse_2nd = np.sqrt(np.mean((nuts_chain_sqs - true_sq_vec) ** 2, axis=1))
    phr_rmse_2nd = np.sqrt(np.mean((phr_chain_sqs - true_sq_vec) ** 2, axis=1))

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        f"Chain-wise RMSE Distribution across {num_runs} Runs",
        fontsize=16,
        fontweight='bold'
    )

    box_labels = ['NUTS', 'Projected slice/HnR']
    box_colors = ['#1f77b4', '#2ca02c']

    def make_rmse_boxplot(ax, data, title, ylabel, use_log=False):
        bp = ax.boxplot(
            data,
            labels=box_labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            widths=0.55,
            boxprops=dict(linewidth=1.5),
            medianprops=dict(color='black', linewidth=2),
            meanprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linewidth=1.3),
            capprops=dict(linewidth=1.3),
            flierprops=dict(
                marker='o',
                markerfacecolor='gray',
                markeredgecolor='black',
                alpha=0.6,
                markersize=5
            )
        )

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        if use_log:
            ax.set_yscale('log')

        for i, vals in enumerate(data, start=1):
            mean_val = np.mean(vals)
            median_val = np.median(vals)
            ymax = np.max(vals)

            ax.text(
                i,
                ymax * 1.08,
                f"mean={mean_val:.3g}\nmed={median_val:.3g}",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    make_rmse_boxplot(
        axes3[0],
        [nuts_rmse_1st, phr_rmse_1st],
        r"RMSE of First Moment Estimates $\hat{\mathbb{E}}[X]$",
        "RMSE",
        use_log=False
    )

    make_rmse_boxplot(
        axes3[1],
        [nuts_rmse_2nd, phr_rmse_2nd],
        r"RMSE of Second Moment Estimates $\hat{\mathbb{E}}[X^2]$",
        "RMSE",
        use_log=True
    )

    fig3.tight_layout(rect=[0, 0.03, 1, 0.93])


    # ==========================================================================
    # --- PRINT RMSE SUMMARY ---
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CHAIN-WISE RMSE SUMMARY".center(80))
    print("=" * 80)

    print("\nFirst Moment RMSE:")
    print(f"  NUTS    mean={np.mean(nuts_rmse_1st):.6f}, median={np.median(nuts_rmse_1st):.6f}")
    print(f"  Projected slice/HnR mean={np.mean(phr_rmse_1st):.6f}, median={np.median(phr_rmse_1st):.6f}")

    print("\nSecond Moment RMSE:")
    print(f"  NUTS    mean={np.mean(nuts_rmse_2nd):.6f}, median={np.median(nuts_rmse_2nd):.6f}")
    print(f"  Projected slice/HnR mean={np.mean(phr_rmse_2nd):.6f}, median={np.median(phr_rmse_2nd):.6f}")

    print("=" * 80)

    finalize_experiment(
        EXPERIMENT_ID,
        {"second_moment_qq": fig, "moment_mse": fig2, "chain_rmse": fig3},
        {
            "average_nuts_chain_seconds": avg_time,
            "first_moment_mse": {"nuts": mse_nuts_1st, "projected_slice_hnr": mse_phr_1st},
            "second_moment_mse": {"nuts": mse_nuts_2nd, "projected_slice_hnr": mse_phr_2nd},
            "average_iterations": {"nuts": avg_iters_nuts, "projected_slice_hnr": avg_iters_phr},
        },
        {
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
