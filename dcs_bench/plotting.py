"""Rebuild every figure and table using saved chains; no simulation occurs here."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .targets import Target
from .references import functional_draws, mcmc_diagnostics

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
LABELS = {'tail_probability': r'$P(\|X\|\geq50)$', 'mean_norm': r'$E\|X\|$',
          'mean_log10_one_plus_norm': r'$E\log_{10}(1+\|X\|)$'}


def mse_summary(estimates, reference, seed=44):
    """Average squared errors of independent-chain estimates, not pooled means."""
    estimates, reference = np.asarray(estimates), np.asarray(reference)
    errors = (estimates-reference)**2
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(errors), size=(4000, len(errors)))
    bootstrap = errors[draws].mean(axis=1)
    mse = errors.mean(axis=0)
    return {'mse': mse.tolist(), 'rmse': np.sqrt(mse).tolist(),
            'mse_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975], axis=0).tolist()}


def save_figure(fig, folder, name, quick=False):
    if quick:
        fig.suptitle('QUICK CHECK — not publication results', fontsize=11, color='#9C2F00')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'{name}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def rebuild(output):
    from .engine import read_chain, write_json
    output = Path(output)
    config = json.loads((output/'configuration.json').read_text())
    labels = dict(LABELS)
    labels['tail_probability'] = rf'$P(\|X\|\geq {config["threshold"]:g})$'
    manifest = json.loads((output/'manifest.json').read_text())
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    all_summary, csv_rows, all_data = [], [], []
    for s, spec in enumerate(config['targets']):
        target = Target(spec)
        directory = output/f'scenario-{s}'
        ref = json.loads((directory/'reference.json').read_text())
        setup = json.loads((directory/'setup.json').read_text())
        summary = {'target': spec, 'reference': ref, 'algorithms': {}}
        data = {}
        for algorithm in config['algorithms']:
            pairs = [read_chain(directory/'chains'/f'{algorithm}-{i:03d}.npz', manifest['fingerprint']) for i in range(config['runs'])]
            chains, records = list(zip(*pairs))
            functions = [functional_draws(target, x, config['threshold']) for x in chains]
            estimates = np.array([f.mean(axis=0) for f in functions])
            quantiles = np.array([np.quantile(x[:, :2], config['q_levels'], axis=0) for x in chains])
            # Unequal wall-time runs retain unequal lengths. Only multichain
            # diagnostics use an explicitly recorded common prefix.
            length = min(map(len, functions))
            diagnostics = mcmc_diagnostics(np.stack([f[:length] for f in functions]), ref['functional_names'])
            indiv = [mcmc_diagnostics(f[None, ...], ref['functional_names']) for f in functions]
            errors = mse_summary(estimates, ref['functional_truth'])
            family = 'sphere' if algorithm == 'SCS' else 'ball'
            setup_seconds = setup.get(family, {}).get('optimization', {}).get('seconds', 0.) if algorithm == 'SCS' or algorithm.startswith('DCS') else 0.
            compilation_path = directory/'nuts_compilation.json'
            if algorithm == 'NUTS' and compilation_path.exists():
                setup_seconds += json.loads(compilation_path.read_text())['seconds']
            totals = [r['sampling_seconds']+r.get('warmup_seconds', r.get('warmup_seconds_including_compilation', 0.))
                      +r.get('production_compile_and_pilot_seconds', 0.)+setup_seconds/config['runs'] for r in records]
            algo_summary = dict(errors, chain_functional_estimates=estimates.tolist(),
                chain_quantiles=quantiles.tolist(), cross_chain_diagnostics=diagnostics,
                diagnostic_draws_per_chain=length, independent_chain_diagnostics=indiv,
                mean_sampling_seconds=float(np.mean([r['sampling_seconds'] for r in records])),
                mean_seconds_including_setup=float(np.mean(totals)),
                setup_cost_charging='Full shared fit cost amortized over this method\'s runs',
                records=list(records))
            if target.kind in ('garch', 'eight_schools'):
                algo_summary['first_moment_rmse_across_chains_and_coordinates'] = float(np.sqrt(np.mean((estimates[:, :target.d]-ref['functional_truth'][:target.d])**2)))
                algo_summary['second_moment_rmse_across_chains_and_coordinates'] = float(np.sqrt(np.mean((estimates[:, target.d:]-ref['functional_truth'][target.d:])**2)))
            summary['algorithms'][algorithm] = algo_summary
            data[algorithm] = {'estimates': estimates, 'quantiles': quantiles, 'records': records, 'summary': algo_summary}
            for i, (est, record, diag) in enumerate(zip(estimates, records, indiv)):
                for k, name in enumerate(ref['functional_names']):
                    ess = diag['ess_bulk'][name]
                    csv_rows.append(dict(scenario=s, algorithm=algorithm, chain=i, seed=record['seed'], functional=name,
                        estimate=est[k], reference=ref['functional_truth'][k], squared_error=(est[k]-ref['functional_truth'][k])**2,
                        mcse=diag['mcse_mean'][name], ess_bulk=ess, ess_tail=diag['ess_tail'][name],
                        ess_per_sampling_second=None if ess is None else ess/record['sampling_seconds'],
                        sampling_seconds=record['sampling_seconds'], seconds_including_setup=totals[i],
                        iterations=record['iterations'], retained_draws=record['retained_draws']))
        all_summary.append(summary)
        all_data.append((target, ref, data))
    write_json(output/'summary.json', {'schema_version': 2, 'preset': config['preset'], 'scenarios': all_summary,
        'interpretation': 'MSE uses independent chain estimates. PosteriorDB and logistic references have Monte Carlo uncertainty. Quick results test execution only.'})
    with (output/'chain_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    quick = config['preset'] == 'quick'
    if config['experiment'] == 'logistic-regression':
        fig, axes = plt.subplots(len(all_data), len(config['algorithms']), figsize=(10, 8), squeeze=False)
        for row, (target, ref, data) in enumerate(all_data):
            truth = np.array(ref['quantiles'])[:, 0]
            for col, (algorithm, values) in enumerate(data.items()):
                ax = axes[row, col]
                qs = values['quantiles'][:, :, 0]
                low, median, high = np.quantile(qs, [.025, .5, .975], axis=0)
                ax.fill_between(truth, low, high, color=COLORS[col], alpha=.18, label='2.5–97.5% across chains')
                ax.plot(truth, median, 'o-', color=COLORS[col], ms=3, label='Median chain quantile')
                limits = [min(truth.min(), low.min()), max(truth.max(), high.max())]
                ax.plot(limits, limits, '--', color='.4', lw=1)
                ax.set(title=f'{algorithm}, prior ν={target.nu:g}', xlabel='Independent reference quantile of β₁', ylabel='Estimated quantile of β₁')
        axes[0, 0].legend(fontsize=8)
        save_figure(fig, folder, config['figure'], quick)
    else:
        target, ref, data = all_data[0]
        algorithms = list(data)
        if target.kind == 'skew_t' and config['experiment'] == 'skew-t-200d-nuts':
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for k, ax in enumerate(axes):
                ax.boxplot([data[a]['estimates'][:, k] for a in algorithms], labels=algorithms, showmeans=True)
                ax.axhline(ref['functional_truth'][k], color='#B2182B', ls='--', lw=1.2)
                ax.set_title(labels.get(ref['functional_names'][k], ref['functional_names'][k]))
                ax.set_ylabel('Independent-chain ergodic average')
        elif target.kind == 'skew_t':
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            runtime = config['mode'] == 'iterations'
            metric = 'sampling_seconds' if runtime else 'iterations'
            axes[0].bar(algorithms, [np.mean([r[metric] for r in data[a]['records']]) for a in algorithms], color=COLORS[:len(algorithms)])
            axes[0].set_title('Mean production seconds' if runtime else 'Mean production iterations')
            for k, ax in enumerate(axes[1:]):
                mse = np.array([data[a]['summary']['mse'][k] for a in algorithms])
                interval = np.array([data[a]['summary']['mse_bootstrap_95_interval'] for a in algorithms])[:, :, k]
                ax.bar(algorithms, mse, color=COLORS[:len(algorithms)])
                ax.errorbar(np.arange(len(algorithms)), mse, yerr=[np.maximum(0, mse-interval[:, 0]), np.maximum(0, interval[:, 1]-mse)], fmt='none', color='.2', capsize=3)
                ax.set_yscale('log')
                ax.set_title(f'MSE: {labels.get(ref["functional_names"][k], ref["functional_names"][k])}')
            for ax in axes:
                ax.tick_params(axis='x', rotation=20)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for moment, ax in enumerate(axes):
                for j, algorithm in enumerate(algorithms):
                    vals = data[algorithm]['summary']['rmse'][moment*target.d:(moment+1)*target.d]
                    x = np.arange(target.d)+(j-.5)*.35
                    ax.bar(x, vals, width=.35, label=algorithm, color=COLORS[j])
                ax.set_xticks(np.arange(target.d), target.names, rotation=45)
                ax.set(ylabel='RMSE relative to PosteriorDB reference', title=f'Moment {moment+1}')
                ax.legend()
            rows = []
            for a in algorithms:
                summ = data[a]['summary']
                rows.append(f'{a} & {summ["first_moment_rmse_across_chains_and_coordinates"]:.6g} & {summ["second_moment_rmse_across_chains_and_coordinates"]:.6g} \\\\')
            (output/'posterior_table.tex').write_text('% RMSE across independent chains and physical coordinates; reference is estimated.\n'+'\n'.join(rows)+'\n')
        save_figure(fig, folder, config['figure'], quick)
        if target.kind == 'skew_t':
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for k, ax in enumerate(axes, 2):
                ax.boxplot([data[a]['estimates'][:, k] for a in algorithms], labels=algorithms, showmeans=True)
                ax.axhline(ref['functional_truth'][k], color='#B2182B', ls='--')
                mcse = ref['functional_reference_mcse'][k]
                if mcse:
                    ax.axhspan(ref['functional_truth'][k]-1.96*mcse, ref['functional_truth'][k]+1.96*mcse, color='#B2182B', alpha=.1)
                ax.set_title(ref['functional_names'][k].replace('_', ' '))
                ax.tick_params(axis='x', rotation=20)
            save_figure(fig, folder, config['figure']+'_directional', quick)
    return output/'summary.json'
