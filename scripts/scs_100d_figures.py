"""Original-style sampler comparison figures and exact numerical tables.

Rendering never selects methods, changes tuning, or substitutes thinned
estimates for the recorded all-iteration chain means.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np
from dcs_bench.display import number, cell, figure_ticks
from dcs_bench.simulation_error import mse_variability, paired_mse_differences

METRICS = ('tail_probability', 'mean_norm')
ALIASES = (('tail_probability', 'tail'), ('mean_norm', 'norm'))
METHODS = ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'SCS')
COLORS = {'DCS-RWM': '#FF8C00', 'DCS-BW': '#9370db', 'DCS-BW-reflect': '#159a8c', 'SCS': '#ff1493'}


def _pair(value):
    if isinstance(value, dict):
        value = [next(value[key] for key in keys if key in value) for keys in ALIASES]
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError('Each metric pair must contain two finite values')
    return result


def _means(case):
    values = case['chain_means']
    if isinstance(values, dict):
        values = np.column_stack([next(values[k] for k in keys if k in values) for keys in ALIASES])
    result = np.asarray(values, dtype=float)
    if result.shape != (case['chains'], 2) or not np.all(np.isfinite(result)):
        raise ValueError('chain_means must contain one finite metric pair per chain')
    return result


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _text(value):
    value = _plain(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, default=_plain)
    return '' if value is None else str(value)


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _text(value) for key, value in row.items()} for row in rows)


def _table(rows):
    if not rows:
        return '<p>No values available.</p>'
    fields = list(dict.fromkeys(key for row in rows for key in row))
    header = ''.join('<th>'+html.escape(key.replace('_', ' '))+'</th>' for key in fields)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(cell(row.get(key), key))+'</td>' for key in fields)+'</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def _policy(name, timed=False, matched=False, ball_esjd=False):
    if ball_esjd and name == 'DCS-BW-reflect':
        return 'Ball-movement pilot selection by mean squared jumping distance; selected step frozen during timed sampling'
    if matched:
        return 'Shared dual-averaging step calibration toward 0.234, then frozen during timed sampling'
    if timed:
        return ('Author SCS: fixed h = 0.10 with separate untimed equilibration' if name == 'SCS' else
                'Separate proposal-step tuning, then frozen during timed sampling')
    return 'Paper-date SCS: fixed h = 0.1' if name == 'SCS' else 'Continuous smoothed dual averaging; target acceptance 0.234'


def _counts(case, config, timed):
    records = case.get('records', [])
    if not records:
        if timed:
            raise ValueError('Timed reports require exact per-chain iteration and burn-in records')
        records = [{'iterations': config['iterations']} for _ in range(case['chains'])]
    generated = [int(row.get('iterations', config.get('iterations'))) for row in records]
    if sum(generated) != case['total_iterations']:
        raise ValueError('Per-chain generated counts must sum to total_iterations')
    discarded = [int(row.get('discarded_iterations', row.get('burnin', 0))) for row in records]
    if timed and any('discarded_iterations' not in row and 'burnin' not in row for row in records):
        raise ValueError('Timed reports require recorded discarded_iterations or burnin')
    used = [int(row.get('estimation_iterations', total-burn)) for row, total, burn in zip(records, generated, discarded)]
    if any(total <= 0 or burn < 0 or remaining <= 0 or remaining != total-burn
           for total, burn, remaining in zip(generated, discarded, used)):
        raise ValueError('Generated, discarded, and estimation counts must agree')
    retained = [row.get('retained_draws', total//config['thin'] if not timed else None)
                for row, total in zip(records, generated)]
    return {'generated': generated, 'discarded': discarded, 'used': used, 'retained': retained,
            'mean_generated': float(np.mean(generated)), 'mean_used': float(np.mean(used))}


def render_report(root, summary):
    """Render every provided case, returning the created figure and table names."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    cases, config = summary['cases'], summary['config']
    suite = config.get('protocol') in ('timed-sampler-comparison-v1', 'timed-numba-comparison-v1')
    optimized = config.get('protocol') == 'timed-numba-comparison-v1'
    vi_mode = config.get('vi', 'on')
    boundary = config.get('protocol') in ('timed-boundary-dcs-scs-v1', 'timed-boundary-dcs-scs-v2', 'timed-boundary-dcs-scs-v3')
    restored_controls = config.get('protocol') == 'timed-boundary-dcs-scs-v2'
    ball_esjd = config.get('protocol') == 'timed-boundary-dcs-scs-v3' or suite
    matched = config.get('protocol') == 'timed-matched-dcs-scs-v3' or boundary or suite
    timed = config.get('mode') == 'time' or config.get('protocol') == 'timed-dcs-author-scs-v2' or matched
    if not cases or any(name not in METHODS for name in cases):
        raise ValueError('Expected completed DCS-RWM, DCS-BW, or SCS cases')
    names = [name for name in (config.get('algorithms', METHODS) if suite else METHODS) if name in cases]
    if len(names) != len(cases) or len(set(names)) != len(names):
        raise ValueError('Configured methods must list every completed case exactly once')
    counts = {name: _counts(cases[name], config, timed) for name in names}
    truth = _pair(summary['truth'])
    means = {name: _means(cases[name]) for name in names}
    mses = {name: _pair(cases[name]['mse']) for name in names}
    biases = {name: _pair(cases[name]['bias_squared']) for name in names}
    variances = {name: _pair(cases[name]['variance_ddof0']) for name in names}
    for name in names:
        if np.any(mses[name] < 0) or np.any(biases[name] < 0) or np.any(variances[name] < 0):
            raise ValueError('MSE, squared bias, and variance must be nonnegative')
        if not np.allclose(mses[name], biases[name]+variances[name], rtol=1e-9, atol=1e-15):
            raise ValueError('Recorded MSE must equal empirical squared bias plus variance (ddof=0)')
    target = config.get('target', summary.get('target', {}))
    dimension, nu = target.get('dimension', 100), target.get('nu', 3.)
    threshold = config.get('threshold', 50.)
    smoke = config.get('preset') == 'quick'
    title = ('EXECUTION CHECK — ' if smoke else '')+f'{dimension}D skewed Student-t, ν = {nu:g}'
    if suite:
        target_title = config.get('target_title') or f'{dimension}D skewed Student-t, ν = {nu:g}'
        title = ('EXECUTION CHECK — ' if smoke else '') + target_title
        title += ' — optimized Numba DCS / author SCS' if optimized else ' — four samplers, JAX'
        title += ' — VI '+str(vi_mode)
    elif boundary:
        title += ' — boundary comparison'
    budget = (f"{config['runs']} chains per method; {number(config['time_budget'])} s of timed sampling per chain; "
              'first third of generated iterations discarded' if timed else
              f"{config['runs']} chains × {config['iterations']} production iterations per method; "
              f"{config['warmup']} warmup iterations discarded")
    estimation_note = ('Estimates use the remaining two thirds, including repeated states after rejection.' if timed else
                       'All production iterations contribute, including rejected states.')
    tuning_note = ('All methods use matched map fitting and step tuning toward 0.234; sampling steps stay fixed.' if matched else
                  'DCS steps are pretuned; author SCS uses h = 0.10 after untimed equilibration. All sampling steps stay fixed.' if timed else
                   'DCS adapts during sampling; SCS uses h = 0.1.')
    if boundary and 'step_scale_ceiling' in config:
        tuning_note = (f"All methods tune toward 0.234 with a shared step ceiling of {number(config['step_scale_ceiling'])}; "
                       'the ceiling may bind. Sampling steps stay fixed.')
    if restored_controls:
        tuning_note = 'Original BW/SCS calibration retained; reflected BW alone has step ceiling 2.00. All tune toward 0.234, then freeze.'
    if ball_esjd:
        tuning_note = 'BW/SCS retain acceptance calibration; reflected BW uses ball-movement pilots. Equal warmup budgets; all sampling steps frozen.'
    metric_labels = (f'P(‖X‖ ≥ {number(threshold) if suite else format(threshold, "g")})', 'E[‖X‖]')
    if suite:
        tuning_note = ('VI '+str(vi_mode)+' for both maps. Acceptance calibration for original methods; '
                       'ball-movement pilots for reflected BW; all steps frozen.')
    x = np.arange(1, len(names)+1)
    colors = [COLORS[name] for name in names]
    figure_names = []

    def save(fig, name):
        figure_ticks(fig)
        if smoke:
            fig.text(.5, .012, 'Quick execution check only — not evidence of sampler performance',
                     ha='center', fontsize=10, color='#995b08')
        fig.savefig(figures/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
        figure_names.append(name)

    # The archived VsSCS figure has three linear-scale panels. Preserve that
    # family; estimator distributions and log errors are additional figures.
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.8))
    beta = summary.get('maps', {}).get('ball', {}).get('beta', 2.)
    latitude = summary.get('maps', {}).get('sphere', {}).get('latitude', 1.1)
    original_labels = [name+(f' (β={beta:g})' if name != 'SCS' else f' (l₀={latitude:.2f})') for name in names]
    panels = ([mses[name][0] for name in names], [mses[name][1] for name in names],
              [counts[name]['mean_generated'] if timed else cases[name]['mean_seconds'] for name in names])
    for ax, values, label, ylabel in zip(axes, panels,
            ('Tail-probability MSE', 'Mean-norm MSE', 'Average Iterations Generated' if timed else 'Average Runtime (s)'),
            ('Mean Squared Error', 'Mean Squared Error', 'Generated Iterations per Chain' if timed else 'Seconds')):
        bars = ax.bar(x, values, color=colors, alpha=.8, edgecolor='#333333', width=.8)
        ax.bar_label(bars, labels=[number(value, scientific='MSE' in label, integer='Iterations' in label) for value in values], padding=5, fontsize=8)
        ax.set_xticks(x, original_labels, rotation=15, ha='right', fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        largest = max(float(value) for value in values)
        ax.set_ylim(0, largest*1.18 if largest > 0 else 1.)
        ax.grid(axis='y', linestyle='--', alpha=.5)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    comparison_title = ('Boundary Handling: Performance & Mean Squared Error' if boundary else
                        'Performance & Mean Squared Error Comparison')
    comparison_target_title = target_title if suite else f'{dimension}D skewed Student-t, ν = {nu:g}'
    fig.suptitle(('EXECUTION CHECK — ' if smoke else '')+comparison_title+'\n'+
                 comparison_target_title, fontsize=14, fontweight='bold')
    fig.text(.5, .079, budget+'.', ha='center', fontsize=9)
    fig.text(.5, .049, tuning_note+' '+estimation_note,
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .9))
    save(fig, 'VsSCS')

    # Keep every chain point and log-MSE comparison in the added 2×2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for metric, label in enumerate(metric_labels):
        ax = axes[0, metric]
        values = [means[name][:, metric] for name in names]
        boxes = ax.boxplot(values, patch_artist=True, showfliers=False, widths=.5)
        for box, color in zip(boxes['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(.35)
        for position, values_, color in zip(x, values, colors):
            offsets = np.linspace(-.11, .11, len(values_)) if len(values_) > 1 else np.zeros(1)
            ax.scatter(position+offsets, values_, color=color, s=23, alpha=.8, zorder=3)
        ax.axhline(truth[metric], color='#27384a', linestyle='--', linewidth=1.4,
                   label='True value = '+number(truth[metric]))
        ax.set_xticks(x, names)
        ax.set_ylabel('Per-chain estimate')
        ax.set_title(label+' estimates')
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(axis='y', alpha=.2)
        ax = axes[1, metric]
        values = np.asarray([mses[name][metric] for name in names])
        positive = values[values > 0]
        floor = float(positive.min()/4) if len(positive) else 1e-16
        ax.set_yscale('log')
        for position, value, color in zip(x, values, colors):
            if value > 0:
                ax.bar(position, value, color=color, width=.58)
            else:
                ax.plot(position, floor, marker='v', color=color)
            ax.text(position, max(float(value), floor)*1.15, number(value, scientific=True),
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylim(floor, max(float(values.max())*4, floor*10))
        ax.set_xticks(x, [name+'\n'+('Mean used N = ' if timed else 'N = ')+
                         number(counts[name]['mean_used'], integer=True) for name in names])
        ax.set_title(label+' MSE across chains')
        ax.set_ylabel('MSE (logarithmic scale)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if np.any(values == 0):
            ax.text(.02, .02, 'Zero values marked at the lower axis edge', transform=ax.transAxes, fontsize=8)
    fig.suptitle(title+' — estimates and logarithmic MSE\n'+budget, fontsize=14)
    fig.text(.5, .046, estimation_note+' '+tuning_note,
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .085, 1, .92))
    save(fig, 'scs_100d_estimates')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, label in zip(axes, ('mean_seconds', 'iterations_per_second'),
                              ('Mean production seconds per chain', 'Production iterations per second')):
        values = np.asarray([cases[name][key] for name in names], dtype=float)
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError('Measured runtime and throughput must be finite and positive')
        bars = ax.bar(x, values, color=colors, width=.58)
        ax.bar_label(bars, labels=[number(value) for value in values], fontsize=9, padding=5)
        ax.set_xticks(x, names)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, float(values.max())*1.28)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.suptitle(title+(' — measured runtime and output at equal sampling time\n' if timed else ' — measured runtime at equal iteration counts\n')+budget, fontsize=14)
    fig.text(.5, .055, 'Clock includes '+('discarded sampling burn-in and estimation' if timed else 'all-iteration estimation')+
             '; excludes map fitting, compilation, step tuning/equilibration and saving. Hardware and load affect timing.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .88))
    save(fig, 'scs_100d_runtime')

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    for metric, (ax, label) in enumerate(zip(axes, metric_labels)):
        variance = np.asarray([variances[name][metric] for name in names])
        bias = np.asarray([biases[name][metric] for name in names])
        ax.bar(x, variance, color=colors, width=.58, label='Across-chain variance (ddof = 0)')
        ax.bar(x, bias, bottom=variance, color=colors, width=.58, hatch='///', alpha=.5,
               label='Squared empirical bias')
        for position, variance_, bias_, name in zip(x, variance, bias, names):
            ax.text(position, variance_+bias_, ' MSE = '+number(mses[name][metric], scientific=True)+
                    '\n bias² = '+number(bias_, scientific=True)+'\n variance = '+number(variance_, scientific=True),
                    ha='center', va='bottom', fontsize=8)
        ymax = float((variance+bias).max())
        ax.set_ylim(0, ymax*1.8 if ymax > 0 else 1.)
        ax.set_xticks(x, names)
        ax.set_ylabel('Squared estimation error')
        ax.set_title(label)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False)
    fig.suptitle(title+' — empirical MSE decomposition\n'+budget, fontsize=14)
    fig.text(.5, .055, 'MSE = squared empirical bias + across-chain variance (ddof = 0). This finite-run identity is not a stationarity diagnostic.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .88))
    save(fig, 'scs_100d_bias_variance')

    # Prefix means come from streaming accumulators in the run, never by
    # treating a thinned trace as all production states.
    prefix_rows, checkpoints = [], {}
    for name in names:
        records = cases[name].get('records', [])
        checkpoints[name] = [{int(row['iterations']): _pair(row['mean_all_iterations'])
                              for row in record.get('checkpoints', [])} for record in records] if not timed else []
    dictionaries = [row for name in names for row in checkpoints[name]]
    common = sorted(set.intersection(*(set(row) for row in dictionaries))) if dictionaries else []
    if any(len(checkpoints[name]) != cases[name]['chains'] for name in names):
        common = []
    if common:
        for name in names:
            for count in common:
                values = np.asarray([row[count] for row in checkpoints[name]])
                error = np.mean((values-truth)**2, axis=0)
                for metric, key in enumerate(METRICS):
                    prefix_rows.append({'method': name, 'iterations_per_chain': count,
                                        'chains': cases[name]['chains'], 'metric': key,
                                        'truth': truth[metric], 'mse': error[metric]})
    if len(common) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for metric, (ax, label) in enumerate(zip(axes, metric_labels)):
            for name in names:
                rows = [row for row in prefix_rows if row['method'] == name and row['metric'] == METRICS[metric]]
                values = [row['mse'] for row in rows]
                ax.plot(common, values, 'o-', color=COLORS[name], label=name)
            ax.set_xscale('log')
            if all(row['mse'] > 0 for row in prefix_rows if row['metric'] == METRICS[metric]):
                ax.set_yscale('log')
            ax.set_xticks(common, [str(count) for count in common], fontsize=8)
            ax.set_xlabel('Exact production iterations per chain')
            ax.set_ylabel('MSE across chains')
            ax.set_title(label)
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(title+' — error at recorded production checkpoints', fontsize=14)
        fig.text(.5, .055, 'Each point uses all iterations through that checkpoint. Points share the same chains; exact MSEs are in CSV.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, .9))
        save(fig, 'scs_100d_prefix_mse')

    case_rows, chain_rows, tuning_rows, tuning_summaries = [], [], [], []
    for name in names:
        case = cases[name]
        count = counts[name]
        per_chain = count['generated'][0] if len(set(count['generated'])) == 1 else 'varies; see chain table'
        retained = (count['retained'][0] if len(set(count['retained'])) == 1 else 'varies; see chain table')
        row = {'method': name, 'production_step_policy': _policy(name, timed, matched, ball_esjd), 'chains': case['chains'],
               'iterations_per_chain': per_chain, 'mean_iterations': count['mean_generated'],
               'total_iterations': case['total_iterations'],
               'mean_estimation_iterations': count['mean_used'], 'total_estimation_iterations': sum(count['used']),
               'total_discarded_iterations': sum(count['discarded']),
               'warmup_iterations': config['warmup'], 'thin': config['thin'],
               'retained_per_chain': retained,
               'mean_seconds': case['mean_seconds'], 'median_seconds': case['median_seconds'],
               'iterations_per_second': case['iterations_per_second'],
               'mean_acceptance': case['mean_acceptance'], 'final_scales': case['final_scales']}
        if suite:
            row.update(target=target_title, dimension=dimension, vi=vi_mode, dcs_beta=config.get('beta'),
                       execution_backend='Numba' if optimized and name != 'SCS' else 'JAX')
        for metric, key in enumerate(METRICS):
            row[key+'_truth'] = truth[metric]
            row[key+'_mse'] = mses[name][metric]
            row[key+'_bias_squared'] = biases[name][metric]
            row[key+'_variance_ddof0'] = variances[name][metric]
        variability = mse_variability(means[name], truth)
        for metric, key in enumerate(METRICS):
            row[key+'_mse_empirical_mcse'] = variability['mse_empirical_mcse'][metric]
            row[key+'_largest_chain_error_share'] = variability['largest_chain_squared_error_share'][metric]
        case_rows.append(row)
        records = case.get('records', [])
        if records and len(records) != case['chains']:
            raise ValueError('Each chain must have exactly one record')
        for index, estimate in enumerate(means[name]):
            row = dict(records[index]) if records else {'chain': index}
            history = row.pop('tuning_history', None)
            row.update(method=name, production_step_policy=_policy(name, timed, matched, ball_esjd))
            if ball_esjd and name == 'DCS-BW-reflect':
                for pilot in history or []:
                    tuning_rows.append(dict(method=name, chain=row['chain'], seed=row.get('seed'),
                        **{key: pilot.get(key) for key in ('round', 'candidate', 'scale', 'iterations',
                                                         'ball_esjd', 'acceptance', 'selected')}))
                tuning_summaries.append({'method': name, 'chain': row['chain'],
                    'warmup_iterations': row.get('warmup_iterations', config['warmup']),
                    'frozen_step': row.get('frozen_scale', row.get('final_scale')),
                    'selected_ball_esjd': row.get('tuning_selected_esjd'),
                    'search_status': row.get('tuning_search_status')})
            row.setdefault('iterations', count['generated'][index])
            row.setdefault('discarded_iterations', count['discarded'][index])
            row.setdefault('estimation_iterations', count['used'][index])
            for metric, key in enumerate(METRICS):
                row[key+'_estimate'] = estimate[metric]
                row[key+'_truth'] = truth[metric]
                row[key+'_squared_error'] = (estimate[metric]-truth[metric])**2
            chain_rows.append(row)
    paired_rows = paired_mse_differences(means, {name: cases[name].get('records', []) for name in names}, truth)
    parameter_rows = []
    for family, parameters in summary.get('maps', {}).items():
        for key, value in parameters.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for index, coordinate in enumerate(value):
                    parameter_rows.append({'map': family, 'parameter': key, 'coordinate': index+1, 'value': coordinate})
            else:
                parameter_rows.append({'map': family, 'parameter': key, 'coordinate': '', 'value': value})
    setup_rows = []
    for family, fitting in summary.get('fitting', {}).items():
        total_draws = fitting.get('total_training_draws')
        count_source = 'recorded' if total_draws is not None else 'unavailable'
        if total_draws is None and fitting.get('steps') is not None and fitting.get('samples_per_step') is not None:
            total_draws = fitting['steps']*fitting['samples_per_step']
            count_source = 'steps × samples_per_step'
        setup_rows.append({'map': family, 'used_by': ' and '.join(n for n in names if n != 'SCS') if family == 'ball' else 'SCS',
                           'fitting_seconds': fitting.get('seconds'), 'steps': fitting.get('steps'),
                           'training_draws': fitting.get('training_draws'),
                           'samples_per_step': fitting.get('samples_per_step'),
                           'total_training_draws': total_draws, 'training_count_source': count_source,
                           'training_batch_policy': fitting.get('training_batch_policy', fitting.get('batch_policy'))})
    count_rows = [{key: row[key] for key in ('method', 'chains', 'iterations_per_chain', 'mean_iterations',
                   'total_iterations', 'mean_estimation_iterations', 'total_estimation_iterations',
                   'total_discarded_iterations', 'warmup_iterations', 'thin', 'retained_per_chain',
                   'mean_seconds', 'median_seconds', 'iterations_per_second')} for row in case_rows]
    csvs = {'scs_100d_values.csv': case_rows, 'paper_values.csv': case_rows,
            'sample_counts.csv': count_rows, 'scs_100d_chains.csv': chain_rows,
            'tuned_parameters.csv': parameter_rows}
    if paired_rows:
        csvs['paired_mse_differences.csv'] = paired_rows
    if prefix_rows:
        csvs['scs_100d_prefix_mse.csv'] = prefix_rows
    if setup_rows:
        csvs['setup_times.csv'] = setup_rows
    if tuning_rows:
        csvs['reflected_tuning.csv'] = tuning_rows
    for filename, rows in csvs.items():
        _write_csv(root/filename, rows)

    notes = [
        'This comparison uses equal production iteration counts, not equal elapsed time. Every production state enters the estimates, including repetitions after rejection. Thinning only controls stored traces.',
        'DCS-RWM and DCS-BW use the original VI fitting and continuously adapt their proposal steps with smoothed dual averaging targeting acceptance 0.234. The first '+str(config['warmup'])+' iterations are discarded; adaptation continues afterward.',
        'SCS uses the paper-date author implementation at fixed proposal step h = 0.1. This reproduces the declared historical protocol; it is not a comparison in which both methods receive identical proposal tuning.',
        'DCS proposal steps and the SCS step have different geometric meanings. Final DCS scales are endpoints of changing sequences, not scales held throughout production.',
        'Production time excludes map fitting, compilation, warmup and saving, and includes estimation using all iterations. The reported times describe this hardware, worker count and competing load.',
        'The setup table reports excluded map-fitting costs once per map; DCS-RWM and DCS-BW share the ball fit. Per-chain records separately report measured compilation and warmup. Training counts distinguish reused base draws from fresh samples at every optimizer step.',
        'Squared empirical bias is the square of the mean chain estimate minus truth. Across-chain variance uses ddof = 0, so the two terms sum exactly, up to floating-point arithmetic, to empirical MSE. These numbers do not establish stationarity or exclude initialization bias.',
        'The SCS author VI and paper-date sampler are applied to the shared normal-CDF-skewed Student-t target at ν = 3. This is not a literal reproduction of the SCS publication target, which uses a Student-t CDF. The two reported functions are tail probability and mean norm.',
    ]+list(summary.get('notes', []))
    if timed:
        notes[:4] = [
            'Every chain receives the same timed sampling budget. Generated iteration counts can differ. Each chain discards the first third of its actual generated iterations; only the remaining states enter the estimates, including rejection repeats.',
            'DCS has a separate step-tuning phase before the sampling clock starts. Author SCS keeps its fixed h = 0.10 and receives separate untimed equilibration. Proposal steps are held fixed during timed sampling. The first-third discard is additional sampling burn-in inside that clock.',
            'The SCS author transition and its fixed paper step are applied to the shared normal-CDF-skewed Student-t target. This default comparison intentionally preserves the author step; it does not give SCS the proposal pretuning used by the other benchmark experiments.',
            'The sample-count panel reports all generated iterations. Exact generated, discarded and estimation counts are listed for every chain below. Final scales are the steps used during timed sampling.',
        ]
        notes[4] = 'Production time includes sampling burn-in and estimation, and excludes map fitting, compilation, separate step tuning or equilibration, and saving. Times describe this hardware, worker count and competing load.'
    if matched:
        notes[1:4] = [
            'DCS-RWM, DCS-BW and SCS all calibrate their proposal steps with the same dual-averaging rule toward acceptance 0.234, using observed accept/reject indicators during separate warmup. Each resulting step is held fixed during timed sampling. The first-third discard is additional sampling burn-in inside that clock.',
            'Both map families use the same fresh-batch reverse-KL fitting framework, optimizer budget, Adam updates and gradient clipping. All three samplers use compiled JAX CPU scans with the same density-evaluation and state-recording policy. Physical projection and radial estimation run after each batch for every method. DCS-RWM and DCS-BW share the ball map.',
            'The sample-count panel reports all generated iterations. Exact generated, discarded and estimation counts are listed for every chain below. Matching tuning rules does not assert that acceptance 0.234 is optimal for every method; the frozen steps differ because their geometries differ.',
        ]
        notes[7] = 'The unmodified author SCS transition is called directly under the common benchmark fitting and tuning protocol. DCS uses the same compiled scan and recording structure. All methods sample the same normal-CDF-skewed Student-t target at ν = 3; this is not a literal reproduction of the SCS publication target, which uses a Student-t CDF.'
        if config.get('implementation_revision') == 'jax-dcs-exact-ball-v1' and not suite:
            notes[2] = 'Both map families use the same fresh-batch reverse-KL fitting framework, optimizer budget, Adam updates and gradient clipping. All three samplers use compiled JAX CPU scans and the same full-state recording policy. DCS rejects outside-ball proposals before either density evaluation and recomputes both current and proposed densities for every admissible proposal. Physical projection and radial estimation run after each batch for every method. DCS-RWM and DCS-BW share the ball map.'
            notes.append('This implementation generates the same uniform-ball BW proposal distribution from d+2 Gaussian draws and simplifies the β=2 map using an exact Möbius identity. The BW random-number sequence for a given seed changes, while its proposal law is preserved. The author SCS transition, shared fitting and tuning budgets, recording and timing remain unchanged.')
    if boundary:
        notes[1] = 'All three methods use the same separate dual-averaging calibration toward acceptance 0.234 and freeze their scales during timed sampling. The first third of actual generated iterations is then discarded.'
        if 'step_scale_ceiling' in config:
            notes[1] += ' All three methods have a common upper step limit of '+number(config['step_scale_ceiling'])+'.'
        if restored_controls:
            notes[1] += ' Ordinary BW and SCS retain exactly the original main-comparison calibration, including its existing numerical guard exp(15). Only reflected BW has a step ceiling of 2. This method-specific constraint is explicit because its acceptance target can be unattainable; it is not a claim of optimal tuning.'
        if ball_esjd:
            notes[1] = 'Ordinary BW and SCS retain their original separate acceptance-target 0.234 calibration. Reflected BW spends the same warmup transition budget on ball-movement pilots at h/2, h and 2h in up to eight rounds, rotating candidate order. It selects by observed mean squared displacement inside the ball, with rejected moves contributing zero. Every method freezes its selected step before timed sampling; the first third of timed iterations is then discarded.'
            notes[3] = 'The sample-count panel reports all generated iterations. Exact generated, discarded and estimation counts are listed below. Reflected BW uses a different tuning objective, with no reflected-only geometric step ceiling; only the numerical log-step guard of 15 remains. Equal warmup transition budgets do not establish equally effective or optimal tuning.'
            notes[7] = 'The unmodified author SCS transition is called directly with the recorded benchmark fitting and acceptance calibration. Both BW variants share their fitted map; reflected BW uses its recorded ball-movement pilot rule. All methods sample the same normal-CDF-skewed Student-t target at ν = 3; this does not reproduce the SCS publication target, which uses a Student-t CDF.'
        notes[2] = 'Standard DCS-BW and DCS-BW-reflect share the same fitted ball map and initial physical point. The reflected variant folds an outside proposal back along its chord, then applies the transformed-density Metropolis test. Author SCS keeps its original cap step-out. Map fitting, recording and time budgets are matched.'
        notes[5] = 'The setup table reports fitting once per map; both BW variants share the ball fit. Compilation and step calibration are excluded from the sampling clock for all methods.'
        notes.append('Reflected-chord BW is a separate reversible kernel, not literal SCS step-out or specular billiards. No clipping onto the ball boundary is used. A numerical invalid endpoint becomes a rejected move.')
        reflected = cases.get('DCS-BW-reflect', {})
        if ball_esjd:
            notes.append('The pilot score measures movement in ball coordinates, not production MSE or radial-estimation efficiency. A finite noisy local search is not a proof of an optimal scale. All pilot observations are discarded and the selected scale is fixed throughout production. The pilot table records every tested candidate; selected flags identify the winner within each round.')
            if any(row.get('scale_ceiling_reached') for row in reflected.get('records', [])):
                notes.append('Reflected BW reached the numerical step-size guard in at least one chain. Inspect the saved pilot candidates and search status; this is not evidence of successful or optimal tuning.')
        elif any(row.get('scale_ceiling_reached') for row in reflected.get('records', [])):
            notes.append('Reflected BW reached '+('its method-specific' if restored_controls else 'the common')+' step-size ceiling in at least one chain. Its acceptance target may be unattainable; do not interpret saturation as successful calibration. Inspect final scales and acceptance rates.')
    if suite:
        acceptance_methods = ('DCS-RWM, ordinary DCS-BW and SCS' if 'DCS-RWM' in names
                              else 'Ordinary DCS-BW and SCS')
        notes[1] = (acceptance_methods+' use separate acceptance-target 0.234 calibration. '
                    'Reflected BW uses its same warmup-transition budget to select by actual squared movement in the ball, '
                    'including zero movement after rejection. The selected steps are frozen before timing; '
                    'the first-third timed burn-in is additional. Equal resources do not mean identical or optimal tuning criteria.')
        notes[2] = ('VI is enabled for both map families: they receive the same optimizer and fresh-draw budgets; '
                    'all DCS methods share the ball map. Maps remain fixed after fitting.' if vi_mode == 'on' else
                    'VI is disabled for both map families: unit scale, zero centre and zero deformation. '
                    'DCS retains the target-specific radial exponent and SCS retains latitude 1.1. '
                    'No fitting draws or optimization are used; proposal-step tuning remains enabled for every method.')
        notes[3] = ('Every post-burn-in iteration contributes equally to the radial estimates, including rejected repeats. '
                    'Saved trace thinning does not change those estimates. Counts and full-precision errors are exported.')
        notes[5] = ('Map fitting, compilation and step calibration are reported separately and excluded from production timing. '
                    'All DCS methods share one ball map; SCS uses the recorded sphere map.')
        notes[7] = ('Every method samples the same recorded target: '+target_title+'. This is a benchmark of radial '
                    'tail-probability and mean-norm estimation, not a reproduction of the SCS publication figures or '
                    'a general mixing guarantee. The unmodified author SCS transition is retained.')
        if optimized:
            notes.append('Implementation comparison: DCS-BW and reflected BW use optimized Numba loops with cached '
                         'current density and radius, fused radial accumulation and no full physical-state recording. '
                         'SCS uses its unmodified author JAX transition and full-state scan. DCS uses the same JAX pilot '
                         'calibration as the common-framework mode, then hands its frozen state and step to Numba. '
                         'The 300-transition timing batches, statistic definitions and first-third discard are common. '
                         'This compares the recorded implementations; it does not isolate the geometric algorithm alone.')
        else:
            notes.append('Common-framework comparison: every method uses JAX CPU scans with full-state recording and '
                         'the same 300-transition timing batches and radial reduction. Outside-ball rejections can skip '
                         'density evaluation in ordinary BW/RWM; this is part of their specified transition cost.')
        notes.append('Reflected BW keeps the reversible chord-fold proposal and its Metropolis correction. Its '
                     'ball-movement pilot search uses up to eight rotating h/2, h, 2h rounds with no special bound of 2. '
                     'A finite noisy local search does not certify an optimal step or optimal radial MSE. '
                     'Every trial is recorded in reflected_tuning.csv; a final-round edge winner is reported explicitly.')
    notes.append('Empirical MSE MCSE is the sample standard deviation of chain squared errors divided by the square root of the number of chains. Largest-chain error share measures how much one chain contributes to total squared error. Paired differences align methods by saved seed; negative differences favor the named method over SCS. These are descriptive, conditional on the fitted maps, not formal confidence intervals or convergence diagnostics. Heavy-tailed norm errors can make variance-based uncertainty unreliable; fresh seed batches are needed to assess repeatability.')
    notes += [summary[key] for key in ('target_note', 'mse_note', 'time_scope') if isinstance(summary.get(key), str)]
    warning = '<p class="warning">Quick execution check only — do not interpret sampler rankings.</p>' if smoke else ''
    guard_hits = [(name, [row for row in cases[name].get('records', []) if row.get('scale_ceiling_reached')])
                  for name in names]
    guard_hits = [(name, rows) for name, rows in guard_hits if rows]
    if guard_hits:
        numerical = all(np.isclose(row.get('step_scale_ceiling', 0.), np.exp(15.), rtol=1e-10, atol=0.)
                        for _, rows in guard_hits for row in rows)
        heading = 'Numerical step-size guard reached.' if numerical else 'Recorded step-size limit reached.'
        details = '; '.join(f'{name}: {len(rows)}/{cases[name]["chains"]} chains' for name, rows in guard_hits)
        warning += ('<p class="warning"><strong>'+heading+'</strong> '+html.escape(details)+'. '
                    'This records calibration saturation, not evidence of an optimal step. '
                    'Inspect the frozen steps and production acceptance before interpreting comparisons.</p>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sampler comparison</title><style>body{font:15px/1.55 system-ui,sans-serif;max-width:1450px;margin:auto;padding:28px;color:#24354b;background:#f6f8fb}h1,h2{line-height:1.2}h2{margin-top:32px}a{color:#5a429c}figure{margin:24px 0;padding:10px;background:white;border-radius:8px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#566579}.table-wrap{overflow:auto;background:white}table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}td,th{padding:8px;border:1px solid #dee4ee;text-align:left;white-space:nowrap}td{font-family:ui-monospace,monospace}th{background:#eaf0f8}.warning{padding:12px;background:#fff0cf;border-left:4px solid #a4670d}li{margin-bottom:8px}</style></head><body>'''
    page += '<h1>'+html.escape(title)+(' — equal sampling time' if timed else ' — original comparison protocol')+'</h1>'+warning+'<p>'+html.escape(budget)+'.</p>'
    if suite:
        page += '<p>Target: '+html.escape(target_title)+'; VI '+html.escape(str(vi_mode))+' for both map families; DCS β = '+number(config.get('beta', 2.))+'.</p>'
    if config.get('protocol'):
        page += '<p>Recorded protocol: <code>'+html.escape(config['protocol'])+'</code>.</p>'
    if config.get('implementation_revision'):
        page += '<p>Implementation revision: <code>'+html.escape(config['implementation_revision'])+'</code>.</p>'
    if 'seed' in config:
        seed_label = ('unrecorded' if 'seed_batch' not in config else
                      'custom' if config['seed_batch'] is None else str(config['seed_batch']))
        page += '<p>Seed batch: '+html.escape(seed_label)+'; first chain seed: '+str(config['seed'])+'; map-fitting seed: '+str(config.get('fit_seed', 'unrecorded'))+'. Repeating a batch reuses its random streams; timed sample counts may still differ.</p>'
    page += '<p>'+' · '.join('<a href="'+filename+'">'+html.escape(filename)+'</a>' for filename in csvs)+'</p>'
    for filename in figure_names:
        page += '<figure><a href="figures/'+filename+'.pdf"><img src="figures/'+filename+'.png" alt="'+html.escape(filename.replace('_', ' '))+'"></a><figcaption>Click the figure to open its PDF.</figcaption></figure>'
    page += '<h2>Counts, errors and runtime</h2>'+_table(case_rows)
    if paired_rows:
        page += '<h2>Paired MSE differences from SCS</h2><p>Negative differences favor the named method. Empirical MCSE is descriptive; it is not a confidence interval.</p>'+_table(paired_rows)
    if setup_rows:
        page += '<h2>Map fitting excluded from production time</h2>'+_table(setup_rows)
    if tuning_summaries:
        page += '<details><summary>Reflected BW pilot tuning and frozen steps</summary><p>The score is the mean squared jump in ball coordinates, including zero movement after rejection. It is not an MSE estimate. Full pilot candidates are available in <a href="reflected_tuning.csv">reflected_tuning.csv</a> and the saved summary.</p>'+_table(tuning_summaries)+'</details>'
    page += '<h2>Every chain</h2>'+_table(chain_rows)
    page += '<h2>Full map parameters</h2>'+_table(parameter_rows)
    if prefix_rows:
        page += '<h2>Exact checkpoint MSE values</h2>'+_table(prefix_rows)
    page += '<h2>Protocol and interpretation</h2><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul>'
    page += '<p>Plots and tables use two decimals, with scientific notation for MSE and small values. Integer sample counts remain exact. CSV and JSON retain full numerical precision.</p></body></html>'
    (root/'paper_values.html').write_text(page, encoding='utf-8')
    return {'figures': figure_names, 'csv_files': list(csvs), 'html': 'paper_values.html'}
