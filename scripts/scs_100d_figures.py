"""Original-style 100D SCS comparison figures and exact numerical tables.

Rendering never selects methods, changes tuning, or substitutes thinned
estimates for the recorded all-iteration chain means.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

METRICS = ('tail_probability', 'mean_norm')
ALIASES = (('tail_probability', 'tail'), ('mean_norm', 'norm'))
METHODS = ('DCS-RWM', 'DCS-BW', 'SCS')
COLORS = {'DCS-RWM': '#FF8C00', 'DCS-BW': '#9370db', 'SCS': '#ff1493'}


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
    body = ''.join('<tr>'+''.join('<td>'+html.escape(_text(row.get(key)))+'</td>' for key in fields)+'</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def _policy(name):
    return 'Paper-date SCS: fixed h = 0.1' if name == 'SCS' else 'Continuous smoothed dual averaging; target acceptance 0.234'


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
    if not cases or any(name not in METHODS for name in cases):
        raise ValueError('Expected completed DCS-RWM, DCS-BW, or SCS cases')
    names = [name for name in METHODS if name in cases]
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
    target = summary.get('target', {})
    dimension, nu = target.get('dimension', 100), target.get('nu', 3.)
    threshold = config.get('threshold', 50.)
    smoke = config.get('preset') == 'quick'
    title = ('EXECUTION CHECK — ' if smoke else '')+f'{dimension}D skewed Student-t, ν = {nu:g}'
    budget = (f"{config['runs']} chains × {config['iterations']} production iterations per method; "
              f"{config['warmup']} warmup iterations discarded")
    metric_labels = (f'P(‖X‖ ≥ {threshold:g})', 'E[‖X‖]')
    x = np.arange(1, len(names)+1)
    colors = [COLORS[name] for name in names]
    figure_names = []

    def save(fig, name):
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
              [cases[name]['mean_seconds'] for name in names])
    for ax, values, label, ylabel in zip(axes, panels,
            ('Tail-probability MSE', 'Mean-norm MSE', 'Average Runtime (s)'),
            ('Mean Squared Error', 'Mean Squared Error', 'Seconds')):
        bars = ax.bar(x, values, color=colors, alpha=.8, edgecolor='#333333', width=.8)
        ax.bar_label(bars, labels=[repr(float(value)) for value in values], padding=5, fontsize=8)
        ax.set_xticks(x, original_labels, rotation=15, ha='right', fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        largest = max(float(value) for value in values)
        ax.set_ylim(0, largest*1.18 if largest > 0 else 1.)
        ax.grid(axis='y', linestyle='--', alpha=.5)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.suptitle(('EXECUTION CHECK — ' if smoke else '')+'Performance & Mean Squared Error Comparison\n'+
                 f'{dimension}D skewed Student-t, ν = {nu:g}', fontsize=14, fontweight='bold')
    fig.text(.5, .079, budget+'. All production states enter the estimates.', ha='center', fontsize=9)
    fig.text(.5, .049, 'DCS steps adapt continuously; SCS uses h = 0.1. Runtime includes estimation and excludes fitting, compilation and warmup.',
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
                   label='True value = '+repr(float(truth[metric])))
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
            ax.text(position, max(float(value), floor)*1.15, repr(float(value)),
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylim(floor, max(float(values.max())*4, floor*10))
        ax.set_xticks(x, [name+'\nN = '+str(config['iterations']) for name in names])
        ax.set_title(label+' MSE across chains')
        ax.set_ylabel('MSE (logarithmic scale)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if np.any(values == 0):
            ax.text(.02, .02, 'Zero values marked at the lower axis edge', transform=ax.transAxes, fontsize=8)
    fig.suptitle(title+' — estimates and logarithmic MSE\n'+budget, fontsize=14)
    fig.text(.5, .046, 'All production iterations contribute, including rejected states. DCS adapts during sampling; SCS uses h = 0.1.',
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
        ax.bar_label(bars, labels=[repr(float(value)) for value in values], fontsize=9, padding=5)
        ax.set_xticks(x, names)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, float(values.max())*1.28)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.suptitle(title+' — measured runtime at equal iteration counts\n'+budget, fontsize=14)
    fig.text(.5, .055, 'Clock includes all-iteration estimation; excludes map fitting, compilation, warmup and saving. Hardware and load affect timing.',
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
            ax.text(position, variance_+bias_, ' MSE = '+repr(float(mses[name][metric]))+
                    '\n bias² = '+repr(float(bias_))+'\n variance = '+repr(float(variance_)),
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
                              for row in record.get('checkpoints', [])} for record in records]
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

    case_rows, chain_rows = [], []
    for name in names:
        case = cases[name]
        row = {'method': name, 'production_step_policy': _policy(name), 'chains': case['chains'],
               'iterations_per_chain': config['iterations'], 'total_iterations': case['total_iterations'],
               'warmup_iterations': config['warmup'], 'thin': config['thin'],
               'retained_per_chain': config['iterations']//config['thin'],
               'mean_seconds': case['mean_seconds'], 'median_seconds': case['median_seconds'],
               'iterations_per_second': case['iterations_per_second'],
               'mean_acceptance': case['mean_acceptance'], 'final_scales': case['final_scales']}
        for metric, key in enumerate(METRICS):
            row[key+'_truth'] = truth[metric]
            row[key+'_mse'] = mses[name][metric]
            row[key+'_bias_squared'] = biases[name][metric]
            row[key+'_variance_ddof0'] = variances[name][metric]
        case_rows.append(row)
        records = case.get('records', [])
        if records and len(records) != case['chains']:
            raise ValueError('Each chain must have exactly one record')
        for index, estimate in enumerate(means[name]):
            row = dict(records[index]) if records else {'chain': index}
            row.update(method=name, production_step_policy=_policy(name))
            row.setdefault('iterations', config['iterations'])
            for metric, key in enumerate(METRICS):
                row[key+'_estimate'] = estimate[metric]
                row[key+'_truth'] = truth[metric]
                row[key+'_squared_error'] = (estimate[metric]-truth[metric])**2
            chain_rows.append(row)
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
        setup_rows.append({'map': family, 'used_by': 'DCS-RWM and DCS-BW' if family == 'ball' else 'SCS',
                           'fitting_seconds': fitting.get('seconds'), 'steps': fitting.get('steps'),
                           'training_draws': fitting.get('training_draws'),
                           'samples_per_step': fitting.get('samples_per_step'),
                           'total_training_draws': total_draws, 'training_count_source': count_source,
                           'training_batch_policy': fitting.get('training_batch_policy', fitting.get('batch_policy'))})
    count_rows = [{key: row[key] for key in ('method', 'chains', 'iterations_per_chain',
                   'total_iterations', 'warmup_iterations', 'thin', 'retained_per_chain',
                   'mean_seconds', 'median_seconds', 'iterations_per_second')} for row in case_rows]
    csvs = {'scs_100d_values.csv': case_rows, 'paper_values.csv': case_rows,
            'sample_counts.csv': count_rows, 'scs_100d_chains.csv': chain_rows,
            'tuned_parameters.csv': parameter_rows}
    if prefix_rows:
        csvs['scs_100d_prefix_mse.csv'] = prefix_rows
    if setup_rows:
        csvs['setup_times.csv'] = setup_rows
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
    notes += [summary[key] for key in ('target_note', 'mse_note', 'time_scope') if isinstance(summary.get(key), str)]
    warning = '<p class="warning">Quick execution check only — do not interpret sampler rankings.</p>' if smoke else ''
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>100D original SCS comparison</title><style>body{font:15px/1.55 system-ui,sans-serif;max-width:1450px;margin:auto;padding:28px;color:#24354b;background:#f6f8fb}h1,h2{line-height:1.2}h2{margin-top:32px}a{color:#5a429c}figure{margin:24px 0;padding:10px;background:white;border-radius:8px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#566579}.table-wrap{overflow:auto;background:white}table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}td,th{padding:8px;border:1px solid #dee4ee;text-align:left;white-space:nowrap}td{font-family:ui-monospace,monospace}th{background:#eaf0f8}.warning{padding:12px;background:#fff0cf;border-left:4px solid #a4670d}li{margin-bottom:8px}</style></head><body>'''
    page += '<h1>'+html.escape(title)+' — original comparison protocol</h1>'+warning+'<p>'+html.escape(budget)+'.</p>'
    page += '<p>'+' · '.join('<a href="'+filename+'">'+html.escape(filename)+'</a>' for filename in csvs)+'</p>'
    for filename in figure_names:
        page += '<figure><a href="figures/'+filename+'.pdf"><img src="figures/'+filename+'.png" alt="'+html.escape(filename.replace('_', ' '))+'"></a><figcaption>Click the figure to open its PDF.</figcaption></figure>'
    page += '<h2>Exact counts, errors and runtime</h2>'+_table(case_rows)
    if setup_rows:
        page += '<h2>Map fitting excluded from production time</h2>'+_table(setup_rows)
    page += '<h2>Every chain</h2>'+_table(chain_rows)
    page += '<h2>Full map parameters</h2>'+_table(parameter_rows)
    if prefix_rows:
        page += '<h2>Exact checkpoint MSE values</h2>'+_table(prefix_rows)
    page += '<h2>Protocol and interpretation</h2><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul>'
    page += '<p>CSV and HTML retain every stored numerical digit. Plot MSE labels use exact floating-point representations.</p></body></html>'
    (root/'paper_values.html').write_text(page, encoding='utf-8')
    return {'figures': figure_names, 'csv_files': list(csvs), 'html': 'paper_values.html'}
