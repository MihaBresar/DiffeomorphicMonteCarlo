"""Equal-time skewed Student-t figures and exact, auditable result tables.

The renderer consumes a completed summary. It does not tune samplers, select
cases, change budgets, or infer finite-variance uncertainty for the raw norm.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import textwrap

import numpy as np
from dcs_bench.display import number, cell, figure_ticks

METRICS = ('tail_probability', 'mean_norm')
COLORS = {'DCS-BW': '#6f4bb0', 'SCS': '#d13d77'}


def _text(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def _csv(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _table(rows):
    if not rows:
        return '<p>No values are available.</p>'
    columns = list(dict.fromkeys(key for row in rows for key in row))
    header = ''.join('<th>'+html.escape(key.replace('_', ' '))+'</th>' for key in columns)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(cell(row.get(key), key))+'</td>'
                                for key in columns)+'</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def _label(name, case):
    return case.get('label', name)


def _plot_label(name, case):
    return '\n'.join(textwrap.wrap(_label(name, case), width=27))


def _line_style(case):
    untuned = case.get('map_kind') in ('vanilla', 'untuned')
    adaptive = case.get('policy') == 'adaptive'
    return ('-.' if adaptive else '--') if untuned else (':' if adaptive else '-')


def render_report(root, summary):
    """Write four figure families, full-precision CSV files, and a local report."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    cases, config = summary['cases'], summary['config']
    if not cases:
        raise ValueError('The timed report requires at least one completed case')
    target = summary['target']
    truth = summary['truth']
    names = list(cases)
    labels = [_plot_label(name, cases[name]) for name in names]
    colors = [COLORS.get(cases[name]['algorithm'], '#526d82') for name in names]
    nu, dimension = target['nu'], target['dimension']
    if nu <= 1:
        raise ValueError('This report requires a finite mean norm (nu > 1)')
    threshold = config.get('threshold', 50.)
    budget_seconds = config['time_budget']
    smoke = config.get('preset') == 'quick'
    title = summary.get('target_label', f'{dimension}D skewed Student-t, ν = {nu:g}')
    if smoke:
        title = 'EXECUTION CHECK — '+title
    budget = (f"{config['runs']} chains per method; {budget_seconds:g} seconds of production per chain; "
              'all completed iterations contribute')
    labels_by_metric = {'tail_probability': f'P(‖X‖ ≥ {threshold:g})', 'mean_norm': 'E[‖X‖]'}
    norm_note = (f'At ν = {nu:g}, the norm has an infinite second moment. Its average squared error below is a '
                 'descriptive finite-run statistic; no finite-variance norm MSE interval is claimed.'
                 if nu <= 2 else 'Average squared errors are conditional on these fitted maps and pilot selections.')
    figure_names = []

    def save(fig, name):
        figure_ticks(fig)
        if smoke:
            fig.text(.5, .014, 'Short execution check only — not evidence of sampler performance',
                     ha='center', fontsize=10, color='#8c510a')
        fig.savefig(figures/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
        figure_names.append(name)

    # Original-style estimator distributions are retained, with exact errors
    # and sample counts immediately below each method rather than rounded ticks.
    fig, axes = (plt.subplots(2, 1, figsize=(20, 13.5)) if len(names) > 4
                 else plt.subplots(1, 2, figsize=(19, 8.5)))
    for ax, metric in zip(axes, METRICS):
        values = [np.asarray(cases[name]['estimates'][metric], dtype=float) for name in names]
        if any(len(v) == 0 or not np.all(np.isfinite(v)) for v in values):
            raise ValueError('Finite per-chain estimates are required for every method')
        bp = ax.boxplot(values, patch_artist=True, showfliers=False, widths=.52)
        for box, color, name in zip(bp['boxes'], colors, names):
            box.set_facecolor(color)
            box.set_alpha(.28)
            if cases[name].get('policy') == 'adaptive':
                box.set_hatch('//')
        for index, (v, color) in enumerate(zip(values, colors), 1):
            # Deterministic offsets avoid rendering RNGs and retain all chains.
            ax.scatter(index+np.linspace(-.11, .11, len(v)), v, color=color, s=18, alpha=.75, zorder=3)
        ax.axhline(truth[metric], color='#233344', linestyle='--', linewidth=1.3,
                   label='Target value: '+number(truth[metric]))
        ticks = []
        for name, label in zip(names, labels):
            case = cases[name]
            ticks.append(label+'\n'+('Empirical MSE =\n' if metric == 'mean_norm' and nu <= 2 else 'MSE =\n')+
                         number(case['empirical_mse'][metric], scientific=True)+'\nMean N = '+number(case['mean_iterations'], integer=True))
        ax.set_xticks(range(1, len(names)+1), ticks, fontsize=8)
        ax.tick_params(axis='x', pad=10)
        ax.set_ylabel('Per-chain estimate of '+labels_by_metric[metric])
        ax.set_title(labels_by_metric[metric]+' estimates', fontsize=12)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='best', fontsize=8)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.suptitle(title+' — estimates at equal sampling time\n'+budget, fontsize=14)
    fig.text(.5, .06, norm_note, ha='center', fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, .14, 1, .91))
    save(fig, 'skew_t_timed_estimates')

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5.8, .8*len(names)+2)))
    for ax, metric in zip(axes, METRICS):
        values = np.asarray([cases[name]['empirical_mse'][metric] for name in names], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Average squared errors must be finite and nonnegative')
        all_positive = bool(np.all(values > 0))
        for index, value in enumerate(values):
            ax.barh(index, value, color=colors[index], height=.6,
                    hatch='//' if cases[names[index]].get('policy') == 'adaptive' else None)
            ax.text(value, index, ' '+number(value, scientific=True), va='center', fontsize=9)
        ax.set_yticks(range(len(names)), labels, fontsize=10)
        ax.set_ylim(len(names)-.4, -.6)
        if all_positive:
            ax.set_xscale('log')
            ax.set_xlim(float(values.min())/2, float(values.max())*60)
        else:
            ax.set_xlim(0, float(values.max())*1.8 if values.max() > 0 else 1.)
        ax.set_title(('Descriptive average squared error: ' if metric == 'mean_norm' and nu <= 2 else 'MSE: ')+
                     labels_by_metric[metric], fontsize=11)
        ax.set_xlabel('MSE (two-decimal scientific labels); '+('logarithmic axis' if all_positive else 'linear axis (includes zero)'))
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(title+' — errors at equal sampling time\n'+budget, fontsize=14)
    fig.text(.5, .06, norm_note, ha='center', fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, .13, 1, .89))
    save(fig, 'skew_t_timed_errors')

    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, .85*len(names)+2)),
                             gridspec_kw={'width_ratios': [1.1, 1.]})
    for name, label, color in zip(names, labels, colors):
        records = sorted(cases[name]['records'], key=lambda row: row['chain'])
        chain_numbers = [row['chain']+1 for row in records]
        axes[0].plot(chain_numbers, [row['iterations'] for row in records], marker='o', markersize=4,
                     color=color, linestyle=_line_style(cases[name]),
                     label=label)
    from matplotlib.ticker import MaxNLocator
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_xlabel('Chain number')
    axes[0].set_ylabel('Completed production iterations, including rejections')
    axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=9)
    for index, name in enumerate(names):
        case = cases[name]
        value = case['mean_iterations']
        axes[1].barh(index, value, color=colors[index], height=.62,
                     hatch='//' if case.get('policy') == 'adaptive' else None)
        annotation = ('Mean N = '+number(value, integer=True)+'\nTotal N = '+str(case['total_iterations'])+
                      '\nMean seconds = '+number(case['mean_sampling_seconds']))
        axes[1].text(value, index, ' '+annotation.replace('\n', '\n '), va='center', fontsize=8)
    axes[1].set_yticks(range(len(names)), labels, fontsize=9)
    axes[1].set_ylim(len(names)-.4, -.6)
    axes[1].set_xlim(0, max(float(cases[n]['mean_iterations']) for n in names)*2.1)
    axes[1].set_xlabel('Mean completed production iterations per chain')
    axes[1].ticklabel_format(axis='x', style='plain', useOffset=False)
    axes[1].grid(axis='x', alpha=.2)
    axes[1].set_axisbelow(True)
    fig.suptitle(title+' — actual sample counts\n'+budget, fontsize=14)
    fig.text(.5, .065, 'All states, including rejection repeats, enter the estimates. Every chain count and measured time is saved in CSV.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .91))
    save(fig, 'skew_t_timed_samples')

    pilot = summary.get('pilots', {})
    grid = pilot.get('grid', [])
    if grid:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        for name in dict.fromkeys(row['case'] for row in grid):
            rows = sorted([r for r in grid if r['case'] == name], key=lambda r: r['scale'])
            case = cases.get(name) or next((c for n, c in cases.items()
                                           if c.get('map_group', n) == name), {})
            color = COLORS.get(case.get('algorithm'), '#526d82')
            style = '--' if case.get('map_kind') in ('vanilla', 'untuned') else '-'
            label = {'mobius-DCS-BW': 'DCS-BW: Möbius map', 'fitted-SCS': 'SCS: fitted map',
                     'vanilla-DCS-BW': 'DCS-BW: untuned map', 'untuned-SCS': 'SCS: untuned map'}.get(name, name)
            for ax, key in zip(axes, ('score', 'mean_seconds')):
                ax.plot([r['scale'] for r in rows], [r.get(key, np.nan) for r in rows],
                        color=color, linestyle=style, marker='o', label=label, markersize=4)
            choice = pilot.get('selected', {}).get(name, {})
            selected = next((r for r in rows if r['scale'] == choice.get('scale')), None)
            if selected is not None:
                axes[0].plot(selected['scale'], selected['score'], marker='*', markersize=15,
                             color=color, markeredgecolor='#24394f')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlabel('Proposal scale: DCS ball radius; SCS tangent coordinate SD')
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Declared independent-pilot score; larger is better')
        axes[0].legend(fontsize=9)
        axes[1].set_ylabel('Mean measured pilot production seconds')
        fig.suptitle(title+' — independent proposal pretuning', fontsize=14)
        fig.text(.5, .065, 'Stars mark the declared selection. A grid-endpoint winner is only the best tested scale; see exact pilot tables.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, .91))
        save(fig, 'skew_t_timed_proposal_scales')

    case_rows, chain_rows, metric_rows = [], [], []
    for name, case in cases.items():
        row = {'case': name, 'label': _label(name, case), 'algorithm': case['algorithm'],
               'map_kind': case['map_kind'], 'map_group': case.get('map_group', name),
               'production_step_policy': case.get('policy', 'frozen'),
               'selected_pilot_scale': case.get('pilot_scale'),
               'mean_initial_production_scale': case['scale'],
               'chains': len(case['records']), 'requested_seconds_per_chain': budget_seconds,
               'mean_sampling_seconds': case['mean_sampling_seconds'],
               'mean_iterations': case['mean_iterations'], 'total_iterations': case['total_iterations']}
        for metric in METRICS:
            row[metric+'_truth'] = truth[metric]
            row[metric+'_empirical_mse'] = case['empirical_mse'][metric]
            row[metric+'_median_absolute_error'] = case['median_absolute_error'][metric]
            metric_rows.append({'case': name, 'metric': metric, 'truth': truth[metric],
                                'empirical_mse': case['empirical_mse'][metric],
                                'median_absolute_error': case['median_absolute_error'][metric],
                                'interpretation': 'Descriptive finite-run squared errors; target second moment diverges'
                                if metric == 'mean_norm' and nu <= 2 else 'Empirical MSE across chains'})
        case_rows.append(row)
        for record in sorted(case['records'], key=lambda r: r['chain']):
            record_row = {k: v for k, v in record.items() if k != 'estimates'}
            record_row.update(case=name, label=_label(name, case),
                              production_step_policy=case.get('policy', 'frozen'),
                              requested_seconds=budget_seconds)
            for metric in METRICS:
                estimate = record['estimates'][metric]
                record_row[metric+'_estimate'] = estimate
                record_row[metric+'_truth'] = truth[metric]
                record_row[metric+'_squared_error'] = (estimate-truth[metric])**2
                record_row[metric+'_absolute_error'] = abs(estimate-truth[metric])
            chain_rows.append(record_row)
    pilot_rows = []
    for record in grid:
        row = {key: value for key, value in record.items() if key not in ('bulk_ess', 'rank_rhat')}
        for metric, value in record.get('bulk_ess', {}).items():
            row['bulk_ess_'+metric] = value
        for metric, value in record.get('rank_rhat', {}).items():
            row['rank_rhat_'+metric] = value
        choice = pilot.get('selected', {}).get(record['case'], {})
        row['selected'] = choice.get('scale') == record['scale']
        row['selected_grid_endpoint'] = choice.get('endpoint_winner') if row['selected'] else False
        pilot_rows.append(row)
    uncertainty_rows = []
    for contrast in summary.get('contrasts', []):
        if contrast['metric'] != 'tail_probability':
            raise ValueError('This renderer only accepts uncertainty intervals for the bounded tail indicator')
        uncertainty_rows.append({key: value for key, value in contrast.items() if key != 'ci95'} |
                                {'ci95_low': contrast['ci95'][0], 'ci95_high': contrast['ci95'][1]})
    csvs = {'skew_t_timed_cases.csv': case_rows, 'skew_t_timed_chains.csv': chain_rows,
            'skew_t_timed_values.csv': metric_rows, 'skew_t_timed_pilots.csv': pilot_rows,
            'skew_t_timed_uncertainty.csv': uncertainty_rows}
    for name, rows in csvs.items():
        _csv(root/name, rows, ['a', 'b', 'metric', 'difference', 'ci95_low', 'ci95_high']
             if not rows and name.endswith('uncertainty.csv') else None)
    image_sections = ''.join('<figure><a href="figures/'+name+'.pdf"><img src="figures/'+name+
                             '.png" alt="'+html.escape(name.replace('_', ' '))+'"></a><figcaption>'+
                             html.escape(name.replace('_', ' '))+': click for PDF.</figcaption></figure>'
                             for name in figure_names)
    notes = [
        'Every method receives proposal pretuning after map preparation. Untuned refers to the map only. Frozen cases keep the selected pilot scale. Adaptive cases update that scale during discarded warmup and continue updating it during production; selected pilot, initial production and final production scales are distinguished in the tables.',
        'The adaptive cases use changing transition kernels. They are a separate algorithmic comparison, not evidence about the frozen kernel or a stationary MCMC error formula. Initial and final production scales and adaptation flags are recorded for each chain.',
        'These are fresh equal-wall-time runs, not time-rescaled errors from a fixed-iteration run.',
        'Estimates use every completed production iteration, including repeated states after rejected proposals. Thinning affects only retained diagnostics.',
        'Each chain receives its own production-time budget. Map fitting, proposal pilots, warmup, imports and result saving are outside that clock; setup is recorded separately.',
        'The final transition can finish after the requested deadline. Exact actual elapsed times, iteration counts and available CPU-time measurements are listed below.',
        norm_note,
        'Only the bounded tail indicator receives a descriptive across-run uncertainty interval. Intervals are conditional on the fitted maps and selected pilot scales; they do not establish stationarity, optimizer robustness or convergence of an adaptive algorithm.',
        'Timing depends on this hardware, worker count, operating-system scheduling and competing load. Sample counts are exact for this run, not guaranteed on another computer.',
    ]+list(summary.get('notes', []))
    setup = {key: summary[key] for key in ('setup', 'fit_metadata', 'timing_protocol', 'environment', 'hardware')
             if key in summary}
    details = ('<details><summary>Recorded setup and timing context (full precision in summary.json)</summary><pre>'+html.escape(
        cell(setup))+'</pre></details>') if setup else ''
    warning = '<p class="warning">Quick execution check only — do not interpret sampler rankings.</p>' if smoke else ''
    report = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equal-time Student-t results</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#203147;background:#f6f8fb;margin:0 auto;max-width:1500px;padding:28px}
h1,h2{line-height:1.2}h2{margin-top:36px}a{color:#4e46a5}figure{margin:24px 0;background:white;padding:12px;border-radius:9px}img{max-width:100%;height:auto}
figcaption{font-size:13px;color:#576478}.table-wrap{overflow-x:auto;background:#fff;border:1px solid #dde3ed;border-radius:6px}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #e4e8ef;white-space:nowrap;text-align:left}th{background:#eaf0f8}td{font-family:ui-monospace,monospace}pre{overflow:auto;background:white;padding:16px}
.warning{background:#fff1ce;padding:12px;border-left:4px solid #a57316}.note{background:#e9eef9;padding:14px}li{margin-bottom:6px}</style></head><body>'''
    report += '<h1>'+html.escape(title)+' — equal sampling time</h1>'+warning
    report += '<p>'+html.escape(budget)+'.</p><p class="note">'+html.escape(norm_note)+'</p>'
    report += '<p>'+ ' · '.join('<a href="'+name+'">'+html.escape(name)+'</a>' for name in csvs)+'</p>'
    report += image_sections+'<h2>Exact case values</h2>'+_table(case_rows)
    report += '<h2>Every chain: estimates, counts and measured times</h2>'+_table(chain_rows)
    report += '<h2>All independent pilot candidates</h2>'+_table(pilot_rows)
    report += '<h2>Tail-indicator uncertainty comparisons</h2>'+_table(uncertainty_rows)
    report += '<h2>Interpretation and timing</h2><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul>'+details
    report += '<p>Plots and tables use two decimals, with scientific notation for MSE and small values. Integer sample counts remain exact. CSV and JSON preserve full numerical precision.</p></body></html>'
    (root/'paper_values.html').write_text(report, encoding='utf-8')
    return {'figures': figure_names, 'csv_files': list(csvs), 'html': 'paper_values.html'}
