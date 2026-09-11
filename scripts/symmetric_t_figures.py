"""Figures and exact numerical tables for the symmetric Student-t control.

Rendering reads the supplied summary only; it never selects or reruns a case.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np
from dcs_bench.display import number, cell, figure_ticks


FUNCTIONS = ['tail_probability', 'mean_norm', 'radial_95',
             'asinh_x1', 'asinh_x2', 'asinh_x3', 'asinh_x4']
COLORS = {'DCS-BW': '#7852b4', 'SCS': '#dc367a'}
MAP_LABELS = {'untuned': 'Untuned map', 'vi_scale': 'VI scale only'}
POLICY_LABELS = {'warmup_frozen': 'step warmup, then frozen',
                 'pilot_selected': 'five-function pilot step',
                 'pilot_radial': 'radial-focused pilot step'}


def case_label(name, case):
    return (f"{MAP_LABELS.get(case['map_kind'], case['map_kind'])} · {case['algorithm']}\n"
            f"{POLICY_LABELS.get(case['policy'], case['policy'])}")


def _number(value):
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _text(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return str(value)


def _table(rows, columns=None):
    if not rows:
        return '<p>No values are available.</p>'
    columns = list(rows[0]) if columns is None else columns
    heading = ''.join('<th>'+html.escape(key.replace('_', ' '))+'</th>' for key in columns)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(cell(row.get(key), key))+'</td>'
                               for key in columns)+'</tr>' for row in rows)
    return ('<div class="table-wrap"><table><thead><tr>'+heading+
            '</tr></thead><tbody>'+body+'</tbody></table></div>')


def _csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def render_report(root, summary):
    """Write readable figures and tables with full-precision CSV exports."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    config, cases = summary['config'], summary['cases']
    if not cases:
        raise ValueError('The symmetric Student-t report requires completed cases')
    target = summary.get('target', config.get('target', {}))
    dimension = target.get('dimension', 100)
    nu = target.get('nu', 3)
    smoke = config.get('preset') == 'quick'
    title = f'{dimension}D symmetric Student-t, ν = {nu}'
    if smoke:
        title = 'EXECUTION CHECK — '+title
    iterations = config['iterations']
    retained = iterations//config['thin']
    budget = (f"{config['runs']} chains × {iterations} production iterations; "
              f"{retained} retained values per chain (thin = {config['thin']})")
    names = list(cases)
    labels = [case_label(name, cases[name]) for name in names]
    colors = [COLORS.get(cases[name]['algorithm'], '#596a79') for name in names]
    mse_functions = list(dict.fromkeys(FUNCTIONS + [function for case in cases.values()
                                                   for function in case['mse']]))
    functions = list(dict.fromkeys(mse_functions + [function for case in cases.values()
                                                    for function in case.get('bulk_ess', {})]))

    def save(fig, name):
        figure_ticks(fig)
        if smoke:
            fig.text(.5, .015, 'Short execution check only — not evidence of sampler performance',
                     ha='center', fontsize=10, color='#8c510a')
        fig.savefig(figures/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)

    def bars(ax, metric, function, heading, log=False):
        values = [_number(cases[name].get(metric, {}).get(function)) for name in names]
        positive = [value for value in values if value is not None and value > 0]
        left = min(positive)/1.8 if log and positive else 0.
        for index, value in enumerate(values):
            if value is not None and value > 0:
                bar = ax.barh(index, value, color=colors[index], height=.7)[0]
                if cases[names[index]]['map_kind'] == 'vi_scale':
                    bar.set_hatch('//')
                    bar.set_edgecolor('#ffffff')
                ax.text(value, index, ' '+number(value, scientific='mse' in metric), va='center', fontsize=8)
            else:
                label = number(value, scientific='mse' in metric)
                ax.text(left, index, label, va='center', fontsize=8)
        ax.set_yticks(range(len(names)), labels, fontsize=9)
        # Text for an unavailable diagnostic does not expand Matplotlib's data
        # limits; set them explicitly so first and last rows remain visible.
        ax.set_ylim(len(names)-.35, -.65)
        ax.set_title(heading, fontsize=12, pad=12)
        if log and positive:
            ax.set_xscale('log')
            ax.set_xlim(left, max(positive)*30)
        elif positive:
            ax.set_xlim(0, max(positive)*1.6)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(21, max(8, .77*len(names))))
    bars(axes[0], 'mse', 'tail_probability', 'MSE of P(‖X‖ ≥ 50)', log=True)
    bars(axes[1], 'mse', 'mean_norm', 'Mean-norm MSE', log=True)
    for ax in axes:
        ax.set_xlabel('MSE (log scale); two-decimal scientific labels')
    fig.suptitle(title+' — radial estimation\n'+budget, fontsize=14)
    fig.tight_layout(rect=(0, .055, 1, .93))
    save(fig, 'symmetric_t_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(21, max(8, .77*len(names))))
    bars(axes[0], 'mse', 'asinh_x1', 'MSE of the mean of asinh(x₁); exact mean = 0', log=True)
    bars(axes[1], 'bulk_ess', 'asinh_x1', 'Bulk ESS for asinh(x₁) on retained values')
    axes[0].set_xlabel('MSE (log scale); two-decimal scientific labels')
    axes[1].set_xlabel(f"Bulk ESS; nominal total = {config['runs']*retained} retained values")
    fig.suptitle(title+' — coordinate estimation\n'+budget, fontsize=14)
    fig.tight_layout(rect=(0, .055, 1, .93))
    save(fig, 'symmetric_t_coordinate')

    case_rows, metric_rows, mse_rows = [], [], []
    for name, case in cases.items():
        settings = {'case': name, 'algorithm': case['algorithm'], 'map_kind': case['map_kind'],
            'step_policy': case['policy'], 'chains': case['chains'],
            'production_iterations_per_chain': iterations, 'retained_values_per_chain': retained,
            'thin': config['thin'], 'warmup_iterations_per_chain': config['warmup'],
            'step_sizes': case['scales'], 'acceptance_rate': case.get('acceptance_mean'),
            'outside_ball_fraction': case.get('outside_ball_fraction'),
            'cap_stepout_fraction': case.get('cap_stepout_fraction')}
        case_rows.append(settings)
        mse_rows.append({'case': name, 'chains': case['chains'],
                         'production_iterations_per_chain': iterations,
                         'retained_values_per_chain': retained,
                         **{function+'_mse': case['mse'].get(function) for function in mse_functions}})
        for function in functions:
            metric_rows.append({'case': name, 'algorithm': case['algorithm'],
                'map_kind': case['map_kind'], 'step_policy': case['policy'],
                'chains': case['chains'], 'production_iterations_per_chain': iterations,
                'retained_values_per_chain': retained, 'function': function,
                'mse': case['mse'].get(function),
                'bulk_ess': case.get('bulk_ess', {}).get(function),
                'rank_rhat': case.get('rank_rhat', {}).get(function)})
    _csv(root/'symmetric_t_case_settings.csv', case_rows)
    _csv(root/'symmetric_t_values.csv', metric_rows)

    # Keep the retained-trace figures above exactly as before. This additional
    # estimate uses streaming functional sums over every production iteration.
    all_iteration_rows, all_iteration_mse_rows = [], []
    all_iteration_cases = [name for name in names if 'mse_all_iterations' in cases[name]]
    if all_iteration_cases:
        if len(all_iteration_cases) != len(names):
            raise ValueError('All-iteration MSE must be supplied for every production case')
        all_budget = (f"{config['runs']} chains × {iterations} production iterations; "
                      f"all {iterations} values per chain contribute, including repeated states")
        fig, axes = plt.subplots(1, 2, figsize=(21, max(8, .77*len(names))))
        bars(axes[0], 'mse_all_iterations', 'tail_probability', 'MSE of P(‖X‖ ≥ 50)', log=True)
        bars(axes[1], 'mse_all_iterations', 'mean_norm', 'Mean-norm MSE', log=True)
        for ax in axes:
            ax.set_xlabel('MSE using every production iteration (log scale); two-decimal scientific labels')
        fig.suptitle(title+' — estimates using every production iteration\n'+all_budget, fontsize=14)
        fig.tight_layout(rect=(0, .055, 1, .93))
        save(fig, 'symmetric_t_all_iterations')
        all_functions = list(dict.fromkeys(mse_functions + [function for case in cases.values()
                                                            for function in case['mse_all_iterations']]))
        for name, case in cases.items():
            counts = {'case': name, 'chains': case['chains'],
                      'iterations_per_chain': iterations,
                      'total_iterations': case['chains']*iterations}
            all_iteration_mse_rows.append({**counts,
                **{function+'_mse': case['mse_all_iterations'].get(function) for function in all_functions}})
            for function in all_functions:
                all_iteration_rows.append({**counts, 'function': function,
                    'mse': case['mse_all_iterations'].get(function)})
        _csv(root/'symmetric_t_all_iterations.csv', all_iteration_rows)

    pilot = summary.get('pilots', {})
    grid = pilot.get('grid', [])
    pilot_rows, pilot_metric_rows = [], []
    if grid:
        has_radial = any('radial_score' in row for row in grid)
        keys = ('score', 'radial_score', 'acceptance_mean') if has_radial else ('score', 'acceptance_mean')
        fig, axes = plt.subplots(1, len(keys), figsize=(7.5*len(keys), 5.8))
        for group in dict.fromkeys(row['group'] for row in grid):
            data = sorted([row for row in grid if row['group'] == group], key=lambda row: row['scale'])
            algorithm, map_kind = data[0]['algorithm'], data[0]['map_kind']
            label = MAP_LABELS.get(map_kind, map_kind)+' · '+algorithm
            style = '--' if map_kind == 'vi_scale' else '-'
            color = COLORS.get(algorithm, '#596a79')
            for ax, key in zip(axes, keys):
                values = [row.get(key) for row in data]
                ax.plot([row['scale'] for row in data], values, marker='o', linestyle=style,
                        color=color, label=label, linewidth=1.6, markersize=5)
            choices = [('selected', 'score', axes[0])]
            if has_radial:
                choices.append(('selected_radial', 'radial_score', axes[1]))
            for selection_key, score_key, ax in choices:
                selection = pilot.get(selection_key, {}).get(group, {})
                selected_row = next((row for row in data if row['scale'] == selection.get('scale')), None)
                if selected_row is not None:
                    ax.plot(selected_row['scale'], selected_row[score_key],
                            marker='*', markersize=15, markeredgecolor='#172940',
                            markerfacecolor=color, linestyle='none')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlabel('Proposal scale: ball radius (DCS-BW); tangent coordinate SD (SCS)')
            ax.grid(alpha=.18)
            ax.legend(fontsize=9)
        axes[0].set_ylabel('Minimum bulk ESS / retained values')
        axes[0].set_title('Five-function score; stars mark selected steps')
        if has_radial:
            axes[1].set_ylabel('Minimum bulk ESS / retained values')
            axes[1].set_title('Radial score: tail probability and mean norm')
        axes[-1].set_ylabel('Acceptance rate')
        axes[-1].set_ylim(-.03, 1.03)
        axes[-1].set_title('Acceptance diagnostics')
        fig.suptitle(title+' — proposal-scale pilots', fontsize=14)
        fig.tight_layout(rect=(0, .055, 1, .95))
        save(fig, 'symmetric_t_proposal_scales')
        for row in grid:
            selection = pilot.get('selected', {}).get(row['group'], {})
            selected = row['scale'] == selection.get('scale')
            radial_selection = pilot.get('selected_radial', {}).get(row['group'], {})
            selected_radial = row['scale'] == radial_selection.get('scale')
            settings = {'group': row['group'], 'algorithm': row['algorithm'], 'map_kind': row['map_kind'],
                'scale': row['scale'], 'five_function_score': row.get('score'),
                'radial_score': row.get('radial_score'),
                'min_ess_over_retained': row.get('min_ess_over_retained'),
                'acceptance_rate': row.get('acceptance_mean'),
                'outside_ball_fraction': row.get('outside_ball_fraction'),
                'cap_stepout_fraction': row.get('cap_stepout_fraction'),
                'selected_five_function': selected,
                'selected_radial': selected_radial,
                'five_function_endpoint': selected and selection.get('endpoint_winner', False),
                'radial_endpoint': selected_radial and radial_selection.get('endpoint_winner', False),
                'five_function_selection_valid': selection.get('valid_selection'),
                'radial_selection_valid': radial_selection.get('valid_selection')}
            pilot_rows.append(settings)
            for function in functions:
                pilot_metric_rows.append({**settings, 'function': function,
                    'mse': row.get('mse', {}).get(function),
                    'bulk_ess': row.get('bulk_ess', {}).get(function),
                    'rank_rhat': row.get('rank_rhat', {}).get(function)})
        _csv(root/'symmetric_t_pilot_values.csv', pilot_metric_rows)

    contrasts = []
    for row in summary.get('contrasts', []):
        low, high = row['ci95']
        contrasts.append({'case_a': row['a'], 'case_b': row['b'], 'function': row['function'],
            'mse_difference_a_minus_b': row['difference'], 'bootstrap_95_low': low,
            'bootstrap_95_high': high, 'interval_includes_zero': low <= 0 <= high})
    _csv(root/'symmetric_t_uncertainty.csv', contrasts)
    all_iteration_contrasts = []
    for row in summary.get('all_iteration_contrasts', []):
        low, high = row['ci95']
        all_iteration_contrasts.append({'case_a': row['a'], 'case_b': row['b'], 'function': row['function'],
            'mse_difference_a_minus_b': row['difference'], 'bootstrap_95_low': low,
            'bootstrap_95_high': high, 'interval_includes_zero': low <= 0 <= high})
    _csv(root/'symmetric_t_all_iteration_uncertainty.csv', all_iteration_contrasts)

    map_rows = []
    for group, mapping in summary.get('maps', {}).items():
        parameters = mapping.get('parameters', mapping)
        fit = mapping.get('fit_metadata', summary.get('fit_metadata', {}).get(group, {}))
        shift = parameters.get('delta', parameters.get('observer', []))
        map_rows.append({'group': group, 'R': parameters.get('scale'),
            'centre_norm': float(np.linalg.norm(parameters.get('mu', []))),
            'shift_norm': float(np.linalg.norm(shift)),
            'VI_steps': fit.get('steps'), 'fit_seconds': fit.get('seconds')})

    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Symmetric Student-t control: exact values</title><style>'
    page += ('body{font:16px system-ui;margin:32px;color:#172940}h1,h2{line-height:1.25}'
             'table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #ccd5df;text-align:right;vertical-align:top}'
             'td:first-child,th:first-child{text-align:left}th{background:#eef3f8}p,li{max-width:1100px;line-height:1.6}'
             '.table-wrap{overflow-x:auto;margin:20px 0}h2{margin-top:36px}summary{cursor:pointer;font-weight:600;padding:12px 0}'
             '.warning{padding:12px 16px;border-left:4px solid #b76b11;background:#fff7e8}'
             'img{max-width:100%;height:auto}pre{overflow:auto;background:#eef3f8;padding:16px}a{color:#315d99}</style>')
    page += '<h1>'+html.escape(title)+'</h1><p>'+html.escape(budget)+'</p>'
    if smoke:
        page += '<p class="warning">QUICK CHECK: these short runs check execution only and cannot support sampler rankings.</p>'
    page += ('<p>Both untuned maps use R = 1 and zero centres and shifts. DCS uses β = 2; SCS uses cap latitude 1.1. '
             'For this symmetric target, VI fixes centres and shifts to zero and fits R only. '
             'Every production step size is pretuned after the map is chosen, using either warmup or independent pilot traces, and then frozen. '
             'Proposal scales mean a whole-ball increment radius for DCS-BW and a per-coordinate tangent standard deviation for SCS; '
             'these are different proposal geometries.</p>')
    if all_iteration_rows:
        page += '<h2>Estimates using every production iteration</h2>'
        page += '<p>'+html.escape(all_budget)+'. These estimates use every sampled state after warmup, including repeated states from rejected proposals. No thinning is applied to these functional means.</p>'
        page += '<img src="figures/symmetric_t_all_iterations.png" alt="Radial MSE using every production iteration for every case">'
        page += '<h3>Exact MSE and iteration counts</h3>'+_table(all_iteration_mse_rows)
        page += '<p>The existing figures and ESS diagnostics below use the retained traces, with thin = '+str(config['thin'])+'. The additional full-iteration estimates do not change those results or their sample counts.</p>'
        page += ('<h3>Paired differences for estimates using every iteration</h3><p>Descriptive paired-chain bootstrap intervals, conditional on map fits and selected steps, '
                 'with no multiplicity correction. Differences are MSE(case A) minus MSE(case B); negative values favour A. An interval containing zero does not resolve the ranking.</p>')
        page += _table(all_iteration_contrasts)
    page += '<h2>Radial estimation: every production case</h2><img src="figures/symmetric_t_comparison.png" alt="Radial MSE for every map and step policy">'
    page += '<h2>Exact MSE values and sample counts</h2>'+_table(mse_rows)
    page += '<p>Plots and tables use two decimals, with scientific notation for MSE and small values. Integer counts remain exact; CSV and JSON retain full precision. Iteration counts include rejected proposals.</p>'
    if summary.get('truth'):
        page += '<details><summary>Exact reference values</summary>'+_table([
            {'function': function, 'reference_value': value}
            for function, value in summary['truth'].items()])+'</details>'
    page += '<h2>Coordinate estimation</h2><img src="figures/symmetric_t_coordinate.png" alt="Coordinate mean MSE and bulk ESS for every case">'
    page += '<p>Symmetry gives E[asinh(x₁)] = 0. Coordinate mixing and radial estimation can favour different methods. ESS is calculated on the retained traces and can exceed their nominal length.</p>'
    page += '<details><summary>Every production functional: MSE, bulk ESS and rank R-hat</summary>'+_table(metric_rows)+'</details>'
    page += '<h2>Proposal settings and rejection diagnostics</h2>'+_table(case_rows)
    page += '<h2>Map parameters</h2>'+_table(map_rows)
    if summary.get('maps'):
        saved_maps = {'maps': summary['maps'], 'fit_metadata': summary.get('fit_metadata', {})}
        page += '<details><summary>Map parameters and fitting metadata (full precision in summary.json)</summary><pre>'+html.escape(cell(saved_maps))+'</pre></details>'
    page += ('<h2>Paired MSE differences and uncertainty</h2><p>These descriptive paired-chain bootstrap intervals are conditional on the map fits and selected steps, '
             'with no multiplicity correction. Differences are MSE(case A) minus MSE(case B); negative values favour A. '
             'An interval containing zero does not resolve a ranking. Map-fitting and pilot-selection uncertainty are not included.</p>')
    page += _table(contrasts)
    if pilot_rows:
        page += '<h2>Every independent proposal-scale pilot</h2><img src="figures/symmetric_t_proposal_scales.png" alt="Every pilot scale, ESS score and acceptance">'
        page += ('<p>Pilot selection uses separate seeds from the production evaluation. Every tested candidate appears below, including its MSE and each functional diagnostic. '
                 'Pilot results diagnose step-size sensitivity; production MSE does not select a winner.</p>')
        if pilot.get('selected_radial'):
            page += ('<p>The radial-focused criterion maximizes the smaller bulk ESS per retained value for the tail-probability indicator P(‖X‖ ≥ 50) and the mean-norm observable ‖X‖. '
                     'The five-function criterion maximizes the minimum over log(1 + ‖X‖), the radial 95th-percentile indicator, asinh(x₁), asinh(x₃) and asinh(x₄). '
                     'Both choices use the same independent pilot traces, and each choice has a separately labelled production case. '
                     'This keeps the paper’s radial objectives visible alongside the broader mixing criterion.</p>')
        page += ('<p class="warning">Zero or unavailable ESS can arise from constant or poorly explored traces. '
                 'Low ESS or large rank R-hat limits what a short pilot can establish. A selected grid endpoint requires a wider search before claiming optimal tuning. '
                 'Equal numerical proposal scales have different meanings in the two algorithms.</p>')
        page += _table(pilot_rows)
        page += '<details><summary>Every pilot functional: exact MSE, bulk ESS and rank R-hat</summary>'+_table(pilot_metric_rows)+'</details>'
    page += '<h2>Numerical downloads</h2><ul>'
    for filename, label in [('symmetric_t_values.csv', 'All production MSE and diagnostics'),
                            ('symmetric_t_case_settings.csv', 'Exact sample counts and step sizes'),
                            ('symmetric_t_pilot_values.csv', 'Every pilot candidate and functional'),
                            ('symmetric_t_uncertainty.csv', 'Paired MSE differences and intervals'),
                            ('symmetric_t_all_iterations.csv', 'MSE using every production iteration'),
                            ('symmetric_t_all_iteration_uncertainty.csv', 'All-iteration MSE differences and intervals')]:
        if (root/filename).exists():
            page += '<li><a href="'+filename+'">'+html.escape(label)+'</a></li>'
    page += '</ul><p>Diagnostic cases can share a process pool. Their recorded times do not establish a throughput ranking.</p></html>'
    (root/'paper_values.html').write_text(page, encoding='utf-8')
