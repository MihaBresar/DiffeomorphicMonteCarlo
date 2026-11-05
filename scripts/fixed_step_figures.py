"""Figures and full-precision tables for the separate fixed-step tuning audit."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from dcs_bench.display import number, cell, figure_ticks


def case_label(name, author_scs=False):
    source, rest = name.split('-', 1)
    algorithm = 'DCS-BW' if rest.startswith('DCS-BW-') else 'SCS'
    policy = rest[len(algorithm)+1:]
    fit = {'current': 'Current fit', 'original': 'Original fit', 'vanilla': 'Vanilla'}[source]
    if algorithm == 'SCS' and author_scs:
        fit = 'Author SCS fit'
    tuning = {'raw_frozen': 'warmup, frozen', 'smoothed_frozen': 'smoothed warmup, frozen',
              'smoothed_continuing': 'continued adaptation', 'fixed_0.1': 'fixed h = 0.1',
              'fixed_pilot': 'independent pilot'}[policy]
    return f'{fit} · {algorithm}\n{tuning}'


def render_report(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    figures = root/'figures'; figures.mkdir(exist_ok=True)
    config = summary['config']; cases = summary['cases']
    smoke = config.get('preset') == 'quick'
    title = ('EXECUTION CHECK — ' if smoke else '')+'100D fixed-step tuning audit'
    colors = {'DCS-BW': '#9370DB', 'SCS': '#FF1493', 'vanilla': '#4682B4'}
    def color(name):
        return colors['vanilla'] if name.startswith('vanilla') else colors['DCS-BW' if 'DCS-BW' in name else 'SCS']
    def save(fig, name):
        figure_ticks(fig)
        fig.savefig(figures/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    selected = [name for name in ('current-DCS-BW-fixed_pilot', 'current-SCS-fixed_pilot',
                                  'vanilla-DCS-BW-fixed_pilot') if name in cases]
    if not selected:
        selected = ['current-DCS-BW-raw_frozen', 'current-SCS-raw_frozen', 'vanilla-DCS-BW-raw_frozen']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, ax in enumerate(axes):
        values = [cases[name]['mse'][i] for name in selected]
        bars = ax.bar(range(len(selected)), values, color=[color(name) for name in selected])
        ax.set_xticks(range(len(selected)), [case_label(name, bool(config.get('scs_vi'))) for name in selected], fontsize=8)
        ax.set_title(['Tail-probability MSE', 'Mean-norm MSE'][i]); ax.set_ylabel('MSE')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.bar_label(bars, labels=[number(value, scientific=True) for value in values], fontsize=8, padding=5, rotation=8)
        ax.margins(y=.3)
    fig.suptitle(title+'\n'+str(config['runs'])+' chains × '+str(config['iterations'])+' production iterations; '+str(config['iterations']//config['thin'])+' retained per chain', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .90)); save(fig, 'fixed_step_comparison')
    names = list(cases)
    fig, axes = plt.subplots(1, 2, figsize=(17, max(6, .47*len(names))))
    for i, ax in enumerate(axes):
        values = [cases[name]['mse'][i] for name in names]
        bars = ax.barh(range(len(names)), values, color=[color(name) for name in names])
        ax.set_yticks(range(len(names)), [case_label(name, bool(config.get('scs_vi'))) for name in names], fontsize=8)
        ax.invert_yaxis(); ax.set_xscale('log'); ax.set_title(['Tail-probability MSE', 'Mean-norm MSE'][i])
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y()+bar.get_height()/2, ' '+number(value, scientific=True), va='center', fontsize=7)
        positive = [v for v in values if v > 0]
        if positive: ax.set_xlim(min(positive)/2, max(positive)*30)
    fig.suptitle(title+' — all map/tuning cases', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96)); save(fig, 'map_tuning_bridge')
    rows = []
    for name, case in cases.items():
        rows.append({'case': name, 'chains': case['chains'], 'iterations_per_chain': config['iterations'],
                     'retained_draws_per_chain': config['iterations']//config['thin'],
                     'mse_tail_probability': case['mse'][0], 'mse_mean_norm': case['mse'][1],
                     'mean_acceptance': case['acceptance_mean'],
                     'mean_frozen_or_final_scale': float(np.mean(case['final_scales']))})
    with (root/'audit_values.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    uncertainty_rows = []
    for contrast in summary.get('paired_mse_differences', []):
        interval = contrast['bootstrap_95_percentile']
        for index, functional in enumerate(('tail_probability', 'mean_norm')):
            low, high = interval[0][index], interval[1][index]
            uncertainty_rows.append({'case_a': contrast['a'], 'case_b': contrast['b'],
                'functional': functional, 'mse_difference_a_minus_b': contrast['a_minus_b'][index],
                'bootstrap_95_low': low, 'bootstrap_95_high': high,
                'interval_includes_zero': low <= 0 <= high})
    if uncertainty_rows:
        with (root/'audit_uncertainty.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(uncertainty_rows[0]))
            writer.writeheader(); writer.writerows(uncertainty_rows)
    pilot_rows, pilot_function_rows = [], []
    pilot_path = root/'pilots/summary.json'
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        grid = pilot['grid']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in dict.fromkeys(row['case'] for row in grid):
            data = [row for row in grid if row['case'] == name]
            axes[0].plot([r['scale'] for r in data], [r['score'] for r in data], 'o-', label=name)
            axes[1].plot([r['scale'] for r in data], [r['mean_acceptance'] for r in data], 'o-', label=name)
        for ax in axes:ax.set_xscale('log');ax.set_xlabel('Proposal scale (ball radius or tangent coordinate SD)');ax.legend(fontsize=8)
        axes[0].set_ylabel('Minimum bulk ESS / retained draws');axes[0].set_title('Independent pilot selection')
        axes[1].set_ylabel('Acceptance rate');axes[1].set_title('Acceptance is a diagnostic, not the selection target')
        fig.suptitle(title);fig.tight_layout();save(fig, 'proposal_scale_pilots')
        fields = ['case', 'scale', 'score', 'mean_acceptance', 'mean_outside_ball_fraction',
                  'mean_cap_stepout_fraction', 'selected', 'selected_endpoint', 'invalid_functions']
        functions = list(dict.fromkeys(name for row in grid for name in row['bulk_ess']))
        fields += [field for name in functions for field in ('bulk_ess_'+name, 'rank_rhat_'+name)]
        for row in grid:
            selection = pilot.get('selected', {}).get(row['case'], {})
            selected_row = row['scale'] == selection.get('scale')
            result = {key: row[key] for key in fields[:6]}
            result.update(selected=selected_row,
                          selected_endpoint=selected_row and selection.get('endpoint_winner', False),
                          invalid_functions=', '.join(row.get('invalid_functions', [])))
            for name in functions:
                ess, rhat = row['bulk_ess'].get(name), row['rank_rhat'].get(name)
                result['bulk_ess_'+name], result['rank_rhat_'+name] = ess, rhat
                pilot_function_rows.append({'case': row['case'], 'scale': row['scale'],
                    'functional': name, 'bulk_ess': ess, 'rank_rhat': rhat,
                    'unavailable_or_constant': name in row.get('invalid_functions', []),
                    'selected': selected_row})
            pilot_rows.append(result)
        with (root/'pilot_values.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(pilot_rows)

    def table(values, columns=None):
        if not values:
            return '<p>No values are available.</p>'
        columns = columns or list(values[0])
        header = ''.join('<th>'+html.escape(key)+'</th>' for key in columns)
        body = ''.join('<tr>'+''.join('<td>'+html.escape(cell(row[key], key))+'</td>' for key in columns)+'</tr>' for row in values)
        return '<div class="table-wrap"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'

    notes = 'All cases share the same target and physical starting point. Pilot choices use separate seeds. Continued adaptation is a historical sensitivity check. These diagnostic cases share a process pool, so their recorded times are not a throughput comparison. Different observables can favour different methods.'
    page = '<!doctype html><meta charset="utf-8"><title>Fixed-step audit values</title><style>body{font:16px system-ui;margin:32px;color:#172940}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #ccd5df;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef3f8}p{max-width:1050px;line-height:1.6}.table-wrap{overflow-x:auto;margin:20px 0}h2{margin-top:36px}summary{cursor:pointer;font-weight:600;padding:12px 0}.warning{padding:12px 16px;border-left:4px solid #b76b11;background:#fff7e8}</style>'
    page += '<h1>'+html.escape(title)+'</h1><p>'+html.escape(notes)+'</p>'
    page += '<p>Plots and tables use two decimals, with scientific notation for MSE and small values. Integer counts remain exact; CSV and JSON retain full precision.</p>'
    page += '<h2>Exact MSE values for every case</h2>'+table(rows)
    uncertainty_note = 'Descriptive paired-chain bootstrap intervals, conditional on the fitted maps, with no multiplicity correction. Differences are MSE(case A) minus MSE(case B); negative values favour A. An interval containing zero does not resolve the ranking. These intervals do not measure uncertainty from fitting the maps.'
    page += '<h2>Paired MSE differences and uncertainty</h2><p>'+html.escape(uncertainty_note)+'</p>'+table(uncertainty_rows)
    if pilot_rows:
        pilot_note = 'Every tested candidate is listed below. Selection uses the minimum bulk ESS per recorded value across five predeclared functions, using independent pilot seeds and shared exact-target initial draws. This is an oracle-assisted synthetic diagnostic. Scale values mean a ball proposal radius for DCS-BW and a tangent coordinate standard deviation for SCS.'
        warning = 'A zero score indicates an unavailable or constant-function diagnostic; it does not mean zero estimation error. Low ESS or large rank R-hat means the short pilot may not adequately explore that function. A selected grid endpoint calls for a wider pilot search before claiming optimal tuning.'
        if smoke:
            warning = 'QUICK CHECK: these short pilots test execution only and cannot support sampler rankings. '+warning
        page += '<h2>All proposal-scale pilot candidates</h2><p>'+html.escape(pilot_note)+'</p><p class="warning">'+html.escape(warning)+'</p>'
        candidate_columns = ['case', 'scale', 'score', 'mean_acceptance', 'mean_outside_ball_fraction',
                             'mean_cap_stepout_fraction', 'selected', 'selected_endpoint', 'invalid_functions']
        page += table(pilot_rows, candidate_columns)
        page += '<details open><summary>Bulk ESS and rank R-hat for every candidate and function</summary>'+table(pilot_function_rows)+'</details>'
    (root/'paper_values.html').write_text(page, encoding='utf-8')
