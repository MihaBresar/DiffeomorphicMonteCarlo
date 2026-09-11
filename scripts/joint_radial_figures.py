"""Figures and full-precision tables for held-out joint R/step calibration.

Selection is supplied by the runner. This module only profiles the declared
pilot scores for display; it never changes a chosen production case.
"""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
import re
import textwrap

import numpy as np
from dcs_bench.display import number, cell, figure_ticks

from scripts.skew_t_timed_figures import render_report as render_timed_report


ALGORITHMS = ('DCS-BW', 'SCS')
COLORS = {'DCS-BW': '#6f4bb0', 'SCS': '#d13d77'}
CONTROL_MARKERS = {'joint': '*', 'unit': 's', 'vi': 'D', 'sqrt-d-plus-one': '^'}
CONTROL_LABELS = {'joint': 'Joint pilot choice', 'unit': 'R = 1',
                  'vi': 'VI-fitted R', 'sqrt-d-plus-one': 'R = √(d + 1)'}
BOUNDARY_COUNTS = ('initial_boundary_crossings', 'stepout_calls', 'final_infeasible',
                   'nonfinite_density', 'density_rejections', 'accepted', 'accepted_after_stepout')
BOUNDARY_NOTE = (
    'Boundary counts come from separate fixed-count probes, outside all pilot and production clocks. '
    'Each rate divides its event count by that case’s total probe iterations. '
    'Initial crossings, step-out calls and accepted-after-step-out events overlap with final outcomes; '
    'their rates must not be added as a partition. An initial SCS cap crossing can be corrected by '
    'step-out and accepted. Final infeasible proposals, nonfinite-density proposals, finite-density '
    'Metropolis rejections and accepted proposals describe the final outcomes.')


def _plain(value):
    """Keep stored precision while giving missing/nonfinite values explicit blanks."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cell(value):
    value = _plain(value)
    if value is None:
        return ''
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _csv(path, rows, fallback):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)


def _table(rows):
    if not rows:
        return '<p>No values are available.</p>'
    columns = list(dict.fromkeys(key for row in rows for key in row))
    head = ''.join('<th>'+html.escape(key.replace('_', ' '))+'</th>' for key in columns)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(cell(row.get(key), key))+'</td>'
                                for key in columns)+'</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def _target_label(target_id, target):
    spec = target['target']
    supplied = target.get('target_label')
    if supplied:
        return str(supplied)
    alpha = spec.get('alpha', {})
    values = list(alpha.values()) if isinstance(alpha, dict) else np.asarray(alpha).ravel()
    symmetric = not any(float(value) != 0 for value in values)
    shape = 'symmetric' if symmetric else 'skewed'
    return f"{spec['dimension']}D {shape} Student-t, ν = {spec['nu']:g}"


def _score(row):
    value = row.get('score')
    if not row.get('valid_selection', False) or value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _matches(row, case):
    return (row.get('algorithm') == case.get('algorithm') and
            row.get('R') == case.get('R') and row.get('scale') == case.get('scale'))


def _candidate_rows(target_id, target):
    rows = []
    for candidate in target.get('pilots', {}).get('grid', []):
        row = {'target': target_id}
        row.update({key: _plain(value) for key, value in candidate.items()
                    if key not in ('bulk_ess', 'rank_rhat')})
        for metric, value in candidate.get('bulk_ess', {}).items():
            row['bulk_ess_'+metric] = _plain(value)
        for metric, value in candidate.get('rank_rhat', {}).items():
            row['rank_rhat_'+metric] = _plain(value)
        matched = [(name, case) for name, case in target.get('cases', {}).items() if _matches(candidate, case)]
        row['selected_production_cases'] = [name for name, _ in matched]
        row['selected_controls'] = [case.get('control', name) for name, case in matched]
        selected = target.get('pilots', {}).get('selected', {})
        row['selected_step_endpoint_cases'] = [name for name, _ in matched
                                               if selected.get(name, {}).get('step_endpoint')]
        row['selected_radius_endpoint_cases'] = [name for name, _ in matched
                                                 if selected.get(name, {}).get('radius_endpoint')]
        row['eligible_score'] = _score(candidate)
        rows.append(row)
    return rows


def _selection_rows(target_id, target):
    rows = []
    for name, selection in target.get('pilots', {}).get('selected', {}).items():
        row = {'target': target_id, 'case': name}
        row.update({key: _plain(value) for key, value in selection.items()
                    if key not in ('bulk_ess', 'rank_rhat')})
        for metric, value in selection.get('bulk_ess', {}).items():
            row['bulk_ess_'+metric] = _plain(value)
        for metric, value in selection.get('rank_rhat', {}).items():
            row['rank_rhat_'+metric] = _plain(value)
        rows.append(row)
    return rows


def _profile_rows(target_id, target):
    grid = target.get('pilots', {}).get('grid', [])
    profiles = []
    for algorithm in ALGORITHMS:
        radii = sorted({float(row['R']) for row in grid if row.get('algorithm') == algorithm})
        for radius in radii:
            candidates = [row for row in grid if row.get('algorithm') == algorithm and
                          float(row['R']) == radius and _score(row) is not None]
            # Display ties deterministically; selection itself remains the runner's decision.
            best = min(candidates, key=lambda row: (-_score(row), float(row['scale']))) if candidates else None
            profiles.append({'target': target_id, 'algorithm': algorithm, 'R': radius,
                             'valid_candidates': len(candidates),
                             'best_tested_scale': best['scale'] if best else None,
                             'best_eligible_score': _score(best) if best else None})
    return profiles


def _boundary_rows(target_id, target):
    rows = []
    for name, probe in target.get('boundary', {}).items():
        case = target.get('cases', {}).get(name, {})
        iterations = probe.get('iterations')
        if iterations is not None and (int(iterations) != iterations or iterations < 0):
            raise ValueError('Boundary probe iterations must be a nonnegative integer')
        row = {'target': target_id, 'case': name, 'label': case.get('label', name),
               'algorithm': case.get('algorithm'), 'R': case.get('R'), 'scale': case.get('scale'),
               'probe_iterations': iterations}
        for key in BOUNDARY_COUNTS:
            count = probe.get(key)
            if count is not None and (int(count) != count or count < 0):
                raise ValueError('Boundary event counts must be nonnegative integers')
            row[key] = count
            row[key+'_rate'] = count/iterations if count is not None and iterations else None
        rows.append(row)
    return rows


def render_report(root, summary):
    """Write target subreports, complete pilot CSVs, calibration plots and an index."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    targets = summary.get('targets', {})
    if not targets:
        raise ValueError('The joint calibration report requires completed targets')
    raw_config = summary.get('config', {})
    config = raw_config if isinstance(raw_config, dict) else {'preset': raw_config}
    smoke = config.get('preset') == 'quick'
    all_candidates, all_profiles, all_boundaries, all_cases, all_selections = [], [], [], [], []
    target_reports, figure_files, target_sections = {}, [], []

    def save(fig, name):
        figure_ticks(fig)
        if smoke:
            fig.text(.5, .012, 'Short execution check only — not evidence of sampler performance',
                     ha='center', fontsize=10, color='#8c510a')
        for suffix in ('.png', '.pdf'):
            fig.savefig(figures/(name+suffix), dpi=180, bbox_inches='tight')
        plt.close(fig)
        figure_files.append(name)
        return ('<figure><a href="figures/'+name+'.pdf"><img src="figures/'+name+'.png" alt="'+
                html.escape(name.replace('_', ' '))+'"></a><figcaption>Click for PDF; exact values are in the CSV tables.</figcaption></figure>')

    for target_id, target in targets.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', target_id):
            raise ValueError('Target IDs must be simple directory names')
        label = _target_label(target_id, target)
        child = dict(target)
        child['config'] = config | target.get('config', {})
        child['target_label'] = label
        child['pilots'] = {'grid': [], 'selected': {}}
        child['notes'] = list(target.get('notes', []))+[
            'R and proposal step are chosen on independent equal-time radial pilots; all production maps and steps are frozen.',
            'The complete two-dimensional pilot grid and separately measured boundary events are in the parent report.',
        ]
        target_reports[target_id] = render_timed_report(root/target_id, child)
        candidates = _candidate_rows(target_id, target)
        profiles = _profile_rows(target_id, target)
        boundaries = _boundary_rows(target_id, target)
        all_candidates.extend(candidates)
        all_profiles.extend(profiles)
        all_boundaries.extend(boundaries)
        selections = _selection_rows(target_id, target)
        all_selections.extend(selections)
        for name, case in target.get('cases', {}).items():
            selection = target.get('pilots', {}).get('selected', {}).get(name, {})
            row = {'target': target_id, 'case': name, 'label': case.get('label', name),
                   'algorithm': case['algorithm'], 'control': case.get('control'),
                   'R': case.get('R'), 'scale': case.get('scale'),
                   'step_endpoint': selection.get('step_endpoint'),
                   'radius_endpoint': selection.get('radius_endpoint'),
                   'chains': len(case.get('records', [])), 'total_iterations': case.get('total_iterations'),
                   'mean_iterations': case.get('mean_iterations'),
                   'mean_sampling_seconds': case.get('mean_sampling_seconds')}
            for metric, value in case.get('empirical_mse', {}).items():
                row[metric+'_empirical_mse'] = value
            row['norm_error_interpretation'] = ('Descriptive finite-run squared errors; norm variance is infinite'
                                               if target['target']['nu'] <= 2 else 'Empirical MSE across chains')
            all_cases.append(row)
        sections = '<section><h2>'+html.escape(label)+'</h2><p><a href="'+target_id+'/paper_values.html">Open estimator plots, exact sample counts and every production chain</a></p>'
        grid = target.get('pilots', {}).get('grid', [])
        if grid:
            fig, axes = plt.subplots(1, 2, figsize=(15, 8.6))
            positive = [_score(row) for row in grid if _score(row) is not None and _score(row) > 0]
            zero_present = any(_score(row) == 0 for row in grid)
            norm = (LogNorm(vmin=min(positive), vmax=max(positive) if max(positive) > min(positive) else min(positive)*1.01)
                    if positive and not zero_present else Normalize(vmin=0, vmax=max(positive, default=1)))
            for ax, algorithm in zip(axes, ALGORITHMS):
                rows = [row for row in grid if row.get('algorithm') == algorithm]
                radii = sorted({float(row['R']) for row in rows})
                scales = sorted({float(row['scale']) for row in rows})
                if not radii or not scales:
                    ax.text(.5, .5, 'No candidates recorded', transform=ax.transAxes, ha='center')
                    ax.set_axis_off()
                    continue
                values = np.full((len(radii), len(scales)), np.nan)
                for row in rows:
                    value = _score(row)
                    if value is not None:
                        values[radii.index(float(row['R'])), scales.index(float(row['scale']))] = value
                cmap = plt.get_cmap('viridis').copy()
                cmap.set_bad('#dce1e8')
                matrix = ax.imshow(np.ma.masked_invalid(values), origin='lower', aspect='auto', cmap=cmap, norm=norm)
                for yi, xi in zip(*np.where(~np.isfinite(values))):
                    ax.text(xi, yi, '×', color='#6f7886', ha='center', va='center')
                for case in target.get('cases', {}).values():
                    if case.get('algorithm') != algorithm or case.get('R') not in radii or case.get('scale') not in scales:
                        continue
                    control = case.get('control', 'joint')
                    ax.scatter(scales.index(case['scale']), radii.index(case['R']),
                               marker=CONTROL_MARKERS.get(control, 'o'), s=180 if control == 'joint' else 100,
                               facecolors='none' if control != 'joint' else '#ffdb6e',
                               edgecolors='white' if control != 'joint' else '#17283a', linewidths=1.5)
                ax.set_xticks(range(len(scales)), [number(value) for value in scales], rotation=35, ha='right')
                ax.set_yticks(range(len(radii)), [number(value) for value in radii])
                ax.set_xlabel('Proposal step: '+('ball radius' if algorithm == 'DCS-BW' else 'tangent coordinate SD'))
                ax.set_ylabel('Map scale R')
                ax.set_title(algorithm)
                fig.colorbar(matrix, ax=ax, label='Eligible minimum radial bulk ESS / second', shrink=.84)
            legend = [Line2D([], [], color='#233344', marker=CONTROL_MARKERS[key], linestyle='none',
                             markerfacecolor='#ffdb6e' if key == 'joint' else 'none', markersize=10,
                             label=value) for key, value in CONTROL_LABELS.items()]
            fig.legend(handles=legend, loc='lower center', ncol=4, bbox_to_anchor=(.5, .065))
            fig.suptitle(('EXECUTION CHECK — ' if smoke else '')+label+' — independent joint calibration', fontsize=14)
            fig.text(.5, .04, 'Larger score is better. Grey × cells are ineligible or absent. All grid scores, validity flags and exact parameters are saved.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .15, 1, .94))
            sections += save(fig, 'joint_radial_heatmap_'+target_id)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharey=True)
            profile_scores = [row['best_eligible_score'] for row in profiles
                              if row['best_eligible_score'] is not None]
            profile_scores += [_score(row) for row in grid if _score(row) is not None and
                               any(_matches(row, case) for case in target.get('cases', {}).values())]
            if profile_scores and min(profile_scores) > 0:
                axes[0].set_yscale('log')
            for ax, algorithm in zip(axes, ALGORITHMS):
                rows = [row for row in profiles if row['algorithm'] == algorithm]
                good = [row for row in rows if row['best_eligible_score'] is not None]
                ax.plot([row['R'] for row in good], [row['best_eligible_score'] for row in good],
                        marker='o', color=COLORS[algorithm], linewidth=1.5, label='Best eligible tested step at each R')
                for case in target.get('cases', {}).values():
                    if case.get('algorithm') != algorithm:
                        continue
                    candidate = next((row for row in grid if _matches(row, case)), None)
                    if candidate is None or _score(candidate) is None:
                        continue
                    control = case.get('control', 'joint')
                    ax.scatter(case['R'], _score(candidate), marker=CONTROL_MARKERS.get(control, 'o'),
                               s=150 if control == 'joint' else 75, color=COLORS[algorithm],
                               edgecolors='#233344', label=CONTROL_LABELS.get(control, control), zorder=4)
                ax.set_xscale('log')
                ax.set_xlabel('R (logarithmic axis)')
                ax.set_ylabel('Best eligible minimum radial bulk ESS / second')
                ax.set_title(algorithm)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            fig.suptitle(label+' — choose the step separately at every R', fontsize=14)
            fig.text(.5, .055, 'Pilot scores are selection diagnostics. Held-out production results evaluate the chosen settings. Grid-endpoint winners are only best among tested values.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .11, 1, .93))
            sections += save(fig, 'joint_radial_profile_'+target_id)

        if boundaries:
            shown = ('initial_boundary_crossings', 'stepout_calls', 'final_infeasible',
                     'nonfinite_density', 'density_rejections', 'accepted', 'accepted_after_stepout')
            display = ('Initial boundary\ncrossing', 'Step-out\ncalled', 'Final\ninfeasible',
                       'Nonfinite\ndensity', 'Density\nrejection', 'Accepted', 'Accepted after\nstep-out')
            values = np.asarray([[row[key+'_rate'] if row[key+'_rate'] is not None else np.nan
                                  for key in shown] for row in boundaries])
            fig, ax = plt.subplots(figsize=(14, max(6, .8*len(boundaries)+2)))
            cmap = plt.get_cmap('Blues').copy()
            cmap.set_bad('#dce1e8')
            matrix = ax.imshow(np.ma.masked_invalid(values), aspect='auto', vmin=0,
                               vmax=max(1, float(np.nanmax(values))) if np.any(np.isfinite(values)) else 1,
                               cmap=cmap)
            for yi, row in enumerate(boundaries):
                for xi, key in enumerate(shown):
                    count = row[key]
                    rate = row[key+'_rate']
                    text = 'unavailable' if rate is None else f'{100*rate:.2f}%\n{count}/{row["probe_iterations"]}'
                    ax.text(xi, yi, text, ha='center', va='center', fontsize=8,
                            color='white' if rate is not None and rate > .6 else '#183047')
            ax.set_xticks(range(len(shown)), display, fontsize=9)
            ax.set_yticks(range(len(boundaries)), ['\n'.join(textwrap.wrap(row['label'], 28)) for row in boundaries], fontsize=9)
            fig.colorbar(matrix, ax=ax, label='Event count / probe iterations', shrink=.8)
            ax.set_title(label+' — boundary and acceptance events in separate probes', fontsize=13, pad=16)
            fig.text(.5, .045, 'Exact numerators and denominators are shown. Initial crossings, step-outs and accepted-after-step-out events overlap with final outcomes.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .12, 1, .97))
            sections += save(fig, 'joint_radial_boundary_'+target_id)
        sections += '<h3>Selected production settings and errors</h3>'+_table([row for row in all_cases if row['target'] == target_id])
        sections += '<h3>Declared pilot selections and endpoint flags</h3>'+_table(selections)
        sections += '<h3>Profiled pilot scores at every R</h3>'+_table(profiles)
        sections += '<h3>Separate boundary probes</h3><p>'+html.escape(BOUNDARY_NOTE)+'</p>'+_table(boundaries)+'</section>'
        target_sections.append(sections)

    csvs = {'joint_radial_candidates.csv': (all_candidates, ('target', 'algorithm', 'R', 'scale', 'score')),
            'joint_radial_selections.csv': (all_selections, ('target', 'case', 'R', 'scale', 'step_endpoint', 'radius_endpoint')),
            'joint_radial_profiles.csv': (all_profiles, ('target', 'algorithm', 'R', 'best_tested_scale', 'best_eligible_score')),
            'joint_radial_cases.csv': (all_cases, ('target', 'case', 'R', 'scale')),
            'joint_radial_boundary.csv': (all_boundaries, ('target', 'case', 'probe_iterations'))}
    for name, (rows, fallback) in csvs.items():
        _csv(root/name, rows, fallback)
    notes = [
        'Both samplers receive equal pilot grid sizes and sampling-time budgets. The score is the minimum rank-based bulk ESS per second for the bounded tail indicator and log norm. This is a finite-variance radial proxy, not direct optimization of raw-norm MSE.',
        'Every control receives proposal pretuning at its own R. Symmetric centres and deformations are zero. Skewed centres and deformations are fitted once, then held fixed while R and step size are calibrated.',
        'R and proposal steps remain fixed during held-out production. Production seeds are separate from pilot seeds. The production comparison uses fresh equal-time runs, not rescaling an iteration-budget experiment.',
        'Only valid candidates can contribute to the profile. The profile is a display of declared pilot scores, not a new selection. Boundary and endpoint selections must be interpreted against the complete tested grid.',
        'At ν = 2, the raw norm has infinite variance. Its observed average squared error is descriptive; no ordinary finite-variance norm-MSE confidence interval is claimed.',
        'Reported production errors and tail-indicator intervals are conditional on the pilot selections and fitted maps. A single tuning run does not establish optimizer robustness.',
        BOUNDARY_NOTE,
        'Production sampling time excludes map fitting, pilots, discarded warmup and the separate boundary probes. Exact realized costs and sample counts are recorded. Timing results depend on hardware, worker count and competing load.',
    ]+list(summary.get('notes', []))
    warning = '<p class="warning">Quick execution check only — do not interpret sampler rankings.</p>' if smoke else ''
    report = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint R and step calibration</title><style>
body{font:15px/1.55 system-ui,sans-serif;color:#203147;background:#f6f8fb;max-width:1500px;margin:auto;padding:28px}h1,h2{line-height:1.2}h2{margin-top:36px}a{color:#4e46a5}figure{margin:24px 0;background:white;padding:12px;border-radius:9px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#576478}.table-wrap{overflow:auto;background:white;border:1px solid #dde3ed;border-radius:6px}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid #e4e8ef;white-space:nowrap;text-align:left}th{background:#eaf0f8}td{font-family:ui-monospace,monospace}.warning{background:#fff1ce;padding:12px;border-left:4px solid #a57316}li{margin-bottom:6px}summary{cursor:pointer}</style></head><body>'''
    report += '<h1>DCS/SCS: joint R and proposal-step calibration</h1>'+warning
    report += '<p>Independent radial pilots choose settings; fresh equal-time production measures their performance. Every proposal step is pretuned.</p>'
    report += '<p>'+' · '.join('<a href="'+name+'">'+name+'</a>' for name in csvs)+'</p>'
    report += ''.join(target_sections)
    report += '<h2>Complete independent pilot grid</h2><details><summary>Show every candidate, score, validity flag and selected control</summary>'+_table(all_candidates)+'</details>'
    report += '<h2>How to interpret this comparison</h2><ul>'+''.join('<li>'+html.escape(str(note))+'</li>' for note in notes)+'</ul>'
    report += '<p>Plots and tables use two decimals, with scientific notation for small values. Boundary and sample counts remain exact integers. CSV and JSON retain full numerical precision.</p></body></html>'
    (root/'paper_values.html').write_text(report, encoding='utf-8')
    return {'figures': figure_files, 'csv_files': list(csvs), 'html': 'paper_values.html',
            'target_reports': target_reports}
