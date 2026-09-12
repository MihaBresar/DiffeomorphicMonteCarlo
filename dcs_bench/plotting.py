"""Original paper figures and supplementary diagnostics from saved chains."""
import csv
import html
import json
from fractions import Fraction
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .targets import Target
from .processing import process_chains
from .radial_reporting import compact_radial_report, process_radial_chains
from .cpu import cpu_info
from .display import number, cell, figure_ticks

COLORS = ['#9370DB', '#FF8C00', '#228B22', '#FF1493', '#4682B4']
METHOD_COLORS = {'DCS-BW': '#9370DB', 'DCS-RWM': '#FF8C00', 'DCS-HnR': '#9370DB',
                 'SCS': '#FF1493', 'RWM-Gaussian': '#4682B4', 'RWM-stable': '#228B22', 'NUTS': '#FF8C00'}
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
    folder.mkdir(parents=True, exist_ok=True)
    if quick:
        title = fig._suptitle.get_text()+'\n' if fig._suptitle else ''
        fig.suptitle(title+'QUICK CHECK — not publication results', fontsize=14)
    figure_ticks(fig)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/f'{name}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def burnin_counts(records):
    """Bookkeeping for recorded sampling burn-in; absent for historical runs."""
    if not records or not all('burnin_iterations' in row or 'discarded_iterations' in row for row in records):
        return {}
    burns = [int(row.get('burnin_iterations', row.get('discarded_iterations'))) for row in records]
    used = [int(row.get('estimation_iterations', row['iterations']-burn)) for row, burn in zip(records, burns)]
    if any(burn < 0 or count <= 0 or burn+count != row['iterations'] for row, burn, count in zip(records, burns, used)):
        raise ValueError('Sampling burn-in and estimation counts must sum to generated iterations')
    return {'total_burnin_iterations': sum(burns), 'mean_burnin_iterations': sum(burns)/len(burns),
            'total_estimation_iterations': sum(used), 'mean_estimation_iterations': sum(used)/len(used)}


def ball_map_description(config):
    if config.get('targets') and all(t['kind'] in ('garch', 'eight_schools') for t in config['targets']):
        return 'posterior identity map'
    if config.get('ball_scale_policy') == 'sqrt-d':
        return 'historical fixed R=√d'
    if config.get('ball_map_policy') == 'vanilla':
        return 'vanilla: R=1, centre=0, Möbius=0'
    return 'Möbius VI'


def method_labels(algorithms, config):
    description = ball_map_description(config)
    map_note = ', R=√d' if description == 'historical fixed R=√d' else \
        ', vanilla' if config.get('ball_map_policy') == 'vanilla' and description != 'posterior identity map' else \
        ', Möbius VI' if config.get('ball_map_policy') == 'mobius' and description != 'posterior identity map' else ''
    return [f'{a} (β={config["beta"]:g}{map_note})' if a.startswith('DCS') else
            f'SCS (l₀={config["latitude"]:.2f})' if a == 'SCS' else
            'RWM Stable (α=0.5)' if a == 'RWM-stable' else a.replace('-', ' ')
            for a in algorithms]


def bar_panel(ax, algorithms, heights, config, title, ylabel):
    """Original linear bars with readable labels and full-precision CSV data."""
    values = [np.nan if h is None else h for h in heights]
    bars = ax.bar(np.arange(len(algorithms)), values,
                  color=[METHOD_COLORS[a] for a in algorithms], alpha=.8, edgecolor='black')
    ax.set(title=title, ylabel=ylabel)
    ax.set_xticks(np.arange(len(algorithms)), method_labels(algorithms, config), rotation=15, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=.6)
    ax.set_axisbelow(True)
    finite = [v for v in values if np.isfinite(v)]
    ax.set_ylim(0, max(finite)*1.18 if finite and max(finite) > 0 else 1)
    for bar, value in zip(bars, heights):
        ax.annotate(number(value, scientific='MSE' in title,
                           integer='Iterations' in ylabel or 'Draws' in ylabel),
                    (bar.get_x()+bar.get_width()/2, value or 0),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=8)


def radial_boxplots(target, ref, data, config, labels):
    algorithms = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    burn_note = '\nAfter step tuning, recorded sampling burn-in and thinning' if any(burnin_counts(data[a]['records']) for a in algorithms) else ''
    fig.suptitle(f'Ergodic Averages across {config["runs"]} Chains (d={target.d}, ν={target.nu:.1f})'+burn_note,
                 fontsize=16, fontweight='bold')
    for k, ax in enumerate(axes):
        boxes = ax.boxplot([data[a]['estimates'][:, k] for a in algorithms], patch_artist=True,
                           tick_labels=method_labels(algorithms, config), showmeans=True, meanline=True)
        for patch, a in zip(boxes['boxes'], algorithms):
            patch.set_facecolor(METHOD_COLORS[a])
            patch.set_alpha(.6)
        ax.axhline(ref['functional_truth'][k], color='red', ls='--', lw=2,
                   label=f'Reference = {number(ref["functional_truth"][k])}')
        ax.set_title(labels[ref['functional_names'][k]], fontsize=14)
        ax.set_ylabel('Independent-chain ergodic average')
        ax.grid(axis='y', ls=':', alpha=.7)
        ax.tick_params(axis='x', rotation=15)
        ax.legend(fontsize=9)
    return fig


def qq_reference(target, ref, config):
    """Recover the original four QQ coordinates without rerunning MCMC.

    Coordinates with zero skew coefficient have exactly Student-t marginals.
    Existing saved quantiles remain authoritative for the first two coordinates.
    A nonzero skew coefficient in an unsaved coordinate needs deterministic IID
    reference draws only, using the recorded seed and batching.
    """
    from scipy.stats import t
    from .references import skew_t_iid
    dims = min(4, target.d)
    saved = np.asarray(ref['quantiles'])
    result = np.empty((len(config['q_levels']), dims))
    keep = min(saved.shape[1], dims)
    result[:, :keep] = saved[:, :keep]
    sources = ['saved independent reference']*keep
    missing = list(range(keep, dims))
    skewed = [dim for dim in missing if target.alpha[dim] != 0]
    if skewed:
        rng = np.random.default_rng(ref['seed'])
        values, remaining = [], ref['draws']
        while remaining:
            n = min(remaining, config['reference_batch_size'])
            values.append(skew_t_iid(target, rng, n)[:, skewed])
            remaining -= n
        qs = np.quantile(np.concatenate(values), config['q_levels'], axis=0)
    for dim in missing:
        result[:, dim] = qs[:, skewed.index(dim)] if dim in skewed else t.ppf(config['q_levels'], target.nu)
        sources.append('replayed independent IID reference' if dim in skewed else 'analytic Student-t marginal')
    return result, sources


def skew_figures(target, ref, data, config, labels, folder, quick):
    name, algorithms = config['figure'], list(data)
    if not compact_radial_report(config):
        skew_qq_figure(target, ref, data, config, folder, quick)
    is_nuts = config['experiment'] == 'skew-t-200d-nuts'
    save_figure(radial_boxplots(target, ref, data, config, labels), folder,
                name if is_nuts else name+'_ergodic_averages', quick)
    third = ('total_coordinate_bulk_ess' if is_nuts else 'mean_sampling_seconds'
             if config['mode'] == 'iterations' else 'mean_iterations')
    ncols = 2 if config['experiment'] == 'skew-t-200d-baselines' else 3
    fig, axes = plt.subplots(1, ncols, figsize=(16 if ncols == 2 else 18, 6))
    title = ('Performance & Mean Squared Error Comparison' if config['mode'] == 'iterations'
             else f'Sampling Efficiency & MSE ({config["time_budget"]:.1f}s Budget)')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for k, ax in enumerate(axes[:2]):
        bar_panel(ax, algorithms, [data[a]['summary']['mse'][k] for a in algorithms], config,
                  f'MSE of {labels[ref["functional_names"][k]]}', 'Mean Squared Error')
    if ncols == 3:
        heading, units = {'total_coordinate_bulk_ess': ('Total Effective Sample Size (ESS)', 'Sum of chain mean bulk ESS, first 10 coordinates'),
                          'mean_sampling_seconds': ('Average Runtime (s)', 'Production Seconds'),
                          'mean_iterations': ('Average Iterations Generated', 'Production Iterations / Chain')}[third]
        bar_panel(axes[2], algorithms, [data[a]['summary'][third] for a in algorithms], config, heading, units)
    save_figure(fig, folder, name+'_mse_and_ess' if is_nuts else name, quick)

    # Explicitly distinguish generated iterations from retained (thinned) draws.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    burn_note = '\nRetained draws follow recorded sampling burn-in and thinning' if any(burnin_counts(data[a]['records']) for a in algorithms) else ''
    fig.suptitle('Sampling Output and Runtime'+burn_note, fontsize=16, fontweight='bold')
    for ax, metric, heading, units in zip(axes,
            ['mean_iterations', 'mean_retained_draws', 'mean_sampling_seconds'],
            ['Average Iterations Generated', 'Average Samples Retained', 'Average Runtime (s)'],
            ['Production Iterations / Chain', 'Retained Draws / Chain', 'Production Seconds']):
        bar_panel(ax, algorithms, [data[a]['summary'][metric] for a in algorithms], config, heading, units)
    save_figure(fig, folder, name+'_sample_counts', quick)


def skew_qq_figure(target, ref, data, config, folder, quick):
    name, algorithms = config['figure'], list(data)
    reference, sources = qq_reference(target, ref, config)
    # Historical summaries contain only the two QQ coordinates then plotted.
    # Display those cached values instead of inventing unrecorded quantiles.
    available = min(reference.shape[1], *(data[a]['quantiles'].shape[2] for a in algorithms))
    reference, sources = reference[:, :available], sources[:available]
    fig, axes = plt.subplots(1, reference.shape[1], figsize=(20, 5), squeeze=False)
    fig.suptitle(f'Marginal Q–Q Diagnostics: Skew-t (d={target.d}, ν={target.nu:g})', fontsize=16)
    qq_rows = []
    for dim, ax in enumerate(axes[0]):
        truth = reference[:, dim]
        ax.plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'k--', alpha=.5, label='y = x')
        for a in algorithms:
            low, median, high = np.quantile(data[a]['quantiles'][:, :, dim], [.05, .5, .95], axis=0)
            ax.plot(truth, median, 'o-', ms=4, color=METHOD_COLORS[a], label=a, alpha=.8)
            ax.fill_between(truth, low, high, color=METHOD_COLORS[a], alpha=.15)
            for q, x, y, lo, hi in zip(config['q_levels'], truth, median, low, high):
                qq_rows.append(dict(coordinate=dim+1, algorithm=a, probability=q,
                    reference=x, reference_source=sources[dim], median=y, across_chains_05=lo, across_chains_95=hi))
        ax.set(title=f'$x_{dim+1}$', xlabel='Independent Reference Quantiles', ylabel='Estimated Quantiles')
        ax.grid(ls=':', alpha=.6)
        if dim == 0:
            ax.legend(fontsize=9)
    save_figure(fig, folder, name+'_marginal_qq', quick)
    write_csv(folder/(name+'_marginal_qq.csv'), qq_rows)


def export_paper_values(output, config, all_summary):
    """Full-precision CSV values and a table with readable numerical labels."""
    rows, count_rows = [], []
    for s, scenario in enumerate(all_summary):
        ref = scenario['reference']
        for a, summary in scenario['algorithms'].items():
            counts = dict(experiment=config['experiment'], preset=config['preset'], scenario=s,
                ball_map_policy=config.get('ball_map_policy', 'historical-fixed-scale' if config.get('ball_scale_policy') == 'sqrt-d' else 'mobius'),
                ball_map_description=ball_map_description(config),
                ball_scale_policy=config.get('ball_scale_policy', 'fit'),
                algorithm=a, chains=config['runs'], total_iterations=summary['total_iterations'],
                mean_iterations=summary['mean_iterations'], mean_iterations_exact=str(Fraction(summary['total_iterations'], config['runs'])),
                total_retained_samples=summary['total_retained_draws'], mean_retained_samples=summary['mean_retained_draws'],
                mean_retained_samples_exact=str(Fraction(summary['total_retained_draws'], config['runs'])),
                thin=config['thin'], mean_production_seconds=summary['mean_sampling_seconds'],
                total_coordinate_bulk_ess=summary.get('total_coordinate_bulk_ess'))
            counts.update(burnin_counts(summary['records']))
            if compact_radial_report(config):
                counts.pop('total_coordinate_bulk_ess', None)
            count_rows.append(counts)
            for k, functional in enumerate(ref['functional_names']):
                rows.append(dict(counts, functional=functional, reference=ref['functional_truth'][k],
                    reference_mcse=ref['functional_reference_mcse'][k], mse=summary['mse'][k], rmse=summary['rmse'][k],
                    mse_bootstrap_95_low=summary['mse_bootstrap_95_interval'][0][k],
                    mse_bootstrap_95_high=summary['mse_bootstrap_95_interval'][1][k]))
                if compact_radial_report(config):
                    rows[-1]['estimate'] = float(np.mean(np.asarray(summary['chain_functional_estimates'])[:, k]))
    write_csv(output/'paper_values.csv', rows)
    write_csv(output/'sample_counts.csv', count_rows)
    # Display formatting is separate from the full-precision CSV exports.
    columns = ['scenario', 'algorithm', 'functional', 'mse', 'reference', 'mean_iterations',
               'mean_retained_samples', 'total_iterations', 'total_retained_samples', 'mean_production_seconds']
    if compact_radial_report(config):
        columns.insert(3, 'estimate')
    if any('total_burnin_iterations' in row for row in rows):
        columns += ['total_burnin_iterations', 'total_estimation_iterations', 'mean_burnin_iterations', 'mean_estimation_iterations']
    burn_note = ('Separate step tuning precedes sampling. Recorded sampling burn-in is included in generated counts and excluded from estimation counts. Retained samples follow tuning, sampling burn-in and thinning. '
                 if any('total_burnin_iterations' in row for row in rows) else 'Retained samples are after warmup and thinning. ')
    content = ''.join('<tr>'+''.join('<td>'+html.escape(cell(row.get(c), c))+'</td>'
                                  for c in columns)+'</tr>' for row in rows)
    page = ('<!doctype html><meta charset="utf-8"><title>Exact plotted values</title>'
            '<style>body{font:15px system-ui;margin:32px;color:#253047}table{border-collapse:collapse}'
            'th,td{padding:9px;border:1px solid #ccd4e0;text-align:left;white-space:nowrap}th{background:#eef1f7}'
            '.scroll{overflow:auto}td{font-variant-numeric:tabular-nums}</style>'
            f'<h1>{html.escape(config["experiment"])}: exact plotted values</h1>'
            f'<p>Run size: <strong>{html.escape(config["preset"])}</strong>. '
            f'DCS map: <strong>{html.escape(ball_map_description(config))}</strong>. '
            'MSE is the average of independent-chain squared errors. Iterations include repeats after rejection; '
            +burn_note+'These are finite numerical estimates, '
            'displayed with two decimals, using scientific notation for small values. CSV and JSON retain full precision.</p>'
            +('<p>Estimates are averages of independent-chain estimates. This report contains only the norm moment, '
              'tail probability, MSE, sample counts and timing. No ESS, R-hat, coordinate quantiles or directional diagnostics are computed.</p>'
              if compact_radial_report(config) else '')+
            '<p><a href="paper_values.csv">Download all values (CSV)</a> · '
            '<a href="sample_counts.csv">Sample counts and exact mean fractions (CSV)</a></p>'
            '<div class="scroll"><table><thead><tr>'+''.join('<th>'+c.replace('_', ' ')+'</th>' for c in columns)+
            '</tr></thead><tbody>'+content+'</tbody></table></div>')
    (output/'paper_values.html').write_text(page, encoding='utf-8')


def rebuild(output):
    from .engine import write_json
    started = perf_counter()
    output = Path(output)
    config = json.loads((output/'configuration.json').read_text())
    radial_only = compact_radial_report(config)
    labels = dict(LABELS)
    labels['tail_probability'] = rf'$P(\|X\|\geq {config["threshold"]:g})$'
    manifest = json.loads((output/'manifest.json').read_text())
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    all_summary, csv_rows, all_data = [], [], []
    print('Sampling is complete. Collecting moment and tail estimates, MSE, sample counts and timing.'
          if radial_only else 'Sampling is complete. Processing saved chains and convergence diagnostics.', flush=True)
    for s, spec in enumerate(config['targets']):
        target = Target(spec)
        directory = output/f'scenario-{s}'
        ref = json.loads((directory/'reference.json').read_text())
        if radial_only:
            # Historical references also contain optional directional/QQ values.
            # Keep their first two radial values without rewriting the reference.
            ref = dict(ref, **{key: ref[key][:2] for key in
                       ('functional_names', 'functional_truth', 'functional_reference_mcse')})
        setup = json.loads((directory/'setup.json').read_text())
        summary = {'target': spec, 'reference': ref, 'algorithms': {}}
        data = {}
        for algorithm in config['algorithms']:
            paths = [directory/'chains'/f'{algorithm}-{i:03d}.npz' for i in range(config['runs'])]
            arguments = dict(expected=manifest['fingerprint'], label=f'Scenario {s}, {algorithm}',
                             progress=lambda message: print(message, flush=True))
            if radial_only:
                processed = process_radial_chains(paths, target, config['threshold'], **arguments)
            else:
                processed = process_chains(paths, target, config['threshold'], config['q_levels'],
                    ref['functional_names'], temporary_parent=output,
                    diagnostic_workers=min(4, config.get('workers', 1), cpu_info()['capacity']), **arguments)
            estimates, quantiles, records = (processed[key] for key in ('estimates', 'quantiles', 'records'))
            length, diagnostics, indiv = (processed[key] for key in ('length', 'diagnostics', 'indiv'))
            errors = mse_summary(estimates, ref['functional_truth'])
            family = 'sphere' if algorithm == 'SCS' else 'ball'
            setup_seconds = setup.get(family, {}).get('optimization', {}).get('seconds', 0.) if algorithm == 'SCS' or algorithm.startswith('DCS') else 0.
            compilation_path = directory/(('nuts' if algorithm == 'NUTS' else algorithm)+'_compilation.json')
            if compilation_path.exists():
                setup_seconds += json.loads(compilation_path.read_text())['seconds']
            totals = [r['sampling_seconds']+r.get('warmup_seconds', r.get('warmup_seconds_including_compilation', 0.))
                      +r.get('production_compile_and_pilot_seconds', 0.)+setup_seconds/config['runs'] for r in records]
            algo_summary = dict(errors, chain_functional_estimates=estimates.tolist(),
                chain_quantiles=quantiles.tolist(), cross_chain_diagnostics=diagnostics,
                result_processing=processed['processing'],
                diagnostic_draws_per_chain=length, independent_chain_diagnostics=indiv,
                mean_sampling_seconds=float(np.mean([r['sampling_seconds'] for r in records])),
                mean_seconds_including_setup=float(np.mean(totals)),
                setup_cost_charging='Shared fit and summed worker compilation durations amortized over this method\'s runs; not total elapsed wall time',
                records=list(records))
            if radial_only:
                for key in ('cross_chain_diagnostics', 'diagnostic_draws_per_chain', 'independent_chain_diagnostics'):
                    algo_summary.pop(key)
            algo_summary.update(burnin_counts(records))
            for field in ('iterations', 'retained_draws'):
                total = sum(int(r[field]) for r in records)
                algo_summary['total_'+field] = total
                algo_summary['mean_'+field] = total/len(records)
            if target.kind == 'skew_t' and not radial_only:
                coord_names = target.names[:10]
                coordinate_ess = processed['coordinate_ess']
                values = [v for entry in coordinate_ess for v in entry.values()]
                algo_summary['chain_coordinate_bulk_ess'] = coordinate_ess
                algo_summary['total_coordinate_bulk_ess'] = (float(np.sum(values)/len(coord_names))
                    if all(v is not None for v in values) else None)
                algo_summary['coordinate_ess_definition'] = 'Sum over chains of the mean bulk ESS of the first min(10,d) coordinates; unavailable if any coordinate ESS is unavailable'
            if target.kind in ('garch', 'eight_schools'):
                algo_summary['first_moment_rmse_across_chains_and_coordinates'] = float(np.sqrt(np.mean((estimates[:, :target.d]-ref['functional_truth'][:target.d])**2)))
                algo_summary['second_moment_rmse_across_chains_and_coordinates'] = float(np.sqrt(np.mean((estimates[:, target.d:]-ref['functional_truth'][target.d:])**2)))
            summary['algorithms'][algorithm] = algo_summary
            data[algorithm] = {'estimates': estimates, 'quantiles': quantiles, 'records': records, 'summary': algo_summary}
            for i, (est, record) in enumerate(zip(estimates, records)):
                for k, name in enumerate(ref['functional_names']):
                    csv_rows.append(dict(scenario=s, algorithm=algorithm, chain=i, seed=record['seed'], functional=name,
                        estimate=est[k], reference=ref['functional_truth'][k], squared_error=(est[k]-ref['functional_truth'][k])**2,
                        sampling_seconds=record['sampling_seconds'], seconds_including_setup=totals[i],
                        iterations=record['iterations'], retained_draws=record['retained_draws']))
                    if not radial_only:
                        diag = indiv[i]
                        ess = diag['ess_bulk'][name]
                        csv_rows[-1].update(mcse=diag['mcse_mean'][name], ess_bulk=ess, ess_tail=diag['ess_tail'][name],
                            ess_per_sampling_second=None if ess is None else ess/record['sampling_seconds'])
                    if 'burnin_iterations' in record or 'discarded_iterations' in record:
                        burn = record.get('burnin_iterations', record.get('discarded_iterations'))
                        csv_rows[-1].update(burnin_iterations=burn,
                                            estimation_iterations=record.get('estimation_iterations', record['iterations']-burn))
        all_summary.append(summary)
        all_data.append((target, ref, data))
    write_json(output/'summary.json', {'schema_version': 2, 'preset': config['preset'], 'scenarios': all_summary,
        'interpretation': 'MSE uses independent chain estimates. PosteriorDB and logistic references have Monte Carlo uncertainty. Quick results test execution only.'})
    with (output/'chain_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in csv_rows for key in row)))
        writer.writeheader()
        writer.writerows(csv_rows)
    diagnostic_seconds = perf_counter()-started
    print('Estimates collected. Saving tables and figures.' if radial_only else
          'Convergence diagnostics complete. Saving tables and figures.', flush=True)
    rendered = _render_prepared(output, config, all_summary, all_data, labels)
    write_json(output/'processing.json', dict(
        schema_version=1, statistics_seconds=diagnostic_seconds,
        tables_and_figures_seconds=perf_counter()-started-diagnostic_seconds,
        total_seconds=perf_counter()-started,
        sampling_clock_includes_result_processing=False,
        diagnostics_computed=not radial_only,
        diagnostic_subsampling=False))
    print(f'Result processing finished in {perf_counter()-started:.2f} seconds.', flush=True)
    return rendered


def render_saved_summary(output):
    """Refresh display only, using cached statistics without reading chains."""
    output = Path(output)
    config = json.loads((output/'configuration.json').read_text())
    radial_only = compact_radial_report(config)
    saved = json.loads((output/'summary.json').read_text())
    all_summary, all_data = saved['scenarios'], []
    labels = dict(LABELS)
    labels['tail_probability'] = rf'$P(\|X\|\geq {config["threshold"]:g})$'
    for scenario in all_summary:
        target, reference = Target(scenario['target']), scenario['reference']
        reference.setdefault('functional_reference_mcse', [None]*len(reference['functional_names']))
        if radial_only:
            for key in ('functional_names', 'functional_truth', 'functional_reference_mcse'):
                reference[key] = reference[key][:2]
        for result in scenario['algorithms'].values():
            # Older summaries saved exact per-chain counts but not totals.
            # Recover only these bookkeeping aggregates; never recalculate ESS.
            for field in ('iterations', 'retained_draws'):
                total = sum(int(record[field]) for record in result['records'])
                result.setdefault('total_'+field, total)
                result.setdefault('mean_'+field, total/len(result['records']))
            if radial_only:
                for key in ('mse', 'rmse'):
                    result[key] = result[key][:2]
                for key in ('chain_functional_estimates', 'mse_bootstrap_95_interval'):
                    result[key] = [row[:2] for row in result[key]]
            else:
                result.setdefault('total_coordinate_bulk_ess', None)
            result.update(burnin_counts(result['records']))
        data = {algorithm: {'estimates': np.asarray(result['chain_functional_estimates']),
                            'quantiles': np.asarray(result['chain_quantiles']),
                            'records': result['records'], 'summary': result}
                for algorithm, result in scenario['algorithms'].items()}
        all_data.append((target, reference, data))
    return _render_prepared(output, config, all_summary, all_data, labels)


def _render_prepared(output, config, all_summary, all_data, labels):
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    quick = config['preset'] == 'quick'
    export_paper_values(output, config, all_summary)
    if config['experiment'] == 'logistic-regression':
        fig, axes = plt.subplots(len(all_data), len(config['algorithms']), figsize=(12, 10), squeeze=False)
        fig.suptitle(r'Q–Q Plot of $\beta_1$ in Robust Logistic Regression'+
                     ('\n'+ball_map_description(config) if config.get('ball_scale_policy') == 'sqrt-d'
                      or config.get('ball_map_policy') == 'vanilla' else ''), fontsize=16)
        qq_rows = []
        for row, (target, ref, data) in enumerate(all_data):
            truth = np.array(ref['quantiles'])[:, 0]
            for col, (algorithm, values) in enumerate(data.items()):
                ax = axes[row, col]
                qs = values['quantiles'][:, :, 0]
                low, median, high = np.quantile(qs, [.05, .5, .95], axis=0)
                ax.plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'k-.', alpha=.5, label='y = x')
                ax.fill_between(truth, low, high, color=METHOD_COLORS[algorithm], alpha=.15, label='5–95% across chains')
                ax.plot(truth, median, 'o-', color=METHOD_COLORS[algorithm], ms=4, label=algorithm)
                ax.set_title(f'{algorithm}, ν={target.nu:g}')
                ax.grid(ls=':', alpha=.6)
                if row == len(all_data)-1:
                    ax.set_xlabel('Independent Reference Quantiles')
                if col == 0:
                    ax.set_ylabel('Sampler Quantiles')
                for q, x, y, lo, hi in zip(config['q_levels'], truth, median, low, high):
                    qq_rows.append(dict(prior_nu=target.nu, algorithm=algorithm, probability=q,
                        reference=x, median=y, across_chains_05=lo, across_chains_95=hi))
        axes[0, 0].legend(fontsize=8)
        write_csv(folder/(config['figure']+'.csv'), qq_rows)
        save_figure(fig, folder, config['figure'], quick)
    else:
        target, ref, data = all_data[0]
        algorithms = list(data)
        if target.kind == 'skew_t':
            skew_figures(target, ref, data, config, labels, folder, quick)
            if compact_radial_report(config):
                return output/'summary.json'
            # Retain the newer overview as an explicitly supplementary plot.
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.suptitle('Supplementary: MSE uncertainty and sampling output')
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
                rows.append(f'{a} & {summ["first_moment_rmse_across_chains_and_coordinates"]:.2e} & {summ["second_moment_rmse_across_chains_and_coordinates"]:.2e} \\\\')
            (output/'posterior_table.tex').write_text('% RMSE across independent chains and physical coordinates; reference is estimated.\n'+'\n'.join(rows)+'\n')
        save_figure(fig, folder/'supplementary' if target.kind == 'skew_t' else folder,
                    config['figure']+'_mse_uncertainty' if target.kind == 'skew_t' else config['figure'], quick)
        if target.kind == 'skew_t' and min(data[a]['estimates'].shape[1] for a in algorithms) >= 4 and len(ref['functional_truth']) >= 4:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            fig.suptitle('Supplementary directional checks: positive-side probability and bounded projection mean')
            for k, ax in enumerate(axes, 2):
                ax.boxplot([data[a]['estimates'][:, k] for a in algorithms], tick_labels=algorithms, showmeans=True)
                ax.axhline(ref['functional_truth'][k], color='#B2182B', ls='--')
                mcse = ref['functional_reference_mcse'][k]
                if mcse:
                    ax.axhspan(ref['functional_truth'][k]-1.96*mcse, ref['functional_truth'][k]+1.96*mcse, color='#B2182B', alpha=.1)
                ax.set_title(ref['functional_names'][k].replace('_', ' '))
                ax.tick_params(axis='x', rotation=20)
            save_figure(fig, folder/'supplementary', config['figure']+'_directional', quick)
            # Retire only the old generated location, so stale diagnostics do
            # not remain mixed in with the restored original figure collection.
            for suffix in ('png', 'pdf'):
                (folder/(config['figure']+'_directional.'+suffix)).unlink(missing_ok=True)
    return output/'summary.json'
