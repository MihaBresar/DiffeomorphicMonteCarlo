"""Original-style sampler comparison figures and exact numerical tables.

Rendering never selects methods, changes tuning, or substitutes thinned
estimates for the recorded all-iteration chain means.
"""
from __future__ import annotations

import csv
import html
import json
from numbers import Real
from pathlib import Path

import numpy as np
from dcs_bench.display import number, cell, figure_ticks
from dcs_bench.simulation_error import mse_variability, paired_mse_differences

METRICS = ('tail_probability', 'mean_norm')
X1_METRICS = ('mean_x1', 'x1_tail_probability')
X23_METRICS = ('mean_x2', 'x2_left_tail_probability', 'mean_x3', 'x3_tail_probability')
ALIASES = (('tail_probability', 'tail'), ('mean_norm', 'norm'))
METHODS = ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'SCS', 'SCS-calibrated', 'DCS-BW-rotate', 'DCS-RWM-rotate', 'DCS-Billiard')
SPHERE_METHODS = ('SCS', 'SCS-calibrated')
ROTATION_METHODS = ('DCS-BW-rotate', 'DCS-RWM-rotate')
COLORS = {'DCS-RWM': '#FF8C00', 'DCS-BW': '#9370db', 'DCS-BW-reflect': '#159a8c',
          'SCS': '#ff1493', 'SCS-calibrated': '#3267b1', 'DCS-BW-rotate': '#2878b5', 'DCS-RWM-rotate': '#b69a27',
          'DCS-Billiard': '#ba433c'}


def _display_name(name):
    return 'SCS calibrated' if name == 'SCS-calibrated' else name


def _metrics(value, metric_names=METRICS):
    if isinstance(value, dict):
        aliases = dict(zip(METRICS, ALIASES))
        value = [next(value[key] for key in aliases.get(name, (name,)) if key in value)
                 for name in metric_names]
    result = np.asarray(value, dtype=float)
    if result.shape != (len(metric_names),) or not np.all(np.isfinite(result)):
        raise ValueError('Each metric vector must contain one finite value per functional')
    return result


def _pair(value):
    return _metrics(value)


def _means(case, metric_names=METRICS):
    values = case['chain_means']
    if isinstance(values, dict):
        aliases = dict(zip(METRICS, ALIASES))
        values = np.column_stack([next(values[key] for key in aliases.get(name, (name,)) if key in values)
                                  for name in metric_names])
    result = np.asarray(values, dtype=float)
    if result.shape != (case['chains'], len(metric_names)) or not np.all(np.isfinite(result)):
        raise ValueError('chain_means must contain one finite value per functional and chain')
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
    header = ''.join('<th>'+html.escape('sum of per-chain ESS' if key == 'total_ess' else key.replace('_', ' '))+'</th>' for key in fields)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(cell(_display_name(row.get(key)) if key == 'method' else row.get(key), key))+'</td>' for key in fields)+'</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def _policy(name, timed=False, matched=False, ball_esjd=False, own_kernel=False, tuning_policies=None):
    selected = (tuning_policies or {}).get(name)
    if selected == 'fixed_dcs_move_v1':
        return ('Prespecified fixed DCS move: BW, reflected-BW and billiard proposal radius h; '
                'RWM Gaussian coordinate sigma=h/sqrt(d+2); '
                'separate discarded equilibration, no step adaptation')
    if selected == 'author_scs_fixed_01':
        return 'Author skew-t setting: fixed h = 0.1 with separate discarded equilibration; no step adaptation'
    if selected == 'fixed_scs_move_v1':
        return ('Independent joint radius/step pilot selected R and h; original author centre, observer and transition; '
                'frozen step with separate discarded equilibration, no adaptation')
    if selected == 'indicator_dual_averaging_0234':
        return 'Acceptance-target 0.234 dual averaging; selected step frozen before production'
    if selected == 'own_kernel_radial_pilot_v1' or (selected is None and own_kernel and name in ('DCS-Billiard', 'SCS')):
        return 'Own-production-kernel radial pilots; bounded squared physical log(1+norm) movement, equal pilot budget and frozen production step'
    if name == 'DCS-Billiard':
        return 'Reflected-BW ball-movement pilots in post-Mobius coordinates; frozen step used by optimized invariant-chord billiard production'
    if name == 'DCS-RWM-rotate':
        return '50/50 reflected Gaussian RWM and fixed-radius rotation; Gaussian coordinate standard deviation and angular scale pretuned within one shared warmup budget, then frozen'
    if name == 'DCS-BW-rotate':
        return '50/50 reflected BW and fixed-radius rotation; both proposal scales pretuned within one shared warmup budget, then frozen'
    if ball_esjd and name == 'DCS-BW-reflect':
        return 'Ball-movement pilot selection by mean squared jumping distance; selected step frozen during timed sampling'
    if matched:
        return 'Shared dual-averaging step calibration toward 0.234, then frozen during timed sampling'
    if timed:
        return ('Author SCS: fixed h = 0.10 with separate untimed equilibration' if name == 'SCS' else
                'Separate proposal-step tuning, then frozen during timed sampling')
    return 'Paper-date SCS: fixed h = 0.1' if name == 'SCS' else 'Continuous smoothed dual averaging; target acceptance 0.234'


def _validate_v11_dcs(config, cases):
    """Never relabel saved acceptance-tuned chains as fixed-move comparisons."""
    methods = set(cases).intersection(('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'DCS-Billiard'))
    if not methods:
        return
    rule = config.get('dcs_step_rule')
    if rule not in ('fixed', 'acceptance'):
        raise ValueError('Version-11 reports require the recorded DCS step rule')
    historical_policies = {'DCS-RWM': 'indicator_dual_averaging_0234',
                          'DCS-BW': 'indicator_dual_averaging_0234',
                          'DCS-BW-reflect': 'ball_esjd_pilot_v1',
                          'DCS-Billiard': 'own_kernel_radial_pilot_v1'}
    policies = config.get('step_tuning_policies', {})
    radius = config.get('dcs_fixed_move')
    if rule == 'fixed' and (isinstance(radius, bool) or not isinstance(radius, Real)
                            or not np.isfinite(radius) or radius <= 0.):
        raise ValueError('Version-11 fixed DCS reports require a positive finite move radius')
    dimension = config['target']['dimension']
    for method in methods:
        expected_policy = 'fixed_dcs_move_v1' if rule == 'fixed' else historical_policies[method]
        if policies.get(method) != expected_policy:
            raise ValueError('Version-11 DCS step policies disagree with the recorded rule')
        records = cases[method].get('records', [])
        if not records:
            raise ValueError('Version-11 reports require actual DCS step-policy records')
        scale = (radius / np.sqrt(dimension+2.) if method == 'DCS-RWM' else radius) if rule == 'fixed' else None
        for record in records:
            if record.get('tuning_policy') != expected_policy:
                raise ValueError('Version-11 reports require actual DCS step-policy records')
            if rule != 'fixed':
                if method in ('DCS-RWM', 'DCS-BW') and record.get('target_acceptance') != .234:
                    raise ValueError('Version-11 acceptance controls require recorded target 0.234')
                continue
            required = {'fixed_move_radius': radius, 'fixed_step': scale,
                        'initial_scale': scale, 'final_scale': scale, 'frozen_scale': scale,
                        'proposal_coordinate_variance': radius**2/(dimension+2.)}
            if (record.get('step_size_adapted') is not False
                    or record.get('adaptation_during_sampling') is not False
                    or 'target_acceptance' not in record or record['target_acceptance'] is not None
                    or record.get('warmup_iterations') != config['warmup']
                    or record.get('tuning_history') or record.get('radial_tuning_history')):
                raise ValueError('Version-11 fixed DCS reports require discarded warmup without adaptation')
            for field, value in required.items():
                actual = record.get(field)
                if (isinstance(actual, bool) or not isinstance(actual, Real)
                        or not np.isfinite(actual) or not np.isclose(actual, value, rtol=1e-12, atol=0.)):
                    raise ValueError('Version-11 fixed DCS scale provenance is inconsistent: '+field)
        if rule == 'fixed' and not np.allclose(cases[method]['final_scales'], scale, rtol=1e-12, atol=0.):
            raise ValueError('Version-11 fixed DCS final scales disagree with their records')


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


def _validate_calibrated_scs(config, cases):
    """Require actual fixed-step records for the independently calibrated control."""
    method = 'SCS-calibrated'
    scale = config['scs_calibrated_move']
    records = cases[method].get('records', [])
    if not records or config['step_tuning_policies'].get(method) != 'fixed_scs_move_v1':
        raise ValueError('Calibrated SCS requires recorded independent fixed-step provenance')
    for row in records:
        if (row.get('tuning_policy') != 'fixed_scs_move_v1'
                or row.get('step_size_adapted') is not False
                or row.get('adaptation_during_sampling') is not False
                or row.get('target_acceptance', 'missing') is not None
                or row.get('warmup_iterations') != config['warmup']
                or row.get('tuning_history') or row.get('radial_tuning_history')):
            raise ValueError('Calibrated SCS requires frozen steps and nonadaptive discarded warmup')
        for key in ('initial_scale', 'final_scale', 'frozen_scale'):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not np.isclose(value, scale, rtol=1e-12, atol=0.)):
                raise ValueError('Calibrated SCS step records disagree with the selected pilot step')
    if not np.allclose(cases[method]['final_scales'], scale, rtol=1e-12, atol=0.):
        raise ValueError('Calibrated SCS final scales disagree with the selected pilot step')


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
    reported_study = config.get('protocol') in ('timed-reported-zero-centre-v1',
                                               'timed-reported-zero-centre-numba-v1',
                                               'timed-reported-zero-centre-v2',
                                               'timed-reported-zero-centre-numba-v2')
    calibrated_scs = config.get('protocol') in ('timed-reported-zero-centre-v2',
                                              'timed-reported-zero-centre-numba-v2')
    if reported_study:
        from dcs_bench.reported_study import validate_reported_summary
        validate_reported_summary(summary)
    suite = reported_study or config.get('protocol') in ('timed-sampler-comparison-v1', 'timed-numba-comparison-v1',
                                      'timed-sampler-comparison-v2', 'timed-numba-comparison-v2',
                                      'timed-sampler-comparison-v3', 'timed-numba-comparison-v3',
                                      'timed-sampler-comparison-v4', 'timed-numba-comparison-v4',
                                      'timed-sampler-comparison-v5', 'timed-numba-comparison-v5',
                                      'timed-sampler-comparison-v6', 'timed-sampler-comparison-v7', 'timed-numba-comparison-v7',
                                      'timed-sampler-comparison-v8', 'timed-numba-comparison-v8',
                                      'timed-sampler-comparison-v9', 'timed-numba-comparison-v9',
                                      'timed-sampler-comparison-v10', 'timed-numba-comparison-v10',
                                      'timed-sampler-comparison-v11', 'timed-numba-comparison-v11')
    optimized = config.get('protocol') in ('timed-reported-zero-centre-numba-v1', 'timed-reported-zero-centre-numba-v2', 'timed-numba-comparison-v1', 'timed-numba-comparison-v2',
                                          'timed-numba-comparison-v3', 'timed-numba-comparison-v4', 'timed-numba-comparison-v5', 'timed-numba-comparison-v7', 'timed-numba-comparison-v8', 'timed-numba-comparison-v9', 'timed-numba-comparison-v10', 'timed-numba-comparison-v11')
    radial_ess = reported_study or config.get('protocol') in ('timed-sampler-comparison-v7', 'timed-numba-comparison-v7',
                                            'timed-sampler-comparison-v8', 'timed-numba-comparison-v8',
                                      'timed-sampler-comparison-v9', 'timed-numba-comparison-v9',
                                      'timed-sampler-comparison-v10', 'timed-numba-comparison-v10',
                                      'timed-sampler-comparison-v11', 'timed-numba-comparison-v11')
    own_kernel = reported_study or config.get('protocol') in ('timed-sampler-comparison-v8', 'timed-numba-comparison-v8',
                                      'timed-sampler-comparison-v9', 'timed-numba-comparison-v9',
                                      'timed-sampler-comparison-v10', 'timed-numba-comparison-v10',
                                      'timed-sampler-comparison-v11', 'timed-numba-comparison-v11')
    author_vi_protocol = reported_study or config.get('protocol') in ('timed-sampler-comparison-v9', 'timed-numba-comparison-v9',
                                      'timed-sampler-comparison-v10', 'timed-numba-comparison-v10',
                                      'timed-sampler-comparison-v11', 'timed-numba-comparison-v11')
    matched_vi = reported_study or config.get('protocol') in ('timed-sampler-comparison-v10', 'timed-numba-comparison-v10',
                                      'timed-sampler-comparison-v11', 'timed-numba-comparison-v11')
    v11 = reported_study or config.get('protocol') in ('timed-sampler-comparison-v11', 'timed-numba-comparison-v11')
    author_fitting = (author_vi_protocol or config.get('sphere_map_policy') == 'author-paper-vi'
                      or config.get('scs_fitting_revision') == 'author-vi-v1'
                      or summary.get('fitting', {}).get('sphere', {}).get('map_policy') == 'SCS-paper-VI')
    tuning_policies = config.get('step_tuning_policies', {}) if own_kernel else {}
    scs_fixed = tuning_policies.get('SCS') == 'author_scs_fixed_01'
    matched_step_control = bool(config.get('calibration_control_provenance', {}).get('maps_reused_without_refitting'))
    radial_pilot_methods = {name for name in ('DCS-Billiard', 'SCS')
                            if own_kernel and (tuning_policies.get(name) == 'own_kernel_radial_pilot_v1' or
                            (name not in tuning_policies and not author_vi_protocol))}
    vi_mode = config.get('vi', 'on')
    boundary = config.get('protocol') in ('timed-boundary-dcs-scs-v1', 'timed-boundary-dcs-scs-v2', 'timed-boundary-dcs-scs-v3')
    restored_controls = config.get('protocol') == 'timed-boundary-dcs-scs-v2'
    ball_esjd = config.get('protocol') == 'timed-boundary-dcs-scs-v3' or suite
    matched = config.get('protocol') == 'timed-matched-dcs-scs-v3' or boundary or suite
    timed = config.get('mode') == 'time' or config.get('protocol') == 'timed-dcs-author-scs-v2' or matched
    if not cases or any(name not in METHODS for name in cases):
        raise ValueError('Expected completed registered DCS or SCS comparison cases')
    names = [name for name in (config.get('algorithms', METHODS) if suite else METHODS) if name in cases]
    if len(names) != len(cases) or len(set(names)) != len(names):
        raise ValueError('Configured methods must list every completed case exactly once')
    if v11:
        _validate_v11_dcs(config, cases)
    if calibrated_scs:
        _validate_calibrated_scs(config, cases)
    fixed_dcs = {name for name in names if tuning_policies.get(name) == 'fixed_dcs_move_v1'}
    if author_vi_protocol:
        if 'SCS' in names and tuning_policies.get('SCS') not in ('indicator_dual_averaging_0234', 'author_scs_fixed_01'):
            raise ValueError('Author-VI SCS reports require the recorded acceptance or fixed-author step policy')
        if ('DCS-Billiard' in names and tuning_policies.get('DCS-Billiard') != 'own_kernel_radial_pilot_v1'
                and not (v11 and 'DCS-Billiard' in fixed_dcs)):
            raise ValueError('Author-VI billiard reports require own-kernel radial calibration')
    if matched_vi:
        steps, draws = config.get('optimizer_steps'), config.get('optimizer_samples')
        if (config.get('vi') != 'on' or config.get('vi_batch_policy') != 'fixed'
                or config.get('vi_budget_policy') != 'equal_updates_and_fixed_reference_clouds_v1'
                or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
                or not isinstance(draws, int) or isinstance(draws, bool) or draws < 1
                or config.get('scs_vi') != dict(optimizer_steps=steps, optimizer_samples=draws)):
            raise ValueError('Version-10/11 reports require matched fixed-cloud VI budgets')
        for family in ('ball', 'sphere'):
            fit = summary.get('fitting', {}).get(family, {})
            if (fit.get('steps') != steps or fit.get('training_draws') != draws
                    or fit.get('batch_policy') != 'fixed' or fit.get('total_training_draws') != draws
                    or fit.get('gradient_updates') != steps or fit.get('unique_reference_draws') != draws
                    or fit.get('objective_point_evaluations') != steps*draws):
                raise ValueError('Version-10/11 reports require actual matched fixed-cloud fitting records')
    has_rotation = any(name in names for name in ROTATION_METHODS)
    has_rwm_rotation = 'DCS-RWM-rotate' in names
    has_billiard = 'DCS-Billiard' in names
    if has_billiard and 'DCS-Billiard' not in summary.get('method_maps', {}):
        raise ValueError('Billiard reports require the recorded effective post-Mobius map')
    if own_kernel:
        for method in radial_pilot_methods:
            for record in cases.get(method, {}).get('records', []):
                history = record.get('radial_tuning_history', [])
                if (record.get('calibration_algorithm') != method or
                        record.get('radial_tuning_pilot_iterations') != config['warmup'] or
                        not history or sum(row['iterations'] for row in history) != config['warmup']):
                    raise ValueError('Reports require complete own-kernel radial pilot provenance')
    counts = {name: _counts(cases[name], config, timed) for name in names}
    metric_names = tuple(summary.get('functionals', METRICS))
    if metric_names not in (METRICS, METRICS+X1_METRICS, METRICS+X1_METRICS+X23_METRICS):
        raise ValueError('Expected radial functionals followed by the optional X1 or X1/X2/X3 functionals')
    truth = _metrics(summary['truth'], metric_names)
    means = {name: _means(cases[name], metric_names) for name in names}
    mses = {name: _metrics(cases[name]['mse'], metric_names) for name in names}
    biases = {name: _metrics(cases[name]['bias_squared'], metric_names) for name in names}
    variances = {name: _metrics(cases[name]['variance_ddof0'], metric_names) for name in names}
    reference = summary.get('reference', {})
    if len(metric_names) > 2 and not all(key in reference for key in ('truth', 'reference_mcse', 'exact')):
        raise ValueError('Coordinate reports require recorded reference values, MCSE and exact flags')
    reference_mcse = _metrics(reference.get('reference_mcse', np.zeros(len(metric_names))), metric_names)
    reference_exact = reference.get('exact', [True]*len(metric_names))
    if isinstance(reference_exact, bool):
        reference_exact = [reference_exact]*len(metric_names)
    if (len(reference_exact) != len(metric_names) or
            any(not isinstance(value, (bool, np.bool_)) for value in reference_exact) or
            np.any(reference_mcse < 0)):
        raise ValueError('Reference flags and nonnegative MCSE must match the functionals')
    if reference and ('truth' in reference and not np.array_equal(_metrics(reference['truth'], metric_names), truth)):
        raise ValueError('Reference values must match the values used to compute MSE')
    if reference.get('functionals') is not None and tuple(reference['functionals']) != metric_names:
        raise ValueError('Reference functionals must match the report order')
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
        sampler_count = {1: 'one', 2: 'two', 5: 'five', 6: 'six'}.get(len(names), str(len(names)))
        sampler_label = sampler_count+(' sampler' if len(names) == 1 else ' samplers')
        title += (f' — {sampler_label}, optimized Numba DCS / author SCS' if optimized and radial_ess else
                  ' — optimized Numba DCS / author SCS' if optimized else
                  f' — {sampler_label}, JAX' if radial_ess else
                  ' — seven samplers, JAX' if has_billiard else
                  ' — six samplers, JAX' if has_rwm_rotation else
                  ' — five samplers, JAX' if has_rotation else ' — four samplers, JAX')
        title += ' — reported zero-centre study' if reported_study else ' — VI '+str(vi_mode)
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
        if has_rotation:
            tuning_note = ('VI '+str(vi_mode)+' for both maps. The rotation mixture splits its warmup between '
                           'BW and angular pilots; all sampling scales frozen.')
        if has_rwm_rotation:
            tuning_note = ('VI '+str(vi_mode)+' for both maps. Each rotation mixture splits its warmup between '
                           'reflected-step and angular pilots; all sampling scales frozen.')
        if has_billiard:
            tuning_note = ('VI '+str(vi_mode)+'. Equal pilot budgets; all steps frozen. Billiard uses '
                           'reflected-BW pilots after Mobius, with effective deformation zero.')
    if own_kernel:
        tuning_note = ('VI '+str(vi_mode)+'. Billiard and SCS tune their own kernels with equal radial-pilot budgets; '
                       'every production step stays frozen.')
    if author_vi_protocol:
        tuning_note = ('VI '+str(vi_mode)+'. SCS targets acceptance 0.234; billiard tunes its own radial moves. '
                       'Equal pilot-transition budgets; production steps frozen.')
    if scs_fixed:
        tuning_note = ('VI '+str(vi_mode)+'. SCS uses author fixed h=0.1; '+
                       ('DCS proposal scales are pretuned. ' if any(name != 'SCS' for name in names) else '')+
                       'Equal discarded warmup; production steps stay fixed.')
    if fixed_dcs:
        tuning_note = ('VI '+str(vi_mode)+'. All four DCS methods use fixed moves h='+number(config['dcs_fixed_move'])+
                       ' (RWM sigma=h/sqrt(d+2)); no step adaptation. '+
                       ('SCS uses author fixed h=0.1. ' if scs_fixed else 'SCS uses acceptance calibration. ')+
                       'Equal discarded warmup; all production steps frozen.')
    if v11 and names == ['SCS']:
        tuning_note = ('VI '+str(vi_mode)+'. SCS '+
                       ('uses author fixed h=0.1 without adaptation. ' if scs_fixed else
                        'uses acceptance-target 0.234 calibration. ')+
                       'Recorded discarded warmup; production step stays frozen.')
    if matched_vi:
        tuning_note += ' Matched fixed-cloud VI budgets for both maps.'
    if author_fitting and not author_vi_protocol:
        tuning_note = tuning_note.replace('matched map fitting and step tuning', 'method-specific map fitting and step tuning')
        tuning_note += (' SCS uses the original author VI routine.' if vi_mode == 'on'
                        else ' Both maps are untuned; VI is disabled.')
    if reported_study:
        tuning_note = ('Frozen study maps; DCS centre/deformation zero, shared h='+
                       ('0.025' if dimension == 100 else '1/√500')+
                       '; original SCS h=0.1. No refitting or step adaptation. Prior DCS calibration is additional.')
    if calibrated_scs:
        tuning_note = ('Frozen study maps; shared DCS h='+('0.025' if dimension == 100 else '1/√500')+
                       '; original SCS h=0.1; SCS calibrated h='+str(config['scs_calibrated_move'])+
                       '. DCS and calibrated SCS use prior equal-budget R/h pilots. No refitting or adaptation.')
    display_names = [_display_name(name) for name in names]
    x = np.arange(1, len(names)+1)
    colors = [COLORS[name] for name in names]
    figure_names = []

    def save(fig, name):
        if len(names) > 4:
            # Widen expanded plots; historical five-method figure bytes and
            # layouts are preserved when their supplied values are unchanged.
            fig.set_size_inches(fig.get_figwidth()*(1.2 if len(names) == 5 else 1.44 if len(names) == 6 else 1.68), fig.get_figheight())
            for ax in fig.axes:
                if len(ax.get_xticklabels()) == len(names):
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(15)
                        tick.set_horizontalalignment('right')
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
    original_labels = [_display_name(name)+(f' (β={beta:g})' if name not in SPHERE_METHODS else f' (l₀={latitude:.2f})') for name in names]
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
        reference_label = 'True value = ' if reference_exact[metric] else 'Independent reference = '
        ax.axhline(truth[metric], color='#27384a', linestyle='--', linewidth=1.4,
                   label=reference_label+number(truth[metric]))
        ax.set_xticks(x, display_names)
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
        ax.set_xticks(x, [_display_name(name)+'\n'+('Mean used N = ' if timed else 'N = ')+
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
        ax.set_xticks(x, display_names)
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
        ax.set_xticks(x, display_names)
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

    ess_rows = []
    if radial_ess:
        for name in names:
            diagnostics = cases[name].get('functional_ess')
            if not isinstance(diagnostics, dict) or any(key not in diagnostics for key in METRICS):
                raise ValueError('Current radial reports require recorded functional ESS for every method')
            for key in METRICS:
                detail = diagnostics[key]
                ess_rows.append(dict(method=name, functional=key, **detail,
                    diagnostic_seconds=cases[name].get('functional_ess_seconds')))
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for metric, label in enumerate(metric_labels):
            for row_index, (field, ylabel) in enumerate((('mean_ess', 'Mean functional ESS per chain'),
                    ('ess_per_sampling_second', 'Functional ESS per sampling second'))):
                ax = axes[row_index, metric]
                values = [cases[name]['functional_ess'][METRICS[metric]].get(field) for name in names]
                finite = [float(value) for value in values if value is not None]
                ymax = max(finite, default=1.)
                for position, value, color, name in zip(x, values, colors, names):
                    detail = cases[name]['functional_ess'][METRICS[metric]]
                    if value is None:
                        ax.text(position, ymax*.03, 'Unavailable\n'+str(detail['valid_chains'])+'/'+
                                str(detail['chains'])+' valid chains', ha='center', va='bottom', fontsize=8)
                    else:
                        ax.bar(position, value, color=color, width=.58)
                        ax.text(position, value+ymax*.025, number(value), ha='center', va='bottom', fontsize=9)
                ax.set_xticks(x, display_names)
                ax.set_xlim(x[0]-.5, x[-1]+.5)
                ax.set_title(label)
                ax.set_ylabel(ylabel)
                ax.set_ylim(0., max(1., ymax*1.25))
                ax.grid(axis='y', alpha=.2)
                ax.set_axisbelow(True)
        fig.suptitle(title+' — functional ESS and ESS per second', fontsize=14)
        fig.text(.5, .045, 'Unthinned post-burn-in functions, including repeated states. ESS/s uses full production time.\n'
                 'Any unavailable chain makes its aggregate unavailable; ESS is not a convergence guarantee.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .94))
        save(fig, 'scs_100d_ess')

    # These figures supplement the original radial figure family. They measure
    # prespecified physical coordinates, not rotations chosen after sampling.
    if len(metric_names) > 2:
        coordinate_threshold = float(config['coordinate_threshold'])
        coordinate_panels = ((2, 'Mean of the first coordinate: E[X₁]', 'coordinate_mean'),
                             (3, 'First-coordinate tail: P(X₁ > '+number(coordinate_threshold)+')',
                              'coordinate_tail_probability'))
        if len(metric_names) == 8:
            coordinate_panels += (
                (4, 'Mean of the second coordinate: E[X₂]', 'coordinate_x2_mean'),
                (5, 'Mirrored second-coordinate tail: P(X₂ < '+number(-coordinate_threshold)+')',
                 'coordinate_x2_left_tail_probability'),
                (6, 'Mean of the third coordinate: E[X₃]', 'coordinate_x3_mean'),
                (7, 'Third-coordinate tail: P(X₃ > '+number(coordinate_threshold)+')',
                 'coordinate_x3_tail_probability'))
        for metric, label, filename in coordinate_panels:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
            ax = axes[0]
            chain_values = [means[name][:, metric] for name in names]
            boxes = ax.boxplot(chain_values, patch_artist=True, showfliers=False, widths=.5)
            for box, color in zip(boxes['boxes'], colors):
                box.set_facecolor(color)
                box.set_alpha(.35)
            for position, values_, color in zip(x, chain_values, colors):
                offsets = np.linspace(-.11, .11, len(values_)) if len(values_) > 1 else np.zeros(1)
                ax.scatter(position+offsets, values_, color=color, s=23, alpha=.8, zorder=3)
            reference_label = ('True value' if reference_exact[metric] else 'Independent reference')
            reference_label += ' = '+number(truth[metric])
            if not reference_exact[metric]:
                reference_label += '; MCSE = '+number(reference_mcse[metric], scientific=True)
            ax.axhline(truth[metric], color='#27384a', linestyle='--', linewidth=1.4,
                       label=reference_label)
            ax.set_xticks(x, display_names)
            ax.set_title('Estimates from every chain')
            ax.set_ylabel('Per-chain estimate')
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.grid(axis='y', alpha=.2)
            ax = axes[1]
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
            ax.set_xticks(x, display_names)
            ax.set_title('MSE across chains' if reference_exact[metric] else 'MSE against independent reference')
            ax.set_ylabel('MSE (logarithmic scale)')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            fig.suptitle(title+'\n'+label, fontsize=14)
            fig.text(.5, .055, budget+'. '+estimation_note, ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .12, 1, .88))
            save(fig, filename)

    # Prefix means come from streaming accumulators in the run, never by
    # treating a thinned trace as all production states.
    prefix_rows, checkpoints = [], {}
    for name in names:
        records = cases[name].get('records', [])
        checkpoints[name] = [{int(row['iterations']): _metrics(row['mean_all_iterations'], metric_names)
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
                for metric, key in enumerate(metric_names):
                    prefix_rows.append({'method': name, 'iterations_per_chain': count,
                                        'chains': cases[name]['chains'], 'metric': key,
                                        'truth': truth[metric], 'mse': error[metric]})
    if len(common) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for metric, (ax, label) in enumerate(zip(axes, metric_labels)):
            for name in names:
                rows = [row for row in prefix_rows if row['method'] == name and row['metric'] == METRICS[metric]]
                values = [row['mse'] for row in rows]
                ax.plot(common, values, 'o-', color=COLORS[name], label=_display_name(name))
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
    angular_tuning_rows, angular_tuning_summaries = [], []
    radial_tuning_rows, radial_tuning_summaries = [], []
    for name in names:
        case = cases[name]
        count = counts[name]
        per_chain = count['generated'][0] if len(set(count['generated'])) == 1 else 'varies; see chain table'
        retained = (count['retained'][0] if len(set(count['retained'])) == 1 else 'varies; see chain table')
        row = {'method': name, 'production_step_policy': _policy(name, timed, matched, ball_esjd, own_kernel, tuning_policies), 'chains': case['chains'],
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
                       execution_backend='Numba' if optimized and name not in SPHERE_METHODS else 'JAX')
        if calibrated_scs and name in SPHERE_METHODS:
            row.update(sphere_radius=summary['method_maps'][name]['scale'],
                       fixed_step=.1 if name == 'SCS' else config['scs_calibrated_move'],
                       step_size_adapted=False, target_acceptance=None,
                       scale_parameter='author sphere proposal scale',
                       radius_step_preparation='original author settings' if name == 'SCS'
                           else 'independent joint radius/step pilot, 524288 transitions')
        if v11 and name in ('DCS-RWM', 'DCS-BW', 'DCS-BW-reflect', 'DCS-Billiard'):
            row.update(dcs_step_rule=config['dcs_step_rule'])
            if name in fixed_dcs:
                radius = config['dcs_fixed_move']
                row.update(dcs_fixed_move=radius, step_size_adapted=False, target_acceptance=None,
                           fixed_step=radius/np.sqrt(dimension+2.) if name == 'DCS-RWM' else radius,
                           scale_parameter='Gaussian coordinate standard deviation' if name == 'DCS-RWM'
                               else 'uniform-ball proposal radius',
                           fixed_move_normalization='BW, reflected BW and billiard radius h; RWM coordinate sigma=h/sqrt(d+2)',
                           proposal_coordinate_variance=radius**2/(dimension+2.))
            else:
                row.update(target_acceptance=.234 if name in ('DCS-RWM', 'DCS-BW') else None)
        if name in ROTATION_METHODS:
            records = case.get('records', [])
            row.update(mixture_probability=config.get('mixture_probability', .5),
                       angular_scales=[r.get('angular_scale') for r in records],
                       angular_angles_radians=[r.get('angular_angle') for r in records],
                       angular_angles_degrees=[float(np.degrees(r['angular_angle']))
                                               if r.get('angular_angle') is not None else None for r in records],
                       bw_warmup_iterations=[r.get('bw_warmup_iterations') for r in records],
                       angular_warmup_iterations=[r.get('angular_warmup_iterations') for r in records])
        if name == 'DCS-RWM-rotate':
            row.update(scale_parameter='Gaussian coordinate standard deviation',
                       gaussian_equivalent_bw_radii=[r.get('gaussian_equivalent_bw_radius') for r in records],
                       initial_scale_reference_bw_radii=[r.get('initial_scale_reference_bw_radius') for r in records])
        for metric, key in enumerate(metric_names):
            row[key+'_truth'] = truth[metric]
            if reference:
                row[key+'_reference_mcse'] = reference_mcse[metric]
                row[key+'_reference_exact'] = reference_exact[metric]
            row[key+'_mse'] = mses[name][metric]
            row[key+'_bias_squared'] = biases[name][metric]
            row[key+'_variance_ddof0'] = variances[name][metric]
        variability = mse_variability(means[name], truth)
        for metric, key in enumerate(metric_names):
            row[key+'_mse_empirical_mcse'] = variability['mse_empirical_mcse'][metric]
            row[key+'_largest_chain_error_share'] = variability['largest_chain_squared_error_share'][metric]
        if radial_ess:
            for key in METRICS:
                detail = case['functional_ess'][key]
                for field in ('total_ess', 'mean_ess', 'median_ess', 'ess_per_sampling_second',
                              'valid_chains', 'chains', 'unavailable_reasons', 'draw_strides'):
                    row[key+'_'+field] = detail.get(field)
            row['functional_ess_seconds'] = case.get('functional_ess_seconds')
        case_rows.append(row)
        records = case.get('records', [])
        if records and len(records) != case['chains']:
            raise ValueError('Each chain must have exactly one record')
        for index, estimate in enumerate(means[name]):
            row = dict(records[index]) if records else {'chain': index}
            history = row.pop('tuning_history', None)
            angular_history = row.pop('angular_tuning_history', None)
            radial_history = row.pop('radial_tuning_history', None)
            row.update(method=name, production_step_policy=_policy(name, timed, matched, ball_esjd, own_kernel, tuning_policies))
            if name in radial_pilot_methods:
                for pilot in radial_history or []:
                    radial_tuning_rows.append(dict(method=name, chain=row['chain'], seed=row.get('seed'),
                        pilot_kernel=row.get('calibration_algorithm'),
                        **{key: pilot.get(key) for key in ('stage', 'round', 'candidate', 'trial_order', 'scale', 'iterations',
                                                         'radial_score', 'acceptance', 'selected')}))
                radial_tuning_summaries.append({'method': name, 'chain': row['chain'],
                    'pilot_kernel': row.get('calibration_algorithm'),
                    'pilot_iterations': row.get('radial_tuning_pilot_iterations'),
                    'pilot_rounds': row.get('radial_tuning_rounds'),
                    'pilot_seconds': row.get('warmup_seconds'),
                    'frozen_step': row.get('frozen_scale', row.get('final_scale')),
                    'selected_radial_score': row.get('radial_tuning_selected_score'),
                    'search_status': row.get('radial_tuning_search_status')})
            elif ball_esjd and name not in fixed_dcs and name in ('DCS-BW-reflect', 'DCS-Billiard', *ROTATION_METHODS):
                for pilot in history or []:
                    tuning_rows.append(dict(method=name, chain=row['chain'], seed=row.get('seed'),
                        **{key: pilot.get(key) for key in ('round', 'candidate', 'scale', 'iterations',
                                                         'ball_esjd', 'acceptance', 'selected')}))
                tuning_summaries.append({'method': name, 'chain': row['chain'],
                    'warmup_iterations': row.get('bw_warmup_iterations', row.get('warmup_iterations', config['warmup'])),
                    'frozen_step': row.get('frozen_scale', row.get('final_scale')),
                    'selected_ball_esjd': row.get('tuning_selected_esjd'),
                    'search_status': row.get('tuning_search_status')})
                if name == 'DCS-Billiard':
                    tuning_summaries[-1].update(pilot_kernel='Reflected BW',
                        pilot_coordinates='Post-Mobius; effective delta=0')
                    for pilot_row in tuning_rows:
                        if pilot_row['method'] == name and pilot_row['chain'] == row['chain']:
                            pilot_row.update(pilot_kernel='Reflected BW',
                                pilot_coordinates='Post-Mobius; effective delta=0')
                if has_rwm_rotation:
                    parameter = ('Gaussian coordinate standard deviation' if name == 'DCS-RWM-rotate' else 'Uniform-ball proposal radius')
                    tuning_summaries[-1].update(scale_parameter=parameter,
                        gaussian_equivalent_bw_radius=row.get('gaussian_equivalent_bw_radius'),
                        initial_scale_reference_bw_radius=row.get('initial_scale_reference_bw_radius'))
                    for pilot_row in tuning_rows:
                        if pilot_row['method'] == name and pilot_row['chain'] == row['chain']:
                            pilot_row['scale_parameter'] = parameter
            if name in ROTATION_METHODS:
                for pilot in angular_history or []:
                    angular_tuning_rows.append(dict(method=name, chain=row['chain'], seed=row.get('seed'),
                        **{key: pilot.get(key) for key in ('round', 'candidate', 'scale', 'iterations',
                                                         'angular_esjd', 'acceptance', 'selected')}))
                angle = row.get('angular_angle')
                angular_tuning_summaries.append({'method': name, 'chain': row['chain'],
                    'angular_warmup_iterations': row.get('angular_warmup_iterations'),
                    'frozen_angular_scale': row.get('angular_scale'),
                    'frozen_angle_radians': angle,
                    'frozen_angle_degrees': float(np.degrees(angle)) if angle is not None else None,
                    'mixture_probability': row.get('mixture_probability', .5),
                    'selected_angular_esjd': row.get('angular_tuning_selected_esjd'),
                    'search_status': row.get('angular_tuning_search_status')})
            row.setdefault('iterations', count['generated'][index])
            row.setdefault('discarded_iterations', count['discarded'][index])
            row.setdefault('estimation_iterations', count['used'][index])
            for metric, key in enumerate(metric_names):
                row[key+'_estimate'] = estimate[metric]
                row[key+'_truth'] = truth[metric]
                row[key+'_squared_error'] = (estimate[metric]-truth[metric])**2
            chain_rows.append(row)
    paired_rows = paired_mse_differences(means, {name: cases[name].get('records', []) for name in names}, truth,
                                       functionals=metric_names)
    reference_rows = [{'metric': key, 'reference_value': truth[metric],
                       'reference_mcse': reference_mcse[metric], 'exact': reference_exact[metric],
                       'source': reference.get('source', 'Recorded reference'),
                       'reference_draws': reference.get('draws'), 'reference_seed': reference.get('seed')}
                      for metric, key in enumerate(metric_names)] if reference else []
    parameter_rows = []
    for family, parameters in summary.get('maps', {}).items():
        for key, value in parameters.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for index, coordinate in enumerate(value):
                    parameter_rows.append({'map': family, 'parameter': key, 'coordinate': index+1, 'value': coordinate})
            else:
                parameter_rows.append({'map': family, 'parameter': key, 'coordinate': '', 'value': value})
    effective_rows, effective_summaries = [], []
    for name in names:
        parameters = summary.get('method_maps', {}).get(name)
        if parameters is None:
            continue
        coordinates = (('Post-Mobius ball; retained jointly fitted centre/R/beta, effective delta=0'
                        if vi_mode == 'on' else 'Post-Mobius ball; untuned unit map, effective delta=0')
                       if name == 'DCS-Billiard' else 'Sphere' if name in SPHERE_METHODS else
                       'Fitted ball' if vi_mode == 'on' else 'Untuned ball')
        if reported_study:
            coordinates = ('Frozen original author-fitted sphere' if name == 'SCS' else
                           'Author-fitted sphere centre/observer with pilot-calibrated R' if name == 'SCS-calibrated' else
                           'Frozen study ball: centre=0, delta=0, prior calibrated R, beta=2')
        effective_rows.append(dict(method=name, coordinates=coordinates, **parameters))
        effective_summaries.append(dict(method=name, coordinates=coordinates,
            scale_R=parameters.get('scale'), centre_norm=float(np.linalg.norm(parameters.get('mu', []))),
            deformation_norm=float(np.linalg.norm(parameters.get('observer' if name in SPHERE_METHODS else 'delta', []))),
            beta=parameters.get('beta'), latitude=parameters.get('latitude')))
    setup_rows = []
    for family, fitting in summary.get('fitting', {}).items():
        total_draws = fitting.get('total_training_draws')
        count_source = 'recorded' if total_draws is not None else 'unavailable'
        if total_draws is None and fitting.get('steps') is not None and fitting.get('samples_per_step') is not None:
            total_draws = fitting['steps']*fitting['samples_per_step']
            count_source = 'steps × samples_per_step'
        setup_rows.append({'map': family, 'used_by': ' and '.join(_display_name(n) for n in names if n not in SPHERE_METHODS) if family == 'ball' else ' and '.join(_display_name(n) for n in names if n in SPHERE_METHODS),
                           'fitting_seconds': fitting.get('seconds'), 'steps': fitting.get('steps'),
                           'training_draws': fitting.get('training_draws'),
                           'samples_per_step': fitting.get('samples_per_step'),
                           'total_training_draws': total_draws, 'training_count_source': count_source,
                           'training_batch_policy': fitting.get('training_batch_policy', fitting.get('batch_policy'))})
        if author_fitting:
            setup_rows[-1].update(map_policy=fitting.get('map_policy'), seed=fitting.get('seed'),
                learning_rate=fitting.get('learning_rate'), objective=fitting.get('objective'),
                vi_coordinate_clip=fitting.get('vi_coordinate_clip'),
                gradient_norm_cap=fitting.get('gradient_norm_cap'),
                gradient_clipping=fitting.get('gradient_clipping'), source_commit=fitting.get('source_commit'),
                full_author_vi_budget=fitting.get('full_author_vi_budget'))
        if matched_vi:
            setup_rows[-1].update(gradient_updates=fitting['gradient_updates'],
                unique_reference_draws=fitting['unique_reference_draws'],
                objective_point_evaluations=fitting['objective_point_evaluations'],
                map_reused_without_refitting=fitting.get('map_reused_without_refitting', False),
                original_fit_seconds=fitting.get('original_fit_seconds'))
    count_rows = [{key: row[key] for key in ('method', 'chains', 'iterations_per_chain', 'mean_iterations',
                   'total_iterations', 'mean_estimation_iterations', 'total_estimation_iterations',
                   'total_discarded_iterations', 'warmup_iterations', 'thin', 'retained_per_chain',
                   'mean_seconds', 'median_seconds', 'iterations_per_second')} for row in case_rows]
    csvs = {'scs_100d_values.csv': case_rows, 'paper_values.csv': case_rows,
            'sample_counts.csv': count_rows, 'scs_100d_chains.csv': chain_rows,
            'tuned_parameters.csv': parameter_rows}
    joint_calibration_rows = []
    if calibrated_scs:
        calibration = summary['prepared_input_provenance']['joint_calibration']
        pilot_budget = calibration['budget_per_family_per_dimension']
        calibration_case = calibration['dimensions'][str(dimension)]
        for family, method in (('DCS-BW', 'DCS shared R/h'), ('SCS', 'SCS calibrated')):
            selected = calibration_case['selections'][family]
            costs = calibration_case['historical_pilot_costs'][family]
            joint_calibration_rows.append(dict(method=method, candidates=pilot_budget['candidates'],
                shared_exact_target_starts=pilot_budget['shared_exact_target_starts'],
                transitions_per_start=pilot_budget['transitions_per_start'],
                total_pilot_transitions=pilot_budget['transitions_total'],
                selected_candidate=selected['candidate_id'], selected_R=selected['R'],
                selected_h=selected['h'], selected_score=selected['score'],
                R_grid_edge=selected['R_grid_edge'], step_grid_edge=selected['step_grid_edge'],
                **{'historical_'+key: value for key, value in costs.items()}))
        csvs['joint_calibration.csv'] = joint_calibration_rows
    if ess_rows:
        csvs['functional_ess.csv'] = ess_rows
    if effective_rows:
        csvs['effective_maps.csv'] = effective_rows
    if paired_rows:
        csvs['paired_mse_differences.csv'] = paired_rows
    if prefix_rows:
        csvs['scs_100d_prefix_mse.csv'] = prefix_rows
    if setup_rows:
        csvs['setup_times.csv'] = setup_rows
    if tuning_rows:
        csvs['reflected_tuning.csv'] = tuning_rows
    if angular_tuning_rows:
        csvs['angular_tuning.csv'] = angular_tuning_rows
    if radial_tuning_rows:
        csvs['radial_tuning.csv'] = radial_tuning_rows
    if reference_rows:
        csvs['reference_values.csv'] = reference_rows
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
        if radial_ess:
            notes[1] = (acceptance_methods+' use separate acceptance-target 0.234 calibration. '
                        'DCS-Billiard spends the same warmup-transition budget on reflected-BW ball-movement '
                        'pilots in post-Mobius coordinates. All steps are frozen before production; '
                        'the first-third timed burn-in is additional. Reflected BW uses the same ball-movement '
                        'pilot rule in the shared fitted ball map, without a geometric step ceiling of 2.')
        if has_billiard:
            notes[2] = ('VI is enabled for both map families with equal optimizer and fresh-draw budgets. '
                        'DCS-Billiard retains the jointly fitted centre, R and beta but runs after Mobius, '
                        'with effective delta=0; this is not a new delta=0 VI fit. Other DCS methods retain '
                        'the shared fitted ball map. All effective maps are recorded separately.' if vi_mode == 'on' else
                        'VI is disabled for both map families: unit scale, zero centre and zero deformation. '
                        'DCS-Billiard uses this same untuned radial map. DCS retains beta=2 and SCS latitude 1.1. '
                        'Proposal-step tuning remains enabled for every method.')
            notes[5] = ('Map fitting, compilation and step calibration are reported separately and excluded from '
                        'production timing. The ball and sphere are fitted once each. Billiard reuses the fitted '
                        'ball centre/R/beta with zero effective deformation; no extra map fitting is performed.')
            notes.append('DCS-Billiard uses specular billiard reflections with invariant extra complete chords, '
                         'computed directly without a bounce loop. It uses the same uniform-ball increment law '
                         'and one Metropolis test. Invalid numerical endpoints are rejected. The original BW, '
                         + ('RWM and SCS production transitions are unchanged.' if own_kernel else
                            'RWM and SCS controls are unchanged.'))
            if not own_kernel:
                notes.append('Billiard step calibration runs reflected BW in its own post-Mobius coordinates from '
                             'the common physical start, using the same total warmup-transition budget. The resulting '
                             'state and selected step initialize billiard production; production does not retune. '
                             'The pilot table therefore scores reflected-BW movement, not billiard movement or MSE.')
            notes.append('Complete-chord extensions preserve the proposed ball radius. Accepted radial transition '
                         'laws coincide with reflected BW only for a radial transformed target in the same '
                         'coordinates and at the same frozen step. Skewed targets and comparisons across maps '
                         'do not have this guarantee. Short laptop results do not establish universal dominance.')
        if own_kernel and not author_vi_protocol:
            notes[1] = ('DCS-Billiard and SCS each tune their actual production kernel with the same total '
                        'warmup-transition budget and radial score. DCS-RWM and ordinary DCS-BW retain '
                        'acceptance-target 0.234 calibration; reflected BW retains its ball-movement pilots. '
                        'Every selected scale and map is frozen before production. The first-third timed burn-in is additional.')
            notes.append('The own-kernel pilot score is the average of delta squared divided by (1 + delta squared), '
                         'where delta is the change in physical log(1 + norm). Rejections contribute zero. '
                         'This is a bounded movement proxy, not MSE, ESS or a claim of optimality. '
                         'Pilot observations are discarded and production errors never select the step. '
                         'Five coarse candidates h times 0.25, 1, 4, 16 and 64 are followed by three '
                         'refinement candidates at 0.5, 1 and 2 times the coarse winner. The exact warmup '
                         'budget is divided across eight windows and trial order is randomized within each stage. '
                         'All candidates, scores and selection flags are recorded in radial_tuning.csv.')
            notes.append('Billiard and SCS receive equal pilot transition counts, not necessarily equal pilot seconds. '
                         'Pilot time is reported separately from equal production-time budgets. Numba uses the same '
                         'JAX calibration state and step as its JAX counterpart. The previous reflected-BW billiard '
                         'pilot remains part of the historical version-6/7 protocols only.')
        if author_vi_protocol:
            if 'DCS-Billiard' in radial_pilot_methods:
                notes.append('Billiard pilots score bounded squared changes of physical log(1 + norm), including '
                         'zero after rejection. Five coarse and three refinement trials share the pilot budget; '
                         'all trial scores are saved in radial_tuning.csv. This is a movement proxy, not MSE or ESS. '
                         'Pilot observations are discarded, and production errors never select a step.')
            notes[1] = ('SCS, DCS-RWM and ordinary DCS-BW use acceptance-target 0.234 calibration. '
                        'DCS-Billiard calibrates its actual production kernel by bounded radial movement; '
                        'reflected BW uses ball-movement pilots. Every method receives the same total '
                        'pilot-transition budget and freezes its step before production. Equal resources '
                        'do not establish equally optimal tuning. The first-third production burn-in is additional.')
            if vi_mode == 'on':
                notes[2] = ('SCS uses the unmodified authors’ VI routine: one fixed training batch, seed 0, '
                            'Adam at 0.01 and physical-coordinate clipping at 1000 within the VI objective. '
                            'DCS retains fresh-batch VI, recorded fit seed '+str(config.get('fit_seed', 42))+' and gradient-norm clipping at 10. '
                            'These are distinct fitting protocols; their actual budgets and times are recorded. '
                            'The production target is not clipped. Billiard retains the joint ball fit’s '
                            'centre/R/beta and uses effective delta=0 in post-Mobius coordinates; other DCS '
                            'methods retain the fitted ball map. All effective maps are recorded.')
            notes.append('SCS acceptance calibration follows the paper’s general tuning guidance, with the '
                         'benchmark’s recorded dual-averaging implementation. It is not the fixed h=0.1 '
                         'setting used in the authors’ skew-t example. The author VI routine is unchanged; '
                         'only the target and explicitly recorded training budgets are supplied by this benchmark.')
        if scs_fixed:
            notes[1] = ('SCS uses the authors’ skew-t example step h=0.1 throughout separate discarded '
                        'equilibration and production, with no step adaptation. '+
                        ('DCS-RWM and ordinary BW target acceptance 0.234; billiard calibrates its own '
                         'radial moves and reflected BW uses ball-movement pilots. ' if has_billiard else '')+
                        'All compared methods receive the same discarded warmup-transition and production-time '
                        'budgets. The first-third production discard is additional, and production scales remain fixed.')
            if vi_mode == 'off':
                notes[2] = ('VI is disabled for both map families: unit scale, zero centre and zero deformation. '
                            'DCS retains beta=2 and SCS latitude 1.1. DCS proposal scales are pretuned; '
                            'SCS uses fixed h=0.1 with separate discarded equilibration.')
            notes = [note for note in notes if not note.startswith('SCS acceptance calibration follows')]
            notes.append('The fixed h=0.1 is the step used in the authors’ skew-t example. '
                         'The target, timing and burn-in remain those of this benchmark; '
                         'this does not reproduce their published experiment in full.')
            if matched_step_control:
                notes.append('This step-size sensitivity control uses the same author transition and exact recorded '
                             'SCS map as the primary comparison. It does not refit VI; the saved primary map, '
                             'sampling seeds and budgets are reused and their provenance is recorded.')
        notes[7] = ('Every method samples the same recorded target: '+target_title+'. This is a benchmark of radial '
                    'tail-probability and mean-norm estimation, not a reproduction of the SCS publication figures or '
                    'a general mixing guarantee. The unmodified author SCS transition is retained.')
        if optimized:
            optimized_methods = ('DCS-RWM, DCS-BW, reflected BW and DCS-Billiard' if radial_ess else
                                 'DCS-BW, reflected BW, DCS-BW-rotate and DCS-RWM-rotate' if has_rwm_rotation else
                                 'DCS-BW, reflected BW and DCS-BW-rotate' if has_rotation else
                                 'DCS-BW and reflected BW')
            notes.append('Implementation comparison: '+optimized_methods+' use optimized Numba loops with cached '
                         'current density and radius, fused radial accumulation and no full physical-state recording. '
                         'SCS uses its unmodified author JAX transition and full-state scan. DCS uses the same JAX '
                         'discarded warmup and recorded step policy as the common-framework mode, then hands '
                         'its frozen state and step to Numba. '
                         'The 300-transition timing batches, statistic definitions and first-third discard are common. '
                         'This compares the recorded implementations; it does not isolate the geometric algorithm alone.')
        else:
            notes.append('Common-framework comparison: every method uses JAX CPU scans with full-state recording and '
                         'the same 300-transition timing batches and radial reduction. Outside-ball rejections can skip '
                         'density evaluation in ordinary BW/RWM; this is part of their specified transition cost.')
        if 'DCS-BW-reflect' in names:
            notes.append('Reflected BW keeps the reversible chord-fold proposal and its Metropolis correction.')
        if 'DCS-BW-reflect' in names and 'DCS-BW-reflect' not in fixed_dcs:
            notes.append('Its '
                     'ball-movement pilot search uses up to eight rotating h/2, h, 2h rounds with no special bound of 2. '
                     'A finite noisy local search does not certify an optimal step or optimal radial MSE. '
                     'Every trial is recorded in reflected_tuning.csv; a final-round edge winner is reported explicitly.')
        if has_rotation:
            notes.append('DCS-BW-rotate chooses reflected BW or a fixed-radius angular proposal with equal '
                         'probability at every iteration. Only the chosen transition is performed and counted. '
                         'With VI, the rotation is applied after the Möbius map and preserves physical distance '
                         'from the fitted centre. Its Metropolis test uses the physical-density ratio. '
                         'The angular move alone cannot change radius; it is used only in this mixture.')
            notes.append('The mixture divides the existing warmup-transition budget: floor(warmup/2) for '
                         'the reflected BW step, and the remainder for the angular scale. Angular pilots score '
                         'squared movement of unit directions, including rejection repeats. The angle is '
                         '2 atan(angular_scale), with initial scale 1. Both selected scales are frozen; the '
                         'mixture probability is fixed at 0.5 before examining results. Full angular candidates '
                         'are in angular_tuning.csv. Equal pilot resources do not guarantee optimal tuning or '
                         'an MSE advantage for any method.')
        if has_rwm_rotation:
            notes.append('DCS-RWM-rotate selects reflected isotropic Gaussian RWM or the same angular proposal with '
                         'equal probability. Its Gaussian step sigma is a coordinate standard deviation, not a BW radius. '
                         'Its initial sigma is the configured reference BW radius divided by sqrt(d+2), matching initial '
                         'expected squared increment length. Subsequent Gaussian and angular scales are pretuned within '
                         'the same total pilot budget, then frozen. The equivalent BW radius sigma*sqrt(d+2) is recorded '
                         'only to compare proposal units; it is not a second tuned parameter.')
        if len(metric_names) > 2:
            notes[3] = ('Every post-burn-in iteration contributes to mean norm, radial tail probability, mean first '
                        'coordinate and first-coordinate tail probability, including repeated states after rejection. '
                        'The same four quantities are accumulated inside every sampling clock. Saved trace thinning '
                        'does not change those estimates. Counts and full-precision errors are exported.')
            notes[7] = ('Every method samples the same recorded target: '+target_title+'. The original radial '
                        'measurements are supplemented by E[X₁] and P(X₁ > '+number(config['coordinate_threshold'])+'). '
                        'These test the asymmetric part of the target; no single set of functionals establishes '
                        'overall mixing. The unmodified author SCS transition is retained.')
            notes.append('The coordinate threshold and reference seed are fixed before sampler results are examined. '
                         'The existing targets and physical initialization remain unchanged. '
                         'All methods use the same reference values.')
            notes.append('Reference values marked exact use analytic distribution identities of the original skewed t targets. '
                         'Other reference values are independently estimated and retain their Monte Carlo standard '
                         'errors (MCSE). Reported MSE uses those estimated reference values without pretending they '
                         'are exact; its uncertainty also depends on reference precision.')
            notes.append('The Student-t degrees of freedom are three. The usual third-moment skewness coefficient '
                         'is therefore not reported; the added mean and tail probability are well-defined.')
            if len(metric_names) == 8:
                notes[3] = ('Every post-burn-in iteration contributes to mean norm, radial tail probability and '
                            'the mean and prespecified tail probability of each of X₁, X₂ and X₃, including '
                            'repeated states after rejection. The same eight quantities are accumulated inside '
                            'every sampling clock. Saved trace thinning does not change those estimates. '
                            'Counts and full-precision errors are exported.')
                notes[7] = ('Every method samples the same recorded target: '+target_title+'. The original radial '
                            'measurements are supplemented by E[X₁], E[X₂], E[X₃], P(X₁ > '+
                            number(config['coordinate_threshold'])+'), P(X₂ < '+
                            number(-config['coordinate_threshold'])+') and P(X₃ > '+
                            number(config['coordinate_threshold'])+'). These compare the skewed coordinates '
                            'with an unskewed coordinate; no single set of functionals establishes overall '
                            'mixing. The unmodified author SCS transition is retained.')
                notes.append('For these targets, X₂ has exactly the same marginal distribution as −X₁. '
                             'Its reference mean is the negative of the X₁ reference mean, and its mirrored '
                             'left-tail reference equals the X₁ right-tail reference; their MCSE values are '
                             'shared. Finite-chain estimates need not obey the symmetry. X₃ has an ordinary '
                             'univariate Student-t marginal, so its mean zero and tail reference are analytic '
                             'and marked exact. These relations do not assert independence of coordinates.')
    if matched_vi:
        notes[2] = ('Each map family reuses its own fixed reference cloud with the same number of draws, '
                    'optimizer updates and objective-point evaluations. SCS uses the unmodified authors’ '
                    'VI routine with seed 0, Adam at 0.01 and physical-coordinate clipping at 1000. '
                    'DCS uses recorded fit seed '+str(config.get('fit_seed', 42))+' and gradient-norm clipping at 10, '
                    'without coordinate clipping. Equal budgets do not make the objectives or clipping rules identical, '
                    'and fitting time is recorded separately. The production target is not clipped. '
                    'Billiard retains the ball fit’s centre/R/beta and runs after Mobius with effective delta=0; '
                    'other DCS methods retain the fitted ball map. All effective maps are recorded.')
    if fixed_dcs:
        notes[1] = ('DCS-BW, reflected BW and DCS-Billiard use the shared fixed ball-proposal radius h='+number(config['dcs_fixed_move'])+
                    '; DCS-RWM uses Gaussian coordinate standard deviation sigma=h/sqrt(d+2). '
                    'This matches the unreflected proposal coordinate variance h²/(d+2), not accepted '
                    'movement or mixing. All four DCS methods use their actual production kernels throughout the separate '
                    'discarded warmup without step adaptation. '+
                    ('SCS retains the authors’ fixed h=0.1 with discarded equilibration. ' if scs_fixed else
                     'SCS uses its recorded acceptance-target 0.234 calibration. ')+
                    'All methods receive the same discarded warmup-transition budget; every production '
                    'step is frozen and the first third of production iterations is additionally discarded. '
                    'The fixed DCS move is a benchmark setting, not a claim of optimality.')
    if v11 and names == ['SCS']:
        notes[1] = (('SCS uses the authors’ fixed h=0.1 throughout discarded equilibration and production, '
                     'without step adaptation. ' if scs_fixed else
                    'SCS uses acceptance-target 0.234 calibration during its separate discarded warmup, '
                     'then freezes the selected step throughout production. ')+
                    'This report contains only SCS; it does not rerun or change the primary DCS controls. '
                    'The first third of generated production iterations is additionally discarded.')
    if author_fitting and not author_vi_protocol:
        if vi_mode == 'on':
            dcs_fit_seed = summary.get('fitting', {}).get('ball', {}).get('seed', config.get('fit_seed', 42))
            notes[2] = ('SCS uses the unmodified authors’ VI routine: one fixed training batch, seed 0, '
                        'Adam at 0.01 and physical-coordinate clipping at 1000 within the VI objective, '
                        'without gradient clipping. DCS retains its own VI with recorded fit seed '
                        +str(dcs_fit_seed)+'. These are distinct fitting protocols with independently '
                        'recorded budgets and fitting costs. The production target is not clipped. '
                        'DCS methods share the fitted ball map; SCS uses the author-fitted sphere map.')
        else:
            notes[2] = ('VI is disabled for both map families: unit scale, zero centre and zero deformation. '
                        'No fitting draws or optimization are used; the recorded proposal-step policy is unchanged.')
        notes = [note.replace('common benchmark fitting and tuning protocol',
                              'recorded author fitting and benchmark tuning protocol')
                      .replace('shared fitting and tuning budgets', 'separately recorded fitting and tuning budgets')
                 for note in notes]
    if reported_study:
        notes = [
            'Reported zero-centre study: the original full-budget VI maps are bundled and reused. '
            'No optimizer or radius/step calibration runs during this invocation. The first reported '
            'study used JAX, two workers, two batches of 20 chains and 3 seconds per chain. '
            'The Full paper preset extends the sampling budget with the same frozen parameters.',
            'All four DCS variants use physical centre zero, deformation zero and beta=2. '
            'The frozen radius R and common proposal radius h come from the prior study. '
            'Gaussian RWM uses sigma=h/sqrt(d+2). These are post-fit choices, not a constrained VI refit. '
            'Original fitted vectors and losses remain separate from the actual production maps.',
            'SCS retains the exact original author-VI sphere map, latitude 1.1 and fixed h=0.1. '
            'Its centre, observer and radius are not replaced by the DCS choices. The author '
            'transition and VI sources are pinned; this applies them to the common normal-CDF skew-t target.',
            'The original fits each used 1,000 updates on a fixed 2,000-point reference cloud. '
            'DCS additionally received a radius/step pilot and centre diagnostic during development. '
            'This compares developed DCS settings with original-author SCS settings, not equal total '
            'tuning resources or a change of sampling kernel alone. Preparation costs are excluded from sampling time.',
            'Every method starts its 1,500 fixed-step warmup transitions at the same physical all-ones '
            'point. Warmup is discarded separately. The first third of timed production iterations is '
            'also discarded, and every remaining state, including rejection repeats, enters the estimates.',
            'The two radial reference values are analytic. MSE and functional ESS concern mean norm '
            'and radial-tail probability only; they do not establish coordinate or angular mixing. '
            'New runs assess sampling variability conditional on the frozen maps, not fitting variability.',
            'Setup tables retain the original fitting budgets and historical fitting seconds. '
            'Current fitting seconds are zero because maps are loaded; asset loading, compilation, '
            'warmup, ESS computation and saving are outside the production clock. Actual timing and '
            '300-transition batch overshoot are recorded. Equal seed inputs do not ensure identical '
            'sample counts or MSE on different computers under a time budget.',
            ('Numba extension: DCS uses optimized Numba production and SCS uses original author JAX. '
             'This is an extension of the reported JAX experiment, not a reproduction of its measured '
             'throughput. Both retain the same frozen maps, step rules and summary definitions.' if optimized else
             f'All {sampler_count} methods use the same JAX timing and full-state recording framework. '
             'Methods run in synchronized equal-size worker waves with rotating method order.'),
        ]
    if calibrated_scs:
        notes[0] = ('Reported zero-centre study extension with original and calibrated SCS. '
                    'Bundled full-budget VI maps and independently pilot-selected parameters are reused; '
                    'no optimizer or radius/step pilot runs during this invocation. The earlier five-method '
                    'reported results remain historical; this six-method report is a new assessment.')
        notes[2] += (' SCS calibrated retains the same author-fitted centre, observer and latitude, '
                     'but uses its independently selected radius and step. Both SCS methods call the unchanged '
                     'author JAX transition; the calibrated control is not the authors’ fixed-h=0.1 recipe.')
        notes[3] = ('The original VI fits each used 1,000 updates on a fixed 2,000-point cloud. '
                    'DCS and calibrated SCS each received 16 radius/step candidates, 4,096 exact target starts '
                    'per candidate and eight transitions per start: 524,288 pilot transitions per dimension. '
                    'Equal pilot counts do not imply equal seconds. DCS also received the later zero-centre/'
                    'deformation intervention. Original SCS deliberately retains h=0.1 and the original fitted radius. '
                    'These are declared preparation policies, not identical algorithms or equal total development effort.')
    if radial_ess:
        notes.append('Functional ESS is computed for the unthinned post-burn-in radial-tail indicator and norm, '
                     'including repeated states after rejection. It uses raw mean ESS with split-chain '
                     'autocorrelations and Geyer positive/monotone sequence truncation; it is distinct from '
                     'rank-normalized bulk ESS and quantile-tail ESS. Saved trace thinning does not alter this diagnostic.')
        notes.append('ESS per sampling second divides summed chain ESS by summed full production time, '
                     'including discarded sampling burn-in. The sum is descriptive, not pooled-estimator ESS. Calibration, compilation and ESS computation are excluded; '
                     'diagnostic computation time is shown separately. A method aggregate is unavailable if any chain '
                     'has unavailable ESS, with reasons and valid-chain counts shown. Constant traces do not receive '
                     'an artificial ESS equal to the sample count. ESS alone does not establish convergence or low bias.')
        notes.append('This current comparison reports only radial-tail probability and mean norm. '
                     'Coordinate and marginal statistics are not generated; archived reports retain their recorded measurements.')
    notes.append('Empirical MSE MCSE is the sample standard deviation of chain squared errors divided by the square root of the number of chains. Largest-chain error share measures how much one chain contributes to total squared error. Paired differences align methods by saved seed; negative differences favor the named method over SCS. These are descriptive, conditional on the fitted maps, not formal confidence intervals or convergence diagnostics. Heavy-tailed norm errors can make variance-based uncertainty unreliable; fresh seed batches are needed to assess repeatability.')
    notes += [summary[key] for key in ('target_note', 'mse_note', 'time_scope') if isinstance(summary.get(key), str)]
    warning = '<p class="warning">Quick execution check only — do not interpret sampler rankings.</p>' if smoke else ''
    uncertain_reference = [key for metric, key in enumerate(metric_names)
                           if not reference_exact[metric] and reference_mcse[metric] > 0 and
                           reference_mcse[metric] > .1*np.sqrt(min(mses[name][metric] for name in names))]
    if uncertain_reference:
        warning += ('<p class="warning"><strong>Reference precision matters.</strong> For '+
                    html.escape(', '.join(uncertain_reference))+', the reference MCSE exceeds one tenth '
                    'of the smallest reported sampler RMSE. Inspect reference_values.csv before interpreting '
                    'small MSE differences; this is a descriptive precision check, not a confidence interval.</p>')
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
        fit_description = ('frozen full-budget VI maps with recorded DCS post-fit overrides; no refitting'
                           if reported_study else 'VI '+str(vi_mode)+' for both map families')
        page += '<p>Target: '+html.escape(target_title)+'; '+html.escape(fit_description)+'; DCS β = '+number(config.get('beta', 2.))+'.</p>'
    if reported_study:
        page += '<p><strong>Reported study protocol.</strong> This uses the bundled zero-centre DCS settings, not the previous fitted-centre h=0.07 comparison. Full parameters and preparation provenance are saved with this run.</p>'
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
    if reference_rows:
        page += '<h2>Reference values and precision</h2><p>Estimated references are independent of every sampler. MCSE describes reference simulation error; it is separate from between-chain variation.</p>'+_table(reference_rows)
    page += '<h2>Counts, errors and runtime</h2>'+_table(case_rows)
    if ess_rows:
        page += '<h2>Functional ESS and ESS per second</h2><p>These are diagnostics of the radial functions used in the estimates, not coordinate ESS. All post-burn-in iterations enter the diagnostic, including rejection repeats. An unavailable chain makes its method aggregate unavailable; reasons and valid counts are retained. ESS/s uses total production seconds, including burn-in; diagnostic computation time is separate.</p>'+_table(ess_rows)
    if paired_rows:
        page += '<h2>Paired MSE differences from SCS</h2><p>Negative differences favor the named method. Empirical MCSE is descriptive; it is not a confidence interval.</p>'+_table(paired_rows)
    if setup_rows:
        page += ('<h2>Original map fitting provenance — maps reused without refitting</h2>' if reported_study else
                 '<h2>Map fitting excluded from production time</h2>')+_table(setup_rows)
    if joint_calibration_rows:
        page += ('<h2>Archived joint radius/step calibration</h2><p>DCS and calibrated SCS used equal '
                 'candidate and transition counts with the same exact physical starting points. '
                 'The table records the prior diagnostic pilot, not calibration performed during this run. '
                 'Original SCS keeps its original fitted radius and h=0.1. DCS also received the later '
                 'centre/deformation intervention. Matching pilot counts does not match all preparation '
                 'seconds or historical development choices.</p>'+_table(joint_calibration_rows)+
                 '<p>Shared exact-state generation took '+number(calibration_case['shared_exact_state_generation_seconds'])+
                 ' seconds on the original computer, recorded once for both families. '+
                 html.escape(calibration_case['cost_scope'])+'</p>')
    if effective_summaries:
        map_description = ('All four DCS methods use the same frozen zero-centre, zero-deformation map; SCS retains its original author-fitted map.' if reported_study else
                           'Shared fitted parameters and production coordinates are distinct for DCS-Billiard.')
        if calibrated_scs:
            map_description += (' SCS calibrated retains the same author centre and observer, '
                                'with its separately pilot-selected sphere radius.')
        page += '<h2>Effective maps used by each sampler</h2><p>'+map_description+' Full vectors are in <a href="effective_maps.csv">effective_maps.csv</a> and the saved summary.</p>'+_table(effective_summaries)
    if tuning_summaries:
        reflected_heading = 'Reflected-step pilot tuning and frozen steps' if has_rwm_rotation else 'Reflected BW pilot tuning and frozen steps'
        page += '<details><summary>'+reflected_heading+'</summary><p>The score is the mean squared jump in ball coordinates, including zero movement after rejection. It is not an MSE estimate. Full pilot candidates are available in <a href="reflected_tuning.csv">reflected_tuning.csv</a> and the saved summary.</p>'+_table(tuning_summaries)+'</details>'
    if radial_tuning_summaries:
        radial_intro = ('Billiard tunes its actual production kernel. SCS uses its separately recorded step policy; '
                        'all methods receive equal discarded warmup-transition budgets. ' if author_vi_protocol else
                        'Billiard and SCS each tune their actual production kernel using equal pilot transition budgets. ')
        page += '<details><summary>Own-kernel radial pilots and frozen steps</summary><p>'+radial_intro+'The score is bounded squared movement of physical log(1 + norm), including zero after rejection; it is not MSE or ESS. Pilot seconds can differ and are excluded from production time. Full candidates are available in <a href="radial_tuning.csv">radial_tuning.csv</a>.</p>'+_table(radial_tuning_summaries)+'</details>'
    if angular_tuning_summaries:
        page += '<details><summary>Rotation pilot tuning and frozen angles</summary><p>The score is the mean squared jump of unit directions, including zero movement after rejection. The angle is 2 atan(angular scale). Reflected-step and angular pilots share the existing warmup budget; both scales are frozen before sampling. Full candidates are available in <a href="angular_tuning.csv">angular_tuning.csv</a>.</p>'+_table(angular_tuning_summaries)+'</details>'
    page += '<h2>Every chain</h2>'+_table(chain_rows)
    page += '<h2>Full map parameters</h2>'+_table(parameter_rows)
    if prefix_rows:
        page += '<h2>Exact checkpoint MSE values</h2>'+_table(prefix_rows)
    page += '<h2>Protocol and interpretation</h2><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul>'
    page += '<p>Plots and tables use two decimals, with scientific notation for MSE and small values. Integer sample counts remain exact. CSV and JSON retain full numerical precision.</p></body></html>'
    (root/'paper_values.html').write_text(page, encoding='utf-8')
    return {'figures': figure_names, 'csv_files': list(csvs), 'html': 'paper_values.html'}
