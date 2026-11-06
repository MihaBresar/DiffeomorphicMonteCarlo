"""Portable frozen inputs for the reported zero-centre, original-SCS study.

The original joint VI artifacts are retained intact. Production DCS maps use
separately declared zero centres/deformations and previously selected radii;
their original loss histories do not describe constrained VI fits.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'studies' / 'zero_centre' / 'maps'
LEGACY_PROTOCOLS = {'jax': 'timed-reported-zero-centre-v1',
                    'numba': 'timed-reported-zero-centre-numba-v1'}
CALIBRATED_PROTOCOLS = {'jax': 'timed-reported-zero-centre-v2',
                        'numba': 'timed-reported-zero-centre-numba-v2'}
ZERO_PROTOCOLS = {'jax': 'timed-reported-zero-centre-v3',
                  'numba': 'timed-reported-zero-centre-numba-v3'}
PROTOCOLS = {'jax': 'timed-reported-zero-centre-v5',
             'numba': 'timed-reported-zero-centre-numba-v5'}
MATCHED_CALIBRATION_SHA256 = '828bb0e8a861dcc8078e7f9b1ff5de5a553634a2d4da74f450bc06e7c57d16a9'
MATCHED = {
    100: {'DCS-BW': dict(R=.6941299233184104, h=.025, candidate='DCS-BW-r0-h0'),
          'SCS': dict(R=1.1048295589441204, h=.2, candidate='SCS-r1-h3')},
    500: {'DCS-BW': dict(R=.6941116209824034, h=.022360679774997897, candidate='DCS-BW-r0-h1'),
          'SCS': dict(R=2.2313627251649697, h=.1, candidate='SCS-r1-h2')},
}
SUPPORTED_PROTOCOLS = (tuple(LEGACY_PROTOCOLS.values()) + tuple(CALIBRATED_PROTOCOLS.values())
                       + tuple(ZERO_PROTOCOLS.values()) + tuple(PROTOCOLS.values()))
CALIBRATED_METHOD = 'SCS-calibrated'
CALIBRATION_SHA256 = '02744e053bc5a39d1f35a677cde5a49c76f83e6d0832ab669e8152cd3f431f64'
CALIBRATED_SCS = {100: dict(R=.5524147794720602, h=.05, candidate='SCS-r0-h1'),
                  500: dict(R=1.1156813625824848, h=.025, candidate='SCS-r0-h0')}
PRESETS = {'quick': dict(runs=2, time_budget=.05),
           'reproduce': dict(runs=20, time_budget=3.),
           'paper': dict(runs=50, time_budget=20.)}
LEGACY_METHODS = ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect', 'DCS-Billiard']
METHODS = ['DCS-RWM', 'DCS-BW', 'SCS', CALIBRATED_METHOD, 'DCS-BW-reflect', 'DCS-Billiard']
TARGETS = {'skew-t-100d': 100, 'skew-t-500d': 500}
FROZEN = {
    100: dict(scale=.6941299233184104, h=.025,
              sha256='dfbc1d209e47aae0364bb3d456d9325719a770981769764da5c37cb6c8cc344b', bytes=78386),
    500: dict(scale=1.3882232419648068, h=.044721359549995794,
              sha256='469fda353c1c3afe4187473cbaf876b67e569f7961ed8e6e3f311e27460b9322', bytes=127567),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_asset(dimension):
    _require(dimension in FROZEN, 'Reported study supports dimensions 100 and 500')
    manifest_path = ASSETS / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _require(manifest['schema'] == 'reported-zero-centre-map-assets-v1', 'Unknown map-asset schema')
    entry = manifest['dimensions'][str(dimension)]
    filename = f'full_budget_{dimension}d.json'
    _require(entry['file'] == filename, 'Unexpected bundled-map filename')
    path = ASSETS / filename
    raw = path.read_bytes()
    frozen = FROZEN[dimension]
    _require(len(raw) == entry['bytes'] == frozen['bytes']
             and hashlib.sha256(raw).hexdigest() == entry['sha256'] == frozen['sha256']
             and entry['original_source_sha256'] == frozen['sha256'], 'Bundled map hash or length differs from the reported study')
    _require(entry['production_dcs'] == dict(scale=frozen['scale'], h=frozen['h'],
             beta=2., mu_policy='zero', delta_policy='zero'), 'Frozen DCS production parameters changed')
    _require(entry['production_scs'] == dict(mapping='unchanged bundled original sphere map', h=.1),
             'Frozen original SCS parameters changed')
    saved = json.loads(raw)
    _require(saved['config']['dimension'] == dimension, 'Bundled map dimension mismatch')
    for family in ('ball', 'sphere'):
        fit = saved['fitting'][family]
        _require(fit['steps'] == fit['gradient_updates'] == 1000
                 and fit['training_draws'] == fit['unique_reference_draws'] == fit['total_training_draws'] == 2000
                 and fit['objective_point_evaluations'] == 2_000_000 and fit['batch_policy'] == 'fixed',
                 'Original fitting metadata differs from the fixed study')
        _require(len(fit['loss']) == 1000 and all(math.isfinite(x) for x in fit['loss']),
                 'Original fitting history is incomplete or nonfinite')
    for name, digest in saved['fitting']['sphere']['source_sha256'].items():
        _require(_sha(ROOT / 'third_party' / 'scs_paper' / name) == digest,
                 'Pinned original SCS source differs from the bundled fitting provenance')
    return saved, manifest, entry


def legacy_configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
                  target='skew-t-100d', seed_batch=22):
    """Build a fresh run; only the seed batch may override the frozen preset."""
    from scripts import reproduce_comparison as suite
    _require(preset in PRESETS, 'Choose quick, reproduce, or paper')
    _require(comparison in PROTOCOLS, 'comparison must be jax or numba')
    _require(target in TARGETS, 'Choose the 100D or 500D skew-t target')
    extra = dict(overrides or {})
    _require(not (set(extra) - {'seed_batch'}), 'Reported-study parameters are frozen; only seed_batch may be overridden')
    if 'seed_batch' in extra:
        seed_batch = extra['seed_batch']
    dimension = TARGETS[target]
    _, manifest, entry = _load_asset(dimension)
    frozen = FROZEN[dimension]
    config = suite.configuration(preset, workers, overrides={**PRESETS[preset],
        'warmup': 1500, 'thin': 50, 'optimizer_steps': 1000, 'optimizer_samples': 2000,
        'seed_batch': seed_batch, 'dcs_step_rule': 'fixed', 'dcs_fixed_move': frozen['h']},
        comparison=comparison, target=target)
    config.update(experiment='reported-zero-centre' + ('-numba' if comparison == 'numba' else ''),
        protocol=LEGACY_PROTOCOLS[comparison], implementation_revision='reported-zero-centre-frozen-inputs-v1',
        study_id='zero-centre-original-scs-v1', fit_reused_without_refitting=True,
        fit_constrained_mu_zero=False, fit_constrained_delta_zero=False,
        production_ball_coordinate_policy='zero_mu_zero_delta_frozen_pilot_radius_all_dcs',
        production_scs_centre_policy='original_author_vi',
        ball_map_policy='reported-zero-centre-frozen',
        billiard_map_policy='shared_zero_centre_zero_deformation_ball_map',
        prepared_input_policy='bundled_original_fits_with_declared_production_overrides',
        reported_map_asset=f'studies/zero_centre/maps/{entry["file"]}',
        reported_map_asset_sha256=entry['sha256'],
        reported_map_manifest_sha256=_sha(ASSETS / 'manifest.json'),
        reported_production_parameters=dict(dcs_R=frozen['scale'], dcs_h=frozen['h'],
            dcs_mu='zero', dcs_delta='zero', dcs_beta=2., scs_map='unchanged original author VI', scs_h=.1),
        study_note=manifest['development_provenance']['scope'],
        fitting_policy='Reuse the bundled original 1000-update, 2000-point fixed-cloud joint VI artifacts. '
            'DCS production uses zero centre/deformation and a separately selected frozen radius. '
            'SCS retains its exact original author fit. No new fitting or constrained VI refit occurs.',
        tuning_note='All four DCS methods share the recorded fixed move; Gaussian RWM uses h/sqrt(d+2). '
            'SCS retains fixed h=0.1. All methods receive 1500 discarded, nonadaptive warmup transitions '
            'and start from the same physical all-ones point. No new step or radius tuning occurs.',
        reported_baseline='JAX reported study' if comparison == 'jax' else 'Numba DCS extension; not the reported JAX backend',
        seed_note='Default batch22 is a fresh assessment. Batches20/21 reproduce the reported random-stream '
            'inputs; wall-clock stopping and hardware can change counts and estimates.')
    validate_configuration(config)
    return config



def _load_calibration():
    path = ASSETS / 'joint_calibration.json'
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == CALIBRATION_SHA256,
             'Bundled joint-calibration hash differs from the frozen pilot')
    value = json.loads(raw)
    _require(value['schema'] == 'reported-joint-radius-step-calibration-v1'
             and value['frozen_before_assessment'] is True, 'Invalid calibration provenance')
    budget = value['budget_per_family_per_dimension']
    _require(budget == dict(candidates=16, shared_exact_target_starts=4096,
             transitions_per_start=8, transitions_total=524288,
             same_physical_starts_reused_across_families_and_candidates=True),
             'Joint-calibration resources changed')
    for dimension, frozen in CALIBRATED_SCS.items():
        entry = value['dimensions'][str(dimension)]
        chosen = entry['selections']['SCS']
        _require(chosen['R'] == frozen['R'] and chosen['h'] == frozen['h']
                 and chosen['candidate_id'] == frozen['candidate'], 'Frozen SCS calibration changed')
        _require(entry['original_map_sha256'] == FROZEN[dimension]['sha256']
                 and entry['pilot_transitions_per_family'] == {'DCS-BW': 524288, 'SCS': 524288},
                 'Joint-calibration source maps or resources changed')
    return value


def calibrated_configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
                           target='skew-t-100d', seed_batch=22):
    """Historical v2 preset with the author-fitted SCS centre and observer."""
    config = legacy_configuration(preset, workers, overrides, comparison=comparison,
                                  target=target, seed_batch=seed_batch)
    calibration = _load_calibration()
    frozen = CALIBRATED_SCS[config['dimension']]
    config.update(protocol=CALIBRATED_PROTOCOLS[comparison],
        algorithms=list(METHODS), implementation_revision='reported-zero-centre-two-scs-controls-v2',
        study_id='zero-centre-original-and-calibrated-scs-v2',
        method_display_labels={CALIBRATED_METHOD: 'SCS calibrated'},
        scs_calibrated_move=frozen['h'],
        reported_calibrated_scs_parameters=dict(R=frozen['R'], h=frozen['h'],
            centre='original author VI', observer='original author VI'),
        reported_calibration_asset='studies/zero_centre/maps/joint_calibration.json',
        reported_calibration_asset_sha256=CALIBRATION_SHA256,
        calibration_budget_per_family=copy.deepcopy(calibration['budget_per_family_per_dimension']),
        step_tuning_policies={**config['step_tuning_policies'], CALIBRATED_METHOD: 'fixed_scs_move_v1'},
        study_note='Original SCS retains author VI and h=0.1. Calibrated SCS uses the same original centre/observer '
            'with R/h selected by the same 16-candidate, 4096-start, 8-transition pilot budget as DCS. '
            'DCS also retains its later zero-centre/deformation choices; equal pilot counts do not mean equal '
            'elapsed preparation cost or identical historical development. All settings are loaded frozen.',
        fitting_policy=config['fitting_policy'] + ' The additional SCS control shares that original VI fit '
            'and changes only its frozen production radius; it is not another VI fit.',
        tuning_note='All four DCS methods share their recorded fixed move; RWM uses h/sqrt(d+2). '
            'Original SCS uses h=0.1 and calibrated SCS uses its prior pilot-selected R/h. '
            'All six receive 1500 discarded nonadaptive warmup transitions from the same physical all-ones point. '
            'No new radius/step selection is performed.',
        seed_note='Default batch22 is a fresh six-method assessment. Batches20/21 repeat recorded stream inputs; '
            'method-order changes, hardware and timed stopping can change counts and estimates. '
            'Historical reported results used a five-method primary run plus separate SCS controls.')
    validate_configuration(config)
    return config


def _zero_scs_intervention():
    return dict(centre_override='zero', observer_override='zero', latitude='unchanged at 1.1',
        radius_step_selection_map='original author VI centre and observer',
        radius_step_recalibrated=False, constrained_vi_refit=False,
        new_fitting=False, new_radius_step_pilot=False,
        scope='SCS has an observer displacement rather than a DCS Mobius delta. '
              'Both its physical centre and observer are set to zero after the original fitting '
              'and R/h pilot. Prior R/h values are retained without recalibration for this changed map; '
              'they are not claimed optimal for the zero-centre, zero-observer control.')


def zero_configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
                  target='skew-t-100d', seed_batch=22):
    """Current six-method preset with a zero-centre, zero-observer SCS control."""
    config = calibrated_configuration(preset, workers, overrides, comparison=comparison,
                                      target=target, seed_batch=seed_batch)
    selected = CALIBRATED_SCS[config['dimension']]
    config.update(protocol=ZERO_PROTOCOLS[comparison],
        implementation_revision='reported-zero-centre-zero-observer-scs-v3',
        study_id='zero-centre-original-and-zero-map-calibrated-scs-v3',
        method_display_labels={CALIBRATED_METHOD: 'SCS calibrated (zero)'},
        reported_calibrated_scs_parameters=dict(R=selected['R'], h=selected['h'],
            centre='zero', observer='zero'),
        production_calibrated_scs_map_policy='zero_mu_zero_observer_with_prior_frozen_radius_step',
        calibrated_scs_map_intervention=_zero_scs_intervention(),
        study_note='Original SCS retains its exact author VI map and h=0.1. The calibrated SCS '
            'control now has zero physical centre and zero observer displacement, with the prior '
            'pilot-selected R/h retained. That pilot used the author-fitted centre/observer before '
            'this map intervention; no new pilot or optimization is performed for the zero map. '
            'All DCS settings remain unchanged. Equal historical pilot counts do not establish '
            'optimal tuning of the modified SCS map or identical development procedures.',
        fitting_policy='Reuse the original joint VI artifacts and loss histories. Original SCS '
            'keeps its exact author fit. The calibrated control uses explicit zero-centre and '
            'zero-observer production overrides and a prior selected radius. This is not a '
            'constrained VI refit, and neither family is fitted again.',
        tuning_note='All four DCS methods retain their recorded fixed move; RWM uses h/sqrt(d+2). '
            'Original SCS uses h=0.1. SCS calibrated (zero) retains its prior R/h selection, '
            'made before the zero-centre/zero-observer override. All six receive 1500 discarded '
            'nonadaptive warmup transitions from the same physical all-ones point. No new '
            'calibration or production adaptation occurs.',
        seed_note='Default batch22 supplies the six-method random-stream inputs. Saved v1/v2 '
            'results retain their original maps; the zero-observer intervention requires fresh '
            'sampling. Timed stopping, hardware and map changes can alter counts and estimates.')
    validate_configuration(config)
    return config



def _load_matched_calibration():
    """Load the frozen, equally budgeted pilot on the actual production maps."""
    path = ASSETS / 'matched_calibration.json'
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == MATCHED_CALIBRATION_SHA256,
             'Matched calibration hash differs from the frozen pilot')
    value = json.loads(raw)
    _require(value['schema'] == 'reported-matched-zero-radius-step-calibration-v1'
             and value['frozen_before_assessment'] is True and value['seed'] == 842901174,
             'Invalid matched-calibration provenance')
    _require(value['budget_per_family_per_dimension'] == dict(candidates=16,
             shared_exact_target_starts=4096, transitions_per_start=8, transitions_total=524288,
             same_physical_starts_reused_across_families_and_candidates=True),
             'Matched calibration resources changed')
    _require(value['final_map_policy'] == {'DCS-BW': 'zero_mu_zero_delta',
             'SCS': 'zero_mu_zero_observer'}, 'Matched pilot must use the final zero maps')
    for dimension, methods in MATCHED.items():
        row = value['dimensions'][str(dimension)]
        _require(row['original_map_sha256'] == FROZEN[dimension]['sha256']
                 and row['pilot_transitions_per_family'] == {'DCS-BW': 524288, 'SCS': 524288},
                 'Matched calibration source maps or resources changed')
        for method, frozen in methods.items():
            selected = row['selections'][method]
            _require(selected['R'] == frozen['R'] and selected['h'] == frozen['h']
                     and selected['candidate_id'] == frozen['candidate'],
                     'Matched production selection changed')
    return value


def _matched_intervention():
    return dict(centre_override='zero', observer_override='zero', latitude='unchanged at 1.1',
        radius_step_selection_map='final zero physical centre and zero observer',
        radius_step_recalibrated=True, constrained_vi_refit=False,
        new_fitting=False, new_radius_step_pilot=True,
        scope='Both families set their centre and directional-deformation parameters to zero '
              'before every pilot candidate. The selected radius and step are used unchanged '
              'in production. This is an explicit map policy, not constrained VI refitting.')


def configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
                  target='skew-t-100d', seed_batch=22):
    """Current comparison: original SCS plus equally calibrated final zero maps."""
    config = zero_configuration(preset, workers, overrides, comparison=comparison,
                                target=target, seed_batch=seed_batch)
    calibration = _load_matched_calibration()
    selections = MATCHED[config['dimension']]
    ball, sphere = selections['DCS-BW'], selections['SCS']
    config.update(protocol=PROTOCOLS[comparison],
        implementation_revision='reported-matched-final-zero-map-calibration-v5',
        study_id='zero-centre-matched-preparation-v5',
        dcs_fixed_move=ball['h'], scs_calibrated_move=sphere['h'],
        reported_production_parameters=dict(dcs_R=ball['R'], dcs_h=ball['h'],
            dcs_mu='zero', dcs_delta='zero', dcs_beta=2., scs_map='unchanged original author VI', scs_h=.1),
        reported_calibrated_scs_parameters=dict(R=sphere['R'], h=sphere['h'], centre='zero', observer='zero'),
        reported_calibration_asset='studies/zero_centre/maps/matched_calibration.json',
        reported_calibration_asset_sha256=MATCHED_CALIBRATION_SHA256,
        calibration_budget_per_family=copy.deepcopy(calibration['budget_per_family_per_dimension']),
        production_calibrated_scs_map_policy='zero_mu_zero_observer_calibrated_before_production',
        calibrated_scs_map_intervention=_matched_intervention(),
        preparation_script='scripts/prepare_comparison.py', preparation_seed=calibration['seed'],
        preparation_fit_seeds=dict(ball=42, sphere=0),
        study_note='DCS and calibrated SCS select R/h on their final zero-centre, zero-deformation '
            'maps using the same physical starting cloud, candidate count, transition count, radial '
            'objective and selection rule. The four DCS variants share the DCS-BW choice. Original '
            'SCS retains author VI and h=0.1. Geometry-specific step grids are declared in the pilot. '
            'Equal objective and pilot counts do not imply equal elapsed preparation time.',
        fitting_policy='Reuse the bundled 1000-update, 2000-point fixed-cloud VI artifacts. Original '
            'SCS keeps its exact author fit. Both calibrated families set centres and directional '
            'deformations to zero before the R/h pilot. This is not constrained VI refitting. '
            'The portable preparation script records seeds, all candidates, choices and timings.',
        tuning_note='Both calibrated families use 16 candidates, 4096 shared exact target starts '
            'and eight transitions per start. The same normalized radial-CDF and tail-indicator '
            'objective selects R/h before an independent holdout. All four DCS methods share the '
            'DCS-BW result; RWM uses h/sqrt(d+2). Original SCS retains h=0.1. Production uses frozen '
            'settings and 1500 discarded, nonadaptive warmup transitions from the same all-ones point.',
        seed_note='Seed 1 through Seed 8 vary only production random streams. Preparation uses '
            'separate recorded seeds. Timed stopping and hardware can alter counts and estimates.')
    validate_configuration(config)
    return config

def validate_configuration(config):
    """Validate scientific settings without requiring the original computer."""
    backend = config.get('comparison')
    target_id = config.get('target_id')
    _require(backend in PROTOCOLS and config.get('protocol') in
             (LEGACY_PROTOCOLS[backend], CALIBRATED_PROTOCOLS[backend], ZERO_PROTOCOLS[backend], PROTOCOLS[backend]),
             'Unknown reported-study protocol')
    calibrated = config['protocol'] != LEGACY_PROTOCOLS[backend]
    matched = config['protocol'] == PROTOCOLS[backend]
    zero_scs = matched or config['protocol'] == ZERO_PROTOCOLS[backend]
    methods = METHODS if calibrated else LEGACY_METHODS
    _require(target_id in TARGETS and config.get('preset') in PRESETS, 'Unknown reported-study target or preset')
    d = TARGETS[target_id]
    ball = MATCHED[d]['DCS-BW'] if matched else dict(R=FROZEN[d]['scale'], h=FROZEN[d]['h'])
    _require(config.get('target') == dict(kind='skew_t', dimension=d, nu=3., alpha={'0': 100., '1': -100.})
             and config.get('dimension') == d and config.get('beta') == 2. and config.get('latitude') == 1.1,
             'Reported target or map geometry changed')
    _require(config.get('threshold') == 50. * math.sqrt(d / 100.), 'Reported tail threshold changed')
    for name, value in {**PRESETS[config['preset']], 'warmup': 1500, 'thin': 50,
                        'optimizer_steps': 1000, 'optimizer_samples': 2000}.items():
        _require(config.get(name) == value, f'Reported preset has a different {name}')
    _require(config.get('algorithms') == methods and config.get('dcs_backend') == backend,
             'Reported method list or backend changed')
    _require(config.get('dcs_step_rule') == 'fixed' and config.get('dcs_fixed_move') == ball['h']
             and config.get('scs_step_rule') == 'fixed' and config.get('initial_scale') == .1
             and config.get('step_tuning_policies') == {m: 'author_scs_fixed_01' if m == 'SCS' else 'fixed_scs_move_v1' if m == CALIBRATED_METHOD else 'fixed_dcs_move_v1' for m in methods},
             'Reported proposal steps are frozen')
    _require(not any(key in config for key in ('scs_fixed_move', 'step_scale_ceilings', 'step_scale_ceiling')),
             'Reported steps do not accept alternate scales or ceilings')
    _require(config.get('reported_production_parameters') == dict(dcs_R=ball['R'], dcs_h=ball['h'],
             dcs_mu='zero', dcs_delta='zero', dcs_beta=2., scs_map='unchanged original author VI', scs_h=.1),
             'Reported production parameter metadata changed')
    if calibrated:
        calibration = _load_matched_calibration() if matched else _load_calibration()
        selected = MATCHED[d]['SCS'] if matched else CALIBRATED_SCS[d]
        centre_policy = 'zero' if zero_scs else 'original author VI'
        label = 'SCS calibrated (zero)' if zero_scs else 'SCS calibrated'
        _require(config.get('scs_calibrated_move') == selected['h']
                 and config.get('reported_calibrated_scs_parameters') == dict(R=selected['R'], h=selected['h'],
                     centre=centre_policy, observer=centre_policy)
                 and config.get('method_display_labels') == {CALIBRATED_METHOD: label},
                 'Reported calibrated SCS parameters changed')
        _require(config.get('reported_calibration_asset') == 'studies/zero_centre/maps/' +
                 ('matched_calibration.json' if matched else 'joint_calibration.json')
                 and config.get('reported_calibration_asset_sha256') ==
                 (MATCHED_CALIBRATION_SHA256 if matched else CALIBRATION_SHA256)
                 and config.get('calibration_budget_per_family') == calibration['budget_per_family_per_dimension'],
                 'Reported calibration provenance changed')
        if zero_scs:
            _require(config.get('production_calibrated_scs_map_policy') ==
                     ('zero_mu_zero_observer_calibrated_before_production' if matched else
                      'zero_mu_zero_observer_with_prior_frozen_radius_step')
                     and config.get('calibrated_scs_map_intervention') ==
                     (_matched_intervention() if matched else _zero_scs_intervention()),
                     'Reported zero-map SCS intervention changed')
    if matched:
        _require(config.get('preparation_script') == 'scripts/prepare_comparison.py'
                 and config.get('preparation_seed') == 842901174
                 and config.get('preparation_fit_seeds') == dict(ball=42, sphere=0),
                 'Matched preparation recipe or seed changed')
    _require(config.get('fit_reused_without_refitting') is True
             and config.get('fit_constrained_mu_zero') is False and config.get('fit_constrained_delta_zero') is False,
             'Reported frozen-fit provenance changed')
    _require(config.get('vi') == 'on' and config.get('vi_enabled') is True
             and config.get('sphere_map_policy') == 'author-paper-vi'
             and config.get('scs_vi') == dict(optimizer_steps=1000, optimizer_samples=2000),
             'Reported original fitting settings changed')
    _require(config.get('mode') == 'time' and config.get('burnin_fraction') == 1/3
             and config.get('coordinate_metrics') is False and config.get('ess_enabled') is True,
             'Reported timing, burn-in, or estimands changed')
    batch = config.get('seed_batch')
    _require(type(batch) is int and 0 <= batch <= 999 and config.get('seed') == 901001 + batch * 1_000_000,
             'Invalid reported seed batch')


def load_prepared(config):
    """Return exact production maps, retained original fitting data, and provenance."""
    validate_configuration(config)
    dimension = config['dimension']
    saved, manifest, entry = _load_asset(dimension)
    _require(config.get('reported_map_asset_sha256') == entry['sha256']
             and config.get('reported_map_asset') == f'studies/zero_centre/maps/{entry["file"]}'
             and config.get('reported_map_manifest_sha256') == _sha(ASSETS / 'manifest.json'),
             'Reported configuration refers to different map assets')
    matched = config['protocol'] in PROTOCOLS.values()
    maps = copy.deepcopy(saved['maps'])
    ball_radius = MATCHED[dimension]['DCS-BW']['R'] if matched else FROZEN[dimension]['scale']
    maps['ball'].update(mu=[0.] * dimension, delta=[0.] * dimension, scale=ball_radius)
    fitting = copy.deepcopy(saved['fitting'])
    for family, fit in fitting.items():
        fit.update(original_fit_seconds=fit['seconds'], seconds=0., map_reused_without_refitting=True,
            original_fitted_map=copy.deepcopy(saved['maps'][family]), production_input_map=copy.deepcopy(maps[family]),
            production_map_calibrated_after_vi=(family == 'ball'),
            production_centre_override_applied=(family == 'ball'),
            fit_constrained_delta_zero=False, fit_constrained_mu_zero=False,
            reuse_note='Original joint VI loss history is retained unchanged. Zero-centre/deformation '
                'and prior DCS radius/step selection are production interventions, not constrained VI refitting.')
    calibrated = config['protocol'] in tuple(CALIBRATED_PROTOCOLS.values()) + tuple(ZERO_PROTOCOLS.values()) + tuple(PROTOCOLS.values())
    zero_scs = matched or config['protocol'] in ZERO_PROTOCOLS.values()
    if calibrated:
        maps['sphere_calibrated'] = copy.deepcopy(saved['maps']['sphere'])
        maps['sphere_calibrated']['scale'] = (MATCHED[dimension]['SCS']['R'] if matched else
                                             CALIBRATED_SCS[dimension]['R'])
        if zero_scs:
            maps['sphere_calibrated'].update(mu=[0.] * dimension, observer=[0.] * dimension)
    provenance = dict(schema='reported-zero-centre-prepared-inputs-v1',
        map_asset=config['reported_map_asset'], map_asset_sha256=entry['sha256'], map_asset_bytes=entry['bytes'],
        map_manifest_sha256=config['reported_map_manifest_sha256'],
        original_fitted_maps=copy.deepcopy(saved['maps']),
        original_fitting=copy.deepcopy(saved['fitting']),
        original_author_verification=copy.deepcopy(saved['direct_author_verification']),
        production_input_maps=copy.deepcopy(maps), new_fitting_seconds=0., new_tuning_seconds=0.,
        development_provenance=copy.deepcopy(manifest['development_provenance']),
        production_parameters=copy.deepcopy(config['reported_production_parameters']))
    if calibrated:
        provenance.update(schema='reported-zero-centre-prepared-inputs-v2',
            calibration_asset=config['reported_calibration_asset'],
            calibration_asset_sha256=(MATCHED_CALIBRATION_SHA256 if matched else CALIBRATION_SHA256),
            joint_calibration=(_load_matched_calibration() if matched else _load_calibration()),
            calibrated_scs_production_parameters=copy.deepcopy(config['reported_calibrated_scs_parameters']))
        if zero_scs:
            provenance.update(schema='reported-zero-centre-prepared-inputs-v3',
                calibrated_scs_map_intervention=copy.deepcopy(config['calibrated_scs_map_intervention']))
    if matched:
        provenance.update(schema='reported-zero-centre-prepared-inputs-v5',
            original_asset_development_provenance=copy.deepcopy(manifest['development_provenance']),
            development_provenance=dict(scope='DCS and calibrated SCS use equally budgeted R/h pilots '
                'on their final zero maps, with the same physical starting cloud, radial objective '
                'and selection rule. Native VI optimizers and map geometries are not identical. '
                'Original SCS remains the unchanged author-fitted h=0.1 baseline.',
                calibration_asset=config['reported_calibration_asset'],
                calibration_asset_sha256=MATCHED_CALIBRATION_SHA256),
            preparation_script=config['preparation_script'], preparation_seed=config['preparation_seed'],
            preparation_fit_seeds=copy.deepcopy(config['preparation_fit_seeds']),
            production_maps_calibrated_before_sampling=True, production_map_change_after_pilot=False)
        for family, fit in fitting.items():
            fit['reuse_note'] = ('Original VI losses are retained unchanged. Final zero-map R/h '
                'calibration is recorded separately and uses the exact production map. Original SCS '
                'retains its unchanged author fit; no new fitting runs during sampling.')
    return dict(maps=maps, fitting=fitting, provenance=provenance)


def validate_reported_summary(summary):
    """Reject reports that mislabel the original SCS or declared DCS maps."""
    from scripts import reproduce_scs_100d as runner
    expected = load_prepared(summary['config'])
    _require(summary.get('maps') == expected['maps'] and summary.get('fitting') == expected['fitting'],
             'Reported summary maps or fitting provenance changed')
    _require(summary.get('method_maps') == runner.effective_method_maps(summary['config'], expected['maps']),
             'Reported effective method maps changed')
    _require(summary.get('prepared_input_provenance') == expected['provenance'],
             'Reported prepared-input provenance changed')


def run(config, output):
    from scripts import reproduce_scs_100d as runner
    started = time.perf_counter()
    prepared = load_prepared(config)
    return runner.run(config, output, prepared_inputs=prepared,
                      prepared_input_load_seconds=time.perf_counter() - started)
