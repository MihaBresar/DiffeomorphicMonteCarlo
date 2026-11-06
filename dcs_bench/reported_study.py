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
PROTOCOLS = {'jax': 'timed-reported-zero-centre-v1',
             'numba': 'timed-reported-zero-centre-numba-v1'}
PRESETS = {'quick': dict(runs=2, time_budget=.05),
           'reproduce': dict(runs=20, time_budget=3.),
           'paper': dict(runs=50, time_budget=20.)}
METHODS = ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect', 'DCS-Billiard']
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


def configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
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
        protocol=PROTOCOLS[comparison], implementation_revision='reported-zero-centre-frozen-inputs-v1',
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


def validate_configuration(config):
    """Validate scientific settings without requiring the original computer."""
    backend = config.get('comparison')
    target_id = config.get('target_id')
    _require(backend in PROTOCOLS and config.get('protocol') == PROTOCOLS[backend], 'Unknown reported-study protocol')
    _require(target_id in TARGETS and config.get('preset') in PRESETS, 'Unknown reported-study target or preset')
    d = TARGETS[target_id]
    _require(config.get('target') == dict(kind='skew_t', dimension=d, nu=3., alpha={'0': 100., '1': -100.})
             and config.get('dimension') == d and config.get('beta') == 2. and config.get('latitude') == 1.1,
             'Reported target or map geometry changed')
    _require(config.get('threshold') == 50. * math.sqrt(d / 100.), 'Reported tail threshold changed')
    for name, value in {**PRESETS[config['preset']], 'warmup': 1500, 'thin': 50,
                        'optimizer_steps': 1000, 'optimizer_samples': 2000}.items():
        _require(config.get(name) == value, f'Reported preset has a different {name}')
    _require(config.get('algorithms') == METHODS and config.get('dcs_backend') == backend,
             'Reported method list or backend changed')
    _require(config.get('dcs_step_rule') == 'fixed' and config.get('dcs_fixed_move') == FROZEN[d]['h']
             and config.get('scs_step_rule') == 'fixed' and config.get('initial_scale') == .1
             and config.get('step_tuning_policies') == {m: 'author_scs_fixed_01' if m == 'SCS' else 'fixed_dcs_move_v1' for m in METHODS},
             'Reported proposal steps are frozen')
    _require(not any(key in config for key in ('scs_fixed_move', 'step_scale_ceilings', 'step_scale_ceiling')),
             'Reported steps do not accept alternate scales or ceilings')
    _require(config.get('reported_production_parameters') == dict(dcs_R=FROZEN[d]['scale'], dcs_h=FROZEN[d]['h'],
             dcs_mu='zero', dcs_delta='zero', dcs_beta=2., scs_map='unchanged original author VI', scs_h=.1),
             'Reported production parameter metadata changed')
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
    maps = copy.deepcopy(saved['maps'])
    maps['ball'].update(mu=[0.] * dimension, delta=[0.] * dimension, scale=FROZEN[dimension]['scale'])
    fitting = copy.deepcopy(saved['fitting'])
    for family, fit in fitting.items():
        fit.update(original_fit_seconds=fit['seconds'], seconds=0., map_reused_without_refitting=True,
            original_fitted_map=copy.deepcopy(saved['maps'][family]), production_input_map=copy.deepcopy(maps[family]),
            production_map_calibrated_after_vi=(family == 'ball'),
            production_centre_override_applied=(family == 'ball'),
            fit_constrained_delta_zero=False, fit_constrained_mu_zero=False,
            reuse_note='Original joint VI loss history is retained unchanged. Zero-centre/deformation '
                'and prior DCS radius/step selection are production interventions, not constrained VI refitting.')
    provenance = dict(schema='reported-zero-centre-prepared-inputs-v1',
        map_asset=config['reported_map_asset'], map_asset_sha256=entry['sha256'], map_asset_bytes=entry['bytes'],
        map_manifest_sha256=config['reported_map_manifest_sha256'],
        original_fitted_maps=copy.deepcopy(saved['maps']),
        original_fitting=copy.deepcopy(saved['fitting']),
        original_author_verification=copy.deepcopy(saved['direct_author_verification']),
        production_input_maps=copy.deepcopy(maps), new_fitting_seconds=0., new_tuning_seconds=0.,
        development_provenance=copy.deepcopy(manifest['development_provenance']),
        production_parameters=copy.deepcopy(config['reported_production_parameters']))
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
