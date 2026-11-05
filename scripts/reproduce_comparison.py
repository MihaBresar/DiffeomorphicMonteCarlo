"""Current comparisons with author SCS VI and method-specific calibration."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_scs_100d as runner
from dcs_bench.comparison_targets import TARGET_IDS, comparison_target

COMPARISONS = ('jax', 'numba')
METHODS = {backend: ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect', 'DCS-Billiard']
           for backend in COMPARISONS}


def configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
                  target='skew-t-100d', vi='on'):
    if comparison not in COMPARISONS:
        raise ValueError('comparison must be jax or numba')
    if vi != 'on':
        raise ValueError('New sampler comparisons require VI for both map families; VI-off runs are no longer offered.')
    selected = comparison_target(target)
    supplied = dict(overrides or {})
    scs_step_rule = supplied.pop('scs_step_rule', None) or 'fixed'
    if scs_step_rule not in ('acceptance', 'fixed'):
        raise ValueError('scs_step_rule must be acceptance or fixed')
    reference_samples = supplied.pop('reference_samples', None)
    reference_seed = supplied.pop('reference_seed', None)
    config = runner.configuration(preset, workers, supplied)
    if config['warmup'] < 8:
        raise ValueError('The radial-pilot comparison requires at least eight warmup transitions')
    reference_samples = ({'quick': 50_000, 'reproduce': 1_000_000, 'paper': 5_000_000}[preset]
                         if reference_samples is None else reference_samples)
    reference_seed = 740317 if reference_seed is None else reference_seed
    if isinstance(reference_samples, bool) or not isinstance(reference_samples, int) or reference_samples < 2:
        raise ValueError('reference_samples must be an integer of at least two')
    if isinstance(reference_seed, bool) or not isinstance(reference_seed, int) or not 0 <= reference_seed <= 2**32-1:
        raise ValueError('reference_seed must be an integer from 0 to 2**32-1')
    spec = selected['target']
    config.update(experiment='sampler-comparison' if comparison == 'jax' else 'sampler-comparison-numba',
        protocol='timed-sampler-comparison-v10' if comparison == 'jax' else 'timed-numba-comparison-v10',
        comparison=comparison, dcs_backend=comparison, target_id=target, target_title=selected['title'],
        target=spec, dimension=spec['dimension'], nu=spec.get('nu'), alpha=spec.get('alpha', {}),
        beta=selected['beta'],
        threshold=selected['threshold'], threshold_policy=selected['threshold_policy'],
        coordinate_metrics=False, ess_enabled=True,
        reference_samples=reference_samples, reference_seed=reference_seed, reference_batch_size=8192,
        vi='on', vi_enabled=True, ball_map_policy='mobius',
        sphere_map_policy='author-paper-vi',
        scs_vi=dict(optimizer_steps=config['optimizer_steps'],
                    optimizer_samples=config['optimizer_samples']),
        algorithms=list(METHODS[comparison]),
        implementation_revision='jax-matched-fixed-vi-v10' if comparison == 'jax' else 'numba-matched-fixed-vi-v10',
        step_policy='separate_method_specific_pretuning_then_frozen',
        step_tuning_policies={'DCS-BW-reflect': 'ball_esjd_pilot_v1',
            'DCS-Billiard': 'own_kernel_radial_pilot_v1',
            'SCS': 'indicator_dual_averaging_0234' if scs_step_rule == 'acceptance' else 'author_scs_fixed_01'},
        scs_step_rule=scs_step_rule,
        fitting_policy=(f'DCS: fixed-cloud reverse KL, seed {config["fit_seed"]}, gradient-norm cap 10, no coordinate/CDF clipping. '
                        'SCS: unchanged author VI, fixed base batch, seed 0, coordinate clip 1000, no gradient clipping. '
                        'Both use Adam 0.01, fixed reference clouds, equal update counts and equal cloud sizes; '
                        'base distributions and parameterizations differ. Fitting costs are recorded per family.'),
        target_note=selected['reference_note'],
        tuning_note='Every method receives the same separate warmup-transition budget. DCS-RWM and '
            'DCS-BW use acceptance-target 0.234 dual averaging. Reflected BW retains its '
            'ball-movement pilots without a geometric step ceiling of 2. DCS-Billiard calibrates its own '
            'production kernel using bounded squared changes of physical log(1+norm), including zero after rejection. '
            'SCS uses acceptance-target 0.234 dual averaging; this follows the paper\'s general acceptance guidance, '
            'not the fixed h=0.1 used in its skew-t experiment. '
            'The final pilot state and selected step initialize production; all steps freeze before sampling.')
    config['billiard_map_policy'] = 'post_mobius_retained_joint_fit_centre_scale_beta_delta_zero'
    if scs_step_rule == 'fixed':
        config['tuning_note'] = config['tuning_note'].replace(
            'SCS uses acceptance-target 0.234 dual averaging; this follows the paper\'s general acceptance guidance, '
            'not the fixed h=0.1 used in its skew-t experiment.',
            'SCS uses the paper skew-t fixed h=0.1, with the same discarded warmup budget and no step adaptation.')
    config['time_scope'] = ('Equal per-chain sampling time includes first-third burn-in, every transition and '
        'all-state radial estimates and unthinned scalar capture for ESS. VI, separate step calibration, compilation, synchronization waits, ESS computation and '
        'file/report generation are excluded and recorded separately. ' +
        ('All methods use JAX full-state scans.' if comparison == 'jax' else
         'DCS uses optimized Numba density/radius caching and fused summaries; SCS uses the unmodified author '
         'JAX transition and full-state processing. This is an implementation-efficiency comparison, '
         'not a common-framework comparison. JAX pretuning is shared with the other comparison.'))
    config['reference_policy'] = 'Analytic Student-t radial references; no coordinate-reference simulation'
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', choices=COMPARISONS, default='jax')
    parser.add_argument('--target', choices=TARGET_IDS, default='skew-t-100d')
    parser.add_argument('--vi', choices=('on',), default='on', help=argparse.SUPPRESS)
    parser.add_argument('--scs-step-rule', choices=('acceptance', 'fixed'))
    parser.add_argument('--preset', choices=runner.PRESETS, default='quick')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--time-budget', type=float)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seed-batch', type=int)
    for name in ('runs', 'warmup', 'thin', 'optimizer-steps', 'optimizer-samples', 'fit-seed',
                 'reference-samples', 'reference-seed'):
        parser.add_argument('--'+name, type=int)
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in vars(args).items() if key not in
                 ('comparison', 'target', 'vi', 'preset', 'workers', 'output_dir', 'dry_run')}
    config = configuration(args.preset, args.workers, overrides, comparison=args.comparison,
                           target=args.target, vi=args.vi)
    output = args.output_dir or ROOT/'results'/config['experiment']/(
        args.target+'-vi-'+args.vi+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    if args.dry_run:
        print(json.dumps({'output_dir': str(output), 'configuration': config}, indent=2))
        return 0
    print(f'Results: {runner.run(config, output)}', flush=True)
    return 0


if __name__ == '__main__':
    if os.name == 'nt':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
