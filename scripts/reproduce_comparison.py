"""Current selectable-target comparisons, with shared VI on or off."""
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
METHODS = {'jax': ['DCS-RWM', 'DCS-BW', 'SCS', 'DCS-BW-reflect'],
           'numba': ['DCS-BW', 'SCS', 'DCS-BW-reflect']}


def configuration(preset='quick', workers=1, overrides=None, *, comparison='jax',
                  target='skew-t-100d', vi='on'):
    if comparison not in COMPARISONS:
        raise ValueError('comparison must be jax or numba')
    if vi not in ('on', 'off'):
        raise ValueError('vi must be on or off')
    selected = comparison_target(target)
    config = runner.configuration(preset, workers, overrides)
    spec = selected['target']
    config.update(experiment='sampler-comparison' if comparison == 'jax' else 'sampler-comparison-numba',
        protocol='timed-sampler-comparison-v1' if comparison == 'jax' else 'timed-numba-comparison-v1',
        comparison=comparison, dcs_backend=comparison, target_id=target, target_title=selected['title'],
        target=spec, dimension=spec['dimension'], nu=spec.get('nu'), alpha=spec.get('alpha', {}),
        beta=selected['beta'],
        threshold=selected['threshold'], threshold_policy=selected['threshold_policy'],
        vi=vi, vi_enabled=vi == 'on', ball_map_policy='mobius' if vi == 'on' else 'vanilla',
        sphere_map_policy='vi' if vi == 'on' else 'vanilla',
        algorithms=list(METHODS[comparison]),
        implementation_revision='shared-jax-reflection-v1' if comparison == 'jax' else 'optimized-numba-ball-v1',
        step_policy='separate_method_specific_pretuning_then_frozen',
        step_tuning_policies={'DCS-BW-reflect': 'ball_esjd_pilot_v1'},
        fitting_policy=('Shared fresh-batch reverse KL, Adam 0.01, gradient-norm cap 10; no coordinate/CDF clipping'
                        if vi == 'on' else 'No VI for either family: R=1, centre=0, deformation=0'),
        target_note=selected['reference_note'],
        tuning_note='Every method receives the same separate warmup-transition budget. Reflected BW selects '
            'by ball-movement pilots; other methods use acceptance-target 0.234 dual averaging. All steps freeze '
            'before sampling. VI off leaves both maps untuned but still calibrates every proposal step.')
    config['time_scope'] = ('Equal per-chain sampling time includes first-third burn-in, every transition and '
        'all-state radial estimates. VI, separate step calibration, compilation, synchronization waits and '
        'file/report generation are excluded and recorded separately. ' +
        ('All methods use JAX full-state scans.' if comparison == 'jax' else
         'DCS uses optimized Numba density/radius caching and fused summaries; SCS uses the unmodified author '
         'JAX transition and full-state processing. This is an implementation-efficiency comparison, '
         'not a common-framework comparison. JAX pretuning is shared with the other comparison.'))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', choices=COMPARISONS, default='jax')
    parser.add_argument('--target', choices=TARGET_IDS, default='skew-t-100d')
    parser.add_argument('--vi', choices=('on', 'off'), default='on')
    parser.add_argument('--preset', choices=runner.PRESETS, default='quick')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--time-budget', type=float)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seed-batch', type=int)
    for name in ('runs', 'warmup', 'thin', 'optimizer-steps', 'optimizer-samples', 'fit-seed'):
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
