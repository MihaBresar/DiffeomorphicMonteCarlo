"""Archived step-policy control using a completed author-VI run's exact maps.

This helper preserves historical sensitivity studies; it is not an active
baseline selector for the current original-SCS-only five-method comparison.
The primary configuration fixes the target, map fit, seeds, chain and pilot
budgets, production clocks, ESS and CPU allocation. Only the SCS step policy changes; warmup retains the same transition budget. This script never refits a map or overwrites an existing output folder.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_scs_100d as runner
from dcs_bench.scs_paper import verify_vendor

METHODS = ['SCS']
PROTOCOLS = {'timed-sampler-comparison-v9': 'jax',
             'timed-numba-comparison-v9': 'numba',
             'timed-sampler-comparison-v10': 'jax',
             'timed-numba-comparison-v10': 'numba',
             'timed-sampler-comparison-v11': 'jax',
             'timed-numba-comparison-v11': 'numba'}


def _read(path):
    def nonfinite(value):
        raise ValueError(f'Nonfinite JSON value in {path.name}: {value}')
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, parse_constant=nonfinite)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_control(primary):
    """Validate saved provenance and return independent control inputs."""
    primary = Path(primary).resolve()
    filenames = ('configuration.json', 'manifest.json', 'maps.json',
                 'method_maps.json', 'summary.json')
    data = {name: _read(primary/name) for name in filenames}
    config, manifest, artifact, summary = (data[name] for name in
        ('configuration.json', 'manifest.json', 'maps.json', 'summary.json'))
    protocol = config.get('protocol')
    if protocol not in PROTOCOLS or manifest.get('status') != 'complete':
        raise ValueError('The primary must be a completed version-9, version-10 or version-11 sampler comparison')
    backend = PROTOCOLS[protocol]
    if config != manifest.get('config') or config != summary.get('config') or manifest.get('protocol') != protocol:
        raise ValueError('Primary configuration, manifest and summary do not agree')
    previous_sources, current_sources = manifest.get('source_hashes'), runner.source_hashes()
    if not isinstance(previous_sources, dict) or set(previous_sources) != set(current_sources):
        raise ValueError('Primary scientific sources differ from this checkout; use the same source version or run a fresh primary')
    source_changes = {name: {'primary': previous_sources[name], 'current': current_sources[name]}
                      for name in current_sources if previous_sources[name] != current_sources[name]}
    if set(source_changes)-{'scripts/scs_100d_figures.py'}:
        raise ValueError('Primary scientific sources differ from this checkout; use the same source version or run a fresh primary')
    verify_vendor()
    if config.get('dcs_backend') != backend or config.get('comparison') != backend:
        raise ValueError('Primary backend does not match its protocol')
    if config.get('coordinate_metrics') is not False or config.get('ess_enabled') is not True:
        raise ValueError('The primary must use the recorded radial and ESS settings')
    if not all(method in config.get('algorithms', []) for method in METHODS):
        raise ValueError('Primary results must include SCS')
    policies = config.get('step_tuning_policies', {})
    primary_policy = policies.get('SCS')
    if primary_policy not in ('indicator_dual_averaging_0234', 'author_scs_fixed_01'):
        raise ValueError('The primary must use SCS acceptance tuning or fixed h=0.1')
    maps, fitting = artifact.get('maps'), artifact.get('fitting')
    if not isinstance(maps, dict) or set(maps) != {'ball', 'sphere'} or not isinstance(fitting, dict) or set(fitting) != {'ball', 'sphere'}:
        raise ValueError('Primary ball and sphere maps and fit records are required')
    if maps != summary.get('maps') or fitting != summary.get('fitting'):
        raise ValueError('Primary map artifact and summary do not agree')
    if protocol.endswith(('-v10', '-v11')):
        steps, draws = config.get('optimizer_steps'), config.get('optimizer_samples')
        if (config.get('vi') != 'on' or config.get('vi_batch_policy') != 'fixed'
                or config.get('vi_budget_policy') != 'equal_updates_and_fixed_reference_clouds_v1'
                or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
                or not isinstance(draws, int) or isinstance(draws, bool) or draws < 1
                or config.get('scs_vi') != dict(optimizer_steps=steps, optimizer_samples=draws)):
            raise ValueError('Version-10/11 primary requires matched fixed-cloud VI budgets')
        for fit in fitting.values():
            if (fit.get('steps') != steps or fit.get('training_draws') != draws
                    or fit.get('batch_policy') != 'fixed' or fit.get('total_training_draws') != draws
                    or fit.get('gradient_updates') != steps or fit.get('unique_reference_draws') != draws
                    or fit.get('objective_point_evaluations') != steps*draws):
                raise ValueError('Version-10/11 primary requires actual matched fixed-cloud fitting records')
    expected = runner.effective_method_maps(config, maps)
    for saved in (artifact.get('method_maps'), data['method_maps.json'],
                  summary.get('method_maps'), manifest.get('method_maps')):
        if expected != saved:
            raise ValueError('Primary effective-map provenance does not agree')
    for method in METHODS:
        case = summary.get('cases', {}).get(method, {})
        rows = case.get('records', [])
        if case.get('chains') != config['runs'] or len(rows) != config['runs']:
            raise ValueError('Primary control methods must have every completed chain')
        if sorted(row.get('chain', -1) for row in rows) != list(range(config['runs'])):
            raise ValueError('Primary chain identifiers are missing or duplicated')
        for row in rows:
            n = row.get('iterations', 0)
            if (row.get('algorithm') != method or row.get('seed') != config['seed']+row['chain'] or
                    row.get('warmup_iterations') != config['warmup'] or
                    row.get('tuning_policy') != primary_policy or
                    n <= 0 or n % 300 or row.get('estimation_iterations') != n-n//3 or
                    row.get('sampling_seconds', 0.) < config['time_budget'] or
                    row.get('adaptation_during_sampling') is not False):
                raise ValueError('Primary chain budget, seed or acceptance-tuning provenance is inconsistent')
    hardware = runner.cpu_info()
    runner.validate_workers(config['workers'], hardware)
    affinity = ('fixed_cpu_prefix_per_wave' if all(cpu is not None for cpu in hardware['cpu_ids'])
                else 'os_managed_no_strict_pinning')
    if (config.get('execution_revision') != runner.EXECUTION_REVISION or
            config.get('wave_cpu_ids') != runner._wave_cpu_ids(config, hardware) or
            config.get('affinity_policy') != affinity):
        raise ValueError('CPU allocation differs from the primary; run a fresh primary on this allocation first')
    control = deepcopy(config)
    alternative = ('author_scs_fixed_01' if primary_policy == 'indicator_dual_averaging_0234'
                   else 'indicator_dual_averaging_0234')
    control.update(algorithms=METHODS.copy(),
        step_tuning_policies={'SCS': alternative}, initial_scale=.1,
        scs_step_rule='fixed' if alternative == 'author_scs_fixed_01' else 'acceptance',
        tuning_note=('Paper skew-t fixed h=0.1. ' if alternative == 'author_scs_fixed_01' else
                     'Acceptance dual averaging toward 0.234. ') + 'The same discarded warmup '
            'transition budget, exact maps, seeds, production settings and ESS are retained.',
        calibration_control='Fixed SCS h=0.1 versus acceptance-tuned SCS; no map refit',
        calibration_control_provenance=dict(primary_directory=str(primary),
            primary_protocol=protocol, primary_files_sha256={name: _digest(primary/name) for name in filenames},
            control_script_sha256=_digest(Path(__file__)), maps_reused_without_refitting=True,
            presentation_source_differences=source_changes,
            cpu_allocation_policy='Require the same ordered CPU allocation and worker count as the primary'))
    reused_fit = deepcopy(fitting)
    for entry in reused_fit.values():
        entry.update(original_fit_seconds=entry.get('seconds', 0.), seconds=0.,
            map_reused_without_refitting=True, reused_map_source=str(primary/'maps.json'))
    return control, deepcopy(maps), reused_fit


def run(primary, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output folder: {output}')
    config, maps, fitting = prepare_control(primary)
    output.mkdir(parents=True, exist_ok=False)
    original_prepare = runner.prepare
    def reuse_maps(_config):
        return deepcopy(maps), deepcopy(fitting)
    try:
        runner.prepare = reuse_maps
        return runner.run(config, output)
    finally:
        runner.prepare = original_prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('primary', type=Path, help='Completed version-9, version-10 or version-11 primary results folder')
    parser.add_argument('--output-dir', required=True, type=Path, help='New folder for the alternative SCS step rule')
    args = parser.parse_args(argv)
    print(f'Control results: {run(args.primary, args.output_dir)}')
    return 0


if __name__ == '__main__':
    if sys.platform == 'win32':
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    raise SystemExit(main())
