"""Diagnose fixed-step ranking changes under a common target and chain budget.

This is a controlled bridge between historical and current fitting/tuning
protocols, not an exact recreation of the archived paper images. It never
changes production samplers to obtain a preferred ranking. Run into a fresh
output directory; completed chain files are retained if execution stops.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import cpu_info, pin_current_process, validate_workers, worker_environment
os.environ.update(worker_environment())  # Before NumPy/JAX imports and spawn.
import numpy as np
from dcs_bench.engine import environment, load_config, write_json
from dcs_bench.references import radial_truth
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import DualAveraging, fit_map
from scripts.legacy_fixed_step_loader import LEGACY, load_legacy

PRESETS = {
    'quick': dict(runs=2, iterations=500, warmup=100, thin=1,
                  optimizer_steps=20, optimizer_samples=128),
    'reproduce': dict(runs=8, iterations=25000, warmup=1000, thin=10,
                      optimizer_steps=400, optimizer_samples=500),
    'paper': dict(runs=20, iterations=500000, warmup=1500, thin=50,
                  optimizer_steps=1000, optimizer_samples=2000),
}
FUNCTIONALS = ['tail_indicator_norm_ge_50', 'norm', 'log_one_plus_norm',
               'coordinate_1', 'coordinate_2', 'coordinate_3', 'coordinate_4']
TIME_SCOPE = 'Diagnostic cases share a worker pool; these times are not a throughput comparison.'
_WORKER_ALLOCATION = {'cpu_id': None, 'pinned': False}
DEFAULT_PRODUCTION_SEED_BASE = 901001


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def audit_sources():
    paths = list((ROOT/'dcs_bench').glob('*.py'))
    paths += [Path(__file__), ROOT/'scripts/legacy_fixed_step_loader.py',
              ROOT/'scripts/fixed_step_pilots.py', ROOT/'scripts/fixed_step_figures.py', LEGACY,
              ROOT/'configs/skew-t-100d-fixed.json', ROOT/'requirements-cpu.txt']
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def production_seeds(config):
    base = config.get('production_seed_base', DEFAULT_PRODUCTION_SEED_BASE)
    if isinstance(base, bool) or not isinstance(base, int) or base < 0:
        raise ValueError('production_seed_base must be a nonnegative integer')
    return [base+chain for chain in range(config['runs'])]


def audit_config(preset, workers, production_seed_base=DEFAULT_PRODUCTION_SEED_BASE):
    config = load_config('skew-t-100d-fixed', 'paper', {'workers': workers})
    config.update(PRESETS[preset])
    config.update(preset=preset, seed=20260911, production_seed_base=production_seed_base, ball_map_policy='mobius',
                  ball_scale_policy='fit', vi_batch_policy='fixed',
                  map_scale_initialization='radial_median')
    # Keep this bridge's target explicit even if the main experiment changes.
    config['targets'] = [{'kind': 'skew_t', 'dimension': 100, 'nu': 3,
                          'alpha': {'0': 100., '1': -100.}}]
    config.update(beta=2., latitude=1.1, threshold=50.)
    config['scs_vi'] = dict(optimizer_steps=config['optimizer_steps'],
                            optimizer_samples=config['optimizer_samples'])
    config['sphere_map_policy'] = 'author-paper-vi'
    production_seeds(config)
    return config


def case_definitions(include_pilots=True):
    cases = []
    for source in ('original', 'current', 'vanilla'):
        for algorithm in (('DCS-BW',) if source == 'vanilla' else ('DCS-BW', 'SCS')):
            policies = (('raw_frozen',) if source == 'vanilla' else
                        ('raw_frozen', 'smoothed_frozen', 'smoothed_continuing')
                        if algorithm == 'DCS-BW' else ('raw_frozen', 'fixed_0.1'))
            for policy in policies:
                cases.append({'case': f'{source}-{algorithm}-{policy}', 'source': source,
                              'algorithm': algorithm, 'policy': policy})
    if include_pilots:
        for source, algorithm, pilot_key in [('current', 'DCS-BW', 'mobius-DCS-BW'),
                ('current', 'SCS', 'current-SCS'), ('vanilla', 'DCS-BW', 'vanilla-DCS-BW')]:
            cases.append({'case': f'{source}-{algorithm}-fixed_pilot', 'source': source,
                          'algorithm': algorithm, 'policy': 'fixed_pilot', 'pilot_key': pilot_key})
    return cases


def prepare(root, config, fingerprint):
    target = Target(config['targets'][0])
    old = load_legacy(nu=3., dimension=100, optimizer_steps=config['optimizer_steps'],
                      optimizer_samples=config['optimizer_samples'])
    fits = {'config': config, 'fingerprint': fingerprint, 'maps': {}, 'fit_metadata': {}}
    for source in ('original', 'current'):
        fits['maps'][source], fits['fit_metadata'][source] = {}, {}
        for family in ('ball', 'sphere'):
            print(f'Fitting {source} {family} map', flush=True)
            started = time.perf_counter()
            if family == 'sphere' or source == 'current':
                mapping, metadata = fit_map(target, config, family, 20260911+811)
                if family == 'sphere':
                    metadata = dict(metadata, audit_fit_note='Original author SCS VI in both DCS fitting-control groups')
            else:
                mu, delta, scale = old['optimize_prwm_hyperparameters']()
                mapping = BallMap(100, beta=2., mu=mu, delta=delta, scale=scale)
                metadata = {'seed': 42, 'steps': config['optimizer_steps']+1,
                            'training_draws': config['optimizer_samples'],
                            'total_training_draws': (config['optimizer_steps']+1)*config['optimizer_samples'],
                            'seconds': time.perf_counter()-started,
                            'source': 'Historical VI function with common nu=3 target',
                            'cdf_argument_clipping': [-35., 35.]}
            fits['maps'][source][family] = mapping.parameters()
            fits['fit_metadata'][source][family] = metadata
    fits['maps']['vanilla'] = {'ball': BallMap(100, beta=2.).parameters()}
    fits['fit_metadata']['vanilla'] = {'ball': {'steps': 0, 'training_draws': 0,
        'total_training_draws': 0, 'mu': 0., 'scale': 1., 'delta': 0.,
        'description': 'Fully untuned map; proposal scale still receives common warmup.'}}
    fits['setup_digest'] = digest(fits)
    write_json(root/'maps.json', fits)
    return fits


def initialize_worker(slots):
    global _WORKER_ALLOCATION
    _WORKER_ALLOCATION = pin_current_process(slots.get())


def make_jobs(root, config, fits):
    jobs = []
    for case in case_definitions():
        for chain, seed in enumerate(production_seeds(config)):
            family = 'sphere' if case['algorithm'] == 'SCS' else 'ball'
            job = {**case, 'chain': chain, 'seed': seed,
                'production_seed_base': config.get('production_seed_base', DEFAULT_PRODUCTION_SEED_BASE),
                'target': config['targets'][0], 'map': fits['maps'][case['source']][family],
                'warmup': config['warmup'], 'iterations': config['iterations'], 'thin': config['thin'],
                'fingerprint': fits['fingerprint'], 'setup_digest': fits['setup_digest'],
                'output': str(root/'chains'/case['case']/f'{chain:03d}.npz')}
            if case['policy'] == 'fixed_pilot':
                selection = fits['pilot_selection'][case['pilot_key']]
                job.update(fixed_scale=selection['scale'],
                           pilot_selection_valid=selection['valid_selection'],
                           pilot_endpoint_winner=selection['endpoint_winner'])
            jobs.append(job)
    return jobs


def chain_job(job):
    output = Path(job['output'])
    if output.exists():
        raise FileExistsError(f'Audit checkpoint already exists: {output}')
    target = Target(job['target'])
    mapping = (SphereMap if job['algorithm'] == 'SCS' else BallMap)(**job['map'])
    kernel = MetropolisKernel(target, job['algorithm'], mapping, job['seed'])
    adapter = DualAveraging(.1)
    policy = job['policy']
    scale = float(job['fixed_scale']) if policy == 'fixed_pilot' else .1
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('The fixed pilot scale must be positive and finite')
    if policy not in ('raw_frozen', 'smoothed_frozen', 'smoothed_continuing', 'fixed_0.1', 'fixed_pilot'):
        raise ValueError(f'Unknown audit adaptation policy: {policy}')
    start = time.perf_counter()
    for _ in range(job['warmup']):
        probability, _ = kernel.step(scale)
        if policy not in ('fixed_0.1', 'fixed_pilot'):
            scale = adapter.update(probability)
            if policy in ('smoothed_continuing', 'smoothed_frozen'):
                scale = adapter.final()
    if policy not in ('fixed_0.1', 'fixed_pilot'):
        scale = adapter.final()
    warmup_seconds, warm_scale = time.perf_counter()-start, scale
    traces, checkpoints, accepted = [], [], 0
    checkpoint_iterations = sorted({job['iterations'],
        *(max(job['thin'], int(job['iterations']*fraction)//job['thin']*job['thin'])
          for fraction in (.05, .2, .5))})
    started = time.perf_counter()
    for iteration in range(1, job['iterations']+1):
        probability, did_accept = kernel.step(scale)
        accepted += did_accept
        if policy == 'smoothed_continuing':
            adapter.update(probability)
            scale = adapter.final()
        if iteration % job['thin'] == 0:
            x = kernel.draw()
            norm = np.linalg.norm(x)
            traces.append([float(norm >= 50.), float(norm), float(np.log1p(norm)), *x[:4].tolist()])
        if iteration in checkpoint_iterations:
            checkpoints.append({'iterations': iteration, 'retained_draws': len(traces),
                                'means': np.mean(traces, axis=0).tolist(), 'scale': scale})
    traces = np.asarray(traces)
    if not np.all(np.isfinite(traces)):
        raise FloatingPointError(f'Nonfinite audit functionals: {job["case"]}, chain {job["chain"]}')
    record = {**{key: value for key, value in job.items() if key not in ('map', 'target', 'output')},
        'map_digest': digest(job['map']), 'target_digest': digest(job['target']),
        'initial_physical_state': 'ones(100), using the current exact inverse for every map',
        'warmup_scale': warm_scale, 'final_scale': scale, 'acceptance': accepted/job['iterations'],
        'retained_draws': len(traces), 'functionals': FUNCTIONALS, 'mean': traces.mean(axis=0).tolist(),
        'checkpoints': checkpoints, 'warmup_seconds': warmup_seconds,
        'seconds': time.perf_counter()-started, 'time_scope': TIME_SCOPE,
        'worker': {'pid': os.getpid(), **_WORKER_ALLOCATION},
        'adaptation_during_sampling': policy == 'smoothed_continuing'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, functionals=traces, metadata=np.array(json.dumps(record, allow_nan=False)))
    os.replace(temporary, output)
    return job['case'], job['chain']


def functional_diagnostics(values):
    """Rank diagnostics from saved thinned traces, without reconstructing draws.

    The first six functions match the held-out scale-evaluation report. Raw
    coordinate diagnostics additionally use all seven saved functional columns.
    """
    import arviz as az
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[-1] != len(FUNCTIONALS):
        raise ValueError('Expected (chains, retained draws, seven functionals)')
    functions = {'tail_50': values[:, :, 0], 'mean_norm': values[:, :, 1],
        'log1p_norm': values[:, :, 2],
        'asinh_contrast': np.arcsinh((values[:, :, 3]-values[:, :, 4])/np.sqrt(2)),
        'asinh_x3': np.arcsinh(values[:, :, 5]), 'asinh_x4': np.arcsinh(values[:, :, 6]),
        **{f'coordinate_{index+1}': values[:, :, index+3] for index in range(4)}}
    ess, rhat, unavailable = {}, {}, {}
    for name, array in functions.items():
        reason = None
        if array.shape[0] < 2 or array.shape[1] < 4:
            reason = 'At least two chains and four retained draws per chain are required.'
        elif not np.all(np.isfinite(array)):
            reason = 'Nonfinite saved or derived functional.'
        elif np.any(np.ptp(array, axis=1) == 0):
            reason = 'At least one chain has a constant functional.'
        if reason is None:
            es = float(az.ess(array, method='bulk'))
            rh = float(az.rhat(array, method='rank'))
            if np.isfinite(es) and np.isfinite(rh):
                ess[name], rhat[name] = es, rh
                continue
            reason = 'ArviZ returned a nonfinite diagnostic.'
        ess[name] = rhat[name] = None
        unavailable[name] = reason
    total = int(values.shape[0]*values.shape[1])
    return {'bulk_ess': ess, 'rank_rhat': rhat,
        'bulk_ess_per_retained_draw': {name: value/total if value is not None else None for name, value in ess.items()},
        'diagnostic_draw_counts': {'chains': int(values.shape[0]),
            'retained_per_chain': int(values.shape[1]), 'total_retained': total,
            'ratio_denominator': 'Total thinned retained draws across chains, not original production iterations.'},
        'unavailable_diagnostics': unavailable,
        'diagnostic_methods': {'ess': 'ArviZ multi-chain bulk ESS', 'rhat': 'ArviZ rank-normalized split R-hat',
            'function_definitions': {'tail_50': 'I(norm >= 50)', 'mean_norm': 'norm',
                'log1p_norm': 'log(1+norm)', 'asinh_contrast': 'asinh((x1-x2)/sqrt(2))',
                'asinh_x3': 'asinh(x3)', 'asinh_x4': 'asinh(x4)'},
            'interpretation': 'Diagnostics use retained thinned traces and are conditional on the fitted maps. '
                              'Continuing-adaptation cases are historical diagnostics, not frozen production kernels.'}}


def summarize(root):
    fits = json.loads((root/'maps.json').read_text(encoding='utf-8'))
    config = fits['config']
    names, truth, errors = radial_truth(Target(config['targets'][0]), 50.)
    summary = {'fingerprint': fits['fingerprint'], 'setup_digest': fits['setup_digest'],
        'production_seed_base': config.get('production_seed_base', DEFAULT_PRODUCTION_SEED_BASE),
        'chain_seeds': production_seeds(config),
        'truth': truth.tolist(), 'reference': {'functionals': names, 'values': truth.tolist(),
            'numerical_errors': errors, 'method': 'Analytic radial Student-t expectations'},
        'config': config, 'cases': {}, 'pilot_selection': fits.get('pilot_selection'),
        'scope': 'Common nu=3 target, physical initial state ones, equal warmup and production iterations, '
                 'paired seeds, current exact transition evaluators; fitting/tuning ablation, not archived PNG reproduction. '
                 'The original/current source labels distinguish DCS fits; both corresponding SCS control groups use the same author fitting routine.',
        'time_scope': TIME_SCOPE, 'preset_warning': 'Quick is a pipeline check only.' if config['preset'] == 'quick' else None}
    for case in case_definitions():
        records, arrays = [], []
        for chain, seed in enumerate(production_seeds(config)):
            path = root/'chains'/case['case']/f'{chain:03d}.npz'
            with np.load(path, allow_pickle=False) as data:
                record = json.loads(str(data['metadata']))
                if (record['fingerprint'] != fits['fingerprint'] or record['setup_digest'] != fits['setup_digest']
                        or record['case'] != case['case'] or record['chain'] != chain or record['seed'] != seed):
                    raise ValueError(f'Audit checkpoint provenance mismatch: {path}')
                if data['functionals'].shape != (config['iterations']//config['thin'], len(FUNCTIONALS)):
                    raise ValueError(f'Wrong audit trace dimensions: {path}')
                records.append(record)
                arrays.append(data['functionals'].copy())
        values = np.asarray(arrays)
        means = values[:, :, :2].mean(axis=1)
        squared_errors = (means-truth)**2
        summary['cases'][case['case']] = {'chains': len(records), 'mse': squared_errors.mean(axis=0).tolist(),
            'estimates': means.tolist(), 'squared_errors': squared_errors.tolist(),
            'acceptance_mean': float(np.mean([record['acceptance'] for record in records])),
            'warmup_scales': [record['warmup_scale'] for record in records],
            'final_scales': [record['final_scale'] for record in records],
            'iterations_per_chain': config['iterations'], 'retained_draws_per_chain': config['iterations']//config['thin'],
            **functional_diagnostics(values)}
    contrasts = [
        ('current-DCS-BW-raw_frozen', 'original-DCS-BW-raw_frozen'),
        ('current-SCS-raw_frozen', 'original-SCS-raw_frozen'),
        ('current-DCS-BW-raw_frozen', 'current-SCS-raw_frozen'),
        ('original-DCS-BW-smoothed_continuing', 'original-SCS-fixed_0.1'),
        ('vanilla-DCS-BW-raw_frozen', 'current-DCS-BW-raw_frozen'),
        ('vanilla-DCS-BW-raw_frozen', 'current-SCS-raw_frozen'),
        ('current-DCS-BW-fixed_pilot', 'current-SCS-fixed_pilot'),
        ('vanilla-DCS-BW-fixed_pilot', 'current-SCS-fixed_pilot'),
        ('vanilla-DCS-BW-fixed_pilot', 'current-DCS-BW-fixed_pilot'),
        ('current-DCS-BW-fixed_pilot', 'current-DCS-BW-raw_frozen'),
        ('current-SCS-fixed_pilot', 'current-SCS-raw_frozen'),
        ('vanilla-DCS-BW-fixed_pilot', 'vanilla-DCS-BW-raw_frozen'),
    ]
    for source in ('original', 'current'):
        contrasts += [(f'{source}-SCS-raw_frozen', f'{source}-SCS-fixed_0.1'),
            (f'{source}-DCS-BW-raw_frozen', f'{source}-DCS-BW-smoothed_frozen'),
            (f'{source}-DCS-BW-smoothed_frozen', f'{source}-DCS-BW-smoothed_continuing')]
    rng = np.random.default_rng(77401)
    indices = rng.integers(0, config['runs'], (10000, config['runs']))
    summary['paired_mse_differences'] = []
    for case, other in contrasts:
        a = np.array(summary['cases'][case]['squared_errors'])
        b = np.array(summary['cases'][other]['squared_errors'])
        difference = (a-b)[indices].mean(axis=1)
        summary['paired_mse_differences'].append({'a': case, 'b': other,
            'a_minus_b': (a-b).mean(axis=0).tolist(),
            'bootstrap_95_percentile': np.quantile(difference, [.025, .975], axis=0).tolist()})
    summary['uncertainty'] = {'method': 'Paired chain bootstrap, percentile intervals',
        'replicates': 10000, 'seed': 77401,
        'interpretation': 'Descriptive intervals conditional on the fitted maps; no multiplicity correction. '
                          'They do not quantify map-fitting uncertainty or establish universal algorithm ordering.'}
    write_json(root/'summary.json', summary)
    for name, case in summary['cases'].items():
        print(name, 'MSE:', repr(case['mse']), 'acceptance:', repr(case['acceptance_mean']), flush=True)
    return summary


def main(argv=None):
    # PYTHONIOENCODING in os.environ affects child processes, not streams that
    # Python already opened. Legacy VI prints Greek map parameters on Windows
    # too, including when stdout/stderr are redirected to the launcher's log.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preset', choices=tuple(PRESETS), default='quick')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--production-seed-base', type=int, default=DEFAULT_PRODUCTION_SEED_BASE,
                        help='First production-chain seed; use 9901001 for the published held-out evaluation.')
    args = parser.parse_args(argv)
    hardware = cpu_info()
    workers = hardware['capacity'] if args.workers is None else validate_workers(args.workers, hardware)
    root = args.output_dir.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        parser.error('Use a new or empty output directory. This audit never reuses saved chains or fitted maps.')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.audit.lock').open('x', encoding='utf-8') as lock:
        lock.write(f'Fresh audit run, parent process {os.getpid()}\n')
    config = audit_config(args.preset, workers, args.production_seed_base)
    provenance = {'source_hashes': audit_sources(), 'environment': environment(), 'config': config}
    fingerprint = digest(provenance)
    protocol = {**provenance, 'fingerprint': fingerprint,
        'registered_before_fitting_and_sampling': True,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'cases': case_definitions(), 'production_seed_base': config['production_seed_base'],
        'chain_seeds': production_seeds(config),
        'initial_physical_state': 'ones(100), current exact inverse for all cases',
        'common_transition_evaluator': 'Current MetropolisKernel; historical DCS VI and adaptation policies are isolated factors. All SCS fits use author VI.',
        'historical_source': {'path': 'validation/legacy_fixed_step.py',
            'sha256': hashlib.sha256(LEGACY.read_bytes()).hexdigest(),
            'use': 'Historical input only, not a current sampler; only its DCS VI function executes.'},
        'fit_note': 'Original DCS VI includes CDF-argument clipping, tanh shift, unit initialization and steps+1 updates; '
                    'current DCS VI uses unclipped target density, gradient-norm clipping, radial-median initialization and steps updates. '
                    'SCS uses the unchanged author VI routine in both groups, with its separately recorded budget and seed 0.',
        'timing_scope': TIME_SCOPE, 'workers': workers, 'hardware': hardware,
        'threads_per_worker': 1, 'schedule': 'Spawned process pool, mixed diagnostic cases',
        'status': 'fitting'}
    write_json(root/'protocol.json', protocol)
    try:
        fits = prepare(root, config, fingerprint)
        write_json(root/'maps_before_pilots.json', fits)
        protocol['status'] = 'independent_pilots'
        write_json(root/'protocol.json', protocol)
        from scripts.fixed_step_pilots import run_pilots, stop_executor
        pilots = run_pilots(fits, root/'pilots', args.preset, workers)
        pilot_protocol = json.loads((root/'pilots/protocol.json').read_text(encoding='utf-8'))
        pilot_seeds = set(pilot_protocol['transition_seeds']) | {pilot_protocol['initial_seed']}
        if pilot_seeds.intersection(production_seeds(config)):
            raise ValueError('Production seeds overlap independent pilot seeds. Choose a different production seed base.')
        fits['pilot_selection'] = pilots['selected']
        fits['pilot_selection_note'] = ('Scales selected on independent pilot chains over the recorded grid. '
            'Endpoint winners and invalid diagnostic scores are reported explicitly; selected scales are not claims of global optimality.')
        fits.pop('setup_digest')
        fits['setup_digest'] = digest(fits)
        write_json(root/'maps.json', fits)
    except BaseException as error:
        protocol.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(root/'protocol.json', protocol)
        raise
    jobs = make_jobs(root, config, fits)
    worker_count = min(workers, len(jobs))
    protocol.update(status='sampling', setup_digest=fits['setup_digest'],
                    total_chains=len(jobs), concurrent_workers=worker_count)
    write_json(root/'protocol.json', protocol)
    print(f'{len(jobs)} diagnostic chains, up to {worker_count} workers', flush=True)
    context = multiprocessing.get_context('spawn')
    slots = context.Queue()
    for cpu in hardware['cpu_ids'][:worker_count]:
        slots.put(cpu)
    try:
        executor, futures = None, []
        try:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=context,
                initializer=initialize_worker, initargs=(slots,))
            futures = [executor.submit(chain_job, job) for job in jobs]
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                print(f'{number}/{len(jobs)} complete:', future.result(), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            if executor is not None:
                stop_executor(executor)
            raise
        else:
            executor.shutdown(wait=True)
        if audit_sources() != provenance['source_hashes']:
            raise RuntimeError('Audit source files changed during this run; results cannot be treated as one verified source version.')
        summary = summarize(root)
        from scripts.fixed_step_figures import render_report
        render_report(root, summary)
    except BaseException as error:
        protocol.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(root/'protocol.json', protocol)
        raise
    finally:
        slots.close()
        slots.join_thread()
    protocol.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'protocol.json', protocol)
    print(f'Audit complete: {root}', flush=True)
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
