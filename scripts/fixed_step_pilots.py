"""Equal-budget independent pilots for the fixed-step diagnostic.

Use run_pilots(fits, output_dir, preset, workers). Only map inputs and this
fresh pilot directory are read; production errors never enter scale selection.
The quick preset exercises the pipeline and is not a scientific tuning run.
"""
from __future__ import annotations

import os
import concurrent.futures
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import cpu_info, pin_current_process, validate_workers, worker_environment
os.environ.update(worker_environment())  # Before NumPy and worker spawning.
os.environ['NUMEXPR_NUM_THREADS'] = os.environ['BLIS_NUM_THREADS'] = '1'

import numpy as np
from scipy.special import ndtr
from scipy.stats import f as f_distribution
from dcs_bench import samplers
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap

ORIGINAL_STEP_OUT = samplers.cap_stepout
FUNCTIONS = ['log1p_norm', 'radial_95_indicator', 'asinh_skew_contrast', 'asinh_x3', 'asinh_x4']
GRIDS = {'DCS-BW': [.025, .05, .075, .1, .15, .2, .3, .45],
         'SCS': [.01, .025, .05, .1, .2, .4, 1., 4.]}
PRESETS = {'paper': {'chains': 4, 'burn': 2000, 'retained': 8000},
           'reproduce': {'chains': 4, 'burn': 1000, 'retained': 4000},
           'quick': {'chains': 2, 'burn': 50, 'retained': 100}}
_WORKER_ALLOCATION = {'cpu_id': None, 'pinned': False}
_WORKER_ALLOCATION_ERROR = None


def initialize_worker(slots):
    global _WORKER_ALLOCATION, _WORKER_ALLOCATION_ERROR
    try:
        _WORKER_ALLOCATION = pin_current_process(slots.get())
    except Exception as error:
        # Report through a job result so Pool cannot repeatedly respawn failed
        # initializers and exhaust the one-slot-per-worker allocation queue.
        _WORKER_ALLOCATION_ERROR = f'{type(error).__name__}: {error}'


def stop_executor(executor):
    """Terminate running work after failure on supported Python 3.12."""
    # Python 3.12 has no public ProcessPoolExecutor.terminate_workers(). Save
    # handles before shutdown clears them, then reap every spawned process.
    processes = list((getattr(executor, '_processes', None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    executor.shutdown(wait=True, cancel_futures=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pilot_chain(job):
    if _WORKER_ALLOCATION_ERROR is not None:
        raise RuntimeError(f'Pilot worker CPU allocation failed: {_WORKER_ALLOCATION_ERROR}')
    previous_stepout = samplers.cap_stepout
    try:
        return _pilot_chain(job)
    finally:
        samplers.cap_stepout = previous_stepout


def _pilot_chain(job):
    output_dir = Path(job['output_dir'])
    start = time.perf_counter()
    target = Target(job['target'])
    mapping = (BallMap if job['algorithm'] == 'DCS-BW' else SphereMap)(**job['map'])
    counters = {'cap_stepouts': 0, 'outside_ball': 0, 'other_nonfinite': 0}
    angles = []

    def stepout(x, proposal, latitude):
        if recording:
            counters['cap_stepouts'] += int(proposal[-1] >= latitude - 1)
            angles.append(float(np.arccos(np.clip(x @ proposal, -1., 1.))))
        return ORIGINAL_STEP_OUT(x, proposal, latitude)

    samplers.cap_stepout = stepout
    recording = False

    class InstrumentedKernel(samplers.MetropolisKernel):
        def evaluate(self, x):
            value = super().evaluate(x)
            if recording and not np.isfinite(value):
                outside = self.algorithm == 'DCS-BW' and x @ x >= 1.
                counters['outside_ball' if outside else 'other_nonfinite'] += 1
            return value

    kernel = InstrumentedKernel(target, job['algorithm'], mapping, job['seed'])
    kernel.state = mapping.inverse(np.asarray(job['initial']))
    kernel.lp = kernel.evaluate(kernel.state)
    if not np.isfinite(kernel.lp):
        raise RuntimeError('Nonfinite independently generated pilot initial density')
    for _ in range(job['burn']):
        kernel.step(job['scale'])
    warmup_seconds = time.perf_counter() - start
    recording = True
    draws = np.empty((job['retained'], target.d))
    accepted = 0
    probability_sum = 0.
    transformed_jump_sum = 0.
    physical_jump_sum = 0.
    previous_draw = kernel.draw().copy()
    start = time.perf_counter()
    for iteration in range(job['retained']):
        previous_state = kernel.state.copy()
        probability, did_accept = kernel.step(job['scale'])
        accepted += did_accept
        probability_sum += probability
        draws[iteration] = kernel.draw()
        transformed_jump_sum += float(np.sum((kernel.state - previous_state)**2))
        physical_jump_sum += float(np.sum((draws[iteration] - previous_draw)**2))
        previous_draw = draws[iteration]
    sampling_seconds = time.perf_counter() - start
    radii = np.linalg.norm(draws, axis=1)
    alpha = target.alpha / np.linalg.norm(target.alpha)
    function_traces = np.column_stack([np.log1p(radii), radii > job['radial_95'],
        np.arcsinh(draws @ alpha), np.arcsinh(draws[:, 2]), np.arcsinh(draws[:, 3])])
    if not np.all(np.isfinite(function_traces)):
        raise RuntimeError('Nonfinite pilot functions')
    result = {k: v for k, v in job.items() if k not in ('map', 'initial', 'output_dir')}
    result.update({'acceptance_rate': accepted / job['retained'],
        'mean_mh_probability': probability_sum / job['retained'],
        'warmup_seconds': warmup_seconds, 'sampling_seconds': sampling_seconds,
        'transformed_esjd': transformed_jump_sum / job['retained'],
        'physical_esjd': physical_jump_sum / job['retained'],
        'production_counters': counters, 'worker': {'pid': os.getpid(), **_WORKER_ALLOCATION},
        'angle_quantiles': np.quantile(angles, [0, .1, .5, .9, 1]).tolist() if angles else None})
    name = f"{job['case']}-scale-{job['candidate']:02d}-chain-{job['chain']:02d}"
    npz_path = output_dir / 'chains' / (name + '.npz')
    if npz_path.exists():
        raise FileExistsError(f'Pilot chain already exists: {npz_path}')
    np.savez_compressed(npz_path, draws=draws, functions=function_traces,
                        initial=np.asarray(job['initial']))
    result['npz_sha256'] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    result['artifact'] = npz_path.relative_to(output_dir).as_posix()
    json_write(output_dir / 'chains' / (name + '.json'), result)
    return result


def summarize(protocol, records, output_dir):
    import arviz as az
    rows = []
    for case in protocol['cases']:
        for candidate, scale in enumerate(GRIDS[case['algorithm']]):
            chains = sorted([r for r in records if r['case'] == case['name'] and r['candidate'] == candidate],
                            key=lambda r: r['chain'])
            traces = []
            for record in chains:
                with np.load(Path(output_dir) / record['artifact'], allow_pickle=False) as data:
                    traces.append(data['functions'].copy())
            values = np.stack(traces)
            ess, rhat, variances = {}, {}, {}
            invalid = []
            for index, name in enumerate(FUNCTIONS):
                array = values[:, :, index]
                variances[name] = np.var(array, axis=1, ddof=1).tolist()
                if np.any(np.var(array, axis=1) == 0) or not np.all(np.isfinite(array)):
                    ess[name] = rhat[name] = None
                    invalid.append(name)
                    continue
                es = float(az.ess(array, method='bulk'))
                rh = float(az.rhat(array, method='rank'))
                ess[name] = es if np.isfinite(es) else None
                rhat[name] = rh if np.isfinite(rh) else None
                if not np.isfinite(es) or not np.isfinite(rh):
                    invalid.append(name)
            finite_ess = [v for v in ess.values() if v is not None]
            row = {'case': case['name'], 'algorithm': case['algorithm'], 'candidate': candidate,
                'scale': scale, 'bulk_ess': ess, 'rank_rhat': rhat, 'per_chain_function_variance': variances,
                'score': min(finite_ess) / values.shape[0] / values.shape[1] if not invalid else 0.,
                'median_ess': float(np.median(finite_ess)) if finite_ess else 0.,
                'invalid_functions': invalid,
                'mean_acceptance': float(np.mean([r['acceptance_rate'] for r in chains])),
                'mean_outside_ball_fraction': float(np.mean([r['production_counters']['outside_ball'] / r['retained'] for r in chains])),
                'mean_cap_stepout_fraction': float(np.mean([r['production_counters']['cap_stepouts'] / r['retained'] for r in chains])),
                'mean_transformed_esjd': float(np.mean([r['transformed_esjd'] for r in chains])),
                'mean_physical_esjd': float(np.mean([r['physical_esjd'] for r in chains]))}
            rows.append(row)
    selected = {}
    for case in protocol['cases']:
        eligible = [r for r in rows if r['case'] == case['name']]
        winner = max(eligible, key=lambda r: (r['score'], r['median_ess'], -r['scale']))
        selected[case['name']] = {'scale': winner['scale'], 'score': winner['score'],
            'candidate': winner['candidate'], 'endpoint_winner': winner['candidate'] in (0, 7),
            'map': case['map'], 'map_sha256': digest(case['map']),
            'algorithm': case['algorithm'], 'rank_rhat': winner['rank_rhat'],
            'bulk_ess': winner['bulk_ess'], 'valid_selection': winner['score'] > 0}
    return {'protocol_sha256': digest(protocol), 'parent_fingerprint': protocol.get('parent_fingerprint'),
        'source_sha256': protocol['source_sha256'], 'preset': protocol['preset'],
        'preset_settings': protocol['preset_settings'], 'smoke_check_only': protocol['preset'] == 'quick',
        'selected': selected, 'grid': rows,
        'notes': ['Selection uses only independent pilots, not production chains or MSE rankings.',
            'Conditional on one fitted map per family; does not establish optimizer robustness.',
            f"Only {protocol['chains_per_candidate']} chains and {protocol['retained']} recorded values per candidate; pilot selection remains uncertain.",
            'IID target starts are oracle-assisted pilot diagnostics; evaluate final common-start transients separately.',
            'An endpoint winner indicates the declared grid needs extension before claiming optimal tuning.',
            'Ranks and ESS for nonstationary or poorly explored pilots require caution; inspect all reported R-hat values.'],
        'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'arviz': az.__version__}


def run_pilots(fits, output_dir, preset, workers):
    """Run fresh pilots and return every candidate plus selected frozen scales.

    fits contains config and maps[current][ball/sphere], maps[vanilla][ball].
    output_dir must not contain prior files. Workers are native-thread-limited
    spawned processes, assigned distinct CPU slots where the OS supports it.
    """
    if preset not in PRESETS:
        raise ValueError(f'Unknown pilot preset: {preset}')
    settings = PRESETS[preset]
    hardware = cpu_info()
    validate_workers(workers, hardware)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Use a new or empty pilot directory: {output_dir}')
    spec = fits['config']['targets'][0]
    expected = {'kind': 'skew_t', 'dimension': 100, 'nu': 3, 'alpha': {'0': 100, '1': -100}}
    if spec != expected or len(fits['config']['targets']) != 1:
        raise ValueError(f'The fixed-step pilot requires the declared 100D nu=3 target: {spec}')
    target = Target(spec)
    chains = settings['chains']
    rng = np.random.default_rng(4_200_001)
    initial = rng.normal(size=(chains, target.d)) / np.sqrt(rng.chisquare(target.nu, size=(chains, 1)) / target.nu)
    r2 = np.sum(initial**2, axis=1)
    p = ndtr(initial @ target.alpha * np.sqrt((target.nu + target.d) / (target.nu + r2)))
    initial *= np.where(rng.random(chains) < p, 1., -1.)[:, None]
    vanilla = fits['maps']['vanilla']['ball']
    if vanilla != BallMap(100, beta=2.).parameters():
        raise ValueError('Vanilla pilot requires R=1, centre zero, shift zero, beta=2')
    cases = [{'name': 'mobius-DCS-BW', 'algorithm': 'DCS-BW', 'map': fits['maps']['current']['ball']},
             {'name': 'current-SCS', 'algorithm': 'SCS', 'map': fits['maps']['current']['sphere']},
             {'name': 'vanilla-DCS-BW', 'algorithm': 'DCS-BW', 'map': vanilla}]
    paths = [*sorted((ROOT / 'dcs_bench').glob('*.py')), Path(__file__), ROOT / 'requirements-cpu.txt']
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    count = min(workers, len(cases) * 8 * chains)
    protocol = {'registered_before_sampling': True, 'target': spec, 'cases': cases,
        'source_sha256': hashes, 'parent_fingerprint': fits.get('fingerprint'),
        'fits_sha256': digest(fits), 'preset': preset, 'preset_settings': dict(settings),
        'grids': GRIDS, 'chains_per_candidate': chains, 'burn': settings['burn'],
        'retained': settings['retained'], 'thin': 1,
        'initial_seed': 4_200_001, 'initials': initial.tolist(), 'transition_seed_base': 4_201_001,
        'functions': FUNCTIONS, 'radial_95': float(np.sqrt(target.d * f_distribution.ppf(.95, target.d, target.nu))),
        'selection': 'maximize minimum multi-chain bulk ESS / total retained over five functions; any constant chain-function or nonfinite ESS/R-hat gives score zero; ties by median ESS, then smaller h',
        'requested_workers': workers, 'workers': count, 'per_worker_native_threads': 1,
        'hardware': hardware, 'cpu_ids': hardware['cpu_ids'][:count],
        'smoke_check_only': preset == 'quick'}
    jobs = []
    for group, case in enumerate(cases):
        for candidate, scale in enumerate(GRIDS[case['algorithm']]):
            for chain in range(chains):
                jobs.append({'case': case['name'], 'algorithm': case['algorithm'], 'map': case['map'],
                    'map_sha256': digest(case['map']), 'target': spec, 'scale': scale, 'candidate': candidate,
                    'chain': chain, 'initial': initial[chain].tolist(), 'output_dir': str(output_dir),
                    'seed': 4_201_001 + group * 10000 + candidate * 100 + chain,
                    'burn': settings['burn'], 'retained': settings['retained'], 'radial_95': protocol['radial_95']})
    protocol['transition_seeds'] = [job['seed'] for job in jobs]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'chains').mkdir()
    json_write(output_dir / 'protocol.json', protocol)
    records = []
    total = len(jobs)
    proposals = total * (settings['burn'] + settings['retained'])
    print(f'Registered {total} pilot chains, {proposals} proposals, {count} workers; production records are not read.', flush=True)
    if preset == 'quick':
        print('Pilot quick preset is a smoke check only; zero or invalid scores are expected with short chains.', flush=True)
    os.environ.update(worker_environment())
    os.environ['NUMEXPR_NUM_THREADS'] = os.environ['BLIS_NUM_THREADS'] = '1'
    context = multiprocessing.get_context('spawn')
    slots = context.Queue()
    for cpu in hardware['cpu_ids'][:count]:
        slots.put(cpu)
    executor = None
    futures = []
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=context,
            initializer=initialize_worker, initargs=(slots,))
        futures = [executor.submit(pilot_chain, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"{len(records)}/{total} {record['case']} h={record['scale']} chain={record['chain']} acceptance={record['acceptance_rate']:.4f}", flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        if executor is not None:
            stop_executor(executor)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        slots.close()
        slots.join_thread()
    result = summarize(protocol, records, output_dir)
    result['chain_count'] = total
    result['total_proposals'] = proposals
    json_write(output_dir / 'summary.json', result)
    print(json.dumps({'selected': {key: {'scale': value['scale'], 'score': value['score'],
        'valid_selection': value['valid_selection'], 'endpoint_winner': value['endpoint_winner']}
        for key, value in result['selected'].items()}}, indent=2), flush=True)
    return result
