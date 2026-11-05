"""Symmetric 100D Student-t: separate map fitting from proposal-step tuning.

Both untuned maps use R=1 and zero centre/shift. DCS symmetric VI fits R
only; SCS uses the authors' unconstrained VI. Every method receives the same independent fixed-grid pilot budget.
Production uses fresh common seeds and the same physical starting point.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dcs_bench.cpu import cpu_info, pin_current_process, validate_workers, worker_environment
os.environ.update(worker_environment())
import numpy as np
from scipy.stats import f as f_distribution
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import fit_map, DualAveraging
from dcs_bench.references import radial_truth
from dcs_bench.engine import load_config, environment
from dcs_bench import samplers
from scripts.fixed_step_pilots import stop_executor

TARGET = {'kind': 'skew_t', 'dimension': 100, 'nu': 3., 'alpha': {}}
FUNCTIONS = ['tail_probability', 'mean_norm', 'log1p_norm', 'radial_95',
             'asinh_x1', 'asinh_x2', 'asinh_x3', 'asinh_x4']
SELECTION_FUNCTIONS = ['log1p_norm', 'radial_95', 'asinh_x1', 'asinh_x3', 'asinh_x4']
RADIAL_SELECTION_FUNCTIONS = ['tail_probability', 'mean_norm']
PRODUCTION_SEED_BASE = 5700001
GRIDS = {'DCS-BW': [.025, .05, .075, .1, .15, .2, .3, .45],
         'SCS': [.01, .025, .05, .1, .2, .4, 1., 4.]}
PRESETS = {
    'quick': dict(runs=2, iterations=500, warmup=100, thin=1,
                  pilot_chains=2, pilot_burn=50, pilot_retained=100,
                  optimizer_steps=20, optimizer_samples=128),
    'reproduce': dict(runs=8, iterations=25000, warmup=1000, thin=10,
                      pilot_chains=4, pilot_burn=1000, pilot_retained=4000,
                      optimizer_steps=400, optimizer_samples=500),
    'paper': dict(runs=20, iterations=500000, warmup=1500, thin=50,
                  pilot_chains=4, pilot_burn=2000, pilot_retained=8000,
                  optimizer_steps=1500, optimizer_samples=2000)}
_ALLOCATION, _ALLOCATION_ERROR = {}, None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [*sorted((ROOT/'dcs_bench').glob('*.py')), Path(__file__),
             ROOT/'scripts/symmetric_t_figures.py', ROOT/'scripts/fixed_step_pilots.py',
             ROOT/'requirements-cpu.txt', ROOT/'configs/skew-t-100d-fixed.json']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def group_definitions():
    return [{'group': kind+'-'+algorithm, 'map_kind': kind, 'algorithm': algorithm}
            for kind in ('untuned', 'vi_fit') for algorithm in ('DCS-BW', 'SCS')]


def case_definitions():
    return [{**g, 'policy': policy, 'case': g['group']+'-'+policy}
            for g in group_definitions()
            for policy in ('warmup_frozen', 'pilot_selected', 'pilot_radial')]


def reference():
    _, radial, _ = radial_truth(Target(TARGET), 50.)
    return {'tail_probability': float(radial[0]), 'mean_norm': float(radial[1]),
            'radial_95': .05, **{f'asinh_x{i}': 0. for i in range(1, 5)}}


def functional_values(x, q95):
    radius = np.linalg.norm(x)
    return [float(radius >= 50.), float(radius), float(np.log1p(radius)),
            float(radius >= q95), *np.arcsinh(x[:4]).tolist()]


def initialize_worker(slots):
    global _ALLOCATION, _ALLOCATION_ERROR
    try:
        _ALLOCATION = pin_current_process(slots.get())
    except Exception as error:
        _ALLOCATION_ERROR = f'{type(error).__name__}: {error}'


def chain_job(job):
    if _ALLOCATION_ERROR:
        raise RuntimeError(_ALLOCATION_ERROR)
    if job['policy'] not in ('fixed_grid', 'warmup_frozen', 'pilot_selected', 'pilot_radial'):
        raise ValueError('Unknown step-size policy')
    output = Path(job['output'])
    if output.exists():
        raise FileExistsError(output)
    target = Target(TARGET)
    mapping = (BallMap if job['algorithm'] == 'DCS-BW' else SphereMap)(**job['map'])
    recording = False
    outside, cap_exits = 0, 0
    original_stepout = samplers.cap_stepout

    class MeasuredKernel(samplers.MetropolisKernel):
        def evaluate(self, z):
            nonlocal outside
            if recording and self.algorithm == 'DCS-BW' and z@z >= 1.:
                outside += 1
            return super().evaluate(z)

    def stepout(z, proposal, latitude):
        nonlocal cap_exits
        if recording:
            cap_exits += int(proposal[-1] >= latitude-1)
        return original_stepout(z, proposal, latitude)

    samplers.cap_stepout = stepout
    try:
        kernel = MeasuredKernel(target, job['algorithm'], mapping, job['seed'])
        if job.get('initial') is not None:
            kernel.state = mapping.inverse(np.asarray(job['initial']))
            kernel.lp = kernel.evaluate(kernel.state)
        scale = job.get('scale', .1)
        adapter = DualAveraging(scale)
        started = time.perf_counter()
        for _ in range(job['warmup']):
            probability, _ = kernel.step(scale)
            if job['policy'] == 'warmup_frozen':
                scale = adapter.update(probability)
        if job['policy'] == 'warmup_frozen':
            scale = adapter.final()
        warmup_seconds = time.perf_counter()-started
        recording = True
        values = []
        accepted = 0
        current_functions = np.asarray(functional_values(kernel.draw(), job['q95']))
        all_totals = np.zeros(len(FUNCTIONS))
        started = time.perf_counter()
        for iteration in range(1, job['iterations']+1):
            _, accept = kernel.step(scale)
            accepted += accept
            if accept:
                current_functions = np.asarray(functional_values(kernel.draw(), job['q95']))
            # Rejections contribute repeated states. Cache functions, never
            # proposals or random draws, to avoid remapping an unchanged state.
            all_totals += current_functions
            if iteration % job['thin'] == 0:
                values.append(current_functions.copy())
        elapsed = time.perf_counter()-started
        values = np.asarray(values)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError('Nonfinite functional traces')
        record = {k: v for k, v in job.items() if k not in ('map', 'initial', 'output')}
        record.update(scale=float(scale), acceptance=accepted/job['iterations'],
                      outside_ball_fraction=outside/job['iterations'],
                      cap_stepout_fraction=cap_exits/job['iterations'],
                      warmup_seconds=warmup_seconds, seconds=elapsed,
                      worker={'pid': os.getpid(), **_ALLOCATION},
                      retained=len(values), map_digest=digest(job['map']),
                      estimates=dict(zip(FUNCTIONS, values.mean(axis=0).tolist())),
                      all_iteration_estimates=dict(zip(FUNCTIONS, (all_totals/job['iterations']).tolist())),
                      all_iteration_count=job['iterations'],
                      adaptation_during_sampling=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, functions=values,
                                metadata=np.array(json.dumps(record, allow_nan=False)))
        os.replace(temporary, output)
        return record
    finally:
        samplers.cap_stepout = original_stepout


def run_jobs(jobs, workers, hardware):
    context = multiprocessing.get_context('spawn')
    slots = context.Queue()
    count = min(workers, len(jobs))
    for cpu in hardware['cpu_ids'][:count]:
        slots.put(cpu)
    executor, futures = None, []
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=context,
                    initializer=initialize_worker, initargs=(slots,))
        futures = [executor.submit(chain_job, job) for job in jobs]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            print(f'{number}/{len(jobs)} {record["case"]} chain {record["chain"]} saved', flush=True)
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


def summarize_jobs(jobs, truth):
    import arviz as az
    arrays, records = [], []
    for job in sorted(jobs, key=lambda j: j['chain']):
        with np.load(job['output'], allow_pickle=False) as saved:
            record = json.loads(str(saved['metadata']))
            for key in ('fingerprint', 'case', 'chain', 'seed', 'iterations', 'warmup', 'thin', 'q95', 'policy'):
                if record[key] != job[key]:
                    raise ValueError(f'Checkpoint provenance mismatch: {key}')
            if record['map_digest'] != digest(job['map']):
                raise ValueError('Checkpoint map mismatch')
            values = saved['functions'].copy()
            if values.shape != (job['iterations']//job['thin'], len(FUNCTIONS)):
                raise ValueError('Wrong trace dimensions')
            records.append(record)
            arrays.append(values)
    values = np.asarray(arrays)
    ess, rhat, invalid = {}, {}, []
    for i, name in enumerate(FUNCTIONS):
        v = values[:, :, i]
        if np.any(np.ptp(v, axis=1) == 0) or not np.all(np.isfinite(v)):
            ess[name] = rhat[name] = None
            invalid.append(name)
            continue
        es, rh = float(az.ess(v, method='bulk')), float(az.rhat(v, method='rank'))
        ess[name], rhat[name] = es if np.isfinite(es) else None, rh if np.isfinite(rh) else None
        if not np.isfinite(es) or not np.isfinite(rh):
            invalid.append(name)
    estimates = {n: values[:, :, FUNCTIONS.index(n)].mean(axis=1) for n in truth}
    errors = {n: (estimates[n]-truth[n])**2 for n in truth}
    all_estimates = {n: np.asarray([r['all_iteration_estimates'][n] for r in records]) for n in truth}
    all_errors = {n: (all_estimates[n]-truth[n])**2 for n in truth}
    return {'chains': len(jobs), 'scales': [r['scale'] for r in records],
            'seeds': [r['seed'] for r in records], 'bulk_ess': ess, 'rank_rhat': rhat,
            'invalid_functions': invalid, 'retained_per_chain': values.shape[1],
            'total_retained': int(values.shape[0]*values.shape[1]),
            'acceptance_mean': float(np.mean([r['acceptance'] for r in records])),
            'outside_ball_fraction': float(np.mean([r['outside_ball_fraction'] for r in records])),
            'cap_stepout_fraction': float(np.mean([r['cap_stepout_fraction'] for r in records])),
            'mse': {n: float(v.mean()) for n, v in errors.items()},
            'squared_errors': {n: v.tolist() for n, v in errors.items()},
            'estimates': {n: v.tolist() for n, v in estimates.items()},
            'mse_all_iterations': {n: float(v.mean()) for n, v in all_errors.items()},
            'all_iteration_squared_errors': {n: v.tolist() for n, v in all_errors.items()},
            'all_iteration_estimates': {n: v.tolist() for n, v in all_estimates.items()}}


def select_pilots(grid, score_name='score'):
    selected = {}
    for group in group_definitions():
        rows = [row for row in grid if row['group'] == group['group']]
        winner = max(rows, key=lambda r: (r[score_name], -r['scale']))
        selected[group['group']] = {'scale': winner['scale'], 'score': winner[score_name],
            'criterion': score_name, 'valid_selection': winner[score_name] > 0,
            'endpoint_winner': winner['scale'] in (GRIDS[group['algorithm']][0], GRIDS[group['algorithm']][-1])}
    return selected


def paired_contrasts(cases, truth, error_key='squared_errors'):
    pairs = [(f'untuned-DCS-BW-{p}', f'untuned-SCS-{p}')
             for p in ('warmup_frozen', 'pilot_selected')]
    for algorithm in ('DCS-BW', 'SCS'):
        for kind in ('untuned', 'vi_fit'):
            pairs.append((f'{kind}-{algorithm}-pilot_selected', f'{kind}-{algorithm}-warmup_frozen'))
        for policy in ('warmup_frozen', 'pilot_selected'):
            pairs.append((f'vi_fit-{algorithm}-{policy}', f'untuned-{algorithm}-{policy}'))
    pairs.append(('vi_fit-DCS-BW-pilot_selected', 'vi_fit-SCS-pilot_selected'))
    for group in group_definitions():
        for alternative in ('pilot_selected', 'warmup_frozen'):
            pairs.append((group['group']+'-pilot_radial', group['group']+'-'+alternative))
    for kind in ('untuned', 'vi_fit'):
        pairs.append((f'{kind}-DCS-BW-pilot_radial', f'{kind}-SCS-pilot_radial'))
    for algorithm in ('DCS-BW', 'SCS'):
        pairs.append((f'vi_fit-{algorithm}-pilot_radial', f'untuned-{algorithm}-pilot_radial'))
    count = next(iter(cases.values()))['chains']
    rng = np.random.default_rng(771231)
    indices = rng.integers(0, count, (20000, count))
    contrasts = []
    for a, b in pairs:
        for name in truth:
            differences = np.asarray(cases[a][error_key][name])-cases[b][error_key][name]
            ci = np.quantile(differences[indices].mean(axis=1), [.025, .975])
            contrasts.append({'a': a, 'b': b, 'function': name, 'difference': float(differences.mean()),
                              'ci95': ci.tolist()})
    return contrasts


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=tuple(PRESETS), default='quick')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    hardware = cpu_info()
    workers = validate_workers(args.workers if args.workers is not None else hardware['capacity'], hardware)
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh output folder.')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.run.lock').open('x', encoding='utf-8') as lock:
        lock.write(str(os.getpid()))
    config = {**PRESETS[args.preset], 'preset': args.preset, 'workers': workers,
              'production_seed_base': PRODUCTION_SEED_BASE}
    config['scs_vi'] = dict(optimizer_steps=1000 if args.preset == 'paper' else config['optimizer_steps'],
                            optimizer_samples=config['optimizer_samples'])
    hashes = source_hashes()
    fingerprint = digest({'source': hashes, 'config': config, 'target': TARGET})
    truth = reference()
    q95 = float(np.sqrt(100*f_distribution.ppf(.95, 100, 3)))
    rng = np.random.default_rng(5300001)
    initials = rng.normal(size=(config['pilot_chains'], 100))
    initials /= np.sqrt(rng.chisquare(3, size=(config['pilot_chains'], 1))/3)
    protocol = {'target': TARGET, 'config': config, 'fingerprint': fingerprint,
        'source_hashes': hashes, 'environment': environment(), 'hardware': hardware,
        'registered_before_fitting_and_sampling': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'case_definitions': case_definitions(), 'grids': GRIDS,
        'pilot_initial_seed': 5300001, 'pilot_initials': initials.tolist(),
        'pilot_seed_rule': '5400001 + group_index*10000 + candidate_index*100 + chain',
        'production_seeds': list(range(PRODUCTION_SEED_BASE, PRODUCTION_SEED_BASE+config['runs'])),
        'selection_functions': SELECTION_FUNCTIONS, 'radial_selection_functions': RADIAL_SELECTION_FUNCTIONS,
        'truth': truth,
        'notes': ['Untuned means R=1 and every centre/shift is zero for BOTH algorithms.',
                  'Every production step is pretuned by warmup or independent pilot chains, then frozen. Fixed grid values are calibration candidates only.',
                  'DCS symmetric VI fits only R. SCS uses unchanged author VI, including unconstrained centre and observer; both receive fresh proposal-scale tuning.',
                  'All pilot starts are shared exact Student-t draws; this is an oracle-assisted synthetic diagnostic.',
                  'Production starts at the same physical ones(100), with equal discarded warmup and retained iteration rules.',
                  'Equal numeric h is not equal geometric step: BW uses total ball radius; SCS uses per-coordinate tangent SD.',
                  'Select maximum minimum bulk ESS per retained pilot value across five functions. Any invalid selected function scores zero.',
                  'A second predeclared rule selects by the minimum bulk ESS for the two paper radial functions. Both methods use the same pilot data and budget for each rule.',
                  'The radial-focused rule was motivated by a preceding exploratory batch. This evaluation uses new production seeds 5700001 onward.',
                  'Every grid point and production case is retained. No production result selects tuning.',
                  'Paired bootstrap: 20000 resamples, seed771231, conditional on fitted maps; no multiplicity adjustment.',
                  'Mixed diagnostic cases share a worker pool: recorded times are not throughput comparisons.',
                  'Production ESS uses thinned retained traces, not the original iteration count.',
                  'Additional MSE estimates use streaming means of every production iteration, including rejected repeats. Original retained-draw figures remain separate.'],
        'status': 'fitting'}
    write_json(root/'protocol.json', protocol)
    try:
        target = Target(TARGET)
        maps, fit_metadata = {}, {}
        vi_config = load_config('skew-t-100d-fixed', 'paper', {'workers': workers})
        vi_config.update(optimizer_steps=config['optimizer_steps'], optimizer_samples=config['optimizer_samples'],
                         ball_map_policy='mobius', scs_vi=config['scs_vi'], sphere_map_policy='author-paper-vi')
        for group in group_definitions():
            family = 'ball' if group['algorithm'] == 'DCS-BW' else 'sphere'
            if group['map_kind'] == 'untuned':
                mapping = BallMap(100, beta=2.) if family == 'ball' else SphereMap(100, latitude=1.1)
                metadata = {'steps': 0, 'training_draws': 0, 'policy': 'fully untuned'}
            else:
                print(f'Fitting {"symmetric DCS scale" if family == "ball" else "SCS author VI"}: {group["algorithm"]}', flush=True)
                mapping, metadata = fit_map(target, vi_config, family, 5600001)
            parameters = mapping.parameters()
            shift = parameters.get('delta', parameters.get('observer'))
            if (group['map_kind'] == 'untuned' or family == 'ball') and (np.any(parameters['mu']) or np.any(shift)):
                raise ValueError('Untuned and symmetric DCS maps require exact zero centre and shift')
            maps[group['group']], fit_metadata[group['group']] = parameters, metadata
        write_json(root/'maps.json', {'maps': maps, 'fit_metadata': fit_metadata})
        pilot_jobs = []
        for gi, group in enumerate(group_definitions()):
            for candidate, scale in enumerate(GRIDS[group['algorithm']]):
                for chain in range(config['pilot_chains']):
                    name = group['group']+f'-grid-{candidate:02d}'
                    pilot_jobs.append({**group, 'case': name, 'chain': chain, 'candidate': candidate,
                        'map': maps[group['group']], 'scale': scale, 'policy': 'fixed_grid',
                        'seed': 5400001+gi*10000+candidate*100+chain,
                        'warmup': config['pilot_burn'], 'iterations': config['pilot_retained'], 'thin': 1,
                        'q95': q95, 'initial': initials[chain].tolist(), 'fingerprint': fingerprint,
                        'output': str(root/'pilots'/name/f'{chain:03d}.npz')})
        write_json(root/'pilot_jobs.json', pilot_jobs)
        protocol['status'] = 'independent_pilots'; write_json(root/'protocol.json', protocol)
        run_jobs(pilot_jobs, workers, hardware)
        grid = []
        for group in group_definitions():
            for candidate, scale in enumerate(GRIDS[group['algorithm']]):
                jobs = [j for j in pilot_jobs if j['group'] == group['group'] and j['candidate'] == candidate]
                row = {**group, 'scale': scale, **summarize_jobs(jobs, truth)}
                valid = all(row['bulk_ess'][n] is not None and row['rank_rhat'][n] is not None for n in SELECTION_FUNCTIONS)
                row['score'] = min(row['bulk_ess'][n] for n in SELECTION_FUNCTIONS)/row['total_retained'] if valid else 0.
                row['min_ess_over_retained'] = row['score']
                radial_valid = all(row['bulk_ess'][n] is not None and row['rank_rhat'][n] is not None for n in RADIAL_SELECTION_FUNCTIONS)
                row['radial_score'] = min(row['bulk_ess'][n] for n in RADIAL_SELECTION_FUNCTIONS)/row['total_retained'] if radial_valid else 0.
                grid.append(row)
        pilots = {'grid': grid, 'selected': select_pilots(grid),
                  'selected_radial': select_pilots(grid, 'radial_score')}
        write_json(root/'pilot_summary.json', pilots)
        print('Pilot selections: '+json.dumps(pilots['selected']), flush=True)
        production_jobs = []
        for case in case_definitions():
            if case['policy'] == 'pilot_selected':
                scale = pilots['selected'][case['group']]['scale']
            elif case['policy'] == 'pilot_radial':
                scale = pilots['selected_radial'][case['group']]['scale']
            else:
                scale = .1
            for chain in range(config['runs']):
                production_jobs.append({**case, 'map': maps[case['group']], 'scale': scale,
                    'chain': chain, 'seed': PRODUCTION_SEED_BASE+chain, 'warmup': config['warmup'],
                    'iterations': config['iterations'], 'thin': config['thin'], 'q95': q95,
                    'fingerprint': fingerprint, 'output': str(root/'chains'/case['case']/f'{chain:03d}.npz')})
        write_json(root/'production_jobs.json', production_jobs)
        protocol['status'] = 'production'; write_json(root/'protocol.json', protocol)
        run_jobs(production_jobs, workers, hardware)
        if hashes != source_hashes():
            raise RuntimeError('Source changed during audit')
        cases = {c['case']: {**c, **summarize_jobs([j for j in production_jobs if j['case'] == c['case']], truth)}
                 for c in case_definitions()}
        summary = {'target': TARGET, 'config': config, 'fingerprint': fingerprint, 'source_hashes': hashes,
                   'maps': maps, 'fit_metadata': fit_metadata, 'truth': truth, 'pilots': pilots, 'cases': cases,
                   'contrasts': paired_contrasts(cases, truth),
                   'all_iteration_contrasts': paired_contrasts(cases, truth, 'all_iteration_squared_errors'),
                   'notes': protocol['notes']}
        write_json(root/'summary.json', summary)
        from scripts.symmetric_t_figures import render_report
        render_report(root, summary)
        protocol['status'] = 'complete'; write_json(root/'protocol.json', protocol)
        print(f'Symmetric audit complete: {root}', flush=True)
    except BaseException as error:
        protocol.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(root/'protocol.json', protocol)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
