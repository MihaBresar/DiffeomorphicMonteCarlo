"""Fresh equal-wall-time comparison on the 100D skewed Student-t with nu=2.

Map fitting and independent proposal pilots precede production. Timed chains
run in synchronized waves, one method/map case at a time. Every iteration,
including rejection repeats, contributes to the two requested estimates.
"""
from __future__ import annotations
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
os.environ.update(worker_environment())
import numpy as np
from scipy.special import ndtr
from scipy.stats import f as f_distribution
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import fit_map, DualAveraging
from dcs_bench.references import radial_truth
from dcs_bench.engine import load_config, environment
from dcs_bench.samplers import MetropolisKernel
from scripts.fixed_step_pilots import stop_executor

TARGET = {'kind': 'skew_t', 'dimension': 100, 'nu': 2., 'alpha': {'0': 100, '1': -100}}
METRICS = ['tail_probability', 'mean_norm']
PILOT_FUNCTIONS = ['log1p_norm', 'radial_95', 'asinh_skew', 'asinh_x3', 'asinh_x4']
GRIDS = {'DCS-BW': [.025, .05, .075, .1, .15, .2, .3, .45],
         'SCS': [.01, .025, .05, .1, .2, .4, 1., 4.]}
PRESETS = {
    'quick': dict(runs=2, time_budget=1., warmup=100, thin=1,
                  pilot_chains=2, pilot_burn=50, pilot_retained=100,
                  optimizer_steps=20, optimizer_samples=128),
    'reproduce': dict(runs=8, time_budget=5., warmup=1000, thin=10,
                      pilot_chains=4, pilot_burn=1000, pilot_retained=4000,
                      optimizer_steps=400, optimizer_samples=500),
    'paper': dict(runs=20, time_budget=20., warmup=1500, thin=50,
                  pilot_chains=4, pilot_burn=2000, pilot_retained=8000,
                  optimizer_steps=1500, optimizer_samples=2000)}
_ALLOCATION, _ALLOCATION_ERROR, _BARRIER = {}, None, None


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes():
    paths = [*sorted((ROOT/'dcs_bench').glob('*.py')), Path(__file__),
             ROOT/'scripts/skew_t_timed_figures.py', ROOT/'scripts/fixed_step_pilots.py',
             ROOT/'requirements-cpu.txt', ROOT/'configs/skew-t-100d-fixed.json']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def map_definitions():
    return [dict(case='mobius-DCS-BW', label='DCS-BW: Mobius VI', algorithm='DCS-BW', map_kind='fitted'),
            dict(case='fitted-SCS', label='SCS: fitted map', algorithm='SCS', map_kind='fitted'),
            dict(case='vanilla-DCS-BW', label='DCS-BW: untuned map', algorithm='DCS-BW', map_kind='untuned'),
            dict(case='untuned-SCS', label='SCS: untuned map', algorithm='SCS', map_kind='untuned')]


def case_definitions():
    return [{**group, 'map_group': group['case'], 'case': group['case']+'-'+policy,
             'label': group['label']+' / '+policy, 'policy': policy}
            for group in map_definitions() for policy in ('frozen', 'adaptive')]


def reference():
    _, radial, _ = radial_truth(Target(TARGET), 50.)
    return dict(zip(METRICS, map(float, radial)))


def radial_values(x):
    radius = float(np.linalg.norm(x))
    return np.array([float(radius >= 50.), radius])


def pilot_values(x, q95):
    radius = np.linalg.norm(x)
    return [np.log1p(radius), float(radius >= q95), np.arcsinh((x[0]-x[1])/np.sqrt(2.)),
            np.arcsinh(x[2]), np.arcsinh(x[3])]


def collect_timed(kernel, scale, seconds, thin, clock=time.perf_counter, cpu_clock=time.process_time,
                  adapter=None):
    """Stop at the first deadline check; record the final transition's overshoot."""
    if seconds <= 0 or thin < 1:
        raise ValueError('Positive time and thinning interval required')
    current = radial_values(kernel.draw())
    totals, traces = np.zeros(2), []
    iterations, accepted = 0, 0
    initial_scale, minimum_scale, maximum_scale, scale_total = scale, scale, scale, 0.
    ceiling_count = 0
    scale_ceiling = float(np.exp(15))
    evaluations = kernel.evaluations
    cpu_started, started = cpu_clock(), clock()
    while clock()-started < seconds:
        minimum_scale, maximum_scale = min(minimum_scale, scale), max(maximum_scale, scale)
        scale_total += scale
        ceiling_count += scale >= scale_ceiling
        probability, accept = kernel.step(scale)
        iterations += 1
        accepted += accept
        if adapter is not None:
            adapter.update(probability)
            scale = adapter.final()
        if accept:
            current = radial_values(kernel.draw())
        # Repeated states carry the same weight as accepted transitions.
        totals += current
        if iterations % thin == 0:
            traces.append(current.copy())
    elapsed, cpu_seconds = clock()-started, cpu_clock()-cpu_started
    if iterations == 0 or len(traces) < 4:
        raise RuntimeError('Too few retained values; increase time or reduce thinning')
    values = np.asarray(traces)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(totals)):
        raise FloatingPointError('Nonfinite radial estimates')
    return values, dict(iterations=iterations, retained=len(traces),
        sampling_seconds=elapsed, cpu_seconds=cpu_seconds,
        overshoot_seconds=max(0., elapsed-seconds), production_started=started,
        production_finished=started+elapsed, acceptance=accepted/iterations,
        density_evaluations=kernel.evaluations-evaluations,
        scale=float(initial_scale), final_scale=float(scale),
        mean_proposal_scale=scale_total/iterations, min_proposal_scale=float(minimum_scale),
        max_proposal_scale=float(maximum_scale), scale_ceiling_fraction=ceiling_count/iterations,
        production_adaptation_updates=iterations if adapter is not None else 0,
        estimates=dict(zip(METRICS, (totals/iterations).tolist())))


def initialize_worker(slots, barrier):
    global _ALLOCATION, _ALLOCATION_ERROR, _BARRIER
    _BARRIER = barrier
    try:
        _ALLOCATION = pin_current_process(slots.get())
    except Exception as error:
        _ALLOCATION_ERROR = f'{type(error).__name__}: {error}'


def chain_job(job):
    if _ALLOCATION_ERROR:
        raise RuntimeError(_ALLOCATION_ERROR)
    output = Path(job['output'])
    if output.exists():
        raise FileExistsError(output)
    if job['phase'] == 'production' and not job.get('pretuned', False):
        raise ValueError('Production requires an independently pretuned proposal step')
    if job['phase'] == 'production' and job.get('policy') not in ('frozen', 'adaptive'):
        raise ValueError('Unknown production adaptation policy')
    if job['phase'] not in ('pilot', 'production'):
        raise ValueError('Unknown phase')
    target = Target(job['target'])
    mapping = (BallMap if job['algorithm'] == 'DCS-BW' else SphereMap)(**job['map'])
    kernel = MetropolisKernel(target, job['algorithm'], mapping, job['seed'])
    if job.get('initial') is not None:
        kernel.state = mapping.inverse(np.asarray(job['initial']))
        kernel.lp = kernel.evaluate(kernel.state)
    if not np.isfinite(kernel.lp):
        raise ValueError('Nonfinite initial density')
    adaptive = job.get('policy') == 'adaptive'
    scale = job['scale']
    adapter = DualAveraging(scale) if adaptive else None
    started = time.perf_counter()
    for _ in range(job['warmup']):
        probability, _ = kernel.step(scale)
        if adapter is not None:
            adapter.update(probability)
            scale = adapter.final()
    warmup_seconds = time.perf_counter()-started
    if job['phase'] == 'pilot':
        values = np.empty((job['iterations'], len(PILOT_FUNCTIONS)))
        current = pilot_values(kernel.draw(), job['q95'])
        started = time.perf_counter()
        for iteration in range(job['iterations']):
            _, accept = kernel.step(job['scale'])
            if accept:
                current = pilot_values(kernel.draw(), job['q95'])
            values[iteration] = current
        record = dict(iterations=job['iterations'], retained=len(values),
                      sampling_seconds=time.perf_counter()-started, scale=job['scale'])
    else:
        if _BARRIER is not None:
            _BARRIER.wait(timeout=180)
        values, record = collect_timed(kernel, scale, job['seconds'], job['thin'], adapter=adapter)
    record.update({k: v for k, v in job.items() if k not in ('map', 'initial', 'output', 'scale')})
    record.update(map_digest=digest(job['map']), warmup_seconds=warmup_seconds,
                  pilot_scale=job['scale'], warmup_adaptation_updates=job['warmup'] if adaptive else 0,
                  adaptation_during_sampling=adaptive, worker={'pid': os.getpid(), **_ALLOCATION})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, functions=values,
                            metadata=np.array(json.dumps(record, allow_nan=False)))
    os.replace(temporary, output)
    return record


def run_jobs(jobs, workers, hardware, synchronize=False):
    context = multiprocessing.get_context('spawn')
    count = min(workers, len(jobs))
    if synchronize and count != len(jobs):
        raise ValueError('A synchronized wave must contain one job per worker')
    slots = context.Queue()
    for cpu in hardware['cpu_ids'][:count]:
        slots.put(cpu)
    barrier = context.Barrier(count) if synchronize else None
    executor, futures = None, []
    records = []
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=context,
                    initializer=initialize_worker, initargs=(slots, barrier))
        futures = [executor.submit(chain_job, job) for job in jobs]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            records.append(record)
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
        slots.close(); slots.join_thread()
    return records


def load_chain(job):
    with np.load(job['output'], allow_pickle=False) as saved:
        record = json.loads(str(saved['metadata']))
        for key in ('fingerprint', 'phase', 'case', 'chain', 'seed', 'warmup', 'target'):
            if record[key] != job[key]:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        if record['map_digest'] != digest(job['map']):
            raise ValueError('Checkpoint map mismatch')
        if record['pilot_scale'] != job['scale']:
            raise ValueError('Checkpoint pilot scale mismatch')
        values = saved['functions'].copy()
    if job['phase'] == 'pilot':
        expected = (job['iterations'], len(PILOT_FUNCTIONS))
    else:
        if (not record['pretuned'] or record['policy'] != job['policy']
                or record['adaptation_during_sampling'] != (job['policy'] == 'adaptive')):
            raise ValueError('Production pretuning or adaptation policy mismatch')
        if record['seconds'] != job['seconds'] or record['thin'] != job['thin']:
            raise ValueError('Checkpoint budget mismatch')
        adaptive = job['policy'] == 'adaptive'
        if (record['production_adaptation_updates'] != (record['iterations'] if adaptive else 0)
                or record['warmup_adaptation_updates'] != (job['warmup'] if adaptive else 0)):
            raise ValueError('Checkpoint adaptation count mismatch')
        if not adaptive and (record['scale'] != job['scale'] or record['final_scale'] != job['scale']):
            raise ValueError('Frozen production scale changed')
        expected = (record['iterations']//job['thin'], len(METRICS))
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError('Invalid trace dimensions or values')
    return values, record


def summarize_pilots(jobs):
    import arviz as az
    rows, selected = [], {}
    for case in map_definitions():
        for candidate, scale in enumerate(GRIDS[case['algorithm']]):
            group = [j for j in jobs if j['case'] == case['case'] and j['candidate'] == candidate]
            loaded = [load_chain(j) for j in group]
            arrays = np.asarray([v for v, _ in loaded])
            ess, rhat = {}, {}
            for i, name in enumerate(PILOT_FUNCTIONS):
                values = arrays[:, :, i]
                if np.any(np.ptp(values, axis=1) == 0):
                    ess[name] = rhat[name] = None
                else:
                    es, rh = float(az.ess(values, method='bulk')), float(az.rhat(values, method='rank'))
                    ess[name] = es if np.isfinite(es) else None
                    rhat[name] = rh if np.isfinite(rh) else None
            valid = all(v is not None for v in [*ess.values(), *rhat.values()])
            rows.append({**case, 'candidate': candidate, 'scale': scale, 'bulk_ess': ess, 'rank_rhat': rhat,
                'valid_selection': valid, 'score': min(ess.values())/arrays.shape[0]/arrays.shape[1] if valid else 0.,
                'median_ess': float(np.median(list(ess.values()))) if valid else 0.,
                'mean_seconds': float(np.mean([r['sampling_seconds'] for _, r in loaded]))})
        choices = [r for r in rows if r['case'] == case['case']]
        winner = max(choices, key=lambda r: (r['score'], r['median_ess'], -r['scale']))
        selected[case['case']] = {k: winner[k] for k in ('scale', 'score', 'candidate', 'valid_selection')}
        selected[case['case']]['endpoint_winner'] = winner['candidate'] in (0, 7)
    return {'grid': rows, 'selected': selected}


def production_schedule(config, workers):
    """Rotate case order across waves, independently of all sampling results."""
    names = [c['case'] for c in case_definitions()]
    schedule = []
    for wave, start in enumerate(range(0, config['runs'], workers)):
        offset = (3*wave) % len(names)
        order = names[offset:]+names[:offset]
        for name in order:
            schedule.append({'wave': wave, 'case': name,
                'chains': list(range(start, min(start+workers, config['runs'])))})
    return schedule


def summarize_production(jobs, truth):
    cases = {}
    for case in case_definitions():
        group = sorted([j for j in jobs if j['case'] == case['case']], key=lambda j: j['chain'])
        records = [load_chain(j)[1] for j in group]
        estimates = {n: [r['estimates'][n] for r in records] for n in METRICS}
        errors = {n: (np.asarray(estimates[n])-truth[n])**2 for n in METRICS}
        counts = [r['iterations'] for r in records]
        cases[case['case']] = {**case, 'scale': float(np.mean([r['scale'] for r in records])),
            'pilot_scale': records[0]['pilot_scale'], 'records': records,
            'estimates': estimates, 'squared_errors': {n: e.tolist() for n, e in errors.items()},
            'empirical_mse': {n: float(e.mean()) for n, e in errors.items()},
            'median_absolute_error': {n: float(np.median(np.abs(np.asarray(estimates[n])-truth[n]))) for n in METRICS},
            'mean_iterations': float(np.mean(counts)), 'total_iterations': sum(counts),
            'mean_sampling_seconds': float(np.mean([r['sampling_seconds'] for r in records]))}
    return cases


def tail_contrasts(cases):
    count = len(next(iter(cases.values()))['records'])
    indices = np.random.default_rng(6410001).integers(0, count, (20000, count))
    rows = []
    pairs = [(a+'-'+policy, b+'-'+policy) for policy in ('frozen', 'adaptive')
             for a, b in [('mobius-DCS-BW', 'fitted-SCS'), ('vanilla-DCS-BW', 'untuned-SCS'),
                          ('mobius-DCS-BW', 'vanilla-DCS-BW'), ('fitted-SCS', 'untuned-SCS')]]
    pairs += [(g['case']+'-adaptive', g['case']+'-frozen') for g in map_definitions()]
    for a, b in pairs:
        delta = np.asarray(cases[a]['squared_errors']['tail_probability'])-np.asarray(cases[b]['squared_errors']['tail_probability'])
        rows.append(dict(a=a, b=b, metric='tail_probability', difference=float(delta.mean()),
                         ci95=np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()))
    return rows


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='quick')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    hardware = cpu_info()
    workers = args.workers or min(hardware['capacity'], PRESETS[args.preset]['runs'])
    validate_workers(workers, hardware)
    config = {**PRESETS[args.preset], 'preset': args.preset, 'workers': workers}
    root = args.output_dir.resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f'Use a fresh output directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    hashes, truth = source_hashes(), reference()
    fingerprint = digest({'target': TARGET, 'config': config, 'source_hashes': hashes})
    schedule = production_schedule(config, min(workers, config['runs']))
    notes = [
        'Fresh actual equal production wall time, not retrospective MSE rescaling from fixed-iteration runs.',
        'Four map groups each have frozen and adaptive production cases. All eight start from independently pretuned proposal sizes; map fitting precedes proposal pilots and maps remain fixed.',
        'The adaptive arms apply the historical DCS smoothed dual-averaging update to BOTH methods, target acceptance 0.234, starting at the pilot-selected step and continuing through discarded warmup and timed production. Update cost is included in the time budget.',
        'The same seed pairs frozen/adaptive cases, but their warmup paths can differ because adaptation begins during warmup. This comparison evaluates the complete adaptation policy, not production adaptation in isolation.',
        'Adaptive transitions depend on sampling history; fixed-kernel convergence results do not automatically justify this empirical adaptive comparison. Initial/final scales and ceiling fractions are recorded.',
        'The five-function ESS-per-iteration pilot rule and eight candidates match the closer skewed diagnostic; target nu is now 2. This rule is not a proven optimum for MSE per second.',
        'Pilot initial points are shared exact target draws: an explicitly oracle-assisted synthetic diagnostic. Production starts at physical ones(100).',
        'One case per synchronized wave, identical worker allocation and chain counts; method order rotates across waves. No algorithms compete in a mixed timed queue.',
        'Production wall time includes transitions, all-iteration radial estimates and thinned trace collection; VI, pilots, warmup, process startup and file saving are excluded and separately reported.',
        'The stopping rule checks elapsed time before each transition. The final completed transition can exceed the deadline; actual elapsed and overshoot are recorded.',
        'Every completed iteration including rejected repeats contributes equally. Saved thinned traces do not determine the reported estimates.',
        'At nu=2, the norm has a finite mean but infinite second moment. Its empirical mean squared error and median absolute error describe these runs; no finite-variance norm MSE inference or norm bootstrap interval is claimed.',
        'Only tail-probability MSE differences have descriptive paired bootstrap intervals: 20000 resamples, seed6410001; conditional on maps and pilots, no multiplicity adjustment.',
        'Wall-time stopping makes sample count random and potentially state-dependent. These are actual fixed-wall-budget algorithm outputs, not fixed-count stationary MSE estimates.',
        'Timing and iteration counts depend on hardware, worker count and load; CPU time is saved alongside wall time. On macOS CPU scheduling is OS-managed.',
        'All cases and pilot candidates are retained. Quick check is a software smoke test, not a scientific ranking.']
    protocol = dict(schema_version=1, registered_before_sampling=True,
        started_utc=datetime.now(timezone.utc).isoformat(), target=TARGET, config=config,
        beta=2., latitude=1.1, threshold=50., cases=case_definitions(), grids=GRIDS,
        selection_functions=PILOT_FUNCTIONS, source_hashes=hashes, fingerprint=fingerprint,
        environment=environment(), hardware=hardware, truth=truth, production_schedule=schedule,
        map_seed=6100001, pilot_initial_seed=6200001,
        pilot_seed_rule='6210001 + case_index*10000 + candidate_index*100 + chain',
        production_seeds=list(range(6300001, 6300001+config['runs'])), notes=notes, status='fitting')
    write_json(root/'protocol.json', protocol)
    invocation = time.perf_counter()
    try:
        target = Target(TARGET)
        vi_config = load_config('skew-t-100d-fixed', 'paper', {'workers': workers})
        vi_config.update(optimizer_steps=config['optimizer_steps'], optimizer_samples=config['optimizer_samples'],
                         ball_map_policy='mobius', beta=2., targets=[TARGET])
        maps, fit_metadata = {}, {}
        for case in map_definitions():
            family = 'ball' if case['algorithm'] == 'DCS-BW' else 'sphere'
            if case['map_kind'] == 'fitted':
                print(f'Fitting nu=2 map: {case["case"]}', flush=True)
                mapping, meta = fit_map(target, vi_config, family, 6100001)
            else:
                mapping = BallMap(100, beta=2.) if family == 'ball' else SphereMap(100, latitude=1.1)
                meta = {'seconds': 0., 'steps': 0, 'policy': 'fully untuned map'}
            maps[case['case']], fit_metadata[case['case']] = mapping.parameters(), meta
        write_json(root/'maps.json', {'maps': maps, 'fit_metadata': fit_metadata})
        rng = np.random.default_rng(6200001)
        initials = rng.normal(size=(config['pilot_chains'], 100))/np.sqrt(rng.chisquare(2., size=(config['pilot_chains'], 1))/2.)
        probabilities = ndtr(initials@target.alpha*np.sqrt(102./(2.+np.sum(initials**2, axis=1))))
        initials *= np.where(rng.random(config['pilot_chains']) < probabilities, 1., -1.)[:, None]
        q95 = float(np.sqrt(100*f_distribution.ppf(.95, 100, 2.)))
        pilot_jobs = []
        for ci, case in enumerate(map_definitions()):
            for candidate, scale in enumerate(GRIDS[case['algorithm']]):
                for chain in range(config['pilot_chains']):
                    pilot_jobs.append({**case, 'phase': 'pilot', 'target': TARGET, 'chain': chain,
                        'candidate': candidate, 'scale': scale, 'map': maps[case['case']],
                        'seed': 6210001+ci*10000+candidate*100+chain, 'initial': initials[chain].tolist(),
                        'warmup': config['pilot_burn'], 'iterations': config['pilot_retained'], 'q95': q95,
                        'fingerprint': fingerprint,
                        'output': str(root/'pilots'/case['case']/f'{candidate:02d}-{chain:03d}.npz')})
        write_json(root/'pilot_jobs.json', pilot_jobs)
        protocol['status'] = 'independent_pilots'; write_json(root/'protocol.json', protocol)
        started = time.perf_counter()
        run_jobs(pilot_jobs, workers, hardware)
        pilots = summarize_pilots(pilot_jobs)
        pilot_wall_seconds = time.perf_counter()-started
        write_json(root/'pilot_summary.json', pilots)
        print('Pretuned steps: '+json.dumps(pilots['selected']), flush=True)
        if args.preset != 'quick' and not all(s['valid_selection'] for s in pilots['selected'].values()):
            raise RuntimeError('A pilot had no valid candidate; inspect pilot_summary.json before production')
        production_jobs = []
        for case in case_definitions():
            selected = pilots['selected'][case['map_group']]
            for chain in range(config['runs']):
                production_jobs.append({**case, 'phase': 'production', 'target': TARGET, 'chain': chain,
                    'scale': selected['scale'], 'pretuned': True, 'pilot_selection_valid': selected['valid_selection'],
                    'map': maps[case['map_group']], 'seed': 6300001+chain, 'warmup': config['warmup'],
                    'seconds': config['time_budget'], 'thin': config['thin'], 'fingerprint': fingerprint,
                    'output': str(root/'chains'/case['case']/f'{chain:03d}.npz')})
        write_json(root/'production_jobs.json', production_jobs)
        protocol['status'] = 'production'; write_json(root/'protocol.json', protocol)
        started = time.perf_counter()
        for wave in schedule:
            jobs = [j for j in production_jobs if j['case'] == wave['case'] and j['chain'] in wave['chains']]
            print(f'Timed wave {wave["wave"]}: {wave["case"]}, {len(jobs)} workers, {config["time_budget"]} seconds per chain', flush=True)
            run_jobs(jobs, len(jobs), hardware, synchronize=True)
        production_invocation_seconds = time.perf_counter()-started
        if hashes != source_hashes():
            raise RuntimeError('Source changed during audit')
        cases = summarize_production(production_jobs, truth)
        summary = dict(target=TARGET, config=config, fingerprint=fingerprint, source_hashes=hashes,
            truth=truth, maps=maps, fit_metadata=fit_metadata, pilots=pilots, cases=cases,
            contrasts=tail_contrasts(cases), notes=notes, setup={'pilot_wall_seconds': pilot_wall_seconds,
                'fit_wall_seconds': sum(m['seconds'] for m in fit_metadata.values()),
                'production_invocation_seconds': production_invocation_seconds,
                'total_invocation_seconds_before_rendering': time.perf_counter()-invocation})
        write_json(root/'summary.json', summary)
        from scripts.skew_t_timed_figures import render_report
        render_report(root, summary)
        protocol['status'] = 'complete'; write_json(root/'protocol.json', protocol)
        print(f'Equal-time skewed audit complete: {root}', flush=True)
    except BaseException as error:
        protocol.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(root/'protocol.json', protocol)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
