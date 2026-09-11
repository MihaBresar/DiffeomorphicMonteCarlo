"""Independent joint R/step pilots followed by fresh equal-time radial tests.

This diagnostic changes neither sampler. It profiles radial ESS/second over
step sizes separately at each R, with equal pilot allocations for both methods.
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
from dcs_bench import samplers
from dcs_bench.samplers import MetropolisKernel
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap
from dcs_bench.tuning import fit_map
from dcs_bench.references import radial_truth
from dcs_bench.engine import load_config, environment
from scripts.audit_skew_t_timed import collect_timed, write_json, digest, METRICS, GRIDS
from scripts.fixed_step_pilots import stop_executor

TARGETS = {
    'symmetric-nu3': {'kind': 'skew_t', 'dimension': 100, 'nu': 3., 'alpha': {}},
    'skewed-nu2': {'kind': 'skew_t', 'dimension': 100, 'nu': 2., 'alpha': {'0': 100, '1': -100}}}
CONTROLS = ('joint', 'unit', 'vi', 'sqrt-d-plus-one')
PRESETS = {
    'quick': dict(runs=2, time_budget=1., warmup=100, thin=1, pilot_chains=2,
        pilot_seconds=.02, pilot_burn=50, optimizer_steps=20, optimizer_samples=128,
        probe_chains=2, probe_steps=100),
    'reproduce': dict(runs=8, time_budget=5., warmup=1000, thin=10, pilot_chains=4,
        pilot_seconds=.5, pilot_burn=1000, optimizer_steps=400, optimizer_samples=500,
        probe_chains=2, probe_steps=2000),
    'paper': dict(runs=20, time_budget=20., warmup=1500, thin=50, pilot_chains=4,
        pilot_seconds=.5, pilot_burn=2000, optimizer_steps=1500, optimizer_samples=2000,
        probe_chains=4, probe_steps=10000)}
PROBE_COUNTS = ('iterations', 'initial_boundary_crossings', 'stepout_calls',
    'final_infeasible', 'nonfinite_density', 'density_rejections', 'accepted',
    'accepted_after_stepout')
_ALLOCATION, _ALLOCATION_ERROR, _BARRIER = {}, None, None
SEED_BASES = {'fit': 710000001, 'pilot_initial': 720000001, 'pilot': 721000001,
              'pilot_order': 722000001, 'production': 730000001,
              'probe_initial': 750000001, 'probe': 751000001}


def seed_for(phase, target=0, method=0, radius=0, candidate=0, chain=0):
    indices = (target, method, radius, candidate, chain)
    if any(not isinstance(i, int) or not 0 <= i < limit for i, limit in zip(indices, (10, 2, 8, 8, 100))):
        raise ValueError('Seed index outside registered disjoint blocks')
    return SEED_BASES[phase]+target*100000+method*10000+radius*1000+candidate*100+chain


def source_hashes():
    paths = [*sorted((ROOT/'dcs_bench').glob('*.py')), Path(__file__),
        ROOT/'scripts/audit_skew_t_timed.py', ROOT/'scripts/skew_t_timed_figures.py',
        ROOT/'scripts/joint_radial_figures.py', ROOT/'scripts/fixed_step_pilots.py',
        ROOT/'requirements-cpu.txt', ROOT/'configs/skew-t-100d-fixed.json']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def radius_grid(dimension, nu, fitted_radius):
    """Seven common anchors plus the method's VI radius; always eight cells."""
    values = []
    for radius in [.5, 1., np.sqrt(nu), 2., 4., np.sqrt(dimension+1.), 20., fitted_radius,
                   40., 80., 160.]:
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('R must be finite and positive')
        if radius not in values:
            values.append(radius)
        if len(values) == 8:
            return sorted(values)
    raise ValueError('Insufficient distinct radii')


def exact_initials(spec, count, seed):
    target, rng = Target(spec), np.random.default_rng(seed)
    x = rng.normal(size=(count, target.d))/np.sqrt(rng.chisquare(spec['nu'], (count, 1))/spec['nu'])
    probability = ndtr(x@target.alpha*np.sqrt((spec['nu']+target.d)/(spec['nu']+np.sum(x*x, axis=1))))
    return x*np.where(rng.random(count) < probability, 1., -1.)[:, None]


def case_definitions(target_id):
    labels = {'joint': 'joint R + step', 'unit': 'R = 1', 'vi': 'VI R',
              'sqrt-d-plus-one': 'R = sqrt(d+1)'}
    return [dict(case=algorithm+'-'+control, target_id=target_id, algorithm=algorithm,
        control=control, label=algorithm+': '+labels[control], map_group=algorithm,
        map_kind='fitted', policy='frozen')
        for algorithm in GRIDS for control in CONTROLS]


def production_schedule(config, workers):
    cases = [c for target_id in TARGETS for c in case_definitions(target_id)]
    schedule = []
    for wave, start in enumerate(range(0, config['runs'], workers)):
        offset = (3*wave) % len(cases)
        for case in cases[offset:]+cases[:offset]:
            schedule.append(dict(wave=wave, target_id=case['target_id'], case=case['case'],
                chains=list(range(start, min(start+workers, config['runs'])))))
    return schedule


class BoundaryKernel(MetropolisKernel):
    """Observe canonical transitions, without consuming any extra randomness."""
    def evaluate(self, x):
        value = super().evaluate(x)
        if getattr(self, 'tracking', False):
            self.infeasible = bool(x@x >= 1.) if self.algorithm == 'DCS-BW' else bool(x[-1] >= self.mapping.latitude-1.)
            self.finite_proposal = bool(np.isfinite(value))
        return value


def boundary_probe(kernel, scale, iterations):
    counts = {name: 0 for name in PROBE_COUNTS}
    original = samplers.cap_stepout
    crossed = False

    def observed_stepout(state, proposal, latitude):
        nonlocal crossed
        crossed = bool(proposal[-1] >= latitude-1.)
        return original(state, proposal, latitude)

    samplers.cap_stepout = observed_stepout
    kernel.tracking = True
    try:
        for _ in range(iterations):
            crossed = False
            kernel.infeasible, kernel.finite_proposal = False, False
            _, accepted = kernel.step(scale)
            if kernel.algorithm == 'DCS-BW':
                crossed = kernel.infeasible
            counts['iterations'] += 1
            counts['initial_boundary_crossings'] += int(crossed)
            counts['stepout_calls'] += int(crossed and kernel.algorithm == 'SCS')
            counts['accepted_after_stepout'] += int(accepted and crossed and kernel.algorithm == 'SCS')
            category = ('accepted' if accepted else 'final_infeasible' if kernel.infeasible
                        else 'nonfinite_density' if not kernel.finite_proposal else 'density_rejections')
            counts[category] += 1
    finally:
        kernel.tracking = False
        samplers.cap_stepout = original
    assert sum(counts[k] for k in ('accepted', 'final_infeasible', 'nonfinite_density', 'density_rejections')) == iterations
    return counts


def initialize_worker(slots, barrier):
    global _ALLOCATION, _ALLOCATION_ERROR, _BARRIER
    _BARRIER, _ALLOCATION_ERROR = barrier, None
    try:
        _ALLOCATION = pin_current_process(slots.get())
    except Exception as error:
        _ALLOCATION_ERROR = f'{type(error).__name__}: {error}'


def chain_job(job):
    if _ALLOCATION_ERROR:
        raise RuntimeError(_ALLOCATION_ERROR)
    if job['phase'] not in ('pilot', 'production', 'probe'):
        raise ValueError('Unknown phase')
    if job['phase'] == 'production' and not job.get('pretuned'):
        raise ValueError('Production requires independently pretuned R and step')
    output = Path(job['output'])
    if output.exists():
        raise FileExistsError(output)
    mapping = (BallMap if job['algorithm'] == 'DCS-BW' else SphereMap)(**job['map'])
    cls = BoundaryKernel if job['phase'] == 'probe' else MetropolisKernel
    kernel = cls(Target(job['target']), job['algorithm'], mapping, job['seed'])
    if job.get('initial') is not None:
        kernel.state = mapping.inverse(np.asarray(job['initial']))
        kernel.lp = kernel.evaluate(kernel.state)
    if not np.isfinite(kernel.lp):
        raise ValueError('Nonfinite starting density')
    started = time.perf_counter()
    for _ in range(job['warmup']):
        kernel.step(job['scale'])
    warmup_seconds = time.perf_counter()-started
    if _BARRIER is not None:
        _BARRIER.wait(timeout=180)
    if job['phase'] == 'probe':
        record = boundary_probe(kernel, job['scale'], job['iterations'])
        values = np.empty((0, 2))
    else:
        values, record = collect_timed(kernel, job['scale'], job['seconds'], job['thin'])
    # Saving/compression by one worker must not overlap another's timed window.
    if _BARRIER is not None:
        _BARRIER.wait(timeout=180)
    record.update({k: v for k, v in job.items() if k not in ('map', 'initial', 'output')})
    record.update(map_digest=digest(job['map']), warmup_seconds=warmup_seconds,
        pilot_scale=job['scale'], adaptation_during_sampling=False,
        warmup_adaptation_updates=0, worker={'pid': os.getpid(), **_ALLOCATION})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, functions=values, metadata=np.array(json.dumps(record, allow_nan=False)))
    os.replace(temporary, output)
    return record


def run_waves(jobs, workers, hardware, waves=None):
    """Reuse a pool across equal-size synchronized waves, never queue timed jobs."""
    context = multiprocessing.get_context('spawn')
    executor, slots, previous_count = None, None, 0
    records, futures = [], []
    waves = [jobs[start:start+workers] for start in range(0, len(jobs), workers)] if waves is None else waves
    total, completed = sum(map(len, waves)), 0
    try:
        for wave_index, wave in enumerate(waves):
            count = len(wave)
            if not 1 <= count <= workers:
                raise ValueError('Invalid synchronized wave size')
            if count != previous_count:
                if executor is not None:
                    executor.shutdown(wait=True)
                    slots.close(); slots.join_thread()
                slots = context.Queue()
                for cpu in hardware['cpu_ids'][:count]:
                    slots.put(cpu)
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=count, mp_context=context,
                    initializer=initialize_worker, initargs=(slots, context.Barrier(count)))
                previous_count = count
            futures = [executor.submit(chain_job, job) for job in wave]
            records.extend(future.result() for future in futures)
            completed += count
            if total > workers and (wave_index % 8 == 0 or completed == total):
                print(f'{completed}/{total} {wave[0]["phase"]} chains saved', flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        if executor is not None:
            stop_executor(executor)
        raise
    else:
        if executor is not None:
            executor.shutdown(wait=True)
    finally:
        if slots is not None:
            slots.close(); slots.join_thread()
    return records


def pilot_waves(jobs, workers, target_index):
    """Alternate homogeneous method blocks, reversing their order every pair."""
    groups = []
    for ai, algorithm in enumerate(GRIDS):
        group = [j for j in jobs if j['algorithm'] == algorithm]
        np.random.default_rng(seed_for('pilot_order', target=target_index, method=ai)).shuffle(group)
        groups.append(group)
    if len(groups[0]) != len(groups[1]):
        raise ValueError('Methods require equal pilot chain counts')
    waves = []
    for index, start in enumerate(range(0, len(groups[0]), workers)):
        for ai in ((0, 1) if index % 2 == 0 else (1, 0)):
            waves.append(groups[ai][start:start+workers])
    return waves


def load_chain(job):
    with np.load(job['output'], allow_pickle=False) as saved:
        values, record = saved['functions'].copy(), json.loads(str(saved['metadata']))
    for key, value in job.items():
        if key not in ('map', 'initial', 'output') and record.get(key) != value:
            raise ValueError(f'Chain provenance mismatch: {key}')
    if record['map_digest'] != digest(job['map']) or record['adaptation_during_sampling']:
        raise ValueError('Chain map or frozen-step mismatch')
    expected = (0, 2) if job['phase'] == 'probe' else (record['iterations']//job['thin'], 2)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError('Invalid radial traces')
    return values, record


def score_pilot(arrays, seconds):
    """Sum individual-chain ESS; no truncation of unequal timed chain lengths."""
    import arviz as az
    if len(arrays) != len(seconds) or not arrays or any(s <= 0 for s in seconds):
        raise ValueError('One positive measured sampling time required per pilot chain')
    functions = [np.column_stack([a[:, 0], np.log1p(a[:, 1])]) for a in arrays]
    bulk, rhat, valid = {}, {}, True
    common = min(map(len, functions))
    for index, name in enumerate(('tail_probability', 'log1p_norm')):
        chains = [a[:, index] for a in functions]
        if common < 4 or any(not np.all(np.isfinite(a)) or np.ptp(a) == 0 for a in chains):
            bulk[name], rhat[name], valid = None, None, False
            continue
        ess = [float(az.ess(a[None, :], method='bulk')) for a in chains]
        rh = float(az.rhat(np.asarray([a[:common] for a in chains]), method='rank'))
        bulk[name] = float(sum(ess)) if np.all(np.isfinite(ess)) else None
        rhat[name] = rh if np.isfinite(rh) else None
        valid = valid and bulk[name] is not None and rhat[name] is not None
    total = float(sum(seconds))
    return dict(bulk_ess=bulk, rank_rhat=rhat, valid_selection=valid,
        score=min(bulk.values())/total if valid else 0., total_sampling_seconds=total,
        iterations_per_chain=list(map(len, arrays)), rhat_common_prefix=common)


def select_profiles(rows, fitted_radii, dimension):
    profiles, selected = [], {}
    for algorithm in GRIDS:
        candidates = [r for r in rows if r['algorithm'] == algorithm]
        for radius in sorted({r['R'] for r in candidates}):
            profile = max((r for r in candidates if r['R'] == radius), key=lambda r: (r['score'], -r['scale']))
            profiles.append({**profile, 'step_endpoint': profile['scale'] in (GRIDS[algorithm][0], GRIDS[algorithm][-1])})
        choices = [p for p in profiles if p['algorithm'] == algorithm]
        joint = max(choices, key=lambda r: (r['score'], -r['R']))
        radii = {'unit': 1., 'vi': fitted_radii[algorithm], 'sqrt-d-plus-one': float(np.sqrt(dimension+1))}
        for control in CONTROLS:
            choice = joint if control == 'joint' else next(p for p in choices if p['R'] == radii[control])
            selected[algorithm+'-'+control] = {**choice,
                'radius_endpoint': choice['R'] in (min(p['R'] for p in choices), max(p['R'] for p in choices))}
    return {'grid': rows, 'profiles': profiles, 'selected': selected}


def summarize_cases(jobs, truth, target_id):
    cases = {}
    for case in case_definitions(target_id):
        group = sorted((j for j in jobs if j['case'] == case['case']), key=lambda j: j['chain'])
        records = [load_chain(j)[1] for j in group]
        estimates = {n: [r['estimates'][n] for r in records] for n in METRICS}
        errors = {n: (np.asarray(estimates[n])-truth[n])**2 for n in METRICS}
        counts = [r['iterations'] for r in records]
        cases[case['case']] = {**case, 'R': records[0]['R'], 'scale': records[0]['scale'],
            'pilot_scale': records[0]['scale'], 'records': records, 'estimates': estimates,
            'squared_errors': {n: e.tolist() for n, e in errors.items()},
            'empirical_mse': {n: float(e.mean()) for n, e in errors.items()},
            'median_absolute_error': {n: float(np.median(np.abs(np.asarray(estimates[n])-truth[n]))) for n in METRICS},
            'mean_iterations': float(np.mean(counts)), 'total_iterations': sum(counts),
            'mean_sampling_seconds': float(np.mean([r['sampling_seconds'] for r in records]))}
    return cases


def tail_contrasts(cases):
    count = len(next(iter(cases.values()))['records'])
    indices = np.random.default_rng(7410001).integers(0, count, (20000, count))
    pairs = [('DCS-BW-'+c, 'SCS-'+c) for c in CONTROLS]
    pairs += [(a+'-joint', a+'-'+c) for a in GRIDS for c in CONTROLS if c != 'joint']
    rows = []
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
    hashes = source_hashes()
    fingerprint = digest({'targets': TARGETS, 'config': config, 'source_hashes': hashes})
    schedule = production_schedule(config, min(workers, config['runs']))
    notes = [
        'Joint R/step search uses equal pilot seconds, repetitions, candidate counts and discarded warmup iterations for both methods. Step values differ because the proposal parameterizations differ.',
        'At each R select the step maximizing the minimum summed per-chain bulk ESS for tail(||X||>=50) and log1p(||X||), divided by summed actual sampling wall time. Then select R by this profiled score.',
        'Timed pilot chain lengths can differ: ESS is computed on each full chain separately and summed. Rank R-hat is descriptive on common-length prefixes. A constant or nonfinite pilot function in any chain invalidates that cell.',
        'Eight R values include 0.5, 1, sqrt(nu), 2, 4, sqrt(d+1), 20 and the method-specific VI R. Exact duplicates are replaced by 40, then 80. Eight steps per R; all cells are retained. Endpoint winners are reported, not certified global optima.',
        'Both methods have equal VI optimizer iterations/batch sizes. VI centre and deformation are held fixed during the subsequent R/step search. Thus the skewed comparison is conditional on each fitted shape. Symmetric targets constrain centre and deformation exactly to zero.',
        'All four controls per method have independently pretuned steps: joint R+step, R=1, VI R, and R=sqrt(d+1). These are fixed-R controls with the same fitted shape; R=1 is a fully untuned map only on the symmetric target.',
        'Exact independent target draws initialize pilots and separate boundary probes: an oracle-assisted synthetic diagnostic. Production uses fresh seeds and physical ones(100). Pilot outcomes never use production errors.',
        'Maps and proposal steps are frozen throughout discarded production warmup and sampling. Existing adaptive audits remain available separately.',
        'Homogeneous timed waves have identical worker allocation across cases; production case order rotates. Pilot method blocks alternate, reversing order every pair, to reduce timing drift. Barriers before and after timed windows keep worker warmup and saving outside all simultaneous sampling windows.',
        'Each pilot and production wall budget includes canonical transitions, all-iteration tail/norm accumulation and trace collection. VI, warmup, process startup, ESS computation and file saving are excluded and recorded separately. Final-transition overshoot and CPU seconds are retained.',
        'Separate fixed-count probes observe the unmodified kernels without additional random draws. Initial SCS cap crossings invoke great-circle step-out before MH. Initial DCS ball crossings reject. Final infeasibility, nonfinite interior density, finite-density rejection and acceptance partition probe transitions.',
        'Every completed iteration including rejected repeats contributes to reported production estimates. Saved thinning does not determine the estimates. Actual timed stopping gives random, potentially state-dependent sample counts.',
        'At nu=2 raw norm has a finite mean but infinite second moment. Its average squared error and median absolute error are descriptive; the finite-variance log1p(norm) proxy is used for tuning. Only tail-MSE differences receive descriptive paired bootstrap intervals (20000 resamples, seed7410001), conditional on pilots, no multiplicity adjustment.',
        'Quick check is execution-only. Its short pilots may have invalid cells and deterministic fallback selections; no scientific ranking is claimed. Reproduce and paper stop if any requested production control has no valid pilot.',
        'Timing depends on hardware and OS scheduling. Equal pilot budgets need not imply equal precision of the selected scores, and a finite grid is not proof of optimality.']
    protocol = dict(schema_version=1, registered_before_sampling=True, status='fitting',
        started_utc=datetime.now(timezone.utc).isoformat(), targets=TARGETS, config=config,
        beta=2., latitude=1.1, threshold=50., step_grids=GRIDS, controls=list(CONTROLS),
        R_anchors=[.5, 1., 'sqrt(nu)', 2., 4., 'sqrt(d+1)', 20., 'VI R'],
        source_hashes=hashes, fingerprint=fingerprint, environment=environment(), hardware=hardware,
        production_schedule=schedule, notes=notes,
        seeds={'phase_bases': SEED_BASES,
               'rule': 'base + target_index*100000 + method_index*10000 + R_index*1000 + step_index*100 + chain',
               'sharing': 'Unused indices are zero. Fit/initial/production/probe streams omit method/R/step indices; production and probe retain chain index. Pilot streams use every index; pilot-order streams use target/method only.'})
    write_json(root/'protocol.json', protocol)
    invocation = time.perf_counter()
    try:
        maps, fits, all_pilots, target_data = {}, {}, [], {}
        for ti, (target_id, spec) in enumerate(TARGETS.items()):
            _, values, _ = radial_truth(Target(spec), 50.)
            target_data[target_id] = dict(target=spec, config=config,
                truth=dict(zip(METRICS, map(float, values))), notes=notes)
            maps[target_id], fits[target_id] = {}, {}
            vi_config = load_config('skew-t-100d-fixed', 'paper', {'workers': workers})
            vi_config.update(optimizer_steps=config['optimizer_steps'], optimizer_samples=config['optimizer_samples'],
                ball_map_policy='mobius', beta=2., targets=[spec], map_scale_initialization='radial_median')
            initials = exact_initials(spec, config['pilot_chains'], seed_for('pilot_initial', target=ti))
            for ai, algorithm in enumerate(GRIDS):
                print(f'Fitting {target_id}: {algorithm}', flush=True)
                mapping, meta = fit_map(Target(spec), vi_config, 'ball' if algorithm == 'DCS-BW' else 'sphere', seed_for('fit', target=ti))
                parameters = mapping.parameters()
                maps[target_id][algorithm], fits[target_id][algorithm] = parameters, meta
                for ri, radius in enumerate(radius_grid(spec['dimension'], spec['nu'], parameters['scale'])):
                    for hi, scale in enumerate(GRIDS[algorithm]):
                        for chain in range(config['pilot_chains']):
                            all_pilots.append(dict(phase='pilot', target_id=target_id, target=spec,
                                case=algorithm, algorithm=algorithm, R=radius, R_index=ri, candidate=hi,
                                scale=scale, chain=chain, seed=seed_for('pilot', target=ti, method=ai, radius=ri, candidate=hi, chain=chain),
                                initial=initials[chain].tolist(), map={**parameters, 'scale': radius},
                                seconds=config['pilot_seconds'], warmup=config['pilot_burn'], thin=1,
                                fingerprint=fingerprint, output=str(root/'pilots'/target_id/algorithm/f'{ri}-{hi}-{chain}.npz')))
        write_json(root/'maps.json', {'maps': maps, 'fit_metadata': fits})
        write_json(root/'pilot_jobs.json', all_pilots)
        pilot_schedule = {target_id: pilot_waves([j for j in all_pilots if j['target_id'] == target_id], workers, ti)
                          for ti, target_id in enumerate(TARGETS)}
        write_json(root/'pilot_schedule.json', {t: [[j['output'] for j in wave] for wave in waves]
                                              for t, waves in pilot_schedule.items()})
        protocol['status'] = 'independent_pilots'; write_json(root/'protocol.json', protocol)
        pilot_started = time.perf_counter()
        for target_id, waves in pilot_schedule.items():
            print(f'Equal-time pilots {target_id}: {sum(map(len, waves))} chains, alternating methods, {config["pilot_seconds"]} seconds each', flush=True)
            run_waves([], workers, hardware, waves=waves)
        pilot_sampling_invocation_seconds = time.perf_counter()-pilot_started
        for target_id, spec in TARGETS.items():
            rows = []
            for algorithm in GRIDS:
                radii = radius_grid(spec['dimension'], spec['nu'], maps[target_id][algorithm]['scale'])
                for ri, radius in enumerate(radii):
                    for hi, scale in enumerate(GRIDS[algorithm]):
                        group = sorted((j for j in all_pilots if j['target_id'] == target_id and j['algorithm'] == algorithm
                                        and j['R_index'] == ri and j['candidate'] == hi), key=lambda j: j['chain'])
                        loaded = [load_chain(j) for j in group]
                        rows.append(dict(algorithm=algorithm, R=radius, scale=scale, R_index=ri, candidate=hi,
                            **score_pilot([v for v, _ in loaded], [r['sampling_seconds'] for _, r in loaded])))
            pilots = select_profiles(rows, {a: p['scale'] for a, p in maps[target_id].items()}, spec['dimension'])
            target_data[target_id].update(pilots=pilots, maps=maps[target_id], fit_metadata=fits[target_id])
            write_json(root/(target_id+'-pilot-summary.json'), pilots)
            for name, selection in pilots['selected'].items():
                print(f'Selected {target_id} {name}: R={selection["R"]:.10g}, step={selection["scale"]:g}, radial ESS/s={selection["score"]:.8g}, valid={selection["valid_selection"]}', flush=True)
            if args.preset != 'quick' and not all(s['valid_selection'] for s in pilots['selected'].values()):
                raise RuntimeError(f'{target_id}: a production control has no valid pilot; inspect all candidates')
        pilot_wall_seconds = time.perf_counter()-pilot_started
        jobs, probes = [], []
        for ti, (target_id, spec) in enumerate(TARGETS.items()):
            initials = exact_initials(spec, config['probe_chains'], seed_for('probe_initial', target=ti))
            for case in case_definitions(target_id):
                selected = target_data[target_id]['pilots']['selected'][case['case']]
                base = dict(**case, target=spec, R=selected['R'], scale=selected['scale'],
                    map={**maps[target_id][case['algorithm']], 'scale': selected['R']},
                    pretuned=True, pilot_selection_valid=selected['valid_selection'], fingerprint=fingerprint)
                for chain in range(config['runs']):
                    jobs.append(dict(**base, phase='production', chain=chain, seed=seed_for('production', target=ti, chain=chain),
                        seconds=config['time_budget'], warmup=config['warmup'], thin=config['thin'],
                        output=str(root/'chains'/target_id/case['case']/f'{chain:03d}.npz')))
                for chain in range(config['probe_chains']):
                    probes.append(dict(**base, phase='probe', chain=chain, seed=seed_for('probe', target=ti, chain=chain),
                        initial=initials[chain].tolist(), warmup=0, iterations=config['probe_steps'],
                        output=str(root/'probes'/target_id/case['case']/f'{chain:03d}.npz')))
        write_json(root/'production_jobs.json', jobs); write_json(root/'probe_jobs.json', probes)
        protocol['status'] = 'production'; write_json(root/'protocol.json', protocol)
        started = time.perf_counter()
        for wave in schedule:
            group = [j for j in jobs if j['target_id'] == wave['target_id'] and j['case'] == wave['case'] and j['chain'] in wave['chains']]
            print(f'Timed wave {wave["wave"]}: {wave["target_id"]} {wave["case"]}, {len(group)} chains x {config["time_budget"]} seconds', flush=True)
            run_waves(group, len(group), hardware)
        production_invocation_seconds = time.perf_counter()-started
        protocol['status'] = 'boundary_probes'; write_json(root/'protocol.json', protocol)
        started = time.perf_counter()
        print('Separate boundary probes (excluded from all timed windows)', flush=True)
        run_waves(probes, workers, hardware)
        probe_wall_seconds = time.perf_counter()-started
        if hashes != source_hashes():
            raise RuntimeError('Source changed during audit')
        for target_id in TARGETS:
            data = target_data[target_id]
            data['cases'] = summarize_cases([j for j in jobs if j['target_id'] == target_id], data['truth'], target_id)
            data['contrasts'] = tail_contrasts(data['cases'])
            data['boundary'] = {}
            for case in case_definitions(target_id):
                records = [load_chain(j)[1] for j in probes if j['target_id'] == target_id and j['case'] == case['case']]
                data['boundary'][case['case']] = {k: sum(r[k] for r in records) for k in PROBE_COUNTS}
                data['boundary'][case['case']]['records'] = records
        summary = dict(schema_version=1, config=config, targets=target_data, notes=notes,
            source_hashes=hashes, fingerprint=fingerprint, environment=protocol['environment'], hardware=hardware,
            setup={'fit_wall_seconds': sum(m['seconds'] for t in fits.values() for m in t.values()),
                'pilot_sampling_invocation_seconds': pilot_sampling_invocation_seconds,
                'pilot_wall_seconds': pilot_wall_seconds, 'production_invocation_seconds': production_invocation_seconds,
                'probe_wall_seconds': probe_wall_seconds,
                'total_invocation_seconds_before_rendering': time.perf_counter()-invocation})
        write_json(root/'summary.json', summary)
        from scripts.joint_radial_figures import render_report
        render_report(root, summary)
        protocol['status'] = 'complete'; write_json(root/'protocol.json', protocol)
        print(f'Joint radial audit complete: {root}', flush=True)
    except BaseException as error:
        protocol.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(root/'protocol.json', protocol)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
