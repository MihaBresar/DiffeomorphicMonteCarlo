"""Explicit configurations, atomic chain checkpoints, and provenance."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from .targets import ROOT, Target
from .tuning import fit_map
from .references import make_reference
from .cpu import cpu_info, execution_plan, validate_workers
from .parallel import launch_chains, compilation_summary

ALIASES = {'skew-t-100d-baselines': 'skew-t-200d-baselines'}


def experiment_names():
    return sorted(p.stem for p in (ROOT/'configs').glob('*.json') if p.stem != 'presets')


def load_config(name, preset='quick', overrides=None):
    name = ALIASES.get(name, name)
    config = json.loads((ROOT/'configs'/f'{name}.json').read_text())
    if preset in ('quick', 'reproduce'):
        config.update(json.loads((ROOT/'configs/presets.json').read_text())[preset])
    elif preset != 'paper':
        raise ValueError(f'Unknown preset: {preset}')
    config['preset'] = preset
    overrides = overrides or {}
    irrelevant = []
    if config['mode'] != 'iterations':
        irrelevant.append('iterations')
    if config['mode'] != 'time':
        irrelevant.append('time_budget')
    if config['mode'] != 'nuts_matched_time':
        irrelevant.append('nuts_samples')
    if 'NUTS' not in config['algorithms']:
        irrelevant += ['nuts_max_depth', 'nuts_batch_size', 'nuts_target_accept']
    if config['targets'][0]['kind'] != 'skew_t':
        irrelevant += ['reference_samples', 'reference_batch_size']
    if config['targets'][0]['kind'] in ('garch', 'eight_schools'):
        irrelevant += ['optimizer_steps', 'optimizer_samples']
    unused = [k for k in irrelevant if overrides.get(k) is not None]
    if unused:
        raise ValueError(f'Overrides not used by {name}: {unused}')
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if config.get('ball_scale_policy', 'fit') not in ('fit', 'sqrt-d'):
        raise ValueError('Unknown ball scale policy')
    for field in ('runs', 'warmup', 'thin', 'iterations', 'nuts_samples', 'nuts_batch_size', 'reference_samples', 'reference_batch_size', 'optimizer_samples'):
        if config[field] <= 0:
            raise ValueError(f'{field} must be positive')
    if config['time_budget'] <= 0 or config['optimizer_steps'] < 0:
        raise ValueError('Time budget must be positive and optimizer steps nonnegative')
    validate_workers(config.get('workers', 1))
    return config


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def source_hashes():
    paths = list((ROOT/'dcs_bench').glob('*.py'))+list((ROOT/'configs').glob('*.json'))
    paths += [ROOT/'run_experiment.py', ROOT/'prepare_reference.py', ROOT/'requirements-cpu.txt', ROOT/'data/posteriordb/manifest.json']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fingerprint(config, hashes):
    # Presentation edits can replay saved chains; scientific edits require a new run.
    scientific = {k: v for k, v in hashes.items() if k != 'dcs_bench/plotting.py'}
    return hashlib.sha256(json.dumps([config, scientific], sort_keys=True).encode()).hexdigest()


def environment():
    packages = {p.metadata['Name']: p.version for p in importlib.metadata.distributions()}
    import jax
    result = {'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'cpu_count': os.cpu_count(), 'cpu_allocation': cpu_info(),
        'jax_backend': jax.default_backend(), 'jax_devices': [str(d) for d in jax.devices()],
        'packages': packages, 'threads': {k: os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'XLA_FLAGS')}}
    for name, args in [('git_commit', ['git', 'rev-parse', 'HEAD']), ('git_status', ['git', 'status', '--short'])]:
        try:
            proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
            result[name] = proc.stdout.strip() if proc.returncode == 0 else None
        except OSError:
            result[name] = None  # ZIP downloads do not require Git to be installed.
    return result


def chain_seed(config, scenario, algorithm_index, chain):
    return int(np.random.SeedSequence([config['seed'], scenario, algorithm_index, chain]).generate_state(1)[0])


def save_chain(path, draws, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, draws=draws, metadata=np.array(json.dumps(record, allow_nan=False)))
    os.replace(temporary, path)


def read_chain(path, expected=None):
    with np.load(path, allow_pickle=False) as data:
        draws = data['draws']
        record = json.loads(str(data['metadata']))
    if expected and record['fingerprint'] != expected:
        raise ValueError(f'Checkpoint belongs to a different configuration or source: {path}')
    if draws.ndim != 2 or len(draws) < 4 or not np.all(np.isfinite(draws)):
        raise ValueError(f'Invalid or empty checkpoint: {path}')
    return draws, record


def run(config, output, resume=False, reference_dir=None):
    output = Path(output).resolve()
    plan = execution_plan(config['workers'], config['runs'])
    hashes = source_hashes()
    digest = fingerprint(config, hashes)
    existing = output/'manifest.json'
    if existing.exists():
        if not resume:
            raise FileExistsError(f'{output} already contains a run; use --resume or choose another directory')
        manifest = json.loads(existing.read_text())
        if manifest['fingerprint'] != digest:
            raise ValueError('Configuration or scientific source changed. Use a new output directory.')
        current_environment = environment()
        for key in ('python', 'platform', 'processor', 'cpu_count', 'cpu_allocation', 'jax_backend', 'jax_devices', 'packages', 'threads'):
            if manifest['environment'].get(key) != current_environment.get(key):
                raise ValueError(f'Resume environment changed ({key}); use a new output directory to avoid mixing benchmark conditions')
    else:
        if resume:
            raise FileNotFoundError('--resume requires an existing manifest')
        output.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version': 3, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'fingerprint': digest, 'source_hashes': hashes, 'environment': environment(),
            'execution_plan': plan,
            'timing_policy': 'Independent chains in bounded CPU processes, one method at a time. Per-chain production wall time includes retained-draw collection. Fitting, worker compilation, warmup and total invocation wall time are separately reported. Pinning is recorded; macOS scheduling is OS-managed.',
            'status': 'running'}
        write_json(output/'configuration.json', config)
        write_json(existing, manifest)
    manifest['status'] = 'running'
    manifest.pop('failure', None)
    write_json(existing, manifest)
    invocation_started = time.perf_counter()
    try:
        for s, spec in enumerate(config['targets']):
            target = Target(spec)
            directory = output/f'scenario-{s}'
            directory.mkdir(exist_ok=True)
            if not (directory/'reference.json').exists():
                print(f'Preparing independent reference for scenario {s}', flush=True)
                make_reference(target, config, directory, reference_dir)
            setup_path = directory/'setup.json'
            if setup_path.exists():
                setup = json.loads(setup_path.read_text())
            else:
                setup = {}
                for family, needed in [('ball', any(a.startswith('DCS') for a in config['algorithms'])),
                                       ('sphere', 'SCS' in config['algorithms'])]:
                    if needed:
                        print(f'Fitting {family} map for scenario {s}', flush=True)
                        mapping, meta = fit_map(target, config, family, config['seed']+s+811)
                        setup[family] = {'parameters': mapping.parameters(), 'optimization': meta}
                write_json(setup_path, setup)
            maps = {k: v['parameters'] for k, v in setup.items()}
            matched_seconds = None
            for ai, algorithm in enumerate(config['algorithms']):
                pending = []
                for chain in range(config['runs']):
                    path = directory/'chains'/f'{algorithm}-{chain:03d}.npz'
                    if path.exists():
                        read_chain(path, digest)
                        print(f'Resumed {s}/{algorithm}/{chain}', flush=True)
                        continue
                    pending.append({'index': chain, 'seed': chain_seed(config, s, ai, chain), 'path': str(path)})
                seconds = None if config['mode'] == 'iterations' else (matched_seconds if config['mode'] == 'nuts_matched_time' else config['time_budget'])
                print(f'Scenario {s}, {algorithm}: {len(pending)} pending chains, up to {plan["max_concurrent_chains"]} workers', flush=True)
                launch_chains({'target': spec, 'config': config, 'algorithm': algorithm, 'maps': maps,
                               'seconds': seconds, 'fingerprint': digest, 'scenario': s},
                              pending, directory/'execution', plan)
                if algorithm == 'NUTS':
                    write_json(directory/'nuts_compilation.json', compilation_summary(directory/'execution'))
                if algorithm == 'NUTS' and config['mode'] == 'nuts_matched_time':
                    records = [read_chain(directory/'chains'/f'NUTS-{i:03d}.npz', digest)[1] for i in range(config['runs'])]
                    matched_seconds = float(np.mean([r['sampling_seconds'] for r in records]))
                    if config['preset'] == 'quick':
                        matched_seconds = max(matched_seconds, config['time_budget'])
                    write_json(directory/'matched_budget.json', {'seconds_per_DCS_chain': matched_seconds,
                        'basis': 'Mean post-warmup NUTS sampling time; setup separately reported',
                        'quick_only_minimum_seconds': config['time_budget'] if config['preset'] == 'quick' else None})
        from .plotting import rebuild
        rebuild(output)
        manifest['status'] = 'complete'
        manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        manifest.setdefault('invocations', []).append({'ended_utc': datetime.now(timezone.utc).isoformat(),
            'wall_seconds': time.perf_counter()-invocation_started, 'resume': resume, 'status': manifest['status']})
        write_json(existing, manifest)
    return output
