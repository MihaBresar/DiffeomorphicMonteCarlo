"""Explicit configurations, atomic chain checkpoints, and provenance."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import zipfile
import zlib
import numpy as np
from .targets import ROOT, Target
from .tuning import fit_map, validate_ball_map_policy
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
    config['burnin_fraction'] = 1/3
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
    config['ball_map_policy'] = validate_ball_map_policy(config)
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


def resume_source_update(saved_config, config, manifest, hashes):
    """Accept only explicitly registered source-only checkpoint migrations.

    Each registry transition gives the exact before/after SHA256 for every
    changed source file. The original configuration and cohort fingerprint
    remain unchanged; production provenance is recorded separately. This is
    deliberately not a general ``ignore source changes`` resume option.
    """
    error = 'Configuration or scientific source changed. Use a new output directory.'
    previous = manifest['source_hashes']
    if saved_config != config or manifest['fingerprint'] != fingerprint(config, previous):
        raise ValueError(error)
    current_digest = fingerprint(config, hashes)
    if current_digest == manifest['fingerprint']:
        return None
    changes = {path: {'before': previous.get(path), 'after': hashes.get(path)}
               for path in previous.keys() | hashes.keys()
               if path != 'dcs_bench/plotting.py' and previous.get(path) != hashes.get(path)}
    registry_path = ROOT/'validation/resume_compatibility.json'
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        if registry.get('schema_version') != 1:
            raise ValueError('Unsupported resume compatibility registry schema')
        for transition in registry.get('transitions', []):
            if (transition.get('source_changes') == changes
                    and isinstance(transition.get('id'), str) and transition['id']
                    and isinstance(transition.get('reason'), str) and transition['reason']):
                return {'transition_id': transition['id'], 'reason': transition['reason'],
                        'source_changes': changes, 'producing_source_fingerprint': current_digest,
                        'producing_source_hashes': hashes}
    raise ValueError(error)


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


def _chain_compression(draws):
    """Spend compression effort only when a bounded sample saves disk space.

    The same deterministic rule applies to every sampler. Using ``flat`` slices
    avoids making a full contiguous copy of a strided or disk-backed array.
    """
    max_items = max(1, (1024*1024)//draws.dtype.itemsize)
    count = min(draws.size, max_items)
    if count == draws.size:
        spans = [(0, count)]
    else:
        width = count//3
        spans = [(0, width), ((draws.size-width)//2, width),
                 (draws.size-(count-2*width), count-2*width)]
    compressor = zlib.compressobj(level=1)
    sample_bytes = compressed_bytes = 0
    for start, width in spans:
        sample = draws.flat[start:start+width].tobytes()
        sample_bytes += len(sample)
        compressed_bytes += len(compressor.compress(sample))
    compressed_bytes += len(compressor.flush())
    ratio = compressed_bytes/sample_bytes if sample_bytes else 1.0
    compress = ratio < .9
    return (zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED), {
        'compression_policy': 'sample up to 1 MiB across beginning/middle/end; '
                              'deflate level 1 only when sampled size falls below 90%',
        'compression_sample_bytes': sample_bytes,
        'compression_sample_ratio': ratio,
        'compression': 'deflate' if compress else 'stored',
        'compression_level': 1 if compress else None}


def _mapped_draw_source(draws):
    """Find contiguous mapped bytes, including offsets from nested burn-in views."""
    if not draws.flags.c_contiguous or draws.dtype.hasobject:
        return None
    base, mapping = draws, None
    while base is not None:
        if isinstance(base, np.memmap):
            mapping = base
        base = getattr(base, 'base', None)
    # A copy-on-write mapping may contain values that do not exist in its file.
    # Worker spools are read-only; leave other mappings on NumPy's value path.
    if mapping is None or mapping.filename is None or mapping.mode != 'r':
        return None
    relative = draws.__array_interface__['data'][0]-mapping.__array_interface__['data'][0]
    if relative < 0 or relative+draws.nbytes > mapping.nbytes:
        return None
    return Path(mapping.filename), int(mapping.offset)+int(relative)


def _write_chain_draw_array(member, draws):
    """Stream mapped file bytes rather than faulting the entire mapping into RAM."""
    source = _mapped_draw_source(draws)
    if source is None:
        np.lib.format.write_array(member, draws, allow_pickle=False)
        return 'NumPy bounded array buffers'
    path, offset = source
    # This public helper writes the magic/version as well as the array header.
    np.lib.format.write_array_header_2_0(member, np.lib.format.header_data_from_array_1_0(draws))
    remaining = int(draws.nbytes)
    buffer = bytearray(min(8*1024*1024, remaining))
    view = memoryview(buffer)
    with path.open('rb', buffering=0) as stream:
        stream.seek(offset)
        while remaining:
            count = stream.readinto(view[:min(len(buffer), remaining)])
            if not count:
                raise OSError('Temporary mapped draw file ended before all retained draws were saved')
            member.write(view[:count])
            remaining -= count
    return 'Backing file streamed in at most 8 MiB blocks'


def save_chain(path, draws, record, *, temporary_directory=None):
    """Write a lossless NumPy archive with bounded, selective compression.

    Array members are streamed in bounded buffers, including non-contiguous
    burn-in views. A failed write leaves any previous complete archive intact.
    Storage timings are outside the sampling clock and never replace it.
    """
    started = time.perf_counter()
    path = Path(path)
    draws = np.asarray(draws)
    compression_started = time.perf_counter()
    compression, policy = _chain_compression(draws)
    policy['compression_selection_seconds'] = time.perf_counter()-compression_started
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = path.parent if temporary_directory is None else Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+b', prefix=f'.{path.name}.', suffix='.tmp',
                                         dir=temporary_directory, delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, 'w', compression=compression,
                                 compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None,
                                 allowZip64=True) as archive:
                serialization_started = time.perf_counter()
                with archive.open('draws.npy', 'w', force_zip64=True) as member:
                    serialization_strategy = _write_chain_draw_array(member, draws)
                info = archive.getinfo('draws.npy')
                storage = {'format': 'npz', **policy,
                           'draw_serialization_strategy': serialization_strategy,
                           'draw_serialization_seconds': time.perf_counter()-serialization_started,
                           'draw_array_bytes': int(draws.nbytes),
                           'draw_payload_bytes': int(info.file_size),
                           'compressed_draw_bytes': int(info.compress_size)}
                metadata = np.array(json.dumps({**record, 'storage': storage}, allow_nan=False))
                with io.BytesIO() as buffer:
                    np.lib.format.write_array(buffer, metadata, allow_pickle=False)
                    archive.writestr('metadata.npy', buffer.getvalue(),
                                     compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
        os.replace(temporary, path)
        return {**storage, 'save_seconds': time.perf_counter()-started,
                'archive_bytes': path.stat().st_size}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_chain_metadata(path, expected=None):
    """Read the small metadata member without decompressing the draw array.

    This does not validate draws; use ``read_chain`` when their values are used.
    """
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(str(data['metadata']))
    if expected and record['fingerprint'] != expected:
        raise ValueError(f'Checkpoint belongs to a different configuration or source: {path}')
    return record


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
    producing_digest = digest
    existing = output/'manifest.json'
    if existing.exists():
        if not resume:
            raise FileExistsError(f'{output} already contains a run; use --resume or choose another directory')
        manifest = json.loads(existing.read_text())
        saved_config = json.loads((output/'configuration.json').read_text())
        source_update = resume_source_update(saved_config, config, manifest, hashes)
        current_environment = environment()
        for key in ('python', 'platform', 'processor', 'cpu_count', 'cpu_allocation', 'jax_backend', 'jax_devices', 'packages', 'threads'):
            if manifest['environment'].get(key) != current_environment.get(key):
                raise ValueError(f'Resume environment changed ({key}); use a new output directory to avoid mixing benchmark conditions')
        # Keep all checkpoints in the original cohort, while recording the
        # exact source that produced each resumed invocation and new chain.
        digest = manifest['fingerprint']
        if source_update is not None:
            updates = manifest.setdefault('resume_source_updates', [])
            if not any(item.get('producing_source_fingerprint') == producing_digest for item in updates):
                updates.append({'accepted_utc': datetime.now(timezone.utc).isoformat(), **source_update})
    else:
        if resume:
            raise FileNotFoundError('--resume requires an existing manifest')
        output.mkdir(parents=True, exist_ok=True)
        manifest = {'schema_version': 3, 'started_utc': datetime.now(timezone.utc).isoformat(),
            'fingerprint': digest, 'source_hashes': hashes, 'environment': environment(),
            'execution_plan': plan,
            'timing_policy': 'Independent chains in bounded CPU processes, one method at a time. Per-chain sampling wall time includes thinning, physical-coordinate conversion, bounded disk-backed draw collection and final buffer flush; NUTS scalar diagnostic collection is also included. Mapping finalization, NPZ serialization and report processing are outside this clock. After separate tuning, the first third of actual generated sampling iterations is discarded for every method; thinning uses the original iteration grid. Fitting, worker compilation, warmup and total invocation wall time are separately reported. Pinning is recorded; macOS scheduling is OS-managed.',
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
                        print(f'Preparing {family} map for scenario {s}', flush=True)
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
                               'seconds': seconds, 'fingerprint': digest, 'scenario': s,
                               'producing_source_fingerprint': producing_digest,
                               'producing_source_hashes': hashes},
                              pending, directory/'execution', plan)
                compilation = compilation_summary(directory/'execution', algorithm)
                if compilation['workers']:
                    write_json(directory/(('nuts' if algorithm == 'NUTS' else algorithm)+'_compilation.json'), compilation)
                if algorithm == 'NUTS' and config['mode'] == 'nuts_matched_time':
                    records = [read_chain_metadata(directory/'chains'/f'NUTS-{i:03d}.npz', digest) for i in range(config['runs'])]
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
            'wall_seconds': time.perf_counter()-invocation_started, 'resume': resume, 'status': manifest['status'],
            'producing_source_fingerprint': producing_digest, 'producing_source_hashes': hashes})
        write_json(existing, manifest)
    return output
