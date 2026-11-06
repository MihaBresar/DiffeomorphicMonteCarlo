"""Internal process entry point: constrain CPU use BEFORE importing NumPy/JAX."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

from .cpu import CPU_THREAD_POLICY, THREAD_VARIABLES, pin_current_process, worker_environment


def pending_chains(job):
    """Atomic claims let a free worker take another chain immediately.

    A fresh queue is created on each launch. Failed claims do not remove any
    completed checkpoint; resume rebuilds the queue from missing checkpoints.
    """
    if 'queue_directory' not in job:
        yield from job['chains']
        return
    queue = Path(job['queue_directory'])
    for path in sorted(queue.glob('*.pending.json')):
        # Every contender must create the SAME claim directory. Keep the input
        # immutable: a successful rename is not an exclusive ownership primitive
        # on all platforms, and a competing move can remove its destination.
        claim = path.with_name(path.name.removesuffix('.pending.json')+'.claim')
        try:
            claim.mkdir()
        except FileExistsError:
            continue  # Another worker claimed this chain.
        chain = json.loads(path.read_text(encoding='utf-8'))
        (claim/'owner.json').write_text(json.dumps({
            'worker_slot': job['worker_slot'], 'chain': chain['index']})+'\n', encoding='utf-8')
        yield chain


def main():
    path = Path(sys.argv[1])
    job = json.loads(path.read_text(encoding='utf-8'))
    os.environ.update(worker_environment())
    allocation = pin_current_process(job['cpu_id'])
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    from .engine import write_json
    from .targets import Target
    from .transforms import BallMap, SphereMap
    from .samplers import BlackJAXRunner, NumPyroRunner, run_metropolis, make_metropolis_runner
    target, config = Target(job['target']), job['config']
    algorithm = job['algorithm']
    runner = None
    if algorithm == 'NUTS':
        runner = (NumPyroRunner if job.get('reference') or config['mode'] == 'nuts_matched_time'
                  else BlackJAXRunner)(target, config)
        write_json(path.with_suffix('.compilation.json'), {'seconds': runner.compile_seconds,
                   'worker_slot': job['worker_slot'], 'pid': os.getpid(), **allocation})
    family = 'sphere' if algorithm == 'SCS' else 'ball'
    params = job.get('maps', {}).get(family) if algorithm == 'SCS' or algorithm.startswith('DCS') else None
    mapping = (SphereMap if family == 'sphere' else BallMap)(**params) if params else None
    if algorithm != 'NUTS':
        runner = make_metropolis_runner(target, algorithm, mapping, config)
        if runner is not None:
            write_json(path.with_suffix('.compilation.json'), {'seconds': runner.compile_seconds,
                       'worker_slot': job['worker_slot'], 'pid': os.getpid(), **allocation})
    if 'queue_directory' in job:
        queue = Path(job['queue_directory'])
        (queue/f'worker-{job["worker_slot"]}.ready').touch()
        while not (queue/'start.ready').exists():
            time.sleep(.05)
    for chain in pending_chains(job):
        _run_and_save_chain(job, chain, target, config, algorithm, runner, mapping, allocation)


def _run_and_save_chain(job, chain, target, config, algorithm, runner, mapping, allocation):
    from .draw_storage import DiskDrawCollector
    from .engine import save_chain
    directory = job.get("draw_directory", str(Path(chain["path"]).parent))
    with DiskDrawCollector(target.d, directory=directory) as collector:
        started = datetime.now(timezone.utc).isoformat()
        print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} starting', flush=True)
        if runner is not None:
            if algorithm == 'NUTS' and (job.get('reference') or config['mode'] == 'nuts_matched_time'):
                draws, record = runner.run(chain['seed'], collector=collector)
            else:
                draws, record = runner.run(chain['seed'], seconds=job['seconds'], collector=collector)
        else:
            draws, record = run_metropolis(target, algorithm, mapping, config, chain['seed'], job['seconds'], collector=collector)
        from .burnin import discard_burnin
        draws, record = discard_burnin(draws, record, config)
        record.update(fingerprint=job['fingerprint'], scenario=job['scenario'], chain=chain['index'],
                      parameter_names=target.names, execution={
                          'worker_slot': job['worker_slot'], 'pid': os.getpid(), **allocation,
                          'started_utc': started, 'sampling_finished_utc': datetime.now(timezone.utc).isoformat(),
                          'requested_workers': config['workers'],
                          'cpu_thread_policy': CPU_THREAD_POLICY,
                          'threads': {key: os.environ.get(key) for key in (*THREAD_VARIABLES, 'XLA_FLAGS')}})
        for field in ('producing_source_fingerprint', 'producing_source_hashes'):
            if field in job:
                record[field] = job[field]
        from .radial_reporting import compact_radial_report, summarize_radial
        if compact_radial_report(config):
            print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} '
                  f'summarizing tail probability and radial moment from {len(draws):,} retained draws', flush=True)
            options = (dict(sampling_seconds=record.get('sampling_seconds'), draw_stride=record.get('thin', 1), ess_enabled=True)
                       if config.get('ess_enabled', False) else {})
            record['radial_summary'] = summarize_radial(draws, target, config['threshold'], **options)
        print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} '
              f'sampling complete ({record["sampling_seconds"]:.2f} s); '
              f'saving {len(draws):,} retained draws ({draws.nbytes/1_000_000:.2f} MB)', flush=True)
        storage = save_chain(Path(chain['path']), draws, record, temporary_directory=collector.directory)
        print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} '
              f'saved in {storage["save_seconds"]:.2f} s '
              f'({storage["archive_bytes"]/1_000_000:.2f} MB)', flush=True)
        del draws, record, storage


if __name__ == '__main__':
    main()
