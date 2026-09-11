"""Internal process entry point: constrain CPU use BEFORE importing NumPy/JAX."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

from .cpu import THREAD_VARIABLES, pin_current_process, worker_environment


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
    from .engine import save_chain, write_json
    from .targets import Target
    from .transforms import BallMap, SphereMap
    from .samplers import BlackJAXRunner, NumPyroRunner, run_metropolis
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
    if 'queue_directory' in job:
        queue = Path(job['queue_directory'])
        (queue/f'worker-{job["worker_slot"]}.ready').touch()
        while not (queue/'start.ready').exists():
            time.sleep(.05)
    for chain in pending_chains(job):
        started = datetime.now(timezone.utc).isoformat()
        print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} starting', flush=True)
        if runner is not None:
            if job.get('reference') or config['mode'] == 'nuts_matched_time':
                draws, record = runner.run(chain['seed'])
            else:
                draws, record = runner.run(chain['seed'], seconds=job['seconds'])
        else:
            draws, record = run_metropolis(target, algorithm, mapping, config, chain['seed'], job['seconds'])
        record.update(fingerprint=job['fingerprint'], scenario=job['scenario'], chain=chain['index'],
                      parameter_names=target.names, execution={
                          'worker_slot': job['worker_slot'], 'pid': os.getpid(), **allocation,
                          'started_utc': started, 'sampling_finished_utc': datetime.now(timezone.utc).isoformat(),
                          'requested_workers': config['workers'],
                          'threads': {key: os.environ.get(key) for key in (*THREAD_VARIABLES, 'XLA_FLAGS')}})
        for field in ('producing_source_fingerprint', 'producing_source_hashes'):
            if field in job:
                record[field] = job[field]
        save_chain(Path(chain['path']), draws, record)
        print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} saved', flush=True)


if __name__ == '__main__':
    main()
