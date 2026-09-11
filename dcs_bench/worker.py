"""Internal process entry point: constrain CPU use BEFORE importing NumPy/JAX."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

from .cpu import THREAD_VARIABLES, pin_current_process, worker_environment


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
    for chain in job['chains']:
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
        save_chain(Path(chain['path']), draws, record)
        print(f'Worker {job["worker_slot"]+1}: {algorithm}, chain {chain["index"]+1} saved', flush=True)


if __name__ == '__main__':
    main()
