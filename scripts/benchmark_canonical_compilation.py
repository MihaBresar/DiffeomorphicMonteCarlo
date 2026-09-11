"""Compare native and Python implementations with identical benchmark kernels."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dcs_bench.cpu import worker_environment
os.environ.update(worker_environment())
import numpy as np
from dcs_bench.compiled_sampling import CompiledRunner
from dcs_bench.engine import load_config, source_hashes
from dcs_bench.samplers import run_numpy_metropolis
from dcs_bench.targets import Target
from dcs_bench.transforms import BallMap, SphereMap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/canonical-compilation.json')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=3)
    args=parser.parse_args()
    cases=[]
    for name, algorithms in [('skew-t-200d-baselines',['DCS-RWM','DCS-BW','RWM-Gaussian','RWM-stable']),
                             ('skew-t-50d-nu-0.5',['DCS-RWM','DCS-BW','SCS']),
                             ('logistic-regression',['DCS-BW']),
                             ('eight-schools',['DCS-HnR']),('garch11',['DCS-HnR'])]:
        config=load_config(name,'quick')
        target=Target(config['targets'][0])
        for algorithm in algorithms:
            mapping=(SphereMap(target.d) if algorithm=='SCS' else
                     BallMap(target.d,beta=config['beta']) if algorithm.startswith('DCS') else None)
            c={**config,'iterations':args.iterations,'thin':10,'warmup':1000}
            runner=CompiledRunner(target,algorithm,mapping,c)
            for repeat in range(args.repeats):
                seed=58121+repeat
                functions=[('python',lambda:run_numpy_metropolis(target,algorithm,mapping,c,seed)),
                           ('native',lambda:runner.run(seed))]
                if repeat%2: functions.reverse()
                result={key:func() for key,func in functions}
                record={key:value[1] for key,value in result.items()}
                row=dict(experiment=name,algorithm=algorithm,repeat=repeat,seed=seed,
                         config=c,mapping=mapping.parameters() if mapping else None,records=record,
                         speedup=record['python']['sampling_seconds']/record['native']['sampling_seconds'])
                cases.append(row)
                print(f'{name} {algorithm} repeat {repeat+1}: {row["speedup"]:.2f}x',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),
        source_hashes=source_hashes(),measurements=cases,notes=[
        'Sequential paired runs with identical target, map, warmup, frozen transition and storage interval.',
        'Compilation and warmup excluded from speedup; same generated sampling counts.',
        'This isolates execution speed, not estimator error or superiority to another sampler.',
        'Floating-point reductions may change long trajectories; density and transition parity have separate tests.']),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
