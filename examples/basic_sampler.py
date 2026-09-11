"""Run from repository root: python -m examples.basic_sampler."""
import numpy as np
from dcs_bench.api import sample


def log_student_t(x):
    return -(3+len(x))/2*np.log1p(np.dot(x, x)/3)


if __name__ == '__main__':
    draws, diagnostics = sample(log_student_t, initial=np.zeros(2),
                                draws=10000, warmup=1500, beta=2., seed=81)
    print('Draw shape:', draws.shape)
    print('Estimated P(||X|| >= 3):', np.mean(np.linalg.norm(draws, axis=1) >= 3))
    print('Exact value:', (1+3**2/3)**(-3/2))
    print('Frozen proposal scale:', diagnostics['frozen_scale'])
