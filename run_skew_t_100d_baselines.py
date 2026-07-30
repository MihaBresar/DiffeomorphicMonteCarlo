"""Convenience entry point for the 100D skew-t baseline benchmark.

Use ``python run_experiment.py skew-t-100d-baselines`` for saved artifacts and
explicit configuration.
"""

from experiments.skew_t_100d_pbw_rwm_nuts import main


if __name__ == "__main__":
    main()
