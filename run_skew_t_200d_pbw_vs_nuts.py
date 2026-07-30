"""Convenience entry point for the 200D PBW-versus-NUTS benchmark.

Use ``python run_experiment.py skew-t-200d-nuts`` for saved artifacts and
explicit configuration.
"""

from experiments.skew_t_200d_pbw_vs_nuts import main


if __name__ == "__main__":
    main()
