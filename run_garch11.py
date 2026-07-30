"""Convenience entry point for the PosteriorDB GARCH(1,1) benchmark.

Use ``python run_experiment.py garch11`` for saved artifacts and explicit
configuration.
"""

from experiments.garch11 import main


if __name__ == "__main__":
    main()
