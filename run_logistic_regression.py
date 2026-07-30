"""Convenience entry point for the robust logistic-regression benchmark.

Use ``python run_experiment.py logistic-regression`` for saved artifacts and
explicit configuration.
"""

from experiments.logistic_regression_student_t import main


if __name__ == "__main__":
    main()
