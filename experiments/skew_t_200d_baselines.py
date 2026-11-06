"""Run the current 200D random-walk comparison."""
import sys
from run_experiment import main as run_main
EXPERIMENT_ID = 'skew-t-200d-baselines'

def main():
    return run_main([EXPERIMENT_ID]+sys.argv[1:])

if __name__ == '__main__':
    raise SystemExit(main())
