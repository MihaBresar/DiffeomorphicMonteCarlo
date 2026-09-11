"""Compatibility entry point; scientific implementation lives in dcs_bench."""
import sys
from run_experiment import main as run_main
EXPERIMENT_ID = 'skew-t-100d-fixed'

def main():
    return run_main([EXPERIMENT_ID]+sys.argv[1:])

if __name__ == '__main__':
    raise SystemExit(main())
