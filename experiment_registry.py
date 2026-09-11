"""Named benchmark catalogue, derived from the explicit JSON configurations."""
from dcs_bench.engine import experiment_names, load_config
EXPERIMENTS = {name: load_config(name, 'paper') for name in experiment_names()}
