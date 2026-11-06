"""Named benchmark catalogue, derived from the explicit JSON configurations."""
from dcs_bench.engine import experiment_names, load_config
EXPERIMENTS = {name: load_config(name, 'paper') for name in experiment_names()}

# The two current compactification comparisons share their published settings.
from dcs_bench.reported_study import configuration
EXPERIMENTS["reported-zero-centre"] = configuration("paper", comparison="jax")
EXPERIMENTS["reported-zero-centre-numba"] = configuration("paper", comparison="numba")
