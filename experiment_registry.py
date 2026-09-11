"""Named benchmark catalogue, derived from the explicit JSON configurations."""
from dcs_bench.engine import experiment_names, load_config
EXPERIMENTS = {name: load_config(name, 'paper') for name in experiment_names()}

# Match the executable default comparison rather than its archived engine settings.
from scripts.reproduce_scs_100d import configuration
EXPERIMENTS["skew-t-100d-fixed"] = configuration("paper")
