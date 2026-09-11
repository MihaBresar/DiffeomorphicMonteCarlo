"""Historical input loader for the controlled audit, not a current sampler.

validation/legacy_fixed_step.py preserves the supplied source byte for byte.
Only the two VI functions are extracted; its script entry point, historical
workers, plotting, and reference generator never execute. Its nu=2 default is
explicitly overridden to the audit's common nu=3 target.
"""
import ast
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1]/'validation/legacy_fixed_step.py'


def load_legacy(*, nu=3., dimension=100, optimizer_steps=1500, optimizer_samples=2000):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jgammaln
    import jax.scipy.stats as jstats
    jax.config.update('jax_enable_x64', True)
    alpha = np.zeros(dimension)
    alpha[:2] = (100., -100.)
    namespace = dict(np=np, jax=jax, jnp=jnp, jgammaln=jgammaln,
                     jstats=jstats, D=dimension, NU=nu, ALPHA=alpha, BETA_VAL=2.,
                     OPTIMIZATION_STEPS=optimizer_steps,
                     OPTIMIZATION_SAMPLES=optimizer_samples)
    names = {'optimize_prwm_hyperparameters', 'optimize_scs_hyperparameters'}
    tree = ast.parse(LEGACY.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('The historical audit input lacks its expected VI functions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LEGACY), 'exec'), namespace)
    return namespace
