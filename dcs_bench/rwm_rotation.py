"""Reflected Gaussian random walk mixed with fixed-radius angular updates.

DCS-RWM-rotate replaces the uniform-ball increment of DCS-BW-rotate by an
isotropic Gaussian increment. Each production iteration independently chooses
one reflected Gaussian or one angular update with probability one half.
The angular proposal and its Mobius correction are shared unchanged with BW.
Both scales are separately pretuned within the common pilot budget and frozen.
"""
import math
from numbers import Integral, Real

import numpy as np

METHOD = 'DCS-RWM-rotate'


def initial_sigma(dimension, bw_radius=.1):
    """Match E||increment||^2 of a uniform d-ball of radius bw_radius.

    Uniform-ball E||increment||^2 = d*bw_radius^2/(d+2);
    Gaussian E||increment||^2 = d*sigma^2.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 2:
        raise ValueError('Rotation requires an integer dimension of at least two')
    if isinstance(bw_radius, bool) or not isinstance(bw_radius, Real) or not np.isfinite(bw_radius) or bw_radius <= 0.:
        raise ValueError('Initial BW-equivalent radius must be positive and finite')
    return float(bw_radius)/math.sqrt(int(dimension)+2.)


def build_parts(mapping_dict, spec):
    import jax
    import jax.numpy as jnp
    from . import rotation, timed_radial
    from .ball_boundary import reflect_ball_proposal
    from .targets import Target

    base=timed_radial._components(mapping_dict,spec,'DCS-BW-reflect')
    target=Target(spec)
    parts=rotation.build_components(mapping_dict,target,base)
    density,projection,initial=parts['density'],parts['projection'],parts['initial']
    d=int(mapping_dict['dimension']);delta=jnp.asarray(mapping_dict['delta'])

    def reflected_step(state,normal,scale,logu):
        proposed,valid=reflect_ball_proposal(state,scale*normal)
        # Isotropic Gaussian increments have an even signed length on every
        # unoriented line. Chord folding retains proposal symmetry for ball
        # volume, so this component needs only the target-density difference.
        accept=jax.lax.cond(valid,
            lambda _: (logu<density(proposed)-density(state))*1.,
            lambda _: jnp.asarray(0.),operand=None)
        return jnp.where(accept>0.,proposed,state),accept

    def reflected_block(state,key,count,scale):
        def step(carry,index):
            state,rng=carry
            rng,normal_key,accept_key=jax.random.split(rng,3)
            normal=jax.random.normal(normal_key,(d,))
            state,accepted=reflected_step(state,normal,scale,
                jnp.log(jax.random.uniform(accept_key)))
            return (state,rng),(state,accepted)
        return jax.lax.scan(step,(state,key),jnp.arange(count))[1]

    def angular_step(state,normal,scale,logu):
        proposed,correction,valid=rotation.angular_proposal(state,normal,scale,delta)
        accept=jax.lax.cond(valid,
            lambda _: (logu<density(proposed)-density(state)+correction)*1.,
            lambda _: jnp.asarray(0.),operand=None)
        return jnp.where(accept>0.,proposed,state),accept

    def block(state,key,count,scales):
        def step(carry,index):
            state,rng=carry
            # Same independent selection, Gaussian and MH streams as the
            # existing mixture; only d normals, no BW normalization.
            rng,selection_key,normal_key,accept_key=jax.random.split(rng,4)
            normal=jax.random.normal(normal_key,(d,))
            logu=jnp.log(jax.random.uniform(accept_key))
            state,accepted=jax.lax.cond(jax.random.bernoulli(selection_key,rotation.ANGULAR_PROBABILITY),
                lambda _:angular_step(state,normal,scales[1],logu),
                lambda _:reflected_step(state,normal,scales[0],logu),operand=None)
            return (state,rng),(state,accepted)
        return jax.lax.scan(step,(state,key),jnp.arange(count))[1]

    parts.update(block=block,reflect_block=reflected_block)
    return parts


def build_components(mapping_dict,spec):
    """Four-tuple compatible with timed_radial._radial_batch."""
    parts=build_parts(mapping_dict,spec)
    return parts['block'],parts['projection'],parts['initial'],parts['density']


def calibrate(parts,key,warmup=1000,initial_bw_radius=.1,initial_angular_scale=1.,scale_ceiling=None):
    """Use the existing two-phase pilot with matched initial RMS length."""
    from .rotation import calibrate as shared_calibrate
    return shared_calibrate(parts,parts['initial'],key,warmup,
                            initial_sigma(len(parts['initial']),initial_bw_radius),initial_angular_scale,scale_ceiling)
