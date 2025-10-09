"""Stochastic Simulations

JAX MD includes integrators for stochastic simulations of Langevin dynamics and
Brownian motion for systems in the NVT ensemble with a solvent.
"""

from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools

from jax import grad
from jax import jit
from jax import random
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

import jax_md

# Types

f32 = jax_md.util.f32
f64 = jax_md.util.f64


T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]



@jax_md.simulate.dispatch_by_state
def stochastic_step_pimd(state: jax_md.simulate.NVTLangevinState, dt:float, kT: float, gamma: jnp.ndarray, natoms: int, n_bead: int, nm_trans: jnp.ndarray):
    """A single stochastic step (the `O` step)."""
    nm_momentum = jnp.tensordot(nm_trans.T, state.momentum.reshape(n_bead, natoms, 3), axes=(1, 0)).reshape(-1, 3)    # (n_bead x natoms, dimen)
    c1 = jnp.exp(- gamma * dt)              # (n_bead x natoms, 1)
    c2 = jnp.sqrt(kT * (1 - c1**2))         # (n_bead x natoms, 1)
    key, split = random.split(state.rng)
    nm_momentum = jax_md.simulate.Normal(c1 * nm_momentum, c2**2 * state.mass).sample(split)    # (n_bead x natoms, dimen)
    momentum = jnp.tensordot(nm_trans, nm_momentum.reshape(n_bead, natoms, 3), axes=(1, 0)).reshape(-1, 3)    # (n_bead x natoms, dimen)
    return state.set(momentum=momentum, rng=key)


def nvt_langevin_pimd(energy_or_force_fn: Callable[..., jax_md.util.Array],
                      shift_fn: jax_md.space.ShiftFn,
                      dt: float,
                      kT: float,
                      gamma: jnp.ndarray,
                      natoms: int, 
                      n_bead: int, 
                      nm_trans: jnp.ndarray,
                      center_velocity: bool=True,
                      **sim_kwargs) -> Simulator:
    """Simulation in the NVT ensemble using the OBABO Langevin thermostat.

    Samples from the canonical ensemble in which the number of particles (N),
    the system volume (V), and the temperature (T) are held constant. Langevin
    dynamics are stochastic and it is supposed that the system is interacting
    with fictitious microscopic degrees of freedom. An example of this would be
    large particles in a solvent such as water. Thus, Langevin dynamics are a
    stochastic ODE described by a friction coefficient and noise of a given
    covariance.

    Our implementation follows the paper [#davidcheck] by Davidchack, Ouldridge,
    and Tretyakov.

    Args:
        energy_or_force: A function that produces either an energy or a force from
        a set of particle positions specified as an ndarray of shape
        `[n, spatial_dimension]`.
        shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
        `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
        dt: Floating point number specifying the timescale (step size) of the
        simulation.
        kT: Floating point number specifying the temperature in units of Boltzmann
        constant. To update the temperature dynamically during a simulation one
        should pass `kT` as a keyword argument to the step function.
        gamma: A float specifying the friction coefficient between the particles
        and the solvent.
        center_velocity: A boolean specifying whether or not the center of mass
        position should be subtracted.
    Returns:
        See above.

    .. rubric:: References
    .. [#carlon] R. L. Davidchack, T. E. Ouldridge, and M. V. Tretyakov.
        "New Langevin and gradient thermostats for rigid body dynamics."
        The Journal of Chemical Physics 142, 144114 (2015)
    """
    force_fn = jax_md.quantity.canonicalize_force(energy_or_force_fn)

    @jit
    def init_fn(key, R, mass=f32(1.0), **kwargs):
        _kT = kwargs.pop('kT', kT)
        key, split = random.split(key)
        force = force_fn(R, **kwargs)
        state = jax_md.simulate.NVTLangevinState(R, None, force, mass, key)
        state = jax_md.simulate.canonicalize_mass(state)
        return jax_md.simulate.initialize_momenta(state, split, _kT)

    @jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop('dt', dt)
        _kT = kwargs.pop('kT', kT)
        dt_2 = _dt / 2

        state = stochastic_step_pimd(state, dt_2, _kT, gamma, natoms, n_bead, nm_trans)
        state = jax_md.simulate.momentum_step(state, dt_2)
        state = jax_md.simulate.position_step(state, shift_fn, dt, **kwargs)
        state = state.set(force=force_fn(state.position, **kwargs))
        state = jax_md.simulate.momentum_step(state, dt_2)
        state = stochastic_step_pimd(state, dt_2, _kT, gamma, natoms, n_bead, nm_trans)

        return state

    return init_fn, step_fn