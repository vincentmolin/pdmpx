from .refreshments import ConstantRateRefreshments
from .queues import SimpleFactorQueue
from .dynamics import LinearDynamics
from .pdmp import PDMP, PDMPState, AbstractFactor, Context, PyTree
from .timers import LinearApproxTimer
from .utils.tree import tree_dot, tree_add_scaled, tree_unit_length

import jax
import jax.numpy as jnp
import functools as ft
import tree_math as tm

from typing import NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union, Any


def create_bps_reflection_kernel(potential, normalize_velocities=True):
    @jax.jit
    def reflection(
        rng, state: PDMPState, context: Context = {}
    ) -> Tuple[PDMPState, Context]:
        grads = jax.grad(potential)(state.params, context)
        vs = state.velocities
        dot_prod = tree_dot(grads, vs)
        norm_sq = tree_dot(grads, grads)
        new_vs = tree_add_scaled(vs, grads, 1, -2 * dot_prod / norm_sq)
        if normalize_velocities:
            new_vs = tree_unit_length(new_vs)
        return PDMPState(state.params, new_vs)

    return reflection


def create_rate_fn(potential):
    @jax.jit
    def rate_fn(params, velocities, context={}):
        pot, dpot = jax.jvp(lambda ps: potential(ps, context), (params,), (velocities,))
        return dpot  # , pot

    return rate_fn


class BPSReflectionFactor:  # (AbstractFactor):
    def __init__(self, potential, valid_time=jnp.inf, normalize_velocities=True):
        self.kernel = create_bps_reflection_kernel(potential, normalize_velocities)
        rate_fn = create_rate_fn(potential)
        self.timer = LinearApproxTimer(rate_fn, valid_time)


class BouncyParticleSampler(PDMP):
    def __init__(self, potential, refreshment_rate, valid_time=jnp.inf):
        reflection = BPSReflectionFactor(potential, valid_time)
        refreshments = ConstantRateRefreshments(refreshment_rate)
        queue = SimpleFactorQueue([reflection, refreshments])
        dynamics = LinearDynamics()
        super().__init__(dynamics=dynamics, factor=queue)
