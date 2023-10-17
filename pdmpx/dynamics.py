from .pdmp import PDMPState, AbstractDynamics
import jax
import chex
import functools as ft
import tree_math as tm
from typing import NamedTuple


@ft.partial(tm.wrap, vector_argnames=("x", "v"))
def linear_dynamics(t, x, v):
    return x + t * v, v


class LinearDynamics(AbstractDynamics):
    @ft.partial(jax.jit, static_argnums=(0,))
    def forward(self, t: float, state: PDMPState):
        return PDMPState(*linear_dynamics(t, state.params, state.velocities))
