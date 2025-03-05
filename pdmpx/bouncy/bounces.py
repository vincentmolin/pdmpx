from ..pdmp import (
    PDMP,
    PDMPState,
    AbstractDynamics,
    AbstractKernel,
    AbstractTimer,
    TimerEvent,
    PyTree,
)
import functools as ft
import tree_math as tm
from .bps import BPSState
from typing import Callable, Any
import jax

from .dynamics import LinearDynamics

from ..timers import sample_linear_approx_poisson_clock

from ..utils.tree import tree_dot, tree_add_scaled


class BounceKernel(AbstractKernel):
    def __init__(self, potential: Callable):
        self.potential = potential

    def __call__(self, rng: Any, state: BPSState, **kwargs) -> BPSState:
        grads = jax.grad(self.potential)(state.params)
        vs = state.velocities
        dot_prod = tree_dot(grads, vs)
        norm_sq = tree_dot(grads, grads)
        reflect_vs = tree_add_scaled(vs, grads, 1, -2 * dot_prod / norm_sq)
        return BPSState(state.params, reflect_vs)


class BounceTimer(AbstractTimer):
    def __init__(self, potential: Callable, valid_time: float):
        self.potential = potential
        self.valid_time = valid_time
        self.dynamics = LinearDynamics()

    def __call__(self, rng: Any, state: BPSState) -> TimerEvent:
        def rate_fn(t):
            pot, dpot = jax.jvp(
                lambda s: self.potential(self.dynamics.forward(s, state.params)),
                (t,),
                (1.0,),
            )
            return dpot

        time = sample_linear_approx_poisson_clock(rng, rate_fn)

        event = jax.lax.cond(
            time < self.valid_time,
            lambda: TimerEvent(time, dirty=1.0),
            lambda: TimerEvent(self.valid_time, dirty=0.0),
        )
        return event
