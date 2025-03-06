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
from typing import NamedTuple


class BPSState(NamedTuple):
    params: PyTree
    velocities: PyTree


@ft.partial(tm.wrap, vector_argnames=("x", "v"))
def linear_dynamics(t, x, v):
    return x + t * v, v


class LinearDynamics(AbstractDynamics):
    def forward(self, t: float, state: BPSState) -> BPSState:
        return BPSState(*linear_dynamics(t, state.params, state.velocities))
