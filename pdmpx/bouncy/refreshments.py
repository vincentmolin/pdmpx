import jax
import functools as ft
from ..pdmp import AbstractKernel, PDMPState, RNGKey, TimerEvent

from ..utils.tree import tree_random_normal, tree_unit_length
from .bps import BPSState
from ..queues import Factor


class RefreshmentKernel(AbstractKernel):
    def __init__(self, normalize_velocities=False):
        self.normalize_velocities = normalize_velocities

    def __call__(
        self, rng: RNGKey, state: BPSState, timer_event: TimerEvent
    ) -> BPSState:
        velocities = tree_random_normal(rng, state.velocities)
        if self.normalize_velocities:
            velocities = tree_unit_length(velocities)
        return BPSState(state.params, velocities)