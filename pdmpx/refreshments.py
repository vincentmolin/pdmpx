import jax
import functools as ft
from .pdmp import AbstractKernel, PDMPState, RNGKey, Context, AbstractFactor
from .timers import ConstantTimer
from .utils.tree import tree_random_normal, tree_unit_length


class RefreshmentKernel(AbstractKernel):
    def __init__(self, normalize_velocities=False):
        self.normalize_velocities = normalize_velocities

    @ft.partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> PDMPState:
        velocities = tree_random_normal(rng, state.velocities)
        if self.normalize_velocities:
            velocities = tree_unit_length(velocities)
        return PDMPState(state.params, velocities)


class ConstantRateRefreshments(AbstractFactor):
    def __init__(self, rate: float, normalize_velocities=False):
        self.timer = ConstantTimer(rate)
        self.kernel = RefreshmentKernel(normalize_velocities)
