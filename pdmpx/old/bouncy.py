from collections import namedtuple
from .refreshments import ConstantRateRefreshments
from .queues import SimpleFactorQueue
from .dynamics import LinearDynamics
from .pdmp import (
    PDMP,
    PDMPState,
    AbstractDynamics,
    AbstractKernel,
    AbstractTimer,
    TimerEvent,
    PyTree,
)
from .timers import LinearApproxTimer
from .utils.tree import tree_dot, tree_add_scaled, tree_unit_length
from .utils.func import maybe_add_dummy_args

import jax
import jax.numpy as jnp
import functools as ft
import tree_math as tm

from typing import Sequence, Tuple, Callable, Dict, Optional, Union, Any

BPSState = namedtuple("BPSState", [("params", PyTree), ("velocities", PyTree)])


@ft.partial(tm.wrap, vector_argnames=("x", "v"))
def linear_dynamics(t, x, v):
    return x + t * v, v


class LinearDynamics(AbstractDynamics):
    def forward(self, t: float, state: BPSState) -> BPSState:
        return PDMPState(*linear_dynamics(t, state.params, state.velocities))


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

    def __call__(self, rng: Any, state: BPSState) -> TimerEvent:
        pot, dpot = jax.jvp(lambda t: self.potential(t, state.params), (0.0,), (1.0,))
        return TimerEvent(self.valid_time, False, dpot)


def create_generalized_bounce_kernel(
    potential,
    gradient_mix=None,
    oscn=False,
    normalize_velocities=True,
    fletcher_reeves=False,
):
    """
    TODO: Pretty poor naming and implementation.
    """
    if oscn:  # TODO: Add Orthogonal Subspace Crank-Nicolson bounces
        raise NotImplementedError

    # if gradient_mix and fletcher_reeves:
    #     warn = (
    #         "Gradient mixing and Fletcher-Reeves incompatible. Using Fletcher-Reeves."
    #     )
    #     print(warn)
    #     gradient_mix = 0

    def bounce(
        rng, state: PDMPState, context: Context = {}
    ) -> Tuple[PDMPState, Context]:
        grads = jax.grad(potential)(state.params, context)
        vs = state.velocities
        dot_prod = tree_dot(grads, vs)
        norm_sq = tree_dot(grads, grads)
        reflect_vs = tree_add_scaled(vs, grads, 1, -2 * dot_prod / norm_sq)

        if fletcher_reeves:
            vs_norm_sq = tree_dot(vs, vs)
            alpha = vs_norm_sq / (norm_sq + vs_norm_sq)
            if fletcher_reeves == "no_reflect":
                new_vs = tree_add_scaled(vs, grads, 1 - alpha, -alpha)
            else:
                new_vs = tree_add_scaled(reflect_vs, grads, 1 - alpha, -alpha)
        elif gradient_mix is not None:
            vs_norm_sq = tree_dot(vs, vs)
            new_vs = tree_add_scaled(
                reflect_vs,
                grads,
                (1 - gradient_mix),
                -gradient_mix * jnp.sqrt(vs_norm_sq / norm_sq),
            )
        else:
            new_vs = reflect_vs

        if normalize_velocities:
            new_vs = tree_unit_length(new_vs)
        return PDMPState(state.params, new_vs)

    return bounce


def create_bps_bounce_kernel(potential, normalize_velocities=True):
    return create_generalized_bounce_kernel(
        potential,
        gradient_mix=None,
        oscn=False,
        normalize_velocities=normalize_velocities,
    )


def create_rate_fn(potential, dynamics=LinearDynamics(), return_aux=False):
    def rate_fn(state, context={}):
        pot, dpot = jax.jvp(
            lambda t: potential(dynamics.forward(t, state).params, context),
            (0.0,),
            (1.0,),
        )
        return (dpot, pot) if return_aux else dpot

    return rate_fn


class GeneralizedBounceFactor:
    def __init__(
        self,
        potential,
        valid_time=jnp.inf,
        gradient_mix=0.0,
        oscn=0,
        coldness=1,
        normalize_velocities=True,
        with_aux=False,
        fletcher_reeves=False,
    ):
        self.kernel = create_generalized_bounce_kernel(
            potential,
            gradient_mix,
            oscn,
            normalize_velocities,
            fletcher_reeves=fletcher_reeves,
        )
        rate_fn = create_rate_fn(potential, return_aux=with_aux)
        self.timer = LinearApproxTimer(
            rate_fn, valid_time, has_aux=with_aux, timescale=coldness
        )


class BPSBounceFactor:  # (GeneralizedBounceFactor)  # (AbstractFactor):
    def __init__(
        self,
        potential,
        valid_time=jnp.inf,
        normalize_velocities=True,
        dynamics=LinearDynamics(),
    ):
        self.kernel = create_bps_bounce_kernel(potential, normalize_velocities)
        rate_fn = create_rate_fn(potential, dynamics)
        self.timer = LinearApproxTimer(rate_fn, valid_time, dynamics=dynamics)


class BouncyParticleSampler(PDMP):
    """
    Bouncy Particle Sampler (BPS) for sampling from a target distribution
    with a given potential function. The BPS is a PDMP with two factors:
    bounce and refreshment.

    Currently, the bounce factor is implemented with a linear approximation
    of the rate function. This approximation is valid for a given time interval
    (see `valid_time` argument).
    """

    def __init__(
        self, potential, refreshment_rate, valid_time=jnp.inf, normalize_velocities=True
    ):
        """
        Args:
            potential: Differentiable potential function of the target distribution (up to an additive constant).
            refreshment_rate: Rate of refreshments.
            valid_time: Time to trust the linear approximation of the rate.
            normalize_velocities: Run in unit speed.
        """
        potential = maybe_add_dummy_args(potential)
        bounce = BPSBounceFactor(potential, valid_time, normalize_velocities)
        refreshments = ConstantRateRefreshments(refreshment_rate, normalize_velocities)
        queue = SimpleFactorQueue([bounce, refreshments])
        dynamics = LinearDynamics()
        super().__init__(dynamics=dynamics, factor=queue)


class ColdBouncyParticleSampler(PDMP):
    """
    CBPS
    TODO: Rework or remove.
    """

    def __init__(
        self,
        potential,
        refreshment_rate,
        gradient_mix,
        coldness,
        valid_time=jnp.inf,
        normalize_velocities=True,
        fletcher_reeves=False,
    ):
        """
        Args:
            potential: Differentiable potential function of the target distribution (up to an additive constant).
            refreshment_rate: Rate of refreshments.
            valid_time: Time to trust the linear approximation of the rate.
            normalize_velocities: Run in unit speed.
        """
        potential = maybe_add_dummy_args(potential)
        bounce = GeneralizedBounceFactor(
            potential,
            valid_time=valid_time,
            gradient_mix=gradient_mix,
            coldness=coldness,
            normalize_velocities=normalize_velocities,
            with_aux=True,
            fletcher_reeves=fletcher_reeves,
        )
        refreshments = ConstantRateRefreshments(refreshment_rate, normalize_velocities)
        queue = SimpleFactorQueue([bounce, refreshments])
        dynamics = LinearDynamics()
        super().__init__(dynamics=dynamics, factor=queue)
