import functools as ft
import jax
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
from .poisson_time.linear import ab_poisson_time
from .pdmp import AbstractTimer, TimerEvent, PDMPState, RNGKey, Context, PyTree
from .dynamics import LinearDynamics


class ConstantRateTimer(AbstractTimer):
    def __init__(self, rate: float):
        self.rate = rate

    # @ft.partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        time = jax.random.exponential(rng) / self.rate
        return TimerEvent(time, 0.0), context


class LinearApproxTimer(AbstractTimer):
    def __init__(
        self,
        rate_fn: Callable[[PyTree, PyTree], Union[float, Tuple[float, Any]]],
        valid_time: float,
        has_aux=False,
        dynamics=LinearDynamics(),
        timescale=1.0,
    ):
        self.valid_time = valid_time
        self.rate_fn = rate_fn
        self.has_aux = has_aux
        self.timescale = timescale

    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        rate, drate, *maybe_aux = jax.jvp(
            lambda params: self.rate_fn(params, state.velocities, context),
            (state.params,),
            (state.velocities,),
            has_aux=self.has_aux,
        )
        a = rate * self.timescale
        b = drate * self.timescale
        u = jax.random.uniform(rng)
        time = ab_poisson_time(u, a, b)
        event = jax.lax.cond(
            time < self.valid_time,
            lambda: TimerEvent(time, bound=0.0),
            lambda: TimerEvent(self.valid_time, bound=1.0),
        )
        if self.has_aux:
            return event, {"timer": maybe_aux, **context}
        else:
            return event, context
