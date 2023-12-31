import functools as ft
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
from .pdmp import PDMPState, TimerEvent, AbstractFactor, Factor, Context, RNGKey


class SimpleFactorQueue(AbstractFactor):
    def __init__(self, factors: Sequence[Factor]):
        self.factors = factors

    #   @ft.partial(jax.jit, static_argnums=(0,))
    def timer(
        self, rng: RNGKey, state: PDMPState, context={}
    ) -> Tuple[TimerEvent, Context]:
        keys = jax.random.split(rng, len(self.factors))
        timer_events = [
            factor.timer(key, state, context)[0]
            for key, factor in zip(keys, self.factors)
        ]
        times = jnp.array([tev.time for tev in timer_events])
        bounds = jnp.array([tev.bound for tev in timer_events])
        next_event_idx = jnp.argmin(times)
        return TimerEvent(times[next_event_idx], bounds[next_event_idx]), {
            **context,
            "simple_factor_queue": {"next_event_idx": next_event_idx},
        }

    #   @ft.partial(jax.jit, static_argnums=(0,))
    def kernel(self, rng: RNGKey, state: PDMPState, context={}) -> PDMPState:
        next_event_idx = context["simple_factor_queue"]["next_event_idx"]
        return jax.lax.switch(
            next_event_idx,
            [factor.kernel for factor in self.factors],
            rng,
            state,
            context,
        )
