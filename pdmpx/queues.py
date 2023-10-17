import functools as ft
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
from .pdmp import PDMPState, TimerEvent, AbstractFactor, Factor, Context, RNGKey


class SimpleFactorQueue(Factor):
    def __init__(self, factors: Sequence[Factor]):
        self.factors = factors

    @ft.partial(jax.jit, static_argnums=(0,))
    def timer(
        self, rng: RNGKey, state: PDMPState, context={}
    ) -> Tuple[TimerEvent, Context]:
        keys = jax.random.split(rng, len(self.factors))
        timer_events = [
            factor.timer(key, state, context) for key, factor in zip(keys, self.factors)
        ]
        next_event_idx = jnp.argmin([tev.time for tev in timer_events])
        nev = timer_events[next_event_idx]
        return TimerEvent(nev.time, nev.bound), {
            "simple_factor_queue": {"next_event_idx": next_event_idx},
            **context,
        }

    @ft.partial(jax.jit, static_argnums=(0,))
    def kernel(self, rng: RNGKey, state: PDMPState, context={}) -> PDMPState:
        next_event_idx = context["simple_factor_queue"]["next_event_idx"]
        return jax.lax.switch(
            next_event_idx,
            [factor.kernel for factor in self.factors],
            rng,
            state,
            context,
        )
