import functools as ft
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
from .pdmp import PDMPState, TimerEvent, AbstractKernel, AbstractTimer, RNGKey

class Factor(NamedTuple):
    timer: AbstractTimer
    kernel: AbstractKernel

class SimpleFactorQueue:
    def __init__(self, factors: Sequence[Factor]):
        self.factors = factors

    def timer(
        self, rng: RNGKey, state: PDMPState
    ) -> TimerEvent:
        keys = jax.random.split(rng, len(self.factors))
        timer_events = [
            factor.timer(key, state)
            for key, factor in zip(keys, self.factors)
        ]
        times = jnp.array([tev.time for tev in timer_events])
        next_event_idx = jnp.argmin(times)
        next_event = timer_events[next_event_idx]
        return TimerEvent(next_event.time, next_event.dirty, {
            **next_event.params,
            "simple_factor_queue": {"next_event_idx": next_event_idx},
        })

    def kernel(self, rng: RNGKey, state: PDMPState, timer_event: TimerEvent) -> PDMPState:
        next_event_idx = timer_event.params["simple_factor_queue"]["next_event_idx"]
        return jax.lax.switch(
            next_event_idx,
            [factor.kernel for factor in self.factors],
            rng,
            state,
            timer_event,
        )
