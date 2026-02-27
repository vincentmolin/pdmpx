import functools as ft
from threading import Timer
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

    def timer(self, rng: RNGKey, state: PDMPState) -> TimerEvent:
        keys = jax.random.split(rng, len(self.factors))
        timer_events = [
            factor.timer(key, state) for key, factor in zip(keys, self.factors)
        ]
        times = jnp.array([tev.time for tev in timer_events])
        dirtys = jnp.array([tev.dirty for tev in timer_events])
        params = [tev.params for tev in timer_events]
        next_event_idx = jnp.argmin(times)
        return TimerEvent(
            times[next_event_idx],  # pyright: ignore
            dirtys[next_event_idx],  # pyright: ignore
            {
                "simple_factor_queue": {
                    "next_event_idx": next_event_idx,
                    "params": params,
                },
            },
        )

    def kernel(
        self, rng: RNGKey, state: PDMPState, timer_event: TimerEvent
    ) -> PDMPState:
        next_event_idx = timer_event.params["simple_factor_queue"]["next_event_idx"]
        paramss = timer_event.params["simple_factor_queue"]["params"]

        def mk_partial_kernel(i):
            kernel = self.factors[i].kernel
            params = paramss[i]
            return lambda r, s: kernel(
                r, s, TimerEvent(timer_event.time, timer_event.dirty, params=params)
            )

        partial_kernels = [mk_partial_kernel(i) for i in range(len(self.factors))]
        return jax.lax.switch(
            next_event_idx,
            partial_kernels,  # [factor.kernel for factor in self.factors],
            rng,
            state,
        )
