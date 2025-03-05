import jax.numpy as jnp
import jax
import functools as ft
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
from abc import ABC, abstractmethod

RNGKey = Any
Context = Dict
PyTree = Any
PDMPState = PyTree


class Event(NamedTuple):
    time: float
    new_state: PDMPState


class TimerEvent(NamedTuple):
    time: float
    dirty: bool
    params: PyTree


class AbstractDynamics(ABC):
    def forward(self, dt: float, state: PDMPState) -> PDMPState:
        raise NotImplementedError


class AbstractTimer(ABC):
    def __call__(self, rng: RNGKey, state: PDMPState) -> TimerEvent:
        pass


class AbstractKernel(ABC):
    def __call__(
        self, rng: RNGKey, state: PDMPState, timer_event: TimerEvent
    ) -> PDMPState:
        pass


class PDMP:
    def __init__(
        self, dynamics: AbstractDynamics, timer: AbstractTimer, kernel: AbstractKernel
    ):
        self.dynamics = dynamics
        self.timer = timer
        self.kernel = kernel

    def get_next_event(
        self, rng: RNGKey, state: PDMPState
    ) -> Tuple[Event, float, bool]:
        """Simulates the PDMP forward.

        Args:
            rng: A JAX random key used to generate random numbers.
            state: The current state of the PDMP simulation.

        Returns:
            A tuple containing the next state, elapsed time and a boolean that is False
            if the state has only has evolved deterministically.
        """
        timer_key, kernel_key = jax.random.split(rng, 2)
        timer_event = self.timer(timer_key, state)
        dt = timer_event.time
        state = self.dynamics.forward(dt, state)
        state, dirty = jax.lax.cond(
            timer_event.dirty,
            lambda rng, st: (st, False),
            lambda k, st: (self.kernel(k, st), True),
            kernel_key,
            state,
        )
        return state, dt, dirty

    def simulate(
        self,
        rng: RNGKey,
        state: PDMPState,
        time_max: float,
        save_trajectory=True,
    ) -> Sequence[Event]:
        """Simulates the PDMP forward in time."""
        time = 0.0
        events = [Event(0.0, state)]

        @jax.jit
        def _get_next_event(rng, state):
            rng, key = jax.random.split(rng)
            return rng, *self.get_next_event(key, state)

        while time < time_max:
            rng, state, dt, dirty = _get_next_event(rng, state)
            time += dt
            if dirty and save_trajectory:
                events.append(Event(time, state))
        if not save_trajectory:
            events.append(Event(time, state))
        return events
