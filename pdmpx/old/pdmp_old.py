import numpy as np
import jax.numpy as jnp
import jax
import functools as ft
from typing import Any, NamedTuple, Sequence, Tuple, Callable
import chex


class PDMPState(NamedTuple):
    x: chex.ArrayTree
    v: chex.ArrayTree


class Event(NamedTuple):
    time: float
    new_state: PDMPState


class AbstractDynamics:
    def forward(self, t, state):
        raise NotImplementedError


class AbstractEventHandler:
    def get_next_event(self, state: PDMPState, rng=None) -> Tuple[float, Callable]:
        raise NotImplementedError


class AbstractEventClock:
    def get_next_event_time(self, state: PDMPState, rng=None) -> float:
        raise NotImplementedError

    def get_new_state(self, state: PDMPState, rng=None) -> PDMPState:
        raise NotImplementedError


class SimpleEventQueue(AbstractEventHandler):
    def __init__(self, event_clocks: Sequence[AbstractEventClock]):
        self.event_clocks = event_clocks
        self.next_event = None

    def get_next_event(self, state: PDMPState, rng=None) -> Tuple[float, Callable]:
        event_times = [handler.get_event_time(state) for handler in self.event_clocks]
        next_event = np.argmin(event_times)
        next_event_time = event_times[next_event]
        return next_event_time, self.event_clocks[next_event].get_new_state


class LinearDynamics(AbstractDynamics):
    def forward(self, t: float, state: PDMPState):
        return PDMPState(state.x + t * state.v, state.v)


def rng_wrap(fun):
    def wrapped_fun(rng, *args, **kwargs):
        rng, key = jax.random.split(rng)
        return rng, fun(*args, **kwargs, rng=key)

    return wrapped_fun


class PDMP:
    def __init__(self, dynamics: AbstractDynamics, event_handler: AbstractEventHandler):
        self.dynamics = dynamics
        self.event_handler = event_handler

    def _next_event(self, time: float, state: PDMPState, rng):
        key0, key1 = jax.random.split(rng)
        next_event_time, new_state_fn = self.event_handler.get_next_event(state, key0)
        time += next_event_time
        state = self.dynamics.forward(next_event_time, state)
        state = new_state_fn(state, key1)
        return Event(time, state)

    def simulate(
        self, state: PDMPState, time_max: float, rng=jax.random.PRNGKey(0)
    ) -> Sequence[Event]:
        time = 0.0
        events = []

        get_next_event = rng_wrap(self._next_event)

        while time < time_max:
            rng, event = get_next_event(rng, time, state)
            time = event.time
            events.append(event)
        return events
