import numpy as np
import jax.numpy as jnp
import jax
import functools as ft
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
import chex
from abc import ABC, abstractmethod

RNGKey = Any
Context = Dict


class PDMPState(NamedTuple):
    params: chex.ArrayTree
    velocities: chex.ArrayTree


class Event(NamedTuple):
    time: float
    new_state: PDMPState


class TimerEvent(NamedTuple):
    time: float
    bound: bool


class AbstractDynamics(ABC):
    @abstractmethod
    def forward(self, t: float, state: PDMPState) -> PDMPState:
        raise NotImplementedError


class AbstractTimer(ABC):
    @abstractmethod
    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        pass


class AbstractKernel(ABC):
    @abstractmethod
    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> PDMPState:
        pass


class AbstractFactor(ABC):
    @abstractmethod
    def timer(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        pass

    @abstractmethod
    def kernel(self, rng: RNGKey, state: PDMPState, context: Context = {}) -> PDMPState:
        pass


class FactorTuple(NamedTuple):
    timer: Callable[
        [RNGKey, PDMPState, Context], Tuple[TimerEvent, Context]
    ] | AbstractTimer
    kernel: Callable[[RNGKey, PDMPState, Context], PDMPState] | AbstractKernel


Factor = Union[AbstractFactor, FactorTuple]


class AbstractContextHandler(ABC):
    @abstractmethod
    def __call__(self, context: Context) -> Context:
        pass


class PDMP:
    def __init__(self, dynamics: AbstractDynamics, factor: Factor):
        self.dynamics = dynamics
        self.factor = factor

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_next_event(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[RNGKey, Event, Context, bool]:
        rng, key0, key1 = jax.random.split(rng, 3)
        timer_event, context = self.factor.timer(key0, state, context)
        time = timer_event.time + context["time"]
        state = self.dynamics.forward(timer_event.time, state)
        state, dirty = jax.lax.cond(
            timer_event.bound,
            lambda rng, st, *_: (st, False),
            lambda k, st, ctx: (self.factor.kernel(k, st, ctx), True),
            key1,
            state,
            context,
        )
        return rng, Event(time, state), context, dirty

    def simulate(
        self,
        rng: RNGKey,
        state: PDMPState,
        time_max: float,
        save_trajectory=True,
        context_handler=None,
        callbacks=[],
    ) -> Sequence[Event]:
        context = {"time": 0.0}
        events = [Event(0.0, state)]

        while context["time"] < time_max:
            if context_handler is not None:
                context = context_handler(context)
            rng, event, context, dirty = self.get_next_event(rng, state, context)
            state = event.new_state
            context["time"] = event.time
            if dirty and save_trajectory:
                events.append(event)
            for callback in callbacks:
                callback(state, context)
        if not save_trajectory:
            events.append(event)
        return events
