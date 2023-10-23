import jax.numpy as jnp
import jax
import functools as ft
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union
from abc import ABC, abstractmethod

RNGKey = Any
Context = Dict
PyTree = Any


class PDMPState(NamedTuple):
    """
    PDMP state is a tuple of parameters and velocities
    """

    params: PyTree
    velocities: PyTree


class Event(NamedTuple):
    time: float
    new_state: PDMPState


class TimerEvent(NamedTuple):
    time: float
    bound: bool


class AbstractDynamics(ABC):
    # @abstractmethod
    def forward(self, t: float, state: PDMPState) -> PDMPState:
        raise NotImplementedError


class AbstractTimer(ABC):
    # @abstractmethod
    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        pass


class AbstractKernel(ABC):
    # @abstractmethod
    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> PDMPState:
        pass


class AbstractFactor(ABC):
    # @abstractmethod
    def timer(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        pass

    # @abstractmethod
    def kernel(self, rng: RNGKey, state: PDMPState, context: Context = {}) -> PDMPState:
        pass


class FactorTuple(NamedTuple):
    timer: Callable[
        [RNGKey, PDMPState, Context], Tuple[TimerEvent, Context]
    ] | AbstractTimer
    kernel: Callable[[RNGKey, PDMPState, Context], PDMPState] | AbstractKernel


Factor = Union[AbstractFactor, FactorTuple]


class AbstractContextHandler(ABC):
    # @abstractmethod
    def __call__(self, context: Context) -> Context:
        pass


class PDMP:
    def __init__(self, dynamics: AbstractDynamics, factor: Factor):
        self.dynamics = dynamics
        self.factor = factor

    def get_next_event(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[Event, Context, bool]:
        """Simulates the PDMP forward.

        Args:
            rng: A JAX random key used to generate random numbers.
            state: The current state of the PDMP simulation.
            context: A dictionary of additional information that can be used by the
                simulation components.
                If context contains the "time" key, the returned event will have
                the time field incremented "correctly" (i.e. the time field will
                be the absolute time of the event).

        Returns:
            A tuple containing the next event, the updated context, and a boolean
            indicating whether an event has happened or if the state has evolved deterministically.
        """
        timer_key, kernel_key = jax.random.split(rng, 2)
        timer_event, context = self.factor.timer(timer_key, state, context)
        time = timer_event.time + context.get("time", 0.0)
        state = self.dynamics.forward(timer_event.time, state)
        state, dirty = jax.lax.cond(
            timer_event.bound,
            lambda rng, st, *_: (st, False),
            lambda k, st, ctx: (self.factor.kernel(k, st, ctx), True),
            kernel_key,
            state,
            context,
        )
        return Event(time, state), context, dirty

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

        @jax.jit
        def _get_next_event(rng, state, context):
            rng, key = jax.random.split(rng)
            return rng, *self.get_next_event(key, state, context)

        while context["time"] < time_max:
            if context_handler is not None:
                context = context_handler(context)
            rng, event, context, dirty = _get_next_event(rng, state, context)
            state = event.new_state
            context["time"] = event.time
            if dirty and save_trajectory:
                events.append(event)
            for callback in callbacks:
                callback(state, context)
        if not save_trajectory:
            events.append(event)
        return events
