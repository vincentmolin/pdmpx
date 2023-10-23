import functools as ft
import jax
import jax.numpy as jnp
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
    """
    Approximates the first arrival time of an inhomogeneous Poisson process
    by a first order Taylor approximation of the rate function, trusted for
    a finite interval.
    """

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


class LinearThinningSlack(NamedTuple):
    a: float
    b: float


class LinearThinningTimer(AbstractTimer):
    def __init__(
        self,
        rate_fn: Callable[[PyTree, PyTree], Union[float, Tuple[float, Any]]],
        valid_time: float,
        slack: LinearThinningSlack = LinearThinningSlack(1.0, 1.0),
        adaptive_slack=True,
        has_aux=False,
        dynamics=LinearDynamics(),
    ):
        self.rate_fn = rate_fn
        self.valid_time = valid_time
        self.has_aux = has_aux
        self.slack = slack
        self.adaptive_slack = adaptive_slack
        self.dynamics = dynamics
        raise NotImplementedError

    def _thinning_loop(self, rng, state: PDMPState, slack):
        rate, drate = jax.jvp(lambda t: self.rate_fn(*self.dynamics.forward(t, state)))

        # def body_fn(rng, t, tau, x, v, rate_bound, grad_evals):
        #     t += tau
        #     x += tau * v
        #     rate_bound = rate_fn(x, v) + c * delta_max * jnp.linalg.norm(v)
        #     grad_evals += 1
        #     rng, key = jax.random.split(rng)
        #     tau = jnp.minimum(jax.random.exponential(key) / rate_bound, delta_max)
        #     return rng, t, tau, x, v, rate_bound, grad_evals

        # cond_fn = lambda rng, t, tau, x, v, rate_bound, grad_evals: tau == delta_max
        # init_val = body_fn(rng, 0.0, 0.0, x, v, 0.0, 0)
        # rng, t, tau, x, v, rate_bound, grad_evals = jax.lax.while_loop(
        #     lambda args: cond_fn(*args), lambda args: body_fn(*args), init_val
        # )
        # return t + tau, x + tau * v, rate_bound, grad_evals

    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        slack = context.get("linear_thinning_slack", self.slack)
        event = TimerEvent(0.0, 0.0)
        if self.has_aux:
            return event, {"timer": None, **context}
        else:
            return event, context
