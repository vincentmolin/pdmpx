import functools as ft
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple, Sequence, Tuple, Callable, Dict, Optional, Union

from .poisson_time.linear import ab_poisson_time
from .poisson_time.quadratic import abc_poisson_time
from .pdmp import AbstractTimer, TimerEvent, PDMPState, RNGKey, Context, PyTree
from .dynamics import LinearDynamics
from .utils.dir_derivs import nth_dir_deriv


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
        rate_fn: Callable[[PDMPState, Context], Union[float, Tuple[float, Any]]],
        valid_time: float,
        has_aux=False,
        dynamics=LinearDynamics(),
        timescale=1.0,
    ):
        self.valid_time = valid_time
        self.rate_fn = rate_fn
        self.has_aux = has_aux
        self.timescale = timescale
        self.dynamics = dynamics

    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        rate, drate, *maybe_aux = jax.jvp(
            lambda t: self.rate_fn(self.dynamics.forward(t, state), context),
            (0.0,),
            (1.0,),
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


class QuadraticApproxTimer(AbstractTimer):
    def __init__(
        self,
        rate_fn: Callable[[PDMPState, Context], Union[float, Tuple[float, Any]]],
        valid_time: float,
        has_aux=False,
        dynamics=LinearDynamics(),
        timescale=1.0,
    ):
        self.valid_time = valid_time
        self.rate_fn = rate_fn
        self.has_aux = has_aux
        if has_aux:
            raise NotImplementedError(
                "QuadraticApproxTimer requires has_aux=False for now"
            )
        self.timescale = timescale
        self.dynamics = dynamics

    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        rate, drate, ddrate = nth_dir_deriv(
            lambda t: self.rate_fn(self.dynamics.forward(t, state), context),
            only_n=False,
            ravel_out=True,
        )(0.0, 1.0, 2)
        a = rate * self.timescale
        b = drate * self.timescale
        c = ddrate * self.timescale
        u = jax.random.uniform(rng)
        time = abc_poisson_time(u, a, b, c)
        event = jax.lax.cond(
            time < self.valid_time,
            lambda: TimerEvent(time, bound=0.0),
            lambda: TimerEvent(self.valid_time, bound=1.0),
        )
        if self.has_aux:
            return event, {"timer": None, **context}
        else:
            return event, context


class LinearThinningSlack(NamedTuple):
    a: float
    b: float


class LinearThinningTimer(AbstractTimer):
    def __init__(
        self,
        rate_fn: Callable[[PDMPState], Union[float, Tuple[float, Any]]],
        valid_time: float,
        slack: Union[
            LinearThinningSlack, jnp.array, Tuple[float, float]
        ] = LinearThinningSlack(1.0, 1.0),
        adaptive_slack=True,
        has_aux=False,
        dynamics=LinearDynamics(),
    ):
        """
        Defines a timer that simulates Poisson times by upper bounding the rate
        with a linear function r = (a + bt)+, where (a,b) = (rate,drate) + slack
        """
        self.rate_fn = rate_fn
        self.valid_time = valid_time
        self.has_aux = has_aux
        if not isinstance(slack, LinearThinningSlack):
            assert len(slack) == 2
            slack = LinearThinningSlack(*slack)
        self.slack = slack
        self.adaptive_slack = adaptive_slack
        if self.adaptive_slack:
            self._adapt_slack = lambda slack, broken: LinearThinningSlack(
                slack.a + slack.a * (1.0 * broken), slack.b + slack.b * (1.0 * broken)
            )
        else:
            self._adapt_slack = lambda slack, broken: slack
        self.dynamics = dynamics

        if has_aux:
            raise NotImplementedError(
                "LinearThinningTimer requires has_aux=False for now"
            )

    def _thinning_loop(self, rng, state: PDMPState, slack, context={}):
        rate, drate = jax.jvp(
            lambda t: self.rate_fn(self.dynamics.forward(t, state), context),
            (0.0,),
            (1.0,),
        )
        bound_rate = lambda t: rate + slack.a + (drate + slack.b) * t

        class LoopState(NamedTuple):
            rng: RNGKey
            t: float
            state: PDMPState
            broken: float
            bound: float
            done: float

        def body_fn(ls: LoopState) -> LoopState:
            rng, t, state = ls.rng, ls.t, ls.state
            rng, key_bound, key_a = jax.random.split(rng, 3)
            bound_time = ab_poisson_time(jax.random.uniform(key_bound), rate, drate)

            def accept_reject(t):
                u = jax.random.uniform(key_a)
                a = self.rate_fn(self.dynamics.forward(t, state)) / bound_rate(t)
                return jax.lax.cond(
                    a > 1.0,  # Upper bound violation
                    lambda: LoopState(rng, t, state, broken=1.0, bound=0.0, done=1.0),
                    lambda: jax.lax.cond(
                        u < a,
                        lambda: LoopState(
                            rng,
                            t,
                            state,
                            broken=0.0,
                            bound=0.0,
                            done=1.0,
                        ),
                        lambda: LoopState(
                            rng,
                            t,
                            state,
                            broken=0.0,
                            bound=0.0,
                            done=0.0,
                        ),
                    ),
                )

            return jax.lax.cond(
                t + bound_time > self.valid_time,
                lambda: LoopState(
                    rng, self.valid_time, state, broken=0.0, bound=1.0, done=1.0
                ),
                lambda: accept_reject(t + bound_time),
            )

        def cond_fn(ls: LoopState) -> bool:
            return ls.done == 0.0

        init_val = LoopState(rng, 0.0, state, 0.0, 0.0, 0.0)
        ls = jax.lax.while_loop(cond_fn, body_fn, init_val)

        return ls.t, ls.broken, ls.bound

    def __call__(
        self, rng: RNGKey, state: PDMPState, context: Context = {}
    ) -> Tuple[TimerEvent, Context]:
        slack = context.get("linear_thinning_slack", self.slack)
        time, slack_broken, bound = self._thinning_loop(rng, state, slack, context)
        slack = self._adapt_slack(slack, slack_broken)
        context = {**context, "linear_thinning_slack": slack}

        event = TimerEvent(time, bound=bound)

        if self.has_aux:
            return event, context, None
        else:
            return event, context
