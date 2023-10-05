import numpy as np
import jax.numpy as jnp
import jax
import functools as ft
from typing import Any, NamedTuple, Sequence, Tuple, Callable


class ParticleState(NamedTuple):
    x: jnp.ndarray
    v: jnp.ndarray


class Event(NamedTuple):
    time: float
    new_state: ParticleState


class AbstractDynamics:
    def forward(self, t, state):
        raise NotImplementedError


class AbstractEventHandler:
    def get_next_event(self, state: ParticleState, rng=None) -> Tuple[float, Callable]:
        raise NotImplementedError


class AbstractEventClock:
    def get_next_event_time(self, state: ParticleState, rng=None) -> float:
        raise NotImplementedError

    def get_new_state(self, state: ParticleState, rng=None) -> ParticleState:
        raise NotImplementedError


class SimpleEventQueue(AbstractEventHandler):
    def __init__(self, event_clocks: Sequence[AbstractEventClock]):
        self.event_clocks = event_clocks
        self.next_event = None

    def get_next_event(self, state: ParticleState, rng=None) -> Tuple[float, Callable]:
        event_times = [handler.get_event_time(state) for handler in self.event_clocks]
        next_event = np.argmin(event_times)
        next_event_time = event_times[next_event]
        return next_event_time, self.event_clocks[next_event].get_new_state


class LinearDynamics(AbstractDynamics):
    def forward(self, t: float, state: ParticleState):
        return ParticleState(state.x + t * state.v, state.v)


def rng_wrap(fun):
    def wrapped_fun(rng, *args, **kwargs):
        rng, key = jax.random.split(rng)
        return rng, fun(*args, **kwargs, rng=key)

    return wrapped_fun


class PDMP:
    def __init__(self, dynamics: AbstractDynamics, event_handler: AbstractEventHandler):
        self.dynamics = dynamics
        self.event_handler = event_handler

    def _next_event(self, time: float, state: ParticleState, rng):
        key0, key1 = jax.random.split(rng)
        next_event_time, new_state_fn = self.event_handler.get_next_event(state, key0)
        time += next_event_time
        state = self.dynamics.forward(next_event_time, state)
        state = new_state_fn(state, key1)
        return Event(time, state)

    def simulate(
        self, state: ParticleState, time_max: float, rng=jax.random.PRNGKey(0)
    ) -> Sequence[Event]:
        time = 0.0
        events = []

        get_next_event = rng_wrap(self._next_event)

        while time < time_max:
            rng, event = get_next_event(rng, time, state)
            time = event.time
            events.append(event)
        return events


class Refreshments(AbstractEventHandler):
    def __init__(self, rate):
        self.rate = rate

    def get_event_time(self, state, rng=None):
        return jnp.random.exponential(self.rate)

    def get_new_state(self, state, rng=None):
        return ParticleState(state.x, jax.random.normal(rng, size=state.v.shape))


class Reflection(AbstractEventHandler):
    def __init__(self, clock, kernel):
        self.clock = clock
        self.kernel = kernel

    def get_event_time(self, state, rng=None):
        return self.clock.get_next_event_time(state, rng)

    def get_new_state(self, state, rng=None):
        return self.kernel(state, rng)


class LinearThinner:
    def __init__(self, rate_fn, dynamics=LinearDynamics()):
        # rate_fn =
        self.rate_fn = rate_fn
        self.dynamics = dynamics

    @ft.partial(jax.jit, static_argnums=(0,))
    def _compute_rate_bound(self, state: ParticleState):
        x, v = state
        # get 1st order Taylor approximation of rate function
        rate, drate = jax.jvp(lambda xx: self.rate_fn(xx, v), (x,), (v,))
        return rate, drate

    def sketch_get_next_event_time(
        self, state: ParticleState, slack: Tuple[float, float], rng=None
    ):
        rate, drate = self._compute_rate_bound(state)
        a, b = rate + slack[0], drate + slack[1]
        rng, suggested = self.ab_poisson_time(rng, a, b)
        proposed_state = self.dynamics.forward(suggested, state)
        u = jax.random.uniform(rng)
        bound = a + b * suggested
        if u < self.rate_fn(proposed_state) / bound:
            return suggested, True
        else:
            return suggested, False


class BPSReflectionKernel:
    def __init__(self, grad_psi_fn):
        self.grad_psi_fn = grad_psi_fn

    @ft.partial(jax.jit, static_argnums=(0,))
    def __call__(self, state: ParticleState, rng=None) -> ParticleState:
        x, v = state
        grad = self.grad_psi_fn(x)
        v_p = project_ab(v, grad)
        return ParticleState(x, v - 2 * v_p)


################################


@jax.jit
def project_ab(a, b):
    """
    Project the vector a onto b
    """
    c = jnp.dot(a, b) / jnp.dot(b, b)
    return c * b


@jax.jit
def oscn(key, v, grad_psi, rho):
    """
    Resample v using the oscn method
    """
    v_p = project_ab(v, grad_psi)
    v_t = v - v_p
    z = jax.random.normal(key, v.shape)
    z = z - project_ab(z, grad_psi)
    return -v_p + rho * v_t + jnp.sqrt(1 - rho**2) * z


class LocalBoundThinner:
    def __init__(self, rate_fn, bound_fn, delta_max):
        self.rate_fn = rate_fn
        self.bound_fn = bound_fn
        self.delta_max = delta_max

    @ft.partial(jax.jit, static_argnums=(0,))
    def _next_event_candidate(self, rng, x, v, c):
        """
        Simulates forward until the next event candidate time
        """

        def body_fn(rng, t, tau, x, v, rate_bound, rate_evals):
            t += tau
            x += tau * v
            rate_bound = self.bound_fn(self.rate_fn(x, v), x, v, c)
            rate_evals += 1
            rng, key = jax.random.split(rng)
            tau = jnp.minimum(jax.random.exponential(key) / rate_bound, self.delta_max)
            return rng, t, tau, x, v, rate_bound, rate_evals

        cond_fn = (
            lambda rng, t, tau, x, v, rate_bound, rate_evals: tau == self.delta_max
        )
        init_val = body_fn(rng, 0.0, 0.0, x, v, 0.0, 0)
        rng, t, tau, x, v, rate_bound, rate_evals = jax.lax.while_loop(
            lambda args: cond_fn(*args), lambda args: body_fn(*args), init_val
        )
        return t + tau, x + tau * v, rate_bound, rate_evals

    @ft.partial(jax.jit, static_argnums=(0,))
    def next_event_time(self, rng, x, v, c):
        """
        Simulates forward until the next reflection event is accepted
        """
        rate_evals = 0

        def cond_fn(rng, t, x, v, u, a, sug, rate_evals):
            return u >= a  # or a > 1.0

        def body_fn(rng, t, x, v, u, a, sug, rate_evals):
            rng, ukey, ekey = jax.random.split(rng, 3)
            u = jax.random.uniform(ukey)
            tau, x, rate_bound, evals = self._next_event_candidate(ekey, x, v, c)
            rate_evals += evals
            t += tau
            sug += 1
            a = self.rate_fn(x, v) / rate_bound
            return rng, t, x, v, u, a, sug, rate_evals

        init_val = body_fn(rng, 0.0, x, v, None, None, 0, 0)

        rng, t, x, v, u, a, sug, rate_evals = jax.lax.while_loop(
            lambda args: cond_fn(*args), lambda args: body_fn(*args), init_val
        )

        # rng, key = jax.random.split(rng)
        # v = oscn(key, v, grad_psi_fn(x), rho)

        return rng, t, x, v, sug, rate_evals + 1, a


if __name__ == "__main__":
    mu = jnp.array([1.0, 1.0])
    sig = jnp.array([1.0, 0.5])

    import tensorflow_probability.substrates.jax as tfp  # pyright: ignore
    import matplotlib.pyplot as plt

    tfd = tfp.distributions
    tfb = tfp.bijectors
    tfpk = tfp.math.psd_kernels

    target = tfd.MultivariateNormalDiag(mu, sig)

    grad_psi_fn = jax.grad(lambda x: -target.log_prob(x))

    x = jnp.array([0.0, 0.0])
    v = jnp.array([1.0, 0.0])
    grad_psi_fn(x)

    # %time xs, ts, accepted, suggested = oscnbps(grad_psi_fn, x, v, 100.0, rho=0.8)
    # xs, ts, accepted, suggested = oscnbps(grad_psi_fn, x, v, 25.0, rho=0.8)
    # print(f"{accepted} / {suggested}")
    # jaxy_oscnbps(grad_psi_fn, x, v, 1.0, rho=0.8)
    # %time xs, ts, accepted, suggested = jaxy_oscnbps(grad_psi_fn, x, v, 100.0, rho=0.8)

    psi_fn = lambda x: 0.5 * jnp.sum(x**2)
    rate_fn = lambda x, v: jnp.maximum(jax.jvp(psi_fn, (x,), (v,))[1], 0)
    x = jax.random.normal(jax.random.PRNGKey(0), (2,))
    v = -x
    st = ParticleState(x, v)
    lt = LinearThinner(rate_fn)
    lt._compute_rate_bound(st)

    ts = np.linspace(0, 2, 100)
    rates = [rate_fn(x + t * v, v) for t in ts]
    plt.plot(ts, rates)
