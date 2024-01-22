import pdmpx
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

pdmpx.utils.dir_derivs.nth_dir_deriv(lambda t: t**2, only_n=True)(0.0, 1.0, 2)


class QuadraticDynamics(pdmpx.dynamics.AbstractDynamics):
    def forward(self, t, state: pdmpx.PDMPState):
        v = state.params["v"]
        a = state.velocities["v"]
        x = state.params["x"] + t * v + 0.5 * t**2 * a
        v = v + t * a
        return pdmpx.PDMPState(
            params={"x": x, "v": v},
            velocities={"x": jnp.zeros(x.shape), "v": v},
        )


class QuadraticBounceFactor:  # (pdmpx.pdmp.AbstractFactor):
    def __init__(self, potential, valid_time):
        self.potential = potential
        self.valid_time = valid_time
        self.dynamics = QuadraticDynamics()
        self.rates_fn = lambda state: pdmpx.utils.dir_derivs.nth_dir_deriv(
            lambda t: self.potential(self.dynamics.forward(t, state).params)
        )(0.0, 1.0, 2)

    def timer(self, rng, state, ctx={}):
        time_key, event_key = jax.random.split(rng)
        # time = jax.random.exponential(time_key) / self.rates_fn(state)
        raise NotImplementedError
        return pdmpx.timers.TimerEvent(time, 0.0), ctx


# return pdmpx.PDMPState(
#     params={
#         "x": state.params["x"]
#         + t * state.params["v"]
#         + 0.5 * t**2 * state.velocities["v"],
#         "v": state.params["v"] + t * state.velocities["v"],
#     },
#     velocities={
#         "x": jnp.zeros(
#             state.params["x"].shape
#         ),  # state.velocities["x"] + t * state.velocities["v"],
#         "v": state.velocities["v"],
#     },
# )


x0 = jnp.array([0.5, 1.0])
v0 = jnp.array([0.5, -0.7])
a0 = jnp.array([1.0, 0.0])

state = pdmpx.PDMPState(
    params={"x": x0, "v": v0},
    velocities={"x": v0, "v": a0},
)


dynamics = QuadraticDynamics()
# dynamics = pdmpx.dynamics.LinearDynamics()
dynamics.forward(1.0, state)

mean = jnp.array([1.0, 1.0])
cov = jnp.array([[1.0, 0.5], [0.0, 1.0]])


def x_potential(params, ctx={}):
    # return jax.scipy.stats.multivariate_normal.logpdf(params["x"], mean, cov)
    return jnp.sum((params["x"] - mean) ** 2) / 2


def v_potential(params, ctx={}):
    return jnp.sum(params["v"] ** 2) / 2


def potential(params, ctx={}):
    return x_potential(params, ctx) + v_potential(params, ctx)


# jax.grad(potential)(state.params)


@jax.jit
def next_event(rng, state):
    rng, key = jax.random.split(rng)
    bounce_factor = pdmpx.bouncy.BPSBounceFactor(
        potential, 1.0, normalize_velocities=False, dynamics=dynamics
    )
    refresh_factor = pdmpx.refreshments.ConstantRateRefreshments(
        0.1, normalize_velocities=True
    )
    factor_queue = pdmpx.queues.SimpleFactorQueue([bounce_factor, refresh_factor])

    event, context, dirty = pdmpx.PDMP(dynamics, factor_queue).get_next_event(
        key, state
    )
    new_state = event.new_state
    # new_state.params["v"] = new_state.velocities["x"]
    event = pdmpx.Event(event.time, new_state)
    return rng, event, context, dirty


rng, ev, context, dirty = next_event(jax.random.key(0), state)
context["time"] = ev.time
T = 50.0
events = []
ts = []
t = 0.0
while t < T:
    rng, ev, context, dirty = next_event(rng, ev.new_state)
    t += ev.time
    # print("t = ", t, "dirty = ", dirty)
    events.append(ev)
    ts.append(t)

xs = [ev.new_state.params["x"] for ev in events]
xs = np.array(xs)
vs = [ev.new_state.params["v"] for ev in events]
vs = np.array(vs)
states = [ev.new_state for ev in events]


def discretize(ts, states, n=5000):
    xs = []
    tmesh = np.linspace(0, T, n)
    state_ctr = 0
    for i in range(len(tmesh)):
        while ts[state_ctr + 1] < tmesh[i]:
            state_ctr += 1
        t_ref = ts[state_ctr]
        dt = tmesh[i] - t_ref
        state = states[state_ctr]
        xs.append(
            state.params["x"]
            + dt * state.velocities["x"]
            + 0.5 * dt**2 * state.velocities["v"]
        )
    return tmesh, np.array(xs)


tm, xd = discretize(ts, states)
x0cum_avg = np.cumsum(xd[:, 0]) / np.arange(1, len(xd) + 1)
x1cum_avg = np.cumsum(xd[:, 1]) / np.arange(1, len(xd) + 1)
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
axs[0].plot(xd[:, 0], xd[:, 1])
axs[0].set_title("Trajectory")
axs[0].set_xlabel("x0")
axs[0].set_ylabel("x1")
axs[1].plot(tm, xd[:, 0], label="x0", color="C0")
axs[1].plot(tm, xd[:, 1], label="x1", color="C1")
axs[1].plot(tm, x0cum_avg, "--", label="x0 cum avg", color="C0")
axs[1].plot(tm, x1cum_avg, "--", label="x1 cum avg", color="C1")
axs[1].set_title("Coordinate traces")
axs[1].legend()
axs[2].plot(vs[:, 0], vs[:, 1])
fig.suptitle("Bouncy Particle Sampler with Quadratic Dynamics targeting a Gaussian")
plt.show()
