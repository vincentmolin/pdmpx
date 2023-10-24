import pdmpx
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple

x0 = jnp.array([1.0, 1.0])
v0 = jnp.array([0.5, -0.7])
a0 = jnp.array([1.0, 0.0])

state = pdmpx.PDMPState(
    params={"x": x0, "v": v0},
    velocities={"x": v0, "v": a0},
)


class QuadraticDynamics(pdmpx.dynamics.AbstractDynamics):
    def forward(self, t, state: pdmpx.PDMPState):
        return pdmpx.PDMPState(
            params={
                "x": state.params["x"]
                + t * state.velocities["x"]
                + 0.5 * t**2 * state.velocities["v"],
                "v": state.params["v"] + t * state.velocities["v"],
            },
            velocities={
                "x": state.velocities["x"] + t * state.velocities["v"],
                "v": state.velocities["v"],
            },
        )


dynamics = QuadraticDynamics()
# dynamics = pdmpx.dynamics.LinearDynamics()
dynamics.forward(1.0, state)


def potential(params, ctx={}):
    return jnp.sum(params["x"] ** 2)


@jax.jit
def next_event(rng, state):
    rng, key = jax.random.split(rng)
    bounce_factor = pdmpx.bouncy.BPSBounceFactor(
        potential, 1.0, normalize_velocities=False, dynamics=dynamics
    )
    refresh_factor = pdmpx.refreshments.ConstantRateRefreshments(0.1)
    factor_queue = pdmpx.queues.SimpleFactorQueue([bounce_factor, refresh_factor])
    return rng, *pdmpx.PDMP(dynamics, factor_queue).get_next_event(key, state)


rng, ev, context, dirty = next_event(jax.random.key(0), state)
context["time"] = ev.time
T = 500.0
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
fig, axs = plt.subplots(1, 2, figsize=(16, 9))
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
fig.suptitle("Bouncy Particle Sampler with Quadratic Dynamics targeting a Gaussian")
plt.show()
