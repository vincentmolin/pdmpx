import pdmpx
import jax
import jax.random as jr
import jax.numpy as jnp
from pdmpx.utils.tree import tree_random_normal, tree_unit_length

import numpy as np
import matplotlib.pyplot as pl

N_PARTICLES = 3

xs = {i: jnp.zeros((2,)) for i in range(N_PARTICLES)}
xs = tree_random_normal(jr.key(0), xs)
vs = tree_random_normal(jr.key(1), xs)
state0 = pdmpx.PDMPState(xs, vs)


class RocketRefreshmentFactor(pdmpx.pdmp.AbstractFactor):
    def __init__(self, rate, attract, repel, refresh):
        self.rate = rate
        self.attract = attract
        self.repel = repel
        self.refresh = refresh
        self.event_logits = jnp.log(jnp.array([attract, repel, refresh]))
        self.next_event_idx = 0

        self.interaction_matrix = (1 + 1 / (N_PARTICLES - 1)) * jnp.eye(
            N_PARTICLES
        ) - 1 / (N_PARTICLES - 1)

    def timer(self, rng, state, ctx={}):
        time_key, event_key = jr.split(rng)

        event_idx = jr.categorical(
            event_key,
            logits=self.event_logits,
        )
        time = jr.exponential(time_key) / self.rate

        self.next_event_idx = event_idx

        return pdmpx.timers.TimerEvent(time, 0.0), {**ctx, "rrf_event_idx": event_idx}

    def kernel(self, rng, state, ctx={}):
        # event_idx = ctx["rrf_event_idx"]
        event_idx = self.next_event_idx

        def repel(state):
            X = jnp.array([state.params[i] for i in range(N_PARTICLES)])
            V = self.interaction_matrix @ X
            nV = V / jnp.sqrt(jnp.sum(V**2, axis=1, keepdims=True) + 1e-6)
            nV = (1 / self.repel) * nV
            return pdmpx.PDMPState(state.params, {i: nV[i] for i in range(N_PARTICLES)})

        def attract(state):
            X = jnp.array([state.params[i] for i in range(N_PARTICLES)])
            V = self.interaction_matrix @ X
            nV = V / jnp.sqrt(jnp.sum(V**2, axis=1, keepdims=True) + 1e-6)
            nV = -(1 / self.repel) * nV
            return pdmpx.PDMPState(state.params, {i: nV[i] for i in range(N_PARTICLES)})

        def refresh(state):
            return pdmpx.PDMPState(
                state.params, tree_random_normal(rng, state.velocities)
            )

        return jax.lax.switch(event_idx, [attract, repel, refresh], state)


# state = pdmpx.PDMPState(xs, vs)
# pdmp = pdmpx.PDMP(
#     dynamics=pdmpx.dynamics.LinearDynamics(),
#     factor=RocketRefreshmentFactor(1.0, 0.7, 0.2, 0.1),
# )


# @jax.jit
# def _step(rng, state):
#     rng, key = jr.split(rng)
#     rff = RocketRefreshmentFactor(1.0, 0.7, 0.2, 0.1)
#     pdmp = pdmpx.PDMP(
#         dynamics=pdmpx.dynamics.LinearDynamics(),
#         factor=rff,
#     )
#     pdmp.get_next_event(key, state)
#     return rng, rff.next_event_idx


# nevs = []
# rng = jr.key(0)
# for _ in range(1000):
#     rng, nev = _step(rng, state)
#     nevs.append(nev)
# nevs = np.array(nevs)
# np.bincount(nevs)


def independent_potential(key):
    def potential(params, ctx={}):
        return jnp.sum(params[key] ** 2) / 2

    return potential


@jax.jit
def next_event(rng, state, ctx):
    rng, key = jr.split(rng)

    # state = pdmpx.PDMPState(
    #     state.params,
    #     {
    #         "x0": tree_unit_length(state.velocities["x0"]),
    #         "x1": tree_unit_length(state.velocities["x1"]),
    #     },
    # )

    bounces = [
        pdmpx.bouncy.BPSBounceFactor(
            independent_potential(i),
            valid_time=1.0,
            normalize_velocities=False,
            dynamics=pdmpx.dynamics.LinearDynamics(),
        )
        for i in range(N_PARTICLES)
    ]
    rff = RocketRefreshmentFactor(1.2 * N_PARTICLES, 0.4, 0.4, 0.2)

    pdmp = pdmpx.PDMP(
        dynamics=pdmpx.dynamics.LinearDynamics(),
        factor=pdmpx.queues.SimpleFactorQueue([*bounces, rff]),
    )
    event, ctx, dirty = pdmp.get_next_event(key, state, ctx)

    return (
        rng,
        event,
        {**ctx, "time": event.time},
        dirty,
        ctx["simple_factor_queue"]["next_event_idx"],
        rff.next_event_idx,
    )


import matplotlib.pyplot as plt
import scipy.interpolate as spi
from matplotlib.animation import FuncAnimation

evs = []
evtypes = []
ts = []
rng = jr.key(0)
state = state0  # pdmpx.PDMPState(xs, vs)
ctx = {"time": 0.0}
for _ in range(50):
    rng, ev, ctx, _, evtype, rfftype = next_event(rng, state, ctx)
    state = ev.new_state
    ts.append(ctx["time"])
    evs.append(ev)
    evtypes.append(evtype if evtype < N_PARTICLES else rfftype + N_PARTICLES)

ts = np.array(ts)
evts = np.array(evtypes)
xs = np.array(
    [np.array([ev.new_state.params[i] for ev in evs]) for i in range(N_PARTICLES)]
)
fig, ax = plt.subplots(2, 1, figsize=(6, 8))
for i in range(N_PARTICLES):
    ax[0].plot(xs[i, :, 0], xs[i, :, 1], color=f"C{i}")
    ax[0].plot(xs[i, evts == i, 0], xs[i, evts == i, 1], "x", color=f"C{i}")
ax[1].plot(ts, evts, "x")
ax[1].set_yticks(np.arange(N_PARTICLES + 3))
ax[1].set_yticklabels([*[f"B{i}" for i in range(N_PARTICLES)], "Attr", "Rep", "Refr"])

splines = [spi.make_interp_spline(ts, xs[i], k=1) for i in range(N_PARTICLES)]

tm, tM = ts.min(), ts.max()

fig, ax = plt.subplots()

lns = [ax.plot([], [], color=f"C{i}")[0] for i in range(N_PARTICLES)]
pts = [ax.plot([], [], "o", color=f"C{i}")[0] for i in range(N_PARTICLES)]


def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    return *lns, *pts


t_idx = 0


def update(frame_t):
    global t_idx
    while t_idx < len(ts) and frame_t > ts[t_idx]:
        t_idx += 1

    for i in range(N_PARTICLES):
        xi = splines[i](frame_t)
        pts[i].set_data([xi[0]], [xi[1]])
        lns[i].set_data([*xs[i, 0:t_idx, 0], xi[0]], [*xs[i, 0:t_idx, 1], xi[1]])
    return *lns, *pts


init()
update(2.0)

ani = FuncAnimation(
    fig,
    update,
    frames=np.linspace(tm, tM - 0.5, 500),
    init_func=init,
    interval=50,
    blit=True,
)
ani.save("rocket_n.gif")
# plt.show()

# x0s_bnce.shape, x1s_bnce.shape

# fig, ax = plt.subplots()
# ax.plot(x0s[:, 0], x0s[:, 1], color="C0")
# ax.plot(x0s_bnce[:, 0], x0s_bnce[:, 1], "x", color="C0", label="x0 bounce")

# ax.plot(x1s[:, 0], x1s[:, 1], color="C1")
# ax.plot(x1s_bnce[:, 0], x1s_bnce[:, 1], "x", color="C1", label="x1 bounce")

# ax.plot(x0s[evts == 2, 0], x0s[evts == 2, 1], "x", color="C2", label="attract")
# ax.plot(x1s[evts == 2, 0], x1s[evts == 2, 1], "x", color="C2")
# ax.plot(x0s[evts == 3, 0], x0s[evts == 3, 1], "x", color="C3", label="repel")
# ax.plot(x1s[evts == 3, 0], x1s[evts == 3, 1], "x", color="C3")
# ax.plot(x0s[evts == 4, 0], x0s[evts == 4, 1], "x", color="C4", label="refresh")
# ax.plot(x1s[evts == 4, 0], x1s[evts == 4, 1], "x", color="C4")
# ax.quiver(x0s[:, 0], x0s[:, 1], v0s[:, 0], v0s[:, 1], width=0.005, zorder=10)
# ax.legend()


# fig, ax = plt.subplots()

# x0_spline = spi.make_interp_spline(ts, x0s, k=1)
# x1_spline = spi.make_interp_spline(ts, x1s, k=1)

# tm, tM = ts.min(), ts.max()
# (lnx0,) = ax.plot([], [], "o", color="C0")
# (lnx1,) = ax.plot([], [], "o", color="C1")
# (lnx0s,) = ax.plot([x0s[0, 0]], [x0s[0, 1]], color="C0")
# (lnx1s,) = ax.plot([x1s[0, 0]], [x1s[0, 1]], color="C1")


# def init():
#     ax.set_xlim(-3, 3)
#     ax.set_ylim(-3, 3)
#     return lnx0, lnx1, lnx0s, lnx1s


# t_idx = 0


# def update(frame_t):
#     global t_idx
#     x0 = x0_spline(frame_t)
#     x1 = x1_spline(frame_t)
#     lnx0.set_data([x0[0]], [x0[1]])
#     lnx1.set_data([x1[0]], [x1[1]])
#     while frame_t > ts[t_idx]:
#         t_idx += 1
#         lnx0s.set_data([x0s[0:t_idx, 0]], [x0s[0:t_idx, 1]])
#         lnx1s.set_data([x1s[0:t_idx, 0]], [x1s[0:t_idx, 1]])
#     return lnx0, lnx1, lnx0s, lnx1s


# ani = FuncAnimation(
#     fig,
#     update,
#     frames=np.linspace(tm, tM - 0.5, 500),
#     init_func=init,
#     interval=50,
#     blit=True,
# )
# ani.save("rocket.gif")
# plt.show()
