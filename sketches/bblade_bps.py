import pdmpx
import jax
import jax.random as jr
import jax.numpy as jnp
from pdmpx.utils.tree import tree_random_normal, tree_unit_length

import numpy as np
import matplotlib.pyplot as plt

xs = {
    "x0": jnp.array([-1.0, -1.0]),
    "x1": jnp.array([1.0, 1.0]),
}
vs = tree_random_normal(jr.key(0), xs)


class RocketRefreshmentFactor(pdmpx.pdmp.AbstractFactor):
    def __init__(self, rate, attract, repel, refresh):
        self.rate = rate
        self.attract = attract
        self.repel = repel
        self.refresh = refresh
        self.event_logits = jnp.log(jnp.array([attract, repel, refresh]))
        self.next_event_idx = 0

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
            x0 = state.params["x0"]
            x1 = state.params["x1"]
            v0 = tree_unit_length(x0 - x1) * 1 / self.repel
            v1 = -v0
            return pdmpx.PDMPState({"x0": x0, "x1": x1}, {"x0": v0, "x1": v1})

        def attract(state):
            x0 = state.params["x0"]
            x1 = state.params["x1"]
            v0 = tree_unit_length(x1 - x0) * 1 / self.attract
            v1 = -v0
            return pdmpx.PDMPState({"x0": x0, "x1": x1}, {"x0": v0, "x1": v1})

        def refresh(state):
            return pdmpx.PDMPState(
                state.params, tree_random_normal(rng, state.velocities)
            )

        return jax.lax.switch(event_idx, [attract, repel, refresh], state)


state = pdmpx.PDMPState(xs, vs)
pdmp = pdmpx.PDMP(
    dynamics=pdmpx.dynamics.LinearDynamics(),
    factor=RocketRefreshmentFactor(1.0, 0.7, 0.2, 0.1),
)


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


def get_ctx_event(ctx):
    qev = ctx["simple_factor_queue"]["next_event_idx"]
    if qev >= 2:
        qev += ctx["rff_event_idx"]
    return qev


def get_ctx_event_str(ctx):
    return ["Bounce x0", "Bounce x1", "Attract", "Repel", "Refresh"][get_ctx_event(ctx)]


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

    bounce0 = pdmpx.bouncy.BPSBounceFactor(
        independent_potential("x0"), normalize_velocities=False
    )
    bounce1 = pdmpx.bouncy.BPSBounceFactor(
        independent_potential("x1"), normalize_velocities=False
    )
    rff = RocketRefreshmentFactor(0.5, 0.7, 0.2, 0.1)

    pdmp = pdmpx.PDMP(
        dynamics=pdmpx.dynamics.LinearDynamics(),
        factor=pdmpx.queues.SimpleFactorQueue([bounce0, bounce1, rff]),
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


import scipy.interpolate as spi
from matplotlib.animation import FuncAnimation

evs = []
evtypes = []
ts = []
rng = jr.key(0)
state = pdmpx.PDMPState(xs, vs)
ctx = {"time": 0.0}
for _ in range(50):
    rng, ev, ctx, _, evtype, rfftype = next_event(rng, state, ctx)
    state = ev.new_state
    ts.append(ctx["time"])
    evs.append(ev)
    evtypes.append(evtype if evtype < 2 else rfftype + 2)

ts = np.array(ts)
evts = np.array(evtypes)
x0s = np.array([ev.new_state.params["x0"] for ev in evs])
x1s = np.array([ev.new_state.params["x1"] for ev in evs])
v0s = np.array([ev.new_state.velocities["x0"] for ev in evs])
v1s = np.array([ev.new_state.velocities["x1"] for ev in evs])
evs[0]
x0s_bnce = x0s[evts == 0]
x1s_bnce = x1s[evts == 1]

x0s_bnce.shape, x1s_bnce.shape

fig, ax = plt.subplots()
ax.plot(x0s[:, 0], x0s[:, 1], color="C0")
ax.plot(x0s_bnce[:, 0], x0s_bnce[:, 1], "x", color="C0", label="x0 bounce")

ax.plot(x1s[:, 0], x1s[:, 1], color="C1")
ax.plot(x1s_bnce[:, 0], x1s_bnce[:, 1], "x", color="C1", label="x1 bounce")

ax.plot(x0s[evts == 2, 0], x0s[evts == 2, 1], "x", color="C2", label="attract")
ax.plot(x1s[evts == 2, 0], x1s[evts == 2, 1], "x", color="C2")
ax.plot(x0s[evts == 3, 0], x0s[evts == 3, 1], "x", color="C3", label="repel")
ax.plot(x1s[evts == 3, 0], x1s[evts == 3, 1], "x", color="C3")
ax.plot(x0s[evts == 4, 0], x0s[evts == 4, 1], "x", color="C4", label="refresh")
ax.plot(x1s[evts == 4, 0], x1s[evts == 4, 1], "x", color="C4")
ax.quiver(x0s[:, 0], x0s[:, 1], v0s[:, 0], v0s[:, 1], width=0.005, zorder=10)
ax.legend()


fig, ax = plt.subplots()

x0_spline = spi.make_interp_spline(ts, x0s, k=1)
x1_spline = spi.make_interp_spline(ts, x1s, k=1)

tm, tM = ts.min(), ts.max()
(lnx0,) = ax.plot([], [], "o", color="C0")
(lnx1,) = ax.plot([], [], "o", color="C1")
(lnx0s,) = ax.plot([x0s[0, 0]], [x0s[0, 1]], color="C0")
(lnx1s,) = ax.plot([x1s[0, 0]], [x1s[0, 1]], color="C1")


def init():
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    return lnx0, lnx1, lnx0s, lnx1s


t_idx = 0


def update(frame_t):
    global t_idx
    x0 = x0_spline(frame_t)
    x1 = x1_spline(frame_t)
    lnx0.set_data([x0[0]], [x0[1]])
    lnx1.set_data([x1[0]], [x1[1]])
    while frame_t > ts[t_idx]:
        t_idx += 1
        lnx0s.set_data([x0s[0:t_idx, 0]], [x0s[0:t_idx, 1]])
        lnx1s.set_data([x1s[0:t_idx, 0]], [x1s[0:t_idx, 1]])
    return lnx0, lnx1, lnx0s, lnx1s


ani = FuncAnimation(
    fig,
    update,
    frames=np.linspace(tm, tM - 0.5, 500),
    init_func=init,
    interval=50,
    blit=True,
)
ani.save("rocket.gif")
plt.show()
