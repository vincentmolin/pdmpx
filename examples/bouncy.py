import jax
import jax.numpy as jnp
import jax.random as jr

from pdmpx.bouncy import BouncyParticleSampler, BPSState
from pdmpx.utils import discretize_trajectory

import matplotlib.pyplot as plt


u = lambda x: jnp.sum(x**2) / 2

x0 = jr.normal(jr.key(0), (2,))
v0 = jr.normal(jr.key(5), (2,))
initial_state = BPSState(x0, v0)

bps = BouncyParticleSampler(
    u, refreshment_rate=0.5, valid_time=0.5, normalize_velocities=False
)

events = bps.simulate(jr.key(2), initial_state, 1000)

xs = [ev.new_state.params for ev in events]
xs = jnp.stack(xs)
ts = [ev.time for ev in events]
ts = jnp.stack(ts)
plt.plot(xs[:, 0], xs[:, 1], "o-")

xx = discretize_trajectory(xs, ts, 1000)

jnp.cov(xx.T)
jnp.mean(xx, axis=0)
