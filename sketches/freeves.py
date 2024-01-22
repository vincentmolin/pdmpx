# Cold bouncy particle sampler on toy probs

import pdmpx
from pdmpx import PDMPState

import jax.numpy as jnp
import jax
import numpy as np

import matplotlib.pyplot as plt


def rosenbrock(x):
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


x0 = jnp.array([-1.0, 1.0])
v0 = jax.grad(rosenbrock)(x0)
state = PDMPState(params=x0, velocities=v0)

T = 100.0
# gradient_mix = 0.2
coldness = lambda t: (t + 1) * 0.5
refreshment_rate = 0.1


@jax.jit
def next_event(rng, state, context):
    rng, key = jax.random.split(rng)

    cold = coldness(context["time"])
    # gmix = gradient_mix  # (context["time"])

    return rng, *pdmpx.bouncy.ColdBouncyParticleSampler(
        potential=rosenbrock,
        refreshment_rate=refreshment_rate,
        gradient_mix=None,
        coldness=cold,
        valid_time=0.1,
        normalize_velocities=True,
        fletcher_reeves="no_reflect",
    ).get_next_event(key, state, context)


events = []
context = {"time": 0.0}
rng = jax.random.key(0)

while context["time"] < T:
    rng, ev, context, dirty = next_event(rng, state, context)
    context["time"] = ev.time
    state = ev.new_state
    if dirty:
        events.append(ev)

times, states = zip(*events)
times = np.array(times)
xs = np.array([s.params for s in states])

plt.plot(xs[:, 0], xs[:, 1])
# missing: nice plots


plt.plot(xs[0:50, 0], xs[0:50, 1])
