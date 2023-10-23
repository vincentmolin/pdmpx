import pdmpx
from pdmpx import BouncyParticleSampler, PDMPState

import jax.numpy as jnp
import jax
import numpy as np

import matplotlib.pyplot as plt


def potential(x):
    return jnp.sum(x**2) / 2.0


x0 = jnp.array([1.0, 1.0])
v0 = jnp.array([1.0, 1.0]) / 2 ** (1 / 2)
state = PDMPState(params=x0, velocities=v0)

##############################################
# Using the convenience function PDMP.simulate
##############################################

bps = BouncyParticleSampler(potential=potential, refreshment_rate=0.2)
events = bps.simulate(rng=jax.random.key(0), state=state, time_max=200.0)

times, states = zip(*events)
times = np.array(times)
xs = np.array([s.params for s in states])


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(xs[:, 0], xs[:, 1])
axs[0].set_title("Trajectory")
axs[1].plot(times, xs[:, 0])
axs[1].plot(times, xs[:, 1])
axs[1].set_title("Coordinate traces")
plt.show()

##############################################
# Using the lower level PDMP.get_next_event
# allows for a more JAX-like style
##############################################


@jax.jit
def next_event(rng, state, context, batch):
    rng, key = jax.random.split(rng)

    def minibatch_potential(params):
        """
        Dummy potential function that closes over the batch.
        """
        return jnp.sum((params - batch) ** 2) / 2.0

    return rng, *BouncyParticleSampler(
        minibatch_potential, refreshment_rate=0.2
    ).get_next_event(key, state, context)


T = 200.0
batch = jnp.array([1.0, 1.0])

x0 = jnp.array([1.0, 1.0])
v0 = jnp.array([1.0, 1.0]) / 2 ** (1 / 2)
state = PDMPState(params=x0, velocities=v0)
t = 0.0
rng = jax.random.key(0)

rng, event, context, dirty = next_event(rng, state, {}, batch)

xs = [state.params]
ts = [0.0]

while t < T:
    rng, event, context, dirty = next_event(rng, state, context, batch)
    state = event.new_state
    t += event.time
    if dirty:
        xs.append(np.array(state.params))
        ts.append(t)
