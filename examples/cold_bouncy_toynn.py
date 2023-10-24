# Cold bouncy particle sampler on toy probs
import pdmpx
from pdmpx import PDMPState


import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

import matplotlib.pyplot as plt


def fake_mnist():
    rng = jax.random.key(0)
    while True:
        rng, key = jax.random.split(rng)
        yield jax.random.normal(key, (32, 784)), jax.random.categorical(
            key, jnp.ones((10,)), shape=(32,)
        )


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], 28, 28, 1))
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


dataset = fake_mnist()
batch = next(dataset)
imgs, labels = batch
model = CNN()
params = model.init(jax.random.key(0), imgs)
output = model.apply(params, imgs)


def potential(params, batch):
    x, y = batch
    out = model.apply(params, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(out, y)
    return jnp.mean(loss)


velocities = pdmpx.utils.tree.tree_random_normal(jax.random.key(0), params)
velocities = pdmpx.utils.tree.tree_unit_length(velocities)
state = PDMPState(params=params, velocities=velocities)


T = 100.0
gradient_mix = 0.2
coldness = lambda t: (t + 1) * 0.5
refreshment_rate = 0.1


@jax.jit
def next_event(rng, state, context, batch):
    rng, key = jax.random.split(rng)

    cold = coldness(context["time"])
    gmix = gradient_mix  # (context["time"])

    return rng, *pdmpx.bouncy.ColdBouncyParticleSampler(
        potential=lambda params: potential(params, batch),
        refreshment_rate=refreshment_rate,
        gradient_mix=gmix,
        coldness=cold,
        valid_time=0.1,
        normalize_velocities=True,
    ).get_next_event(key, state, context)


context = {"time": 0.0}
rng = jax.random.key(0)

# Training loop!
while context["time"] < T:
    rng, ev, context, dirty = next_event(rng, state, context, next(dataset))
    context["time"] = ev.time
    state = ev.new_state


l2_dist = jax.tree_util.tree_reduce(
    lambda x, y: x + y,
    jax.tree_util.tree_map(lambda x, y: jnp.sum((x - y) ** 2), params, state.params),
)
print(f"Final L2 distance to initial parameters: {l2_dist}")
