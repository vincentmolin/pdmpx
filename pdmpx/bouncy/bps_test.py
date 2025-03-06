import jax
import jax.numpy as jnp
import jax.random as jr

from pdmpx.bouncy import BPSState, BouncyParticleSampler
from pdmpx.utils import discretize_trajectory


def tree_allclose(a, b):
    return all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.allclose, a, b)))


def test_bouncy_particle_sampler():
    u = lambda x: jnp.sum(x**2) / 2

    x0 = jr.normal(jr.key(0), (2,))
    v0 = jr.normal(jr.key(1), (2,))
    initial_state = BPSState(x0, v0)

    bps = BouncyParticleSampler(
        u, refreshment_rate=0.5, valid_time=0.5, normalize_velocities=False
    )

    events = bps.simulate(jr.key(2), initial_state, 1000)

    xs = [ev.new_state.params for ev in events]
    xs = jnp.stack(xs)
    ts = [ev.time for ev in events]
    ts = jnp.stack(ts)

    xx = discretize_trajectory(xs, ts, 1000)

    assert jnp.abs((jnp.eye(2) - jnp.cov(xx.T))).max() < 0.2
    assert jnp.abs(jnp.mean(xx, axis=0)).max() < 0.1
