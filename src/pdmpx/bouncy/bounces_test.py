import jax
import jax.numpy as jnp
import jax.random as jr


from pdmpx.bouncy import BounceKernel, BounceTimer, BPSState


def tree_allclose(a, b):
    return all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.allclose, a, b)))


def test_bounces():
    u = lambda x: jnp.sum(x**2) / 2

    x0 = jr.normal(jr.key(0), (2,))
    v0 = jr.normal(jr.key(1), (2,))
    initial_state = BPSState(x0, v0)

    bk = BounceKernel(u)
    bt = BounceTimer(u, valid_time=0.5)

    event = bt(jr.key(2), initial_state)
    assert event.time > 0.0
