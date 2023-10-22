import jax
import jax.numpy as jnp

from pdmpx import PDMPState
from pdmpx.dynamics import LinearDynamics


def tree_allclose(a, b):
    return all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.allclose, a, b)))


def test_linear_dynamics():
    simple_state = PDMPState(jnp.zeros((2,)), jnp.array([1.0, 0.1]))
    forw_10 = LinearDynamics().forward(10.0, simple_state)
    assert jnp.all(forw_10.velocities == simple_state.velocities)
    assert jnp.all(
        forw_10.params == simple_state.params + 10.0 * simple_state.velocities
    )

    tree_x = {"a": jnp.zeros((2,)), "b": {"c": jnp.zeros((1, 2, 3, 4))}, "d": 0.0}
    tree_v = {"a": jnp.array([1.0, 0.1]), "b": {"c": jnp.ones((1, 2, 3, 4))}, "d": 0.0}
    tree_state = PDMPState(tree_x, tree_v)
    forw_10 = LinearDynamics().forward(10.0, tree_state)
    assert tree_allclose(forw_10.velocities, tree_state.velocities)
    assert jnp.all(tree_x["a"] + 10.0 * tree_v["a"] == forw_10.params["a"])
    assert jnp.all(
        tree_x["b"]["c"] + 10.0 * tree_v["b"]["c"] == forw_10.params["b"]["c"]
    )
    assert jnp.all(tree_x["d"] + 10.0 * tree_v["d"] == forw_10.params["d"])
