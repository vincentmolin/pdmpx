import jax
import jax.numpy as jnp

from pdmpx.bouncy import BPSState, LinearDynamics


def tree_allclose(a, b):
    return all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.allclose, a, b)))


def test_linear_dynamics():

    ld_forward = LinearDynamics().forward
    jit_ld_forward = jax.jit(ld_forward)

    simple_state = BPSState(jnp.zeros((2,)), jnp.array([1.0, 0.1]))
    forw_10 = ld_forward(10.0, simple_state)
    jit_forw_10 = jit_ld_forward(10.0, simple_state)
    assert jnp.all(forw_10.params == jit_forw_10.params)
    assert jnp.all(forw_10.velocities == jit_forw_10.velocities)
    assert jnp.all(forw_10.velocities == simple_state.velocities)
    assert jnp.all(
        forw_10.params == simple_state.params + 10.0 * simple_state.velocities
    )

    tree_x = {"a": jnp.zeros((2,)), "b": {"c": jnp.zeros((1, 2, 3, 4))}, "d": 0.0}
    tree_v = {"a": jnp.array([1.0, 0.1]), "b": {"c": jnp.ones((1, 2, 3, 4))}, "d": 0.0}
    tree_state = BPSState(tree_x, tree_v)
    forw_10 = ld_forward(10.0, tree_state)
    jit_forw_10 = jit_ld_forward(10.0, tree_state)
    assert tree_allclose(forw_10.params, jit_forw_10.params)
    assert tree_allclose(forw_10.velocities, jit_forw_10.velocities)
    assert tree_allclose(forw_10.velocities, tree_state.velocities)
    assert jnp.all(tree_x["a"] + 10.0 * tree_v["a"] == forw_10.params["a"])
    assert jnp.all(
        tree_x["b"]["c"] + 10.0 * tree_v["b"]["c"] == forw_10.params["b"]["c"]
    )
    assert jnp.all(tree_x["d"] + 10.0 * tree_v["d"] == forw_10.params["d"])


# def test_linear_dynamics_jvp():
#     simple_state = BPSState(jnp.zeros((2,)), jnp.array([1.0, 0.1]))
