import jax
import jax.numpy as jnp
import tree_math as tm
from pdmpx.utils.tree import tree_dot, tree_add_scaled


def test_dot():
    x = {"a": jnp.ones((2, 3)), "b": jnp.ones((3, 4))}
    y = {"a": jnp.ones((2, 3)), "b": -jnp.ones((3, 4))}

    assert tree_dot(x, y) == tm.Vector(x) @ tm.Vector(y)
    assert tree_dot(jnp.eye(5), jnp.eye(5)) == 5.0
