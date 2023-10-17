import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import functools as ft


@ft.partial(jax.jit, static_argnums=(1,))
def split(dct, keys):
    """Partition a dict by keys.

    Args:
        tree: A pytree to partition.
        keys: A list of keys to partition the pytree by.

    Returns:
        A list of pytrees, each with the same structure as `tree`, but with
        subtrees corresponding to the keys in `keys`.
    """
    left = {k: v for k, v in dct.items() if k in keys}
    right = {k: v for k, v in dct.items() if k not in keys}
    return left, right


@jax.jit
def merge(left, right):
    return {**left, **right}


def tree_dot(tree1, tree2):
    return jtu.tree_reduce(
        lambda x, y: x + y,
        jtu.tree_map(lambda x, y: jnp.sum(x * y), tree1, tree2),
    )


def tree_project(tree, onto):
    prod = tree_dot(tree, onto)
    normsq = tree_dot(onto, onto)
    return jtu.tree_map(lambda x, y: x - prod / normsq * y, tree, onto)


def tree_add_scaled(tree1, tree2, scale1=1.0, scale2=1.0):
    return jtu.tree_map(lambda x, y: scale1 * x + scale2 * y, tree1, tree2)


def tree_unit_length(tree):
    tree_length = jnp.sqrt(tree_dot(tree, tree))
    return jtu.tree_map(lambda x: x / tree_length, tree)


@jax.jit
def tree_random_normal(rng, tree, scale=1.0):
    leaves, tree_def = jtu.tree_flatten(tree)
    keys = jax.random.split(rng, len(leaves))
    return jtu.tree_unflatten(
        tree_def, [scale * jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)]
    )
