from pdmpx.utils.dir_derivs import nth_dir_deriv
import jax
import jax.numpy as jnp


def test_nth_dir_deriv():
    def f(x):
        return jnp.sum(x**2)

    x = 1.0

    fx, dfdx, ddfddx, dddfdddx = (1.0, 2.0, 2.0, 0.0)

    dfs = nth_dir_deriv(lambda t: f(x + t))(0.0, 1.0, 3)

    assert jnp.all(jnp.array(dfs) == jnp.array((fx, dfdx, ddfddx, dddfdddx)))

    x = jnp.array([1.0, 1.0])
    v = jnp.array([1.0, -1.0])

    assert jax.grad(f)(x) @ v == nth_dir_deriv(f, only_n=True)(x, v, 1)
