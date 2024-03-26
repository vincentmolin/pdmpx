import jax
import jax.numpy as jnp


def ab_poisson_time(u: float, a: float, b: float) -> float:
    """
    First arrival time of Poisson process with time-dependent rate `rate = (a + b*t)+`
    """
    y = -jnp.log(1.0 - u)
    return jax.lax.select(
        b < 0,
        jax.lax.select(
            a <= 0,
            jnp.inf,
            jax.lax.select(
                y < -(a**2) / (2 * b),
                -a / b - jnp.sqrt(2 * y / b + a**2 / b**2),
                jnp.inf,
            ),
        ),
        jax.lax.select(
            b == 0,
            jax.lax.select(a <= 0, jnp.inf, y / a),
            jax.lax.select(
                a < 0,
                -a / b + jnp.sqrt(2 * y / b),
                -a / b + jnp.sqrt(2 * y / b + a**2 / b**2),
            ),
        ),
    )
