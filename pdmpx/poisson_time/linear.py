import jax
import jax.numpy as jnp


@jax.jit
def ab_poisson_time(u: float, a: float, b: float) -> float:
    """
    First arrival time of Poisson process with time-dependent rate `rate = (a + b*t)+`
    """
    return jax.lax.select(
        b < 0,
        jax.lax.select(
            a <= 0,
            jnp.inf,
            jax.lax.select(
                -a / b < -jnp.log(u),
                (-a - jnp.sqrt(2 * b * -jnp.log(u) + a**2)) / b,
                jnp.inf,
            ),
        ),
        jax.lax.select(
            b == 0,
            jax.lax.select(a <= 0, jnp.inf, -jnp.log(u) / a),
            jax.lax.select(
                a < 0,
                (-a + jnp.sqrt(2 * b * -jnp.log(u))) / b,
                (-a + jnp.sqrt(2 * b * -jnp.log(u) + a**2)) / b,
            ),
        ),
    )
