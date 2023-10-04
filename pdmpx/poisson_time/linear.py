import numpy as np
import numba
import jax
import jax.numpy as jnp


@numba.jit(nopython=True)
def ab_poisson_time(u: float, a: float, b: float) -> float:  # TODO: linear_poisson_time
    """
    First arrival time of Poisson process with time-dependent rate `rate = (a + b*t)+`
    """
    if b < 0:
        if a <= 0:
            return np.inf
        else:
            y = -np.log(u)  # reference time
            if -a / b < y:
                return (-a - np.sqrt(2 * b * y + a**2)) / b  # solve
            else:
                return np.inf
    elif b == 0:
        if a <= 0:
            return np.inf
        else:
            return -np.log(u) / a
    else:  # b > 0
        if a < 0:
            y = -np.log(u)
            return (-a + np.sqrt(2 * b * y)) / b
        else:
            y = -np.log(u)
            return (-a + np.sqrt(2 * b * y + a**2)) / b


linear_poisson_time = ab_poisson_time


@jax.jit
def jax_ab_poisson_time(u: float, a: float, b: float) -> float:
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
