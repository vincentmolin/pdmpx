import jax
import jax.numpy as jnp
import numpy as np
import numba
import functools as ft


def exp_rand(key, rate=1.0):
    """
    Exponential random variable
    """
    return -jnp.log(jax.random.uniform(key)) / rate


def poisson_time(key, rate=1.0):
    """
    First arrival time of Poisson process with constant rate `rate`
    """
    return exp_rand(key, rate)


def poisson_time_stepf(key, rates, ts):
    """
    First arrival time of Poisson process with time-varying piecewise constant rate `rates`.
    """
    # reference time
    x = exp_rand(key)

    dts = jnp.diff(ts)

    return ts + exp_rand(key, rates)


@numba.jit(nopython=True)
def trapezoid(x, y):
    return 0.5 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))


@numba.jit(nopython=True)
def trapezoid_cum(x, y):
    return 0.5 * np.cumsum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))


class TrapezoidTime:
    def __call__(self, ts, rates):
        ref_time = np.random.exponential()
        raise NotImplementedError


# function ab_exp_rand(a, b)
#     # Samples an exponential rv with
#     # rate l_t = a + bt
#     if (a <= 0) & (b <= 0)
#         return Inf
#     end
#     u = exp_rand()
#     if b > 0.0
#         return (sqrt(2*b*u + a^2)-a) / b
#     else
#         return u / a
#     end
# end


@numba.jit(nopython=True)
def ab_poisson_time(u: float, a: float, b: float) -> float:
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


# u = 0.5
# ab_poisson_time(u, 1.0, 1.0)
# jax_ab_poisson_time(u, 1.0, 1.0)
# ab_poisson_time(u, 1.0, 0.0) == jax_ab_poisson_time(u, 1.0, 0.0)
# ab_poisson_time(u, 0.0, 1.0)
# ab_poisson_time(u, 0.0, 0.0)
